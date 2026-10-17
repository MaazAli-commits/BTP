import os
import argparse
import yaml
import json
import torch
from torch.utils.data import DataLoader
from torchvision import transforms
from tqdm import tqdm

from src.dataset.mevid import MEVIDDataset
from transformers import CLIPModel, CLIPProcessor
from src.models.vclip_adaptation import VCLIPVideoEncoder
from src.models.quality_estimator import FrameQualityEstimator
from src.models.reliability_matching import ReliabilityAwareMatching
from experiments.stage2_cross_modal import generate_query_sentence, set_seed

def compute_metrics_score(sim_matrix, query_pids, gallery_pids):
    indices = torch.argsort(sim_matrix, dim=-1, descending=True)
    
    mAP = 0.0
    cmc = torch.zeros(len(query_pids), len(gallery_pids))
    
    for i in range(len(query_pids)):
        q_pid = query_pids[i]
        g_pids_sorted = gallery_pids[indices[i]]
        matches = (g_pids_sorted == q_pid).float()
        
        if matches.sum() > 0:
            first_match = torch.argmax(matches).item()
            cmc[i, first_match:] = 1.0
            
            precision = torch.cumsum(matches, dim=0) / torch.arange(1, len(matches) + 1).float()
            ap = (precision * matches).sum() / matches.sum()
            mAP += ap.item()
            
    mAP = mAP / len(query_pids) if len(query_pids) > 0 else 0.0
    rank1 = cmc[:, 0].mean().item() if len(query_pids) > 0 else 0.0
    rank5 = cmc[:, min(4, len(gallery_pids)-1)].mean().item() if len(query_pids) > 0 else 0.0
    rank10 = cmc[:, min(9, len(gallery_pids)-1)].mean().item() if len(query_pids) > 0 else 0.0
    
    return {'Rank-1': rank1, 'Rank-5': rank5, 'Rank-10': rank10, 'mAP': mAP}

def main(config):
    set_seed(config.get('seed', 42))
    exp_dir = config.get('exp_dir', 'experiments/stage4')
    os.makedirs(exp_dir, exist_ok=True)
    
    with open(os.path.join(exp_dir, 'config.yaml'), 'w') as f:
        yaml.dump(config, f)
        
    # Force CPU to avoid CUDA OOM during rapid script execution
    device = torch.device('cpu')
    print(f"Using device: {device}")
    
    # 1. Dataset Loader
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize((0.48145466, 0.4578275, 0.40821073), (0.26862954, 0.26130258, 0.27577711))
    ])
    
    gallery_dataset = MEVIDDataset(config['data_dir'], transform=transform, split='test', role='gallery', num_frames=config['num_frames'])
    gallery_loader = DataLoader(gallery_dataset, batch_size=config['batch_size'], shuffle=False, num_workers=0)
    
    annotations_path = os.path.join(config['data_dir'], 'annotations.json')
    if os.path.exists(annotations_path):
        with open(annotations_path, 'r') as f:
            attributes_dict = json.load(f)
    else:
        attributes_dict = {}
        
    gallery_pids_raw = [item['pid'] for item in gallery_dataset.tracklets]
    unique_pids = sorted(list(set(gallery_pids_raw)))
    
    queries = []
    query_pids = []
    for pid in unique_pids:
        attr = attributes_dict.get(pid, {})
        queries.append(generate_query_sentence(attr))
        query_pids.append(pid)
        
    # 2. Models
    print("Loading Models...")
    model_name = config.get('model_name', 'openai/clip-vit-base-patch32')
    clip_model = CLIPModel.from_pretrained(model_name).to(device)
    clip_processor = CLIPProcessor.from_pretrained(model_name)
    vclip_encoder = VCLIPVideoEncoder(clip_model, num_attributes=41).to(device)
    quality_estimator = FrameQualityEstimator().to(device)
    
    matching_mode = config.get('matching_mode', 'post_softmax')
    reliability_matcher = ReliabilityAwareMatching(mode=matching_mode).to(device)
    
    vclip_encoder.eval()
    clip_model.eval()
    quality_estimator.eval()
    reliability_matcher.eval()
    
    # 3. Extract Text Sequences and Confidence
    print("Extracting External Query Text Sequences...")
    text_inputs = clip_processor(text=queries, return_tensors="pt", padding=True, truncation=True).to(device)
    with torch.no_grad():
        text_outputs = clip_model.text_model(**text_inputs)
        # Apply projection to all tokens to get (B_q, L_t, D)
        text_seq = clip_model.text_projection(text_outputs.last_hidden_state) 
    
    # Dummy confidence c_i (1.0 for actual tokens, 0.0 for padding)
    c_i = text_inputs['attention_mask'].float() # (B_q, L_t)
    
    # 4. Extract Gallery Video Sequences and Quality
    print("Extracting Gallery Video Sequences & Quality Scores...")
    gallery_seqs = []
    gallery_qualities = []
    g_pids = []
    
    with torch.no_grad():
        for batch in tqdm(gallery_loader):
            frames = batch['frames'].to(device)
            pids = batch['pid']
            
            # Query-Independent Video Embedding Sequence (B, T, D)
            f_video_seq = vclip_encoder(frames, return_seq=True)
            
            # Frame Quality (B, T)
            q_j = quality_estimator(frames)
            
            gallery_seqs.append(f_video_seq.cpu())
            gallery_qualities.append(q_j.cpu())
            g_pids.extend(pids)
            
    gallery_seqs = torch.cat(gallery_seqs, dim=0).to(device)
    gallery_qualities = torch.cat(gallery_qualities, dim=0).to(device)
    
    # 5. Joint Stage: Reliability-Aware Matching
    print(f"Computing Reliability-Aware Matching (Mode: {matching_mode})...")
    with torch.no_grad():
        sim_matrix = reliability_matcher(text_seq, gallery_seqs, c_i, gallery_qualities)
        
    pid_to_idx = {pid: i for i, pid in enumerate(unique_pids)}
    q_labels = torch.tensor([pid_to_idx[pid] for pid in query_pids])
    g_labels = torch.tensor([pid_to_idx[pid] for pid in g_pids])
    
    metrics = compute_metrics_score(sim_matrix.cpu(), q_labels, g_labels)
    
    print(f"\n=== Stage 4 Reliability-Aware Matching ({matching_mode}) Results ===")
    for k, v in metrics.items():
        print(f"{k}: {v:.4f}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, default='configs/stage4.yaml')
    args = parser.parse_args()
    
    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)
        
    main(config)
