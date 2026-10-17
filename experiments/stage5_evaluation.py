import os
import argparse
import yaml
import json
import torch
from torch.utils.data import DataLoader
from torchvision import transforms
from tqdm import tqdm

from src.dataset.mevid import MEVIDDataset
from src.dataset.corruption import VisualCorruptor, TextCorruptor, generate_robustness_query
from transformers import CLIPModel, CLIPProcessor
from src.models.vclip_adaptation import VCLIPVideoEncoder
from src.models.quality_estimator import FrameQualityEstimator
from src.models.reliability_matching import ReliabilityAwareMatching
from experiments.stage2_cross_modal import set_seed
from experiments.stage4_reliability import compute_metrics_score

def run_experiment(config, exp_name, use_degradation, use_reliability):
    print(f"\n{'='*50}\nRunning {exp_name}\nDegradation: {use_degradation} | Reliability Module: {use_reliability}\n{'='*50}")
    
    device = torch.device('cpu') # Smoke test default
    
    # Setup Corruptors
    visual_corruptor = VisualCorruptor(corruption_ratio=0.5, blur_radius=5.0) if use_degradation else None
    text_corruptor = TextCorruptor(drop_prob=0.7) if use_degradation else TextCorruptor(drop_prob=0.0)
    
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize((0.48145466, 0.4578275, 0.40821073), (0.26862954, 0.26130258, 0.27577711))
    ])
    
    gallery_dataset = MEVIDDataset(config['data_dir'], transform=transform, visual_corruptor=visual_corruptor, split='test', role='gallery', num_frames=config['num_frames'])
    gallery_loader = DataLoader(gallery_dataset, batch_size=config['batch_size'], shuffle=False, num_workers=0)
    
    # Setup Text Queries
    annotations_path = os.path.join(config['data_dir'], 'annotations.json')
    if os.path.exists(annotations_path):
        with open(annotations_path, 'r') as f:
            attributes_dict = json.load(f)
    else:
        attributes_dict = {}
        
    unique_pids = sorted(list(set([item['pid'] for item in gallery_dataset.tracklets])))
    queries = []
    query_pids = []
    for pid in unique_pids:
        attr = attributes_dict.get(pid, {})
        corrupted_attr = text_corruptor(attr)
        queries.append(generate_robustness_query(corrupted_attr))
        query_pids.append(pid)
        
    # Setup Models
    model_name = config.get('model_name', 'openai/clip-vit-base-patch32')
    clip_model = CLIPModel.from_pretrained(model_name).to(device)
    clip_processor = CLIPProcessor.from_pretrained(model_name)
    vclip_encoder = VCLIPVideoEncoder(clip_model, num_attributes=41).to(device)
    
    quality_estimator = FrameQualityEstimator().to(device)
    reliability_matcher = ReliabilityAwareMatching(mode=config.get('matching_mode', 'post_softmax')).to(device)
    
    vclip_encoder.eval()
    clip_model.eval()
    quality_estimator.eval()
    reliability_matcher.eval()
        
    # Extract Text Sequences
    text_inputs = clip_processor(text=queries, return_tensors="pt", padding=True, truncation=True).to(device)
    with torch.no_grad():
        text_outputs = clip_model.text_model(**text_inputs)
        text_seq = clip_model.text_projection(text_outputs.last_hidden_state) 
        c_i = text_inputs['attention_mask'].float()
            
    # Extract Video Sequences
    gallery_feats = []
    gallery_qualities = []
    g_pids = []
    
    with torch.no_grad():
        for batch in gallery_loader:
            frames = batch['frames'].to(device)
            pids = batch['pid']
            
            f_video_seq = vclip_encoder(frames, return_seq=True)
            q_j = quality_estimator(frames)
            
            gallery_feats.append(f_video_seq)
            gallery_qualities.append(q_j)
            g_pids.extend(pids)
            
    gallery_feats = torch.cat(gallery_feats, dim=0)
    gallery_qualities = torch.cat(gallery_qualities, dim=0)
        
    pid_to_idx = {pid: i for i, pid in enumerate(unique_pids)}
    q_labels = torch.tensor([pid_to_idx[pid] for pid in query_pids])
    g_labels = torch.tensor([pid_to_idx[pid] for pid in g_pids])
    
    with torch.no_grad():
        # Clean Ablation: Use exactly the same pipeline, just toggle apply_weights
        sim_matrix = reliability_matcher(text_seq, gallery_feats, c_i, gallery_qualities, apply_weights=use_reliability)
            
    metrics = compute_metrics_score(sim_matrix.cpu(), q_labels, g_labels)
    
    print(f"Results for {exp_name}:")
    for k, v in metrics.items():
        print(f"  {k}: {v:.4f}")
    return metrics

def main(config):
    set_seed(config.get('seed', 42))
    
    exp_dir = config.get('exp_dir', 'experiments/stage5')
    os.makedirs(exp_dir, exist_ok=True)
    with open(os.path.join(exp_dir, 'config.yaml'), 'w') as f:
        yaml.dump(config, f)
        
    # Experiment A: Clean Data, Baseline Math
    run_experiment(config, "Experiment A (Normal Baseline)", use_degradation=False, use_reliability=False)
    
    # Experiment B: Degraded Data, Baseline Math
    run_experiment(config, "Experiment B (Vulnerability Test)", use_degradation=True, use_reliability=False)
    
    # Experiment C: Degraded Data, Reliability Math
    run_experiment(config, "Experiment C (Zero-Shot Reliability)", use_degradation=True, use_reliability=True)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, default='configs/stage5.yaml')
    args = parser.parse_args()
    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)
    main(config)
