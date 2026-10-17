import os
import argparse
import yaml
import json
import torch
from torch.utils.data import DataLoader
from torchvision import transforms
from tqdm import tqdm
import random
import numpy as np

from src.dataset.mevid import MEVIDDataset
from transformers import CLIPModel, CLIPProcessor
from src.models.vclip_adaptation import VCLIPVideoEncoder
from experiments.stage2_cross_modal import compute_metrics, generate_query_sentence, set_seed

def main(config):
    set_seed(config.get('seed', 42))
    
    exp_dir = config.get('exp_dir', 'experiments/stage3')
    os.makedirs(exp_dir, exist_ok=True)
    
    with open(os.path.join(exp_dir, 'config.yaml'), 'w') as f:
        yaml.dump(config, f)
        
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # 1. Dataset Loader
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize((0.48145466, 0.4578275, 0.40821073), (0.26862954, 0.26130258, 0.27577711))
    ])
    
    gallery_dataset = MEVIDDataset(config['data_dir'], transform=transform, split='test', role='gallery', num_frames=config['num_frames'])
    gallery_loader = DataLoader(gallery_dataset, batch_size=config['batch_size'], shuffle=False, num_workers=0)
    
    # 2. Text Queries
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
        
    # 3. Models
    print("Loading CLIP Model and VCLIP Video Encoder...")
    model_name = config.get('model_name', 'openai/clip-vit-base-patch32')
    clip_model = CLIPModel.from_pretrained(model_name).to(device)
    clip_processor = CLIPProcessor.from_pretrained(model_name)
    
    # Initialize the query-independent video encoder
    vclip_encoder = VCLIPVideoEncoder(clip_model, num_attributes=41).to(device)
    vclip_encoder.eval()
    clip_model.eval()
    
    # 4. Extract Text Features
    print("Extracting External Query Text Features...")
    text_inputs = clip_processor(text=queries, return_tensors="pt", padding=True, truncation=True).to(device)
    with torch.no_grad():
        text_outputs = clip_model.text_model(**text_inputs)
        text_features = clip_model.text_projection(text_outputs.pooler_output)
        
    # 5. Extract Gallery Visual Features (Query-Independent)
    print("Extracting Gallery Video Features (via VCLIP Co-Attention)...")
    gallery_features = []
    g_pids = []
    
    with torch.no_grad():
        for batch in tqdm(gallery_loader):
            frames = batch['frames'].to(device) # (B, T, C, H, W)
            pids = batch['pid']
            
            # Forward pass entirely isolated from the text queries
            f_video = vclip_encoder(frames)
            
            gallery_features.append(f_video.cpu())
            g_pids.extend(pids)
            
    gallery_features = torch.cat(gallery_features, dim=0)
    
    # Map PIDs
    pid_to_idx = {pid: i for i, pid in enumerate(unique_pids)}
    q_labels = torch.tensor([pid_to_idx[pid] for pid in query_pids])
    g_labels = torch.tensor([pid_to_idx[pid] for pid in g_pids])
    
    print("Computing Cross-Modal Metrics...")
    metrics, indices, sim_matrix = compute_metrics(text_features.cpu(), gallery_features, q_labels, g_labels)
    
    checkpoint = {
        'text_features': text_features.cpu(),
        'gallery_features': gallery_features,
        'q_labels': q_labels,
        'g_labels': g_labels,
        'metrics': metrics
    }
    torch.save(checkpoint, os.path.join(exp_dir, 'checkpoint.pt'))
    
    print("\n=== Stage 3 Cross-Modal (VCLIP-Faithful Baseline) Results ===")
    for k, v in metrics.items():
        print(f"{k}: {v:.4f}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, default='configs/stage3.yaml')
    args = parser.parse_args()
    
    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)
        
    main(config)
