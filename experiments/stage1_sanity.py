import os
import argparse
import yaml
import json
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import transforms
from tqdm import tqdm
import random
import numpy as np
from PIL import Image

from src.dataset.mevid import MEVIDDataset
from transformers import CLIPVisionModel, CLIPImageProcessor

def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True

def compute_metrics(features, pids):
    """
    Computes Rank-1, 5, 10 and mAP.
    Assumes cross-camera evaluation where every sample is evaluated against all others
    (excluding self). This is a simplified metric calculation for the sanity check.
    """
    features = features / features.norm(dim=-1, keepdim=True)
    sim_matrix = torch.matmul(features, features.T)
    
    # Mask out self-similarity
    sim_matrix.fill_diagonal_(-1.0)
    
    indices = torch.argsort(sim_matrix, dim=-1, descending=True)
    
    mAP = 0.0
    cmc = torch.zeros(len(pids), len(pids))
    
    for i in range(len(pids)):
        query_pid = pids[i]
        gallery_pids = pids[indices[i]]
        
        matches = (gallery_pids == query_pid).float()
        
        # CMC
        if matches.sum() > 0:
            first_match = torch.argmax(matches).item()
            cmc[i, first_match:] = 1.0
            
            # AP
            precision = torch.cumsum(matches, dim=0) / torch.arange(1, len(matches) + 1).float()
            ap = (precision * matches).sum() / matches.sum()
            mAP += ap.item()
            
    mAP = mAP / len(pids) if len(pids) > 0 else 0.0
    rank1 = cmc[:, 0].mean().item() if len(pids) > 0 else 0.0
    rank5 = cmc[:, min(4, len(pids)-1)].mean().item() if len(pids) > 0 else 0.0
    rank10 = cmc[:, min(9, len(pids)-1)].mean().item() if len(pids) > 0 else 0.0
    
    return {'Rank-1': rank1, 'Rank-5': rank5, 'Rank-10': rank10, 'mAP': mAP}, indices, sim_matrix

def visualize_retrieval(query_idx, indices, dataset, output_path, top_k=5):
    """
    Generates a basic visualization grid showing the query tracklet and top-K retrieved tracklets.
    Only visualizes the first frame of each tracklet for simplicity.
    """
    query_info = dataset.tracklets[query_idx]
    
    retrieved_infos = [dataset.tracklets[indices[query_idx][i]] for i in range(top_k)]
    
    try:
        query_img = Image.open(query_info['frames'][0]).convert('RGB').resize((224, 224))
        retrieved_imgs = [Image.open(info['frames'][0]).convert('RGB').resize((224, 224)) for info in retrieved_infos]
        
        # Concatenate horizontally
        total_width = 224 * (1 + top_k)
        grid_img = Image.new('RGB', (total_width, 224))
        
        grid_img.paste(query_img, (0, 0))
        for i, img in enumerate(retrieved_imgs):
            grid_img.paste(img, (224 * (i + 1), 0))
            
        grid_img.save(output_path)
    except Exception as e:
        print(f"Visualization error (expected if frames missing): {e}")

def main(config):
    set_seed(config.get('seed', 42))
    
    # Setup exp dir
    exp_dir = config.get('exp_dir', 'experiments/stage1')
    os.makedirs(exp_dir, exist_ok=True)
    
    # Save config
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
    
    dataset = MEVIDDataset(config['data_dir'], transform=transform, split='test', num_frames=config['num_frames'])
    dataloader = DataLoader(dataset, batch_size=config['batch_size'], shuffle=False, num_workers=0)
    
    if len(dataset) == 0:
        print("Dataset is empty. Run dummy_data_gen.py first.")
        return
        
    print(f"Loaded {len(dataset)} tracklets (SOFTWARE VALIDATION MODE).")
    
    # 2. Minimal Visual-only Re-ID Baseline
    print("Loading CLIP Vision Model...")
    model_name = config.get('model_name', 'openai/clip-vit-base-patch32')
    vision_model = CLIPVisionModel.from_pretrained(model_name).to(device)
    vision_model.eval()
    
    all_features = []
    all_pids = []
    
    print("Extracting features (Sanity check on dummy data)...")
    with torch.no_grad():
        for batch in tqdm(dataloader):
            frames = batch['frames'].to(device) # (B, T, C, H, W)
            pids = batch['pid']
            
            B, T, C, H, W = frames.shape
            frames = frames.view(B * T, C, H, W)
            
            outputs = vision_model(pixel_values=frames)
            frame_embeddings = outputs.pooler_output # (B*T, D)
            
            frame_embeddings = frame_embeddings.view(B, T, -1)
            
            # Temporal Aggregation: Mean Pooling
            tracklet_embeddings = frame_embeddings.mean(dim=1) # (B, D)
            
            all_features.append(tracklet_embeddings.cpu())
            all_pids.extend(pids)
            
    all_features = torch.cat(all_features, dim=0)
    
    unique_pids = {pid: i for i, pid in enumerate(set(all_pids))}
    pid_labels = torch.tensor([unique_pids[pid] for pid in all_pids])
    
    print("Computing metrics...")
    metrics, indices, sim_matrix = compute_metrics(all_features, pid_labels)
    
    # Checkpoint saving
    checkpoint = {
        'features': all_features,
        'pids': all_pids,
        'metrics': metrics
    }
    torch.save(checkpoint, os.path.join(exp_dir, 'checkpoint.pt'))
    
    # Visualization
    vis_path = os.path.join(exp_dir, 'retrieval_vis.jpg')
    visualize_retrieval(query_idx=0, indices=indices, dataset=dataset, output_path=vis_path, top_k=min(5, len(dataset)-1))
    print(f"Saved retrieval visualization to {vis_path}")
    
    print("\n=== Stage 1 Sanity Check Results (SOFTWARE VALIDATION on Dummy Data) ===")
    print("NOTE: These results do NOT represent real MEVID performance. This is purely for pipeline validation.")
    for k, v in metrics.items():
        print(f"{k}: {v:.4f}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, default='configs/stage1.yaml')
    args = parser.parse_args()
    
    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)
        
    main(config)
