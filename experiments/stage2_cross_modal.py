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
from transformers import CLIPModel, CLIPProcessor

def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True

def generate_query_sentence(attributes):
    gender = attributes.get('gender', 'person')
    top = attributes.get('top_color', 'clothes')
    bottom = attributes.get('bottom_color', 'clothes')
    return f"A {gender} wearing a {top} top and {bottom} bottom."

def compute_metrics(query_features, gallery_features, query_pids, gallery_pids):
    """
    Computes Rank-1, 5, 10 and mAP for text-to-video retrieval.
    query_features: (Num_Queries, D)
    gallery_features: (Num_Gallery, D)
    """
    query_features = query_features / query_features.norm(dim=-1, keepdim=True)
    gallery_features = gallery_features / gallery_features.norm(dim=-1, keepdim=True)
    
    sim_matrix = torch.matmul(query_features, gallery_features.T) # (Num_Queries, Num_Gallery)
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
    
    return {'Rank-1': rank1, 'Rank-5': rank5, 'Rank-10': rank10, 'mAP': mAP}, indices, sim_matrix

def main(config):
    set_seed(config.get('seed', 42))
    
    exp_dir = config.get('exp_dir', 'experiments/stage2')
    os.makedirs(exp_dir, exist_ok=True)
    
    with open(os.path.join(exp_dir, 'config.yaml'), 'w') as f:
        yaml.dump(config, f)
        
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # 1. Dataset Loaders
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize((0.48145466, 0.4578275, 0.40821073), (0.26862954, 0.26130258, 0.27577711))
    ])
    
    gallery_dataset = MEVIDDataset(config['data_dir'], transform=transform, split='test', role='gallery', num_frames=config['num_frames'])
    gallery_loader = DataLoader(gallery_dataset, batch_size=config['batch_size'], shuffle=False, num_workers=0)
    
    if len(gallery_dataset) == 0:
        print("Dataset is empty. Run dummy data generator first.")
        return
        
    print(f"Loaded {len(gallery_dataset)} gallery tracklets.")
    
    # 2. Load Attributes
    annotations_path = os.path.join(config['data_dir'], 'annotations.json')
    if os.path.exists(annotations_path):
        with open(annotations_path, 'r') as f:
            attributes_dict = json.load(f)
    else:
        attributes_dict = {}
        
    # We define queries based on unique PIDs in the test set.
    # In a real scenario, the query is the text description.
    # We will generate one query per identity in the gallery.
    gallery_pids_raw = [item['pid'] for item in gallery_dataset.tracklets]
    unique_pids = sorted(list(set(gallery_pids_raw)))
    
    queries = []
    query_pids = []
    for pid in unique_pids:
        attr = attributes_dict.get(pid, {})
        sentence = generate_query_sentence(attr)
        queries.append(sentence)
        query_pids.append(pid)
        print(f"Query for {pid}: {sentence}")
        
    # 3. Load CLIP Model
    print("Loading CLIP Model (Text and Vision)...")
    model_name = config.get('model_name', 'openai/clip-vit-base-patch32')
    clip_model = CLIPModel.from_pretrained(model_name).to(device)
    clip_processor = CLIPProcessor.from_pretrained(model_name)
    clip_model.eval()
    
    # 4. Extract Text Features
    print("Extracting Text Features...")
    text_inputs = clip_processor(text=queries, return_tensors="pt", padding=True, truncation=True).to(device)
    with torch.no_grad():
        text_outputs = clip_model.text_model(**text_inputs)
        text_features = clip_model.text_projection(text_outputs.pooler_output) # (Num_Queries, D)
        
    # 5. Extract Gallery Visual Features
    print("Extracting Gallery Visual Features...")
    gallery_features = []
    g_pids = []
    
    with torch.no_grad():
        for batch in tqdm(gallery_loader):
            frames = batch['frames'].to(device) # (B, T, C, H, W)
            pids = batch['pid']
            
            B, T, C, H, W = frames.shape
            frames = frames.view(B * T, C, H, W)
            
            # Extract features using vision encoder directly
            outputs = clip_model.vision_model(pixel_values=frames)
            frame_embeddings = outputs.pooler_output # (B*T, D)
            
            # Project vision features to the shared multimodal space
            frame_embeddings = clip_model.visual_projection(frame_embeddings)
            
            frame_embeddings = frame_embeddings.view(B, T, -1)
            
            # Temporal Aggregation: Mean Pooling
            tracklet_embeddings = frame_embeddings.mean(dim=1) # (B, D)
            
            gallery_features.append(tracklet_embeddings.cpu())
            g_pids.extend(pids)
            
    gallery_features = torch.cat(gallery_features, dim=0)
    
    # Map PIDs to integers
    pid_to_idx = {pid: i for i, pid in enumerate(unique_pids)}
    q_labels = torch.tensor([pid_to_idx[pid] for pid in query_pids])
    g_labels = torch.tensor([pid_to_idx[pid] for pid in g_pids])
    
    print("Computing metrics...")
    metrics, indices, sim_matrix = compute_metrics(text_features.cpu(), gallery_features, q_labels, g_labels)
    
    checkpoint = {
        'text_features': text_features.cpu(),
        'gallery_features': gallery_features,
        'q_labels': q_labels,
        'g_labels': g_labels,
        'metrics': metrics
    }
    torch.save(checkpoint, os.path.join(exp_dir, 'checkpoint.pt'))
    
    print("\n=== Stage 2 Cross-Modal (Text->Video) Results (SOFTWARE VALIDATION) ===")
    for k, v in metrics.items():
        print(f"{k}: {v:.4f}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, default='configs/stage2.yaml')
    args = parser.parse_args()
    
    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)
        
    main(config)
