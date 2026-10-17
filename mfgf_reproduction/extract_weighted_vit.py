import os
import sys
import torch
import torchvision.models as models
from torch.utils.data import DataLoader
from tqdm import tqdm

os.environ['TORCH_HOME'] = 'E:/BTP/.cache/torch'

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from src.dataset.tvpreid import TVPReIDDataset
from src.models.quality_estimator import FrameQualityEstimator

def extract_weighted_features(split, dataset_name):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    data_dir = f"../data/TVPReID/TVPReid-{dataset_name}"
    dataset = TVPReIDDataset(data_dir, split=split, num_frames=8)
    loader = DataLoader(dataset, batch_size=1, shuffle=False, num_workers=0)
    
    mean_features = {}
    weighted_features = {}
    print("\n--- Extracting Quality-Weighted and Mean-Pooled Visual Features (ViT) ---")
    vit = models.vit_b_16(pretrained=True).to(device)
    vit.eval()
    
    q_estimator = FrameQualityEstimator().to(device)
    q_estimator.eval()
    
    with torch.no_grad():
        for batch in tqdm(loader, desc=f"ViT Weighted ({split})"):
            frames = batch['frames'].to(device) # (B, T, C, H, W)
            vid_ids = batch['video_id']
            
            B, T, C, H, W = frames.shape
            
            # 1. Get Quality Weights q_j
            q_j = q_estimator(frames) # (B, T)
            q_weights = q_j / (q_j.sum(dim=1, keepdim=True) + 1e-6) # (B, T)
            
            # 2. Extract ViT features per frame
            frames_flat = frames.view(B * T, C, H, W)
            visual_feat = vit._process_input(frames_flat)
            n = visual_feat.shape[0]
            batch_class_token = vit.class_token.expand(n, -1, -1)
            visual_feat = torch.cat([batch_class_token, visual_feat], dim=1)
            visual_feat = vit.encoder(visual_feat)
            visual_feat = visual_feat[:, 0] # (B*T, 768)
            
            # 3. Apply Both Pooling Strategies
            visual_feat = visual_feat.view(B, T, -1) # (B, T, 768)
            
            mean_visual_feat = visual_feat.mean(dim=1) # (B, 768)
            weighted_visual_feat = (visual_feat * q_weights.unsqueeze(-1)).sum(dim=1) # (B, 768)
            
            # Update the dictionary
            for i, vid in enumerate(vid_ids):
                mean_features[vid] = mean_visual_feat[i].cpu()
                weighted_features[vid] = weighted_visual_feat[i].cpu()
            
            del frames, frames_flat, visual_feat, mean_visual_feat, weighted_visual_feat
            torch.cuda.empty_cache()
                
    # Save the features
    out_dir = f"data/{dataset_name}_features_ablation/{split}"
    os.makedirs(out_dir, exist_ok=True)
    torch.save(mean_features, os.path.join(out_dir, 'mean_visual_features.pt'))
    torch.save(weighted_features, os.path.join(out_dir, 'weighted_visual_features.pt'))
    print(f"Successfully saved {len(mean_features)} pairs of items to {out_dir}")

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, required=True, choices=['PRID', 'iLIDs', 'Duke'])
    args = parser.parse_args()
    extract_weighted_features('train', args.dataset)
    extract_weighted_features('test', args.dataset)
