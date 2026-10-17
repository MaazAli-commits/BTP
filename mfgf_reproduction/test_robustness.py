import os
import sys
import torch
import argparse
import torch.nn.functional as F
import torchvision.models as models
import torchvision.transforms.functional as TF
from torch.utils.data import DataLoader
from tqdm import tqdm
import json
import numpy as np

os.environ['TORCH_HOME'] = 'E:/BTP/.cache/torch'

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from src.dataset.tvpreid import TVPReIDDataset
from src.models.quality_estimator import FrameQualityEstimator
from evaluate_mfgf_offline import calculate_metrics, MFGFHead, OfflineMFGFDataset

def run_robustness_test(seed=42, dataset_name='PRID'):
    device = torch.device('cpu')
    mean_feat_path = f'data/{dataset_name}_features_ablation/test/corrupted_mean.pt'
    weighted_feat_path = f'data/{dataset_name}_features_ablation/test/corrupted_weighted.pt'
    
    mean_features = {}
    weighted_features = {}
    
    q_clean_total = []
    q_blur_total = []
    if os.path.exists(mean_feat_path) and os.path.exists(weighted_feat_path):
        mean_features = torch.load(mean_feat_path)
        weighted_features = torch.load(weighted_feat_path)
        print("Loaded cached corrupted features.")
    else:
        extract_device = torch.device('cpu') # FORCE CPU to avoid fragmentation
        print(f"\n--- Running Robustness Test on Test Split ---")
    
        # 1. Load Data
        data_dir = f"../data/TVPReID/TVPReid-{dataset_name}"
        dataset = TVPReIDDataset(data_dir, split='test', num_frames=8)
        loader = DataLoader(dataset, batch_size=1, shuffle=False, num_workers=0)
        
        # 2. Models
        vit = models.vit_b_16(pretrained=True).to(extract_device)
        vit.eval()
        q_estimator = FrameQualityEstimator().to(extract_device)
        q_estimator.eval()
        
        mean_features = {}
        weighted_features = {}
        
        blur_amount = [15, 25]  # kernel sizes for blur
        
        with torch.no_grad():
            for batch in tqdm(loader, desc="Extracting corrupted features"):
                frames = batch['frames'].to(extract_device) # (B, T, C, H, W)
                vid_ids = batch['video_id']
                
                B, T, C, H, W = frames.shape
                
                # Corrupt half the frames with severe blur
                # Frames 4, 5, 6, 7 are blurred
                corrupted_frames = frames.clone()
                del frames
                half_t = T // 2
                for b in range(B):
                    for t in range(half_t, T):
                        # apply gaussian blur
                        corrupted_frames[b, t] = TF.gaussian_blur(corrupted_frames[b, t], kernel_size=[15, 15], sigma=[5.0, 5.0])
                
                # Get Quality Weights q_j on corrupted frames
                q_j = q_estimator(corrupted_frames) # (B, T)
                
                q_clean = q_j[:, :half_t].mean().item()
                q_blur = q_j[:, half_t:].mean().item()
                q_clean_total.append(q_clean)
                q_blur_total.append(q_blur)
                
                q_weights = q_j / (q_j.sum(dim=1, keepdim=True) + 1e-6) # (B, T)
                
                # Extract ViT features per frame
                frames_flat = corrupted_frames.view(B * T, C, H, W)
                visual_feat = vit._process_input(frames_flat)
                n = visual_feat.shape[0]
                batch_class_token = vit.class_token.expand(n, -1, -1)
                visual_feat = torch.cat([batch_class_token, visual_feat], dim=1)
                visual_feat = vit.encoder(visual_feat)
                visual_feat = visual_feat[:, 0] # (B*T, 768)
                
                # Apply Both Pooling Strategies
                visual_feat = visual_feat.view(B, T, -1) # (B, T, 768)
                mean_visual_feat = visual_feat.mean(dim=1) # (B, 768)
                weighted_visual_feat = (visual_feat * q_weights.unsqueeze(-1)).sum(dim=1) # (B, 768)
                
                # Save
                for i, vid in enumerate(vid_ids):
                    mean_features[vid] = mean_visual_feat[i].cpu()
                    weighted_features[vid] = weighted_visual_feat[i].cpu()
                    
                del corrupted_frames, frames_flat, visual_feat, mean_visual_feat, weighted_visual_feat
                
            torch.save(mean_features, mean_feat_path)
            torch.save(weighted_features, weighted_feat_path)
        
    if len(q_clean_total) > 0:
        print(f"\nAverage q_j for CLEAN frames: {np.mean(q_clean_total):.4f}")
        print(f"Average q_j for BLURRED frames: {np.mean(q_blur_total):.4f}")
    
    # 3. Evaluate the Mean model
    print("\nEvaluating MEAN Model on Blurred Data...")
    mean_ckpt = f"../experiments/checkpoints/mfgf_{dataset_name}_offline_mean_{seed}/offline_mfgf_final.pt"
    metrics_mean = evaluate_robustness(mean_features, mean_ckpt, device, dataset_name)
    
    # 4. Evaluate the Weighted model
    print("\nEvaluating WEIGHTED Model on Blurred Data...")
    weighted_ckpt = f"../experiments/checkpoints/mfgf_{dataset_name}_offline_weighted_{seed}/offline_mfgf_final.pt"
    metrics_weighted = evaluate_robustness(weighted_features, weighted_ckpt, device, dataset_name)
    return metrics_mean, metrics_weighted
    
def evaluate_robustness(ablated_vis, checkpoint_path, device, dataset_name):
    print("Loading base test features...")
    test_features = torch.load(f'data/{dataset_name}_features/test/extracted_features.pt')
    tips_dict = torch.load(f'data/{dataset_name}_tips.pt')
    
    # Override visual features
    for vid, vis_feat in ablated_vis.items():
        if vid in test_features:
            test_features[vid]['visual'] = vis_feat
            
    # Normalize keys
    clean_features = {k.replace('.mp4', ''): v for k, v in test_features.items()}
    clean_tips = {k.replace('.mp4', ''): v for k, v in tips_dict.items()}
    
    test_dataset = OfflineMFGFDataset(clean_features, clean_tips)
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)
    
    d_tips = next(iter(clean_tips.values())).shape[0]
    model = MFGFHead(d_tips=d_tips, embed_dim=256).to(device)
    model.load_state_dict(torch.load(checkpoint_path, map_location=device))
    model.eval()
    
    vid_features = []
    txt_features = []
    
    with torch.no_grad():
        for batch in test_loader:
            vis = batch['visual'].to(device)
            mot = batch['motion'].to(device)
            txt = batch['text'].to(device)
            
            com_vid, com_txt, _, _, _ = model(vis, mot, txt)
            
            vid_features.append(com_vid.cpu())
            txt_features.append(com_txt.cpu())
            
    vid_features = torch.cat(vid_features, dim=0)
    txt_features = torch.cat(txt_features, dim=0)
    
    similarity_matrix = torch.matmul(txt_features, vid_features.t()).numpy()
    metrics = calculate_metrics(similarity_matrix)
    
    print(f"Rank-1:  {metrics['R1']:.2f}%")
    print(f"Rank-5:  {metrics['R5']:.2f}%")
    print(f"Rank-10: {metrics['R10']:.2f}%")
    print(f"mAP:     {metrics['mAP']:.2f}%")
    return metrics

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--dataset', type=str, required=True, choices=['PRID', 'iLIDs', 'Duke'])
    args = parser.parse_args()
    run_robustness_test(seed=args.seed, dataset_name=args.dataset)
