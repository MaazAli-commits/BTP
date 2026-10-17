import os
import json
import argparse
import numpy as np
import torch
from torch.utils.data import DataLoader
from train_unified_mfgf import UnifiedMFGFHead, UnifiedOfflineDataset

def calculate_metrics(similarity_matrix):
    num_queries, num_videos = similarity_matrix.shape
    assert num_queries == num_videos, f"Query ({num_queries}) and gallery ({num_videos}) size mismatch"
    
    ranks = np.zeros(num_queries)
    for i in range(num_queries):
        sorted_indices = np.argsort(similarity_matrix[i])[::-1]
        rank = np.where(sorted_indices == i)[0][0]
        ranks[i] = rank
        
    r1 = (ranks == 0).mean() * 100.0
    r5 = (ranks < 5).mean() * 100.0
    r10 = (ranks < 10).mean() * 100.0
    
    ap = 1.0 / (ranks + 1)
    mAP = ap.mean() * 100.0
    
    return {'R1': float(r1), 'R5': float(r5), 'R10': float(r10), 'mAP': float(mAP)}

def main():
    parser = argparse.ArgumentParser(description="Evaluate Unified MFGF on 820 test videos")
    parser.add_argument('--pool', type=str, choices=['baseline', 'mean', 'weighted'], required=True)
    parser.add_argument('--seed', type=int, required=True)
    parser.add_argument('--split', type=str, choices=['clean', 'corrupted'], default='clean',
                        help="clean = standard clean test set; corrupted = 50% Gaussian blur")
    args = parser.parse_args()
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print("=" * 65)
    print(f"Evaluating Unified MFGF ({args.split.upper()} data, {args.pool} pooling, seed={args.seed})")
    print("=" * 65)
    
    checkpoint_dir = f"../experiments/checkpoints/mfgf_unified_offline_{args.pool}_{args.seed}"
    checkpoint_path = os.path.join(checkpoint_dir, "offline_mfgf_final.pt")
    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")
        
    print("Loading test features...")
    test_features = torch.load('data/Unified_features/test/extracted_features.pt', map_location='cpu')
    
    if args.split == 'clean' and args.pool != 'baseline':
        vis_path = f'data/Unified_features_ablation/test/{args.pool}_visual_features.pt'
        if os.path.exists(vis_path):
            print(f"Applying clean {args.pool} pooled visual features...")
            vis_data = torch.load(vis_path, map_location='cpu')
            for vid, feat in vis_data.items():
                if vid in test_features:
                    test_features[vid]['visual'] = feat
        else:
            raise FileNotFoundError(f"Missing clean ablation features: {vis_path}")
    elif args.split == 'corrupted':
        vis_path = f'data/Unified_features_ablation/test/corrupted_{args.pool}.pt'
        if os.path.exists(vis_path):
            print(f"Applying corrupted (50% blur) {args.pool} visual features from {vis_path}...")
            vis_data = torch.load(vis_path, map_location='cpu')
            for vid, feat in vis_data.items():
                if vid in test_features:
                    test_features[vid]['visual'] = feat
        else:
            raise FileNotFoundError(f"Missing corrupted ablation features: {vis_path}")
            
    tips_dict = torch.load('data/Unified_tips.pt', map_location='cpu')
    
    clean_features = {k: v for k, v in test_features.items() if k in tips_dict}
    clean_tips = {k: tips_dict[k] for k in clean_features.keys()}
    
    print(f"Evaluation gallery size: {len(clean_features)} test videos")
    
    test_dataset = UnifiedOfflineDataset(clean_features, clean_tips)
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)
    
    d_tips = next(iter(clean_tips.values())).shape[0]
    model = UnifiedMFGFHead(d_tips=d_tips, embed_dim=256).to(device)
    model.load_state_dict(torch.load(checkpoint_path, map_location=device))
    model.eval()
    
    all_vid_embeds = []
    all_txt_embeds = []
    
    with torch.no_grad():
        for batch in test_loader:
            vis = batch['visual'].to(device)
            mot = batch['motion'].to(device)
            txt = batch['text'].to(device)
            
            c_vid, c_txt, _, _, _ = model(vis, mot, txt)
            all_vid_embeds.append(c_vid.cpu())
            all_txt_embeds.append(c_txt.cpu())
            
    all_vid_embeds = torch.cat(all_vid_embeds, dim=0) # (820, 256)
    all_txt_embeds = torch.cat(all_txt_embeds, dim=0) # (820, 256)
    
    similarity_matrix = (all_txt_embeds @ all_vid_embeds.t()).numpy()
    metrics = calculate_metrics(similarity_matrix)
    
    print(f"\n--- Unified MFGF Evaluation Results ({args.split.upper()} | {args.pool.upper()} | Seed {args.seed}) ---")
    print(f"Rank-1:  {metrics['R1']:.2f}%")
    print(f"Rank-5:  {metrics['R5']:.2f}%")
    print(f"Rank-10: {metrics['R10']:.2f}%")
    print(f"mAP:     {metrics['mAP']:.2f}%")
    
    res_dir = "../experiments/results"
    os.makedirs(res_dir, exist_ok=True)
    out_filename = f"Unified_{args.split}_{args.pool}_{args.seed}.json"
    out_path = os.path.join(res_dir, out_filename)
    with open(out_path, 'w') as f:
        json.dump(metrics, f, indent=2)
    print(f"Saved results to: {out_path}")
    return metrics

if __name__ == '__main__':
    main()
