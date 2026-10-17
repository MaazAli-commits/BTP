import os
import sys
import torch
from torch.utils.data import DataLoader
import numpy as np
import argparse
import json

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from train_mfgf_offline import MFGFHead, OfflineMFGFDataset

def calculate_metrics(similarity_matrix, num_descriptions=1):
    num_texts, num_videos = similarity_matrix.shape
    num_unique_videos = num_texts // num_descriptions
    assert num_unique_videos == num_videos, "Mismatch in similarity matrix dimensions"
    
    ranks = np.zeros(num_texts)
    for i in range(num_texts):
        correct_vid_idx = i // num_descriptions
        sorted_indices = np.argsort(similarity_matrix[i])[::-1]
        rank = np.where(sorted_indices == correct_vid_idx)[0][0]
        ranks[i] = rank
        
    r1 = (ranks == 0).mean() * 100
    r5 = (ranks < 5).mean() * 100
    r10 = (ranks < 10).mean() * 100
    
    ap = 1.0 / (ranks + 1)
    mAP = ap.mean() * 100
    
    return {'R1': r1, 'R5': r5, 'R10': r10, 'mAP': mAP}

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, required=True, choices=['PRID', 'iLIDs', 'Duke'])
    parser.add_argument('--pool', type=str, choices=['baseline', 'mean', 'weighted'], required=True)
    parser.add_argument('--seed', type=int, required=True)
    args = parser.parse_args()
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Evaluating {args.pool} pooling on {args.dataset} (seed={args.seed}) on: {device}")
    
    checkpoint_path = f"../experiments/checkpoints/mfgf_{args.dataset.lower()}_offline_{args.pool}_{args.seed}/offline_mfgf_final.pt"
    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f"Checkpoint missing: {checkpoint_path}")
    
    print("Loading test features...")
    test_features = torch.load(f'data/{args.dataset}_features/test/extracted_features.pt')
    
    if args.pool != 'baseline':
        ablation_vis_path = f'data/{args.dataset}_features_ablation/test/{args.pool}_visual_features.pt'
        if os.path.exists(ablation_vis_path):
            ablation_vis = torch.load(ablation_vis_path)
            for vid, vis_feat in ablation_vis.items():
                if vid in test_features:
                    test_features[vid]['visual'] = vis_feat
        else:
            raise FileNotFoundError(f"Missing ablation features: {ablation_vis_path}")
                
    tips_dict = torch.load(f'data/{args.dataset}_tips.pt')
    
    # Normalize keys
    clean_features = {k.replace('.mp4', ''): v for k, v in test_features.items()}
    clean_tips = {k.replace('.mp4', ''): v for k, v in tips_dict.items()}
    
    test_dataset = OfflineMFGFDataset(clean_features, clean_tips)
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)
    
    d_tips = next(iter(clean_tips.values())).shape[0]
    model = MFGFHead(d_tips=d_tips, embed_dim=256).to(device)
    model.load_state_dict(torch.load(checkpoint_path, map_location=device))
    model.eval()
    
    all_vid_embeds = []
    all_txt_embeds = []
    
    print("Extracting embeddings for retrieval...")
    with torch.no_grad():
        for batch_idx, batch in enumerate(test_loader):
            visual = batch['visual'].to(device)
            motion = batch['motion'].to(device)
            text = batch['text'].to(device)
            
            c_video, c_text, _, _, _ = model(visual, motion, text)
            
            all_vid_embeds.append(c_video.cpu())
            all_txt_embeds.append(c_text.cpu())
            
    all_vid_embeds = torch.cat(all_vid_embeds, dim=0) # (142, 256)
    all_txt_embeds = torch.cat(all_txt_embeds, dim=0) # (142, 256)
    
    print("Calculating similarity matrix...")
    similarity_matrix = all_txt_embeds @ all_vid_embeds.t() # (142, 142)
    
    print("Calculating metrics...")
    metrics = calculate_metrics(similarity_matrix.numpy(), num_descriptions=1)
    
    print("\n--- TRUE MFGF Evaluation Results (120 Epochs) ---")
    print(f"Rank-1:  {metrics['R1']:.2f}%")
    print(f"Rank-5:  {metrics['R5']:.2f}%")
    print(f"Rank-10: {metrics['R10']:.2f}%")
    print(f"mAP:     {metrics['mAP']:.2f}%")
    
    res_dir = "../experiments/results"
    os.makedirs(res_dir, exist_ok=True)
    out_path = os.path.join(res_dir, f"{args.dataset}_{args.pool}_{args.seed}.json")
    with open(out_path, 'w') as f:
        json.dump(metrics, f)

if __name__ == '__main__':
    main()
