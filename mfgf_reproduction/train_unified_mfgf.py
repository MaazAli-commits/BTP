import os
import random
import argparse
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import torch.nn.functional as F

class UnifiedOfflineDataset(Dataset):
    def __init__(self, features_dict, tips_dict):
        self.features_dict = features_dict
        self.tips_dict = tips_dict
        self.vid_ids = sorted(list(features_dict.keys()))
        
    def __len__(self):
        return len(self.vid_ids)
        
    def __getitem__(self, idx):
        vid = self.vid_ids[idx]
        feat = self.features_dict[vid]
        tip = self.tips_dict[vid]
        
        return {
            'video_id': vid,
            'visual': feat['visual'],
            'motion': feat['motion'],
            'text': feat['text'],
            'tip': tip
        }

class UnifiedMFGFHead(nn.Module):
    def __init__(self, d_tips, embed_dim=256):
        super().__init__()
        # Feature Aggregation (ViT 768 + S3D 1024 = 1792)
        concat_dim = 768 + 1024
        self.bn = nn.BatchNorm1d(concat_dim)
        self.fc = nn.Linear(concat_dim, embed_dim)
        
        # Common Space Projection
        self.common_text_proj = nn.Linear(768, embed_dim)
        self.common_video_proj = nn.Linear(embed_dim, embed_dim)
        
        # D^2 Attribute Space Projection
        self.d2_text_fc = nn.Linear(768, d_tips)
        self.d2_text_bn = nn.BatchNorm1d(d_tips)
        self.d2_video_fc = nn.Linear(embed_dim, d_tips)
        self.d2_video_bn = nn.BatchNorm1d(d_tips)
        
        # Learnable Temperature Scale (init = ln(1 / 0.05) ~ 2.9957)
        self.logit_scale = nn.Parameter(torch.ones([]) * torch.tensor(1 / 0.05).log())
        
    def forward(self, visual_feat, motion_feat, text_feat):
        # Aggregate Visual + Motion
        f_fusion = torch.cat([visual_feat, motion_feat], dim=-1)
        f_ME = self.fc(self.bn(f_fusion))
        
        # Common Space Normalized Embeddings
        c_text = F.normalize(self.common_text_proj(text_feat), dim=-1)
        c_video = F.normalize(self.common_video_proj(f_ME), dim=-1)
        
        # D^2 Attribute Probabilities
        d2_text = torch.sigmoid(self.d2_text_bn(self.d2_text_fc(text_feat)))
        d2_video = torch.sigmoid(self.d2_video_bn(self.d2_video_fc(f_ME)))
        
        return c_video, c_text, d2_video, d2_text, self.logit_scale.exp()

def mfgf_loss(c_video, c_text, d2_video, d2_text, logit_scale, tips_gt, alpha, beta_weight=1.0):
    # Common Space InfoNCE Loss
    scores = logit_scale * c_text @ c_video.t()
    labels = torch.arange(c_video.size(0), device=c_video.device)
    loss_v2t = F.cross_entropy(scores, labels)
    loss_t2v = F.cross_entropy(scores.t(), labels)
    L_common = (loss_v2t + loss_t2v) / 2.0
    
    # D^2 Space Contrastive Loss
    scores_d2 = logit_scale * d2_text @ d2_video.t()
    loss_d2_v2t = F.cross_entropy(scores_d2, labels)
    loss_d2_t2v = F.cross_entropy(scores_d2.t(), labels)
    L_contrast_d2 = (loss_d2_v2t + loss_d2_t2v) / 2.0
    
    # D^2 Space BCE Attribute Supervision
    bce = nn.BCELoss()
    L_bce = bce(d2_text, tips_gt) + bce(d2_video, tips_gt)
    
    L_d2 = L_contrast_d2 + beta_weight * L_bce
    L_mfgf = alpha * L_common + (1 - alpha) * L_d2
    return L_mfgf, L_common, L_d2

def main():
    parser = argparse.ArgumentParser(description="Train Unified MFGF Head on PRID + iLIDS + Duke")
    parser.add_argument('--pool', type=str, choices=['baseline', 'mean', 'weighted'], required=True,
                        help="Visual pooling: baseline (raw ViT), mean (uniform avg), weighted (quality q_j)")
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--epochs', type=int, default=120)
    parser.add_argument('--batch_size', type=int, default=64)
    parser.add_argument('--lr', type=float, default=1e-4)
    args = parser.parse_args()
    
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
        
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print("=" * 65)
    print(f"Training Unified MFGF ({args.pool} pooling, seed={args.seed}) on {device}")
    print("=" * 65)
    
    save_dir = f"../experiments/checkpoints/mfgf_unified_offline_{args.pool}_{args.seed}"
    os.makedirs(save_dir, exist_ok=True)
    
    print("Loading unified train features...")
    train_features = torch.load('data/Unified_features/train/extracted_features.pt', map_location='cpu')
    
    if args.pool != 'baseline':
        ablation_path = f'data/Unified_features_ablation/train/{args.pool}_visual_features.pt'
        if os.path.exists(ablation_path):
            print(f"Applying {args.pool} pooled visual features from {ablation_path}")
            ablation_vis = torch.load(ablation_path, map_location='cpu')
            for vid, vis_feat in ablation_vis.items():
                if vid in train_features:
                    train_features[vid]['visual'] = vis_feat
        else:
            raise FileNotFoundError(f"Missing unified ablation features: {ablation_path}")
            
    print("Loading unified TIPS dictionary...")
    tips_dict = torch.load('data/Unified_tips.pt', map_location='cpu')
    
    # Filter only keys present in both
    clean_features = {k: v for k, v in train_features.items() if k in tips_dict}
    clean_tips = {k: tips_dict[k] for k in clean_features.keys()}
    print(f"Dataset ready: {len(clean_features)} training samples")
    
    train_dataset = UnifiedOfflineDataset(clean_features, clean_tips)
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, drop_last=True)
    
    d_tips = next(iter(clean_tips.values())).shape[0]
    print(f"MFGF Head d_tips dimension: {d_tips}")
    
    model = UnifiedMFGFHead(d_tips=d_tips, embed_dim=256).to(device)
    alpha = nn.Parameter(torch.tensor(0.5, device=device))
    optimizer = optim.Adam(list(model.parameters()) + [alpha], lr=args.lr)
    
    for epoch in range(args.epochs):
        model.train()
        total_loss = 0.0
        
        for batch in train_loader:
            optimizer.zero_grad()
            
            visual = batch['visual'].to(device)
            motion = batch['motion'].to(device)
            text = batch['text'].to(device)
            tips_gt = batch['tip'].to(device)
            
            c_vid, c_txt, d2_vid, d2_txt, logit_scale = model(visual, motion, text)
            loss, l_com, l_d2 = mfgf_loss(c_vid, c_txt, d2_vid, d2_txt, logit_scale, tips_gt, alpha)
            
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
            
        if (epoch + 1) % 30 == 0 or (epoch + 1) == args.epochs:
            avg_loss = total_loss / len(train_loader)
            print(f"Epoch [{epoch+1:3d}/{args.epochs}] | Avg Loss: {avg_loss:.4f} | Alpha: {alpha.item():.4f}")
            
    final_path = os.path.join(save_dir, "offline_mfgf_final.pt")
    torch.save(model.state_dict(), final_path)
    print(f"\nTraining complete! Checkpoint saved to: {final_path}")

if __name__ == '__main__':
    main()
