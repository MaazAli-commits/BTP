import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import torch.nn.functional as F
import argparse
import random
import numpy as np

class OfflineMFGFDataset(Dataset):
    def __init__(self, features_dict, tips_dict):
        self.features_dict = features_dict
        self.tips_dict = tips_dict
        self.vid_ids = list(features_dict.keys())
        
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

class MFGFHead(nn.Module):
    def __init__(self, d_tips, embed_dim=256):
        super().__init__()
        # Feature Aggregation
        concat_dim = 768 + 1024 # ViT + S3D
        self.bn = nn.BatchNorm1d(concat_dim)
        self.fc = nn.Linear(concat_dim, embed_dim)
        
        # Common Space
        self.common_text_proj = nn.Linear(768, embed_dim)
        self.common_video_proj = nn.Linear(embed_dim, embed_dim)
        
        # D^2 Space
        self.d2_text_fc = nn.Linear(768, d_tips)
        self.d2_text_bn = nn.BatchNorm1d(d_tips)
        self.d2_video_fc = nn.Linear(embed_dim, d_tips)
        self.d2_video_bn = nn.BatchNorm1d(d_tips)
        
        self.logit_scale = nn.Parameter(torch.ones([]) * torch.tensor(1 / 0.05).log())
        
    def forward(self, visual_feat, motion_feat, text_feat):
        # Aggregate
        f_fusion = torch.cat([visual_feat, motion_feat], dim=-1)
        f_ME = self.fc(self.bn(f_fusion))
        
        # Common Space
        c_text = F.normalize(self.common_text_proj(text_feat), dim=-1)
        c_video = F.normalize(self.common_video_proj(f_ME), dim=-1)
        
        # D^2 Space
        d2_text = torch.sigmoid(self.d2_text_bn(self.d2_text_fc(text_feat)))
        d2_video = torch.sigmoid(self.d2_video_bn(self.d2_video_fc(f_ME)))
        
        return c_video, c_text, d2_video, d2_text, self.logit_scale.exp()

def mfgf_loss(c_video, c_text, d2_video, d2_text, logit_scale, tips_gt, alpha, beta_weight):
    scores = logit_scale * c_text @ c_video.t()
    labels = torch.arange(c_video.size(0), device=c_video.device)
    loss_v2t = F.cross_entropy(scores, labels)
    loss_t2v = F.cross_entropy(scores.t(), labels)
    L_common = (loss_v2t + loss_t2v) / 2.0
    
    scores_d2 = logit_scale * d2_text @ d2_video.t()
    loss_d2_v2t = F.cross_entropy(scores_d2, labels)
    loss_d2_t2v = F.cross_entropy(scores_d2.t(), labels)
    L_contrast_d2 = (loss_d2_v2t + loss_d2_t2v) / 2.0
    
    bce = nn.BCELoss()
    L_bce = bce(d2_text, tips_gt) + bce(d2_video, tips_gt)
    
    L_d2 = L_contrast_d2 + beta_weight * L_bce
    L_mfgf = alpha * L_common + (1 - alpha) * L_d2
    return L_mfgf, L_common, L_d2

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, required=True, choices=['PRID', 'iLIDs', 'Duke'])
    parser.add_argument('--pool', type=str, choices=['baseline', 'mean', 'weighted'], required=True)
    parser.add_argument('--seed', type=int, default=42)
    args = parser.parse_args()
    
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
        
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Training offline with {args.pool} pooling on {args.dataset}, seed={args.seed} on: {device}")
    
    save_dir = f"../experiments/checkpoints/mfgf_{args.dataset.lower()}_offline_{args.pool}_{args.seed}"
    os.makedirs(save_dir, exist_ok=True)
    
    print("Loading features...")
    train_features = torch.load(f'data/{args.dataset}_features/train/extracted_features.pt')
    
    if args.pool != 'baseline':
        ablation_vis_path = f'data/{args.dataset}_features_ablation/train/{args.pool}_visual_features.pt'
        if os.path.exists(ablation_vis_path):
            ablation_vis = torch.load(ablation_vis_path)
            for vid, vis_feat in ablation_vis.items():
                if vid in train_features:
                    train_features[vid]['visual'] = vis_feat
        else:
            raise FileNotFoundError(f"Missing ablation features: {ablation_vis_path}")
                
    tips_dict = torch.load(f'data/{args.dataset}_tips.pt')
    
    # Map keys correctly by stripping .mp4 from both
    clean_features = {k.replace('.mp4', ''): v for k, v in train_features.items()}
    clean_tips = {k.replace('.mp4', ''): v for k, v in tips_dict.items()}
    
    train_dataset = OfflineMFGFDataset(clean_features, clean_tips)
    train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True, drop_last=True)
    
    d_tips = next(iter(clean_tips.values())).shape[0]
    model = MFGFHead(d_tips=d_tips, embed_dim=256).to(device)
    
    alpha = nn.Parameter(torch.tensor(0.5, device=device))
    beta_weight = 1.0
    
    optimizer = optim.Adam(list(model.parameters()) + [alpha], lr=1e-4)
    
    num_epochs = 120 # TVPR paper uses 120 epochs
    
    for epoch in range(num_epochs):
        model.train()
        total_loss = 0
        
        for batch_idx, batch in enumerate(train_loader):
            optimizer.zero_grad()
            
            visual = batch['visual'].to(device)
            motion = batch['motion'].to(device)
            text = batch['text'].to(device)
            tips_gt = batch['tip'].to(device)
            
            c_video, c_text, d2_video, d2_text, logit_scale = model(visual, motion, text)
            loss, l_com, l_d2 = mfgf_loss(c_video, c_text, d2_video, d2_text, logit_scale, tips_gt, alpha, beta_weight)
            
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            
            if batch_idx % 20 == 0:
                print(f"Epoch {epoch+1} | Batch {batch_idx}/{len(train_loader)} | Loss: {loss.item():.4f} | alpha: {alpha.item():.4f}")
                
        print(f"--- Epoch {epoch+1} Average Loss: {total_loss / len(train_loader):.4f} ---")
        torch.save(model.state_dict(), os.path.join(save_dir, f"offline_mfgf_epoch_{epoch+1}.pt"))
        
    torch.save(model.state_dict(), os.path.join(save_dir, "offline_mfgf_final.pt"))

if __name__ == '__main__':
    main()
