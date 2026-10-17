import os
import sys
import time

os.environ['TORCH_HOME'] = 'E:/BTP/.cache/torch'
os.environ['HF_HOME'] = 'E:/BTP/.cache/huggingface'

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from transformers import BertTokenizer

# Fix import path since we are running from mfgf_reproduction
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from src.dataset.tvpreid import TVPReIDDataset
from real_mfgf import TrueMFGFModel
import torch.nn.functional as F

def mfgf_loss(c_video, c_text, d2_video, d2_text, logit_scale, tips_gt, alpha, beta_weight):
    # --- Common Space Loss (Eq 16) ---
    # c_video: (B, 256), c_text: (B, 256)
    # logit_scale is 1/theta
    scores = logit_scale * c_text @ c_video.t() # (B, B)
    labels = torch.arange(c_video.size(0), device=c_video.device)
    loss_v2t = F.cross_entropy(scores, labels)
    loss_t2v = F.cross_entropy(scores.t(), labels)
    L_common = (loss_v2t + loss_t2v) / 2.0
    
    # --- Dul-Distilled Space Loss (Eq 19) ---
    # d2_video: (B, 410), d2_text: (B, 410)
    scores_d2 = logit_scale * d2_text @ d2_video.t() # (B, B)
    loss_d2_v2t = F.cross_entropy(scores_d2, labels)
    loss_d2_t2v = F.cross_entropy(scores_d2.t(), labels)
    L_contrast_d2 = (loss_d2_v2t + loss_d2_t2v) / 2.0
    
    # --- BCE Distillation Loss (Eq 20) ---
    # BCE(Tips, T^D2) + BCE(Tips, V^D2)
    bce = nn.BCELoss()
    # Note: paper optimizes cross-entropy, PyTorch BCELoss expects probabilities.
    # d2_text and d2_video already passed through sigmoid.
    loss_bce_text = bce(d2_text, tips_gt)
    loss_bce_video = bce(d2_video, tips_gt)
    L_bce = loss_bce_text + loss_bce_video
    
    # Total D^2 Loss (Eq 21)
    L_d2 = L_contrast_d2 + beta_weight * L_bce
    
    # Overall Loss (Eq 22)
    L_mfgf = alpha * L_common + (1 - alpha) * L_d2
    return L_mfgf, L_common, L_d2

def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Training on: {device}")
    
    data_dir = "../data/TVPReID/TVPReid-PRID"
    save_dir = "../experiments/checkpoints/mfgf_prid_true"
    os.makedirs(save_dir, exist_ok=True)
    
    # Load Tips Ground Truth
    tips_dict = torch.load('data/tips.pt')
    # The dataset returns prefixed video_ids but TVPReidDataset uses the base filename prefix if specified.
    # In preprocess_tips we used the raw json keys.
    
    train_dataset = TVPReIDDataset(data_dir, split='train', num_frames=8)
    train_loader = DataLoader(train_dataset, batch_size=8, shuffle=True, num_workers=0, drop_last=True)
    
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    
    # Get D_tips from tips_dict
    sample_tip = next(iter(tips_dict.values()))
    d_tips = sample_tip.shape[0]
    
    model = TrueMFGFModel(d_tips=d_tips, embed_dim=256).to(device)
    
    # Learnable weight parameters
    alpha = nn.Parameter(torch.tensor(0.5, device=device))
    beta_weight = 1.0 # B in Eq 21
    
    optimizer = optim.Adam(
        list(model.parameters()) + [alpha],
        lr=1e-4
    )
    
    num_epochs = 20
    grad_accum_steps = 2
    
    for epoch in range(num_epochs):
        model.train()
        total_loss = 0
        optimizer.zero_grad()
        
        for batch_idx, batch in enumerate(train_loader):
            frames = batch['frames'].to(device)
            texts = batch['text']
            vid_ids = batch['video_id']
            
            # Prefix handling: preprocess_tips.py keys are like "person0001"
            # TVPReIDDataset might append .mp4 or use raw id. 
            # Let's clean vid_ids to match tips_dict keys.
            clean_vids = [vid + '.mp4' if not vid.endswith('.mp4') else vid for vid in vid_ids]
            
            tips_gt = torch.stack([tips_dict[vid] for vid in clean_vids]).to(device)
            
            tokens = tokenizer(texts, padding=True, truncation=True, return_tensors='pt', max_length=77).to(device)
            
            c_video, c_text, d2_video, d2_text, logit_scale = model(frames, tokens['input_ids'], tokens['attention_mask'])
            
            loss, l_com, l_d2 = mfgf_loss(c_video, c_text, d2_video, d2_text, logit_scale, tips_gt, alpha, beta_weight)
            loss = loss / grad_accum_steps
            loss.backward()
            
            if (batch_idx + 1) % grad_accum_steps == 0:
                optimizer.step()
                optimizer.zero_grad()
                
            total_loss += loss.item() * grad_accum_steps
            
            if batch_idx % 10 == 0:
                print(f"Epoch {epoch+1} | Batch {batch_idx}/{len(train_loader)} | Loss: {loss.item()*grad_accum_steps:.4f} | L_com: {l_com.item():.4f} | L_d2: {l_d2.item():.4f} | alpha: {alpha.item():.4f}")
                
        avg_loss = total_loss / len(train_loader)
        print(f"--- Epoch {epoch+1} Average Loss: {avg_loss:.4f} ---")
        
        # Save checkpoint
        torch.save(model.state_dict(), os.path.join(save_dir, f"true_mfgf_epoch_{epoch+1}.pt"))

if __name__ == '__main__':
    main()
