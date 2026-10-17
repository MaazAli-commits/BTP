import os
os.environ['TORCH_HOME'] = 'E:/BTP/.cache/torch'
os.environ['HF_HOME'] = 'E:/BTP/.cache/huggingface'

import torch
import torch.nn as nn
import torchvision.models as models
import torchvision.models.video as video_models
from transformers import BertModel
import torch.nn.functional as F

class TrueMFGFTextEncoder(nn.Module):
    def __init__(self, embed_dim=256):
        super().__init__()
        self.bert = BertModel.from_pretrained('bert-base-uncased')
        # Freeze text encoder to save VRAM and maintain pretrained MLM knowledge
        for param in self.bert.parameters():
            param.requires_grad = False
        
    def forward(self, input_ids, attention_mask):
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        # Use CLS token as per paper Section 3.1.2
        cls_token = outputs.last_hidden_state[:, 0, :]
        return cls_token

class TrueMFGFVideoEncoder(nn.Module):
    def __init__(self, embed_dim=256):
        super().__init__()
        # Visual Encoder: ViT-B/16
        vit = models.vit_b_16(pretrained=True)
        # We only need up to the representation/CLS token
        # Freeze ViT entirely to save VRAM (act as offline feature extractor)
        for param in vit.parameters():
            param.requires_grad = False
        self.vit = vit
        
        # Motion Encoder: S3D
        s3d = video_models.s3d(weights=video_models.S3D_Weights.KINETICS400_V1)
        # Freeze S3D
        for param in s3d.parameters():
            param.requires_grad = False
        # We want the features before the final classifier
        self.s3d_features = s3d.features
        
        # Feature Aggregation: Eq 13 & 14
        # ViT hidden size: 768. S3D feature size: 1024.
        concat_dim = 768 + 1024
        self.bn = nn.BatchNorm1d(concat_dim)
        self.fc = nn.Linear(concat_dim, embed_dim)
        
    def forward(self, frames):
        # frames: (B, T, C, H, W)
        B, T, C, H, W = frames.shape
        
        # --- Visual Encoder (ViT) ---
        # The paper says it randomly selects frames or processes patches.
        # We'll just pass frames independently and mean pool to get video visual feature.
        frames_flat = frames.view(B * T, C, H, W)
        visual_feat = self.vit._process_input(frames_flat)
        # Expand class token
        n = visual_feat.shape[0]
        batch_class_token = self.vit.class_token.expand(n, -1, -1)
        visual_feat = torch.cat([batch_class_token, visual_feat], dim=1)
        visual_feat = self.vit.encoder(visual_feat)
        # Use CLS token
        visual_feat = visual_feat[:, 0] # (B*T, 768)
        visual_feat = visual_feat.view(B, T, -1).mean(dim=1) # (B, 768)
        
        # --- Motion Encoder (S3D) ---
        # S3D expects (B, C, T, H, W)
        frames_s3d = frames.permute(0, 2, 1, 3, 4)
        motion_feat = self.s3d_features(frames_s3d) # (B, 1024, T', H', W')
        # Global Average Pooling for S3D (MaxPool is mentioned in eq 9 but GAP is standard for final feat)
        motion_feat = F.adaptive_avg_pool3d(motion_feat, (1, 1, 1)).view(B, -1) # (B, 1024)
        
        # --- Feature Aggregation ---
        # f_fusion = f_visual + f_motion (concat)
        f_fusion = torch.cat([visual_feat, motion_feat], dim=-1) # (B, 1792)
        
        # f_ME = FC(BN(f_fusion))
        f_ME = self.fc(self.bn(f_fusion))
        return f_ME

class TrueMFGFModel(nn.Module):
    def __init__(self, d_tips=410, embed_dim=256):
        super().__init__()
        self.text_encoder = TrueMFGFTextEncoder(embed_dim)
        self.video_encoder = TrueMFGFVideoEncoder(embed_dim)
        
        # BERT hidden size: 768
        # Video embed size: embed_dim = 256
        
        # --- Common Space Learning ---
        # Project to common space
        self.common_text_proj = nn.Linear(768, embed_dim)
        self.common_video_proj = nn.Linear(embed_dim, embed_dim)
        
        # --- Dul-Distilled Space Learning ---
        # Project to D^2 space (dimension = D_tips)
        self.d2_text_fc = nn.Linear(768, d_tips)
        self.d2_text_bn = nn.BatchNorm1d(d_tips)
        
        self.d2_video_fc = nn.Linear(embed_dim, d_tips)
        self.d2_video_bn = nn.BatchNorm1d(d_tips)
        
        self.logit_scale = nn.Parameter(torch.ones([]) * torch.tensor(1 / 0.05).log()) # Temperature = 0.05 per paper
        
    def forward(self, frames, input_ids, attention_mask):
        # Base features
        f_ME = self.video_encoder(frames) # (B, 256)
        f_text = self.text_encoder(input_ids, attention_mask) # (B, 768)
        
        # --- Common Space ---
        c_text = self.common_text_proj(f_text)
        c_video = self.common_video_proj(f_ME)
        c_text = F.normalize(c_text, dim=-1)
        c_video = F.normalize(c_video, dim=-1)
        
        # --- D^2 Space ---
        d2_text = torch.sigmoid(self.d2_text_bn(self.d2_text_fc(f_text)))
        d2_video = torch.sigmoid(self.d2_video_bn(self.d2_video_fc(f_ME)))
        
        return c_video, c_text, d2_video, d2_text, self.logit_scale.exp()
