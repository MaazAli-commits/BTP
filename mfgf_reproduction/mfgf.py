import torch
import torch.nn as nn
import torchvision.models as models
from transformers import DistilBertModel

class MFGFVideoEncoder(nn.Module):
    def __init__(self, embed_dim=512):
        super().__init__()
        # Standard visual appearance backbone
        resnet = models.resnet50(pretrained=True)
        self.backbone = nn.Sequential(*list(resnet.children())[:-1]) # Output: (B, 2048, 1, 1)
        
        # Freeze early layers to save VRAM
        for name, param in self.backbone.named_parameters():
            if 'layer3' not in name and 'layer4' not in name:
                param.requires_grad = False
                
        # Motion/Temporal capture module
        self.temporal_conv = nn.Conv1d(in_channels=2048, out_channels=2048, kernel_size=3, padding=1)
        
        # Projection to joint space
        self.visual_proj = nn.Linear(2048, embed_dim)
        self.motion_proj = nn.Linear(2048, embed_dim)
        self.fusion = nn.Linear(embed_dim * 2, embed_dim)
        
    def forward(self, frames, q_j=None):
        # frames: (B, T, C, H, W)
        B, T, C, H, W = frames.shape
        frames = frames.view(B * T, C, H, W)
        
        # Appearance Features
        feat = self.backbone(frames) # (B*T, 2048, 1, 1)
        feat = feat.view(B, T, -1) # (B, T, 2048)
        
        # Optional Frame-Quality Reliability (q_j) integration point
        if q_j is not None:
            # q_j: (B, T) - Use it to weight the frames before temporal aggregation
            weight = q_j.unsqueeze(-1) # (B, T, 1)
            feat = feat * weight
        
        # Motion Features
        feat_t = feat.permute(0, 2, 1) # (B, 2048, T)
        motion = self.temporal_conv(feat_t) # (B, 2048, T)
        motion = torch.relu(motion).permute(0, 2, 1) # (B, T, 2048)
        
        # Aggregate temporally (Mean pooling)
        app_pool = feat.mean(dim=1) # (B, 2048)
        mot_pool = motion.mean(dim=1) # (B, 2048)
        
        # Project
        app_embed = self.visual_proj(app_pool)
        mot_embed = self.motion_proj(mot_pool)
        
        # Fuse multielement features
        fused = self.fusion(torch.cat([app_embed, mot_embed], dim=-1))
        return fused

class MFGFTextEncoder(nn.Module):
    def __init__(self, embed_dim=512):
        super().__init__()
        self.bert = DistilBertModel.from_pretrained('distilbert-base-uncased')
        # Freeze text encoder to save VRAM, matching standard cross-modal baseline practices
        for param in self.bert.parameters():
            param.requires_grad = False
        self.text_proj = nn.Linear(self.bert.config.hidden_size, embed_dim)
        
    def forward(self, input_ids, attention_mask):
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        # Use CLS token
        cls_token = outputs.last_hidden_state[:, 0, :]
        embed = self.text_proj(cls_token)
        return embed

class MFGFModel(nn.Module):
    def __init__(self, embed_dim=512):
        super().__init__()
        self.video_encoder = MFGFVideoEncoder(embed_dim)
        self.text_encoder = MFGFTextEncoder(embed_dim)
        self.logit_scale = nn.Parameter(torch.ones([]) * torch.tensor(1 / 0.07).log())
        
    def forward(self, frames, input_ids, attention_mask, q_j=None):
        vid_embeds = self.video_encoder(frames, q_j=q_j)
        txt_embeds = self.text_encoder(input_ids, attention_mask)
        
        vid_embeds = vid_embeds / vid_embeds.norm(dim=-1, keepdim=True)
        txt_embeds = txt_embeds / txt_embeds.norm(dim=-1, keepdim=True)
        
        return vid_embeds, txt_embeds, self.logit_scale.exp()
