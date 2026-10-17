import torch
import torch.nn as nn
from transformers import CLIPModel, CLIPProcessor
import math

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=5000):
        super(PositionalEncoding, self).__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0) # (1, max_len, d_model)
        self.register_buffer('pe', pe)

    def forward(self, x):
        # x: (B, T, D)
        x = x + self.pe[:, :x.size(1), :]
        return x

class CoAttentionFusion(nn.Module):
    def __init__(self, embed_dim=512, num_heads=8):
        super().__init__()
        # Forward attention: query=att, kv=vis
        self.attn_forward = nn.MultiheadAttention(embed_dim, num_heads, batch_first=True)
        # Backward attention: query=vis, kv=att
        self.attn_backward = nn.MultiheadAttention(embed_dim, num_heads, batch_first=True)
        
        # Implementation Decision: VCLIP paper says "and visa versa" but doesn't specify merge.
        # We concatenate and project back to embed_dim.
        self.merge_proj = nn.Linear(embed_dim * 2, embed_dim)
        
    def forward(self, f_vis, f_att):
        # f_vis, f_att: (B*T, 1, D) or (B, T, D)
        # We assume they are (B, T, D) here.
        
        # Forward: query=att, kv=vis
        out_fwd, _ = self.attn_forward(query=f_att, key=f_vis, value=f_vis)
        
        # Backward: query=vis, kv=att
        out_bwd, _ = self.attn_backward(query=f_vis, key=f_att, value=f_att)
        
        # Merge
        merged = torch.cat([out_fwd, out_bwd], dim=-1) # (B, T, 2D)
        out = self.merge_proj(merged) # (B, T, D)
        return out

class VCLIPVideoEncoder(nn.Module):
    def __init__(self, clip_model: CLIPModel, num_attributes=41, embed_dim=512):
        super().__init__()
        self.clip_model = clip_model
        
        # 1. Attribute Inference Module
        # We learn K continuous embeddings representing the attribute prompts.
        # In actual training, these are learned to align with the word embeddings of the attributes.
        self.num_attributes = num_attributes
        self.learned_prompts = nn.Parameter(torch.randn(num_attributes, embed_dim))
        
        # To construct the attribute sentence per frame, VCLIP selects attributes and encodes them.
        # For simplicity in this adaptation (since the exact text reconstruction isn't open source),
        # we weight the learned prompts by the similarity probabilities to form f_att.
        # This is a differentiable soft-selection of attributes.
        self.temperature = nn.Parameter(torch.ones(1))
        
        # 2. Co-Attention Fusion
        self.co_attention = CoAttentionFusion(embed_dim=embed_dim)
        
        # 3. Temporal Transformer
        self.pos_encoder = PositionalEncoding(d_model=embed_dim)
        encoder_layer = nn.TransformerEncoderLayer(d_model=embed_dim, nhead=8, dim_feedforward=2048, batch_first=True)
        self.temporal_transformer = nn.TransformerEncoder(encoder_layer, num_layers=2)
        
    def forward(self, pixel_values, q_j=None):
        """
        pixel_values: (B, T, C, H, W)
        q_j: Optional (B, T) frame quality weights
        Returns query-independent video embedding (B, D)
        """
        B, T, C, H, W = pixel_values.shape
        frames = pixel_values.view(B * T, C, H, W)
        
        # --- 1. Visual Feature Extraction ---
        vis_outputs = self.clip_model.vision_model(pixel_values=frames)
        f_vis = vis_outputs.pooler_output
        f_vis = self.clip_model.visual_projection(f_vis) # (B*T, D)
        
        # Normalize for similarity
        f_vis_norm = f_vis / f_vis.norm(dim=-1, keepdim=True)
        prompts_norm = self.learned_prompts / self.learned_prompts.norm(dim=-1, keepdim=True)
        
        # --- 2. Attribute Inference ---
        sim = torch.matmul(f_vis_norm, prompts_norm.T) * torch.exp(self.temperature) # (B*T, K)
        attn_weights = torch.softmax(sim, dim=-1) # (B*T, K)
        f_att = torch.matmul(attn_weights, self.learned_prompts) # (B*T, D)
        
        # Reshape to sequence
        f_vis_seq = f_vis.view(B, T, -1)
        f_att_seq = f_att.view(B, T, -1)
        
        # --- 3. Co-Attention Fusion ---
        f_co = self.co_attention(f_vis_seq, f_att_seq) # (B, T, D)
        
        # --- 4. Temporal Fusion ---
        f_co = self.pos_encoder(f_co)
        f_temp = self.temporal_transformer(f_co) # (B, T, D)
        
        if q_j is not None:
            # Soft Reliability-Aware Pooling over T
            f_video = (f_temp * q_j.unsqueeze(-1)).sum(dim=1) / (q_j.sum(dim=1, keepdim=True) + 1e-6)
        else:
            # Mean Pooling over T
            f_video = f_temp.mean(dim=1) # (B, D)
        
        return f_video
