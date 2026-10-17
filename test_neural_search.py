import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import BertTokenizer, BertModel
import json

os.environ['HF_HOME'] = 'E:/BTP/.cache/huggingface'

print("=" * 60)
print("Building Genuine Neural Video Embeddings for Demo")
print("Model: Unified MFGF (Checkpoint: seed 42)")
print("Text Encoder: bert-base-uncased (768-d CLS token)")
print("Video Encoders: ViT-B/16 (768-d) + S3D (1024-d)")
print("=" * 60)

# 1. Load MFGF Head Architecture
class MFGFVideoProjector(nn.Module):
    def __init__(self, embed_dim=256):
        super().__init__()
        concat_dim = 768 + 1024
        self.bn = nn.BatchNorm1d(concat_dim)
        self.fc = nn.Linear(concat_dim, embed_dim)
        self.common_video_proj = nn.Linear(embed_dim, embed_dim)
        
    def forward(self, visual_feat, motion_feat):
        f_fusion = torch.cat([visual_feat, motion_feat], dim=-1)
        f_ME = self.fc(self.bn(f_fusion))
        c_video = F.normalize(self.common_video_proj(f_ME), dim=-1)
        return c_video

# Load base test features (motion + keys)
test_base = torch.load('mfgf_reproduction/data/Unified_features/test/extracted_features.pt', map_location='cpu')
vid_ids = sorted(list(test_base.keys()))
print(f"Loaded {len(vid_ids)} test video tracklets")

motion_tensors = torch.stack([test_base[vid]['motion'] for vid in vid_ids]) # (820, 1024)

# Load visual ablation features for the 4 conditions
vis_sources = {
    'clean_mean': 'mfgf_reproduction/data/Unified_features_ablation/test/mean_visual_features.pt',
    'clean_weighted': 'mfgf_reproduction/data/Unified_features_ablation/test/weighted_visual_features.pt',
    'corrupted_mean': 'mfgf_reproduction/data/Unified_features_ablation/test/corrupted_mean.pt',
    'corrupted_weighted': 'mfgf_reproduction/data/Unified_features_ablation/test/corrupted_weighted.pt',
}

# Checkpoints corresponding to mean and weighted models
ckpts = {
    'mean': 'experiments/checkpoints/mfgf_unified_offline_mean_42/offline_mfgf_final.pt',
    'weighted': 'experiments/checkpoints/mfgf_unified_offline_weighted_42/offline_mfgf_final.pt'
}

gallery_embeddings = {
    'vid_ids': vid_ids
}

for cond_name, vis_path in vis_sources.items():
    pool_type = 'mean' if 'mean' in cond_name else 'weighted'
    ckpt_path = ckpts[pool_type]
    
    ckpt = torch.load(ckpt_path, map_location='cpu')
    model = MFGFVideoProjector()
    model.bn.weight.data = ckpt['bn.weight']
    model.bn.bias.data = ckpt['bn.bias']
    model.bn.running_mean.data = ckpt['bn.running_mean']
    model.bn.running_var.data = ckpt['bn.running_var']
    model.fc.weight.data = ckpt['fc.weight']
    model.fc.bias.data = ckpt['fc.bias']
    model.common_video_proj.weight.data = ckpt['common_video_proj.weight']
    model.common_video_proj.bias.data = ckpt['common_video_proj.bias']
    model.eval()
    
    vis_dict = torch.load(vis_path, map_location='cpu')
    vis_tensors = torch.stack([vis_dict[vid] for vid in vid_ids]) # (820, 768)
    
    with torch.no_grad():
        c_video = model(vis_tensors, motion_tensors) # (820, 256)
        
    gallery_embeddings[cond_name] = c_video
    print(f"Generated {cond_name} video embeddings: {c_video.shape}")

# Save gallery embeddings cache
torch.save(gallery_embeddings, 'data/neural_gallery_embeddings.pt')
print("\nSaved neural gallery embeddings to data/neural_gallery_embeddings.pt!")

# Now test Query Retrieval
print("\n--- Testing Custom Query Neural Retrieval ---")
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased', local_files_only=True)
bert = BertModel.from_pretrained('bert-base-uncased', local_files_only=True)
bert.eval()

# Load text projector from trained weighted checkpoint
text_proj = nn.Linear(768, 256)
ckpt_w = torch.load(ckpts['weighted'], map_location='cpu')
text_proj.weight.data = ckpt_w['common_text_proj.weight']
text_proj.bias.data = ckpt_w['common_text_proj.bias']
text_proj.eval()

query = "The woman is wearing a red coat, black trousers and black shoes."
with torch.no_grad():
    tokens = tokenizer([query], padding=True, truncation=True, return_tensors='pt', max_length=77)
    out = bert(**tokens)
    text_feat = out.last_hidden_state[:, 0, :]
    c_text = F.normalize(text_proj(text_feat), dim=-1) # (1, 256)

sims = (c_text @ gallery_embeddings['clean_weighted'].t()).squeeze(0) # (820,)
top_indices = torch.argsort(sims, descending=True)[:5]

print(f"Query: \"{query}\"")
print("Top 5 Retrieved Candidates by Genuine Neural Model:")
for rank, idx in enumerate(top_indices, 1):
    vid = vid_ids[idx]
    score = sims[idx].item()
    print(f"  Rank #{rank}: {vid} (cosine similarity = {score:.4f})")
