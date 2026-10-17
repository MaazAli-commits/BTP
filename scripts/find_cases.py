import os
import json
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import BertTokenizer, BertModel

gallery_path = "data/neural_gallery_embeddings.pt"
gallery = torch.load(gallery_path, map_location='cpu')
meta = gallery['meta']
vids = gallery['vid_ids']
c_mean = gallery['clean_mean']
b_mean = gallery['corrupted_mean']
b_weighted = gallery['corrupted_weighted']

tok = BertTokenizer.from_pretrained('bert-base-uncased', local_files_only=True)
bert = BertModel.from_pretrained('bert-base-uncased', local_files_only=True)
bert.eval()

ckpt_m = torch.load('experiments/checkpoints/mfgf_unified_offline_mean_42/offline_mfgf_final.pt', map_location='cpu')
ckpt_w = torch.load('experiments/checkpoints/mfgf_unified_offline_weighted_42/offline_mfgf_final.pt', map_location='cpu')

proj_m = nn.Linear(768, 256)
proj_m.weight.data = ckpt_m['common_text_proj.weight']
proj_m.bias.data = ckpt_m['common_text_proj.bias']
proj_m.eval()

proj_w = nn.Linear(768, 256)
proj_w.weight.data = ckpt_w['common_text_proj.weight']
proj_w.bias.data = ckpt_w['common_text_proj.bias']
proj_w.eval()

captions = [meta[v].get('caption', '') for v in vids]

# Batch encode all 820 captions
all_cls = []
batch_size = 64
with torch.no_grad():
    for i in range(0, len(captions), batch_size):
        batch = captions[i:i+batch_size]
        inputs = tok(batch, return_tensors='pt', padding=True, truncation=True, max_length=77)
        feat = bert(**inputs).last_hidden_state[:, 0, :]
        all_cls.append(feat)

all_cls = torch.cat(all_cls, dim=0) # (820, 768)
with torch.no_grad():
    txt_m = F.normalize(proj_m(all_cls), dim=-1) # (820, 256)
    txt_w = F.normalize(proj_w(all_cls), dim=-1) # (820, 256)

    # Compute similarity matrices (820, 820)
    sim_c = txt_m @ c_mean.t()
    sim_bm = txt_m @ b_mean.t()
    sim_bw = txt_w @ b_weighted.t()

candidates = []
for i, vid in enumerate(vids):
    cap = captions[i]
    if not cap or len(cap) < 10:
        continue
    
    r_c = (sim_c[i] > sim_c[i, i]).sum().item() + 1
    r_bm = (sim_bm[i] > sim_bm[i, i]).sum().item() + 1
    r_bw = (sim_bw[i] > sim_bw[i, i]).sum().item() + 1
    
    if r_c == 1 and r_bm > 1:
        candidates.append({
            'vid': vid,
            'dataset': meta[vid]['dataset'],
            'video_file': meta[vid]['video_file'],
            'caption': cap,
            'r_c': r_c,
            'r_bm': r_bm,
            'r_bw': r_bw,
            'gain': r_bm - r_bw
        })

print(f"Total candidates where clean is Rank 1 and blur drops: {len(candidates)}")
for c in sorted(candidates, key=lambda x: x['r_bm']):
    print(f"{c['vid']} ({c['dataset']}): Clean #{c['r_c']} -> BlurMean #{c['r_bm']} -> BlurWeighted #{c['r_bw']} (gain {c['gain']:+d}) | {c['caption'][:65]}")
