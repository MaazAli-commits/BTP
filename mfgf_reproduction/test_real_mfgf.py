import os
os.environ['TORCH_HOME'] = 'E:/BTP/.cache/torch'
os.environ['HF_HOME'] = 'E:/BTP/.cache/huggingface'

import torch
from real_mfgf import TrueMFGFModel
import time

def main():
    print("Initializing TrueMFGFModel...")
    t0 = time.time()
    model = TrueMFGFModel(d_tips=410, embed_dim=256)
    print(f"Initialization complete in {time.time() - t0:.2f}s")
    
    model.eval()
    
    # Dummy data
    B = 2
    T = 8
    # Video: (B, T, C, H, W) for ViT and S3D
    frames = torch.randn(B, T, 3, 224, 224)
    # Text: (B, L)
    input_ids = torch.randint(0, 30522, (B, 30))
    attention_mask = torch.ones(B, 30)
    
    print("Running forward pass...")
    with torch.no_grad():
        c_video, c_text, d2_video, d2_text, logit_scale = model(frames, input_ids, attention_mask)
        
    print("Forward pass successful!")
    print(f"c_video shape: {c_video.shape} (Expected: {B}, 256)")
    print(f"c_text shape: {c_text.shape} (Expected: {B}, 256)")
    print(f"d2_video shape: {d2_video.shape} (Expected: {B}, 410)")
    print(f"d2_text shape: {d2_text.shape} (Expected: {B}, 410)")
    print(f"logit_scale: {logit_scale.item()}")

if __name__ == '__main__':
    main()
