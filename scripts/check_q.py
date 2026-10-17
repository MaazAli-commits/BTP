import torch
import torch.nn as nn
import cv2
import numpy as np
from src.dataset.tvpreid import TVPReIDDataset
from src.models.quality_estimator import FrameQualityEstimator
import os

def blur_frames(frames_tensor):
    blurred_frames = []
    half = len(frames_tensor) // 2
    for i, frame in enumerate(frames_tensor):
        if i < half:
            frame_np = frame.permute(1, 2, 0).cpu().numpy()
            blurred = cv2.GaussianBlur(frame_np, (25, 25), 0)
            blurred_tensor = torch.from_numpy(blurred).permute(2, 0, 1).to(frame.device)
            blurred_frames.append(blurred_tensor)
        else:
            blurred_frames.append(frame)
    return torch.stack(blurred_frames)

def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    dataset = TVPReIDDataset(data_dir="data/TVPReID/TVPReid-Duke", split='test', num_frames=8)
    estimator = FrameQualityEstimator().to(device)
    
    print("\n--- Diagnostic Check: Calibrated q_j Scores ---")
    
    for i in range(5):
        item = dataset[i]
        frames_raw = item['frames'].to(device).unsqueeze(0) # (1, 8, C, H, W)
        
        # Heterogeneous Blur (first 4 blurred, last 4 clear)
        frames_blur = torch.stack([blur_frames(f) for f in frames_raw]).to(device)
        
        # Get scores directly from the updated module
        q_j = estimator(frames_blur)
        
        print(f"\nVideo {i+1}:")
        print(f"  Blurred Frames (0-3): {q_j[0, :4].cpu().numpy()}")
        print(f"  Clear Frames   (4-7): {q_j[0, 4:].cpu().numpy()}")

if __name__ == "__main__":
    main()
