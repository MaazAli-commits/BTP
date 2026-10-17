import os
import argparse
import torch
import numpy as np
from torch.utils.data import DataLoader
from transformers import CLIPModel, CLIPProcessor
from src.dataset.tvpreid import TVPReIDDataset
from src.models.vclip_adaptation import VCLIPVideoEncoder
from src.models.reliability_matching import ReliabilityAwareMatching
from src.models.quality_estimator import FrameQualityEstimator as QualityEstimator
import cv2

def calculate_metrics(similarity_matrix):
    num_queries = similarity_matrix.shape[0]
    rank1, rank5, rank10 = 0, 0, 0
    mAP = 0.0
    
    for i in range(num_queries):
        ranked_indices = torch.argsort(similarity_matrix[i], descending=True)
        rank = (ranked_indices == i).nonzero(as_tuple=True)[0].item()
        
        if rank == 0:
            rank1 += 1
        if rank < 5:
            rank5 += 1
        if rank < 10:
            rank10 += 1
            
        # For single ground-truth (identity mapping), AP = 1 / (rank + 1)
        mAP += 1.0 / (rank + 1)
        
    return (rank1/num_queries, rank5/num_queries, rank10/num_queries, mAP/num_queries)

def print_metrics(name, metrics):
    r1, r5, r10, map_score = metrics
    print(f"\n[{name}]\n  Rank-1: {r1*100:.2f}% | Rank-5: {r5*100:.2f}% | Rank-10: {r10*100:.2f}% | mAP: {map_score*100:.2f}%")

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

def main(args):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    clip_model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32").to(device)
    clip_processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
    
    video_encoder = VCLIPVideoEncoder(clip_model=clip_model).to(device)
    checkpoint_path = os.path.join(args.save_dir, "best_model.pth")
    checkpoint = torch.load(checkpoint_path, map_location=device)
    video_encoder.load_state_dict(checkpoint['model_state_dict'])
    video_encoder.eval()
    print(f"Loaded best_model.pth (Trained for {checkpoint['epoch']} epochs)")

    reliability_matcher = ReliabilityAwareMatching().to(device)
    quality_estimator = QualityEstimator().to(device)
    
    test_dataset = TVPReIDDataset(data_dir=args.data_dir, split='test', num_frames=args.num_frames)
    def collate_fn(batch):
        frames = torch.stack([item['frames'] for item in batch])
        texts = [item['text'] for item in batch]
        return frames, texts
        
    test_loader = DataLoader(test_dataset, batch_size=4, shuffle=False, num_workers=0, collate_fn=collate_fn)

    all_vid_pooled_clean, all_vid_pooled_blur_baseline, all_vid_pooled_blur_rel = [], [], []
    all_text_pooled = []
    all_q_j = []
    
    print("\n--- Extracting Features for Test Set ---")
    with torch.no_grad(), torch.autocast(device_type='cuda', dtype=torch.float16):
        for batch_idx, (frames_clean, texts) in enumerate(test_loader):
            frames_clean = frames_clean.to(device)
            
            # Clean
            vid_clean = video_encoder(frames_clean, q_j=None)
            all_vid_pooled_clean.append(vid_clean)
            
            # Blur
            frames_blur = torch.stack([blur_frames(f) for f in frames_clean]).to(device)
            vid_blur_baseline = video_encoder(frames_blur, q_j=None)
            all_vid_pooled_blur_baseline.append(vid_blur_baseline)
            
            # Quality (T=8)
            q_j = quality_estimator(frames_blur) # (B, T)
            all_q_j.append(q_j)
            
            # Blur with internal Reliability applied before Temporal Transformer
            vid_blur_rel = video_encoder(frames_blur, q_j=q_j)
            all_vid_pooled_blur_rel.append(vid_blur_rel)
            
            # Text
            text_inputs = clip_processor(text=texts, return_tensors="pt", padding=True, truncation=True).to(device)
            text_outputs = clip_model.text_model(**text_inputs)
            # Pooled text
            text_pooled = clip_model.text_projection(text_outputs.pooler_output)
            all_text_pooled.append(text_pooled)
            
            print(f"Processed batch {batch_idx+1}/{len(test_loader)}", end='\r')
            
    all_vid_pooled_clean = torch.cat(all_vid_pooled_clean, dim=0).float()
    all_vid_pooled_blur_baseline = torch.cat(all_vid_pooled_blur_baseline, dim=0).float()
    all_vid_pooled_blur_rel = torch.cat(all_vid_pooled_blur_rel, dim=0).float()
    all_text_pooled = torch.cat(all_text_pooled, dim=0).float()
    all_q_j = torch.cat(all_q_j, dim=0).float()
    
    # Normalize pooled
    all_vid_pooled_clean = all_vid_pooled_clean / all_vid_pooled_clean.norm(dim=-1, keepdim=True)
    all_vid_pooled_blur_baseline = all_vid_pooled_blur_baseline / all_vid_pooled_blur_baseline.norm(dim=-1, keepdim=True)
    all_vid_pooled_blur_rel = all_vid_pooled_blur_rel / all_vid_pooled_blur_rel.norm(dim=-1, keepdim=True)
    all_text_pooled = all_text_pooled / all_text_pooled.norm(dim=-1, keepdim=True)
    
    print("\n\n--- Running Pilot Experiments ---")
    print("\nSample q_j (Quality Scores) for the first video (Heterogeneous Corruption):")
    print(all_q_j[0])
    
    
    # Exp 1
    sim_clean = all_text_pooled @ all_vid_pooled_clean.t()
    print_metrics("Experiment 1: Baseline Clean", calculate_metrics(sim_clean))
    
    # Exp 2 (Baseline Blurred)
    sim_blur = all_text_pooled @ all_vid_pooled_blur_baseline.t()
    print_metrics("Experiment 2: Baseline Blurred (q=1)", calculate_metrics(sim_blur))
    
    # Exp 3 (Reliability Blurred)
    sim_rel = all_text_pooled @ all_vid_pooled_blur_rel.t()
    print_metrics("Experiment 3: Blurred Reliability Math (+q)", calculate_metrics(sim_rel))

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", type=str, default="data/TVPReID/TVPReid-Duke")
    parser.add_argument("--save_dir", type=str, default="experiments/checkpoints")
    parser.add_argument("--num_frames", type=int, default=8)
    args = parser.parse_args()
    main(args)
