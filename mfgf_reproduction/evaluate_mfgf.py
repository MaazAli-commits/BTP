import os
import argparse
import torch
from torch.utils.data import DataLoader
from transformers import DistilBertTokenizer
import numpy as np
from src.dataset.tvpreid import TVPReIDDataset
from mfgf_reproduction.mfgf import MFGFModel

def calculate_metrics(similarity_matrix, num_descriptions=2):
    # similarity_matrix is (num_texts, num_videos)
    num_texts, num_videos = similarity_matrix.shape
    num_unique_videos = num_texts // num_descriptions
    assert num_unique_videos == num_videos, "Mismatch in similarity matrix dimensions"
    
    ranks = np.zeros(num_texts)
    for i in range(num_texts):
        correct_vid_idx = i // num_descriptions
        sorted_indices = np.argsort(similarity_matrix[i])[::-1]
        rank = np.where(sorted_indices == correct_vid_idx)[0][0]
        ranks[i] = rank
        
    r1 = (ranks == 0).mean() * 100
    r5 = (ranks < 5).mean() * 100
    r10 = (ranks < 10).mean() * 100
    
    # mAP calculation for single ground-truth retrieval
    ap = 1.0 / (ranks + 1)
    mAP = ap.mean() * 100
    
    return {'R1': r1, 'R5': r5, 'R10': r10, 'mAP': mAP}

def main(args):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    tokenizer = DistilBertTokenizer.from_pretrained('distilbert-base-uncased')
    model = MFGFModel(embed_dim=512)
    
    checkpoint_path = os.path.join(args.save_dir, "mfgf_best_light.pth")
    checkpoint = torch.load(checkpoint_path, map_location='cpu', mmap=True, weights_only=True)
    trained_epoch = checkpoint['epoch']
    model.load_state_dict(checkpoint['model_state_dict'], assign=True)
    del checkpoint
    
    model = model.to(device)
    model.eval()
    print(f"Loaded mfgf_best.pth (Trained for {trained_epoch} epochs)")
    
    test_dataset = TVPReIDDataset(data_dir=args.data_dir, split='test', num_frames=args.num_frames)
    
    def collate_fn(batch):
        frames = torch.stack([item['frames'] for item in batch])
        texts = [item['text'] for item in batch]
        video_ids = [item['video_id'] for item in batch]
        return frames, texts, video_ids
        
    test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False, collate_fn=collate_fn)
    
    all_vid_embeds = []
    all_txt_embeds = []
    
    with torch.no_grad(), torch.autocast(device_type='cuda', dtype=torch.float16):
        for batch_idx, (frames, texts, _) in enumerate(test_loader):
            torch.cuda.empty_cache()
            print(f"Processing batch {batch_idx+1}/{len(test_loader)}... frames shape: {frames.shape}")
            frames = frames.to(device)
            encoded_text = tokenizer(texts, return_tensors="pt", padding=True, truncation=True, max_length=77).to(device)
            
            num_texts = encoded_text['input_ids'].size(0)
            frames = frames.expand(num_texts, -1, -1, -1, -1)
            
            vid_embeds, txt_embeds, _ = model(frames, encoded_text['input_ids'], encoded_text['attention_mask'], q_j=None)
            
            all_vid_embeds.append(vid_embeds[0].cpu())
            for i in range(num_texts):
                all_txt_embeds.append(txt_embeds[i].cpu())
                
    all_vid_embeds = torch.stack(all_vid_embeds)
    all_txt_embeds = torch.stack(all_txt_embeds)
    
    similarity_matrix = all_txt_embeds @ all_vid_embeds.t()
    metrics = calculate_metrics(similarity_matrix.numpy(), num_descriptions=1)
    
    print("\n--- MFGF Evaluation Results ---")
    print(f"Rank-1:  {metrics['R1']:.2f}%")
    print(f"Rank-5:  {metrics['R5']:.2f}%")
    print(f"Rank-10: {metrics['R10']:.2f}%")
    print(f"mAP:     {metrics['mAP']:.2f}%")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", type=str, default="data/TVPReID/TVPReid-PRID")
    parser.add_argument("--save_dir", type=str, default="experiments/checkpoints/mfgf_prid")
    parser.add_argument("--num_frames", type=int, default=8)
    args = parser.parse_args()
    main(args)
