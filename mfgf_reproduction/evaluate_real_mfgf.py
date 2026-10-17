import os
import sys
import torch
from torch.utils.data import DataLoader
from transformers import BertTokenizer
import numpy as np

os.environ['TORCH_HOME'] = 'E:/BTP/.cache/torch'
os.environ['HF_HOME'] = 'E:/BTP/.cache/huggingface'

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from src.dataset.tvpreid import TVPReIDDataset
from real_mfgf import TrueMFGFModel

def calculate_metrics(similarity_matrix, num_descriptions=1):
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
    
    ap = 1.0 / (ranks + 1)
    mAP = ap.mean() * 100
    
    return {'R1': r1, 'R5': r5, 'R10': r10, 'mAP': mAP}

def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    data_dir = "../data/TVPReID/TVPReid-PRID"
    checkpoint_path = "../experiments/checkpoints/mfgf_prid_true/true_mfgf_epoch_20.pt"
    
    test_dataset = TVPReIDDataset(data_dir, split='test', num_frames=8)
    # Test loader with batch size 8 for speed
    test_loader = DataLoader(test_dataset, batch_size=8, shuffle=False, num_workers=2)
    
    tips_dict = torch.load('data/tips.pt')
    sample_tip = next(iter(tips_dict.values()))
    d_tips = sample_tip.shape[0]
    
    model = TrueMFGFModel(d_tips=d_tips, embed_dim=256).to(device)
    model.load_state_dict(torch.load(checkpoint_path, map_location=device))
    model.eval()
    
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    
    all_vid_embeds = []
    all_txt_embeds = []
    
    print("Extracting features for evaluation...")
    with torch.no_grad():
        for batch_idx, batch in enumerate(test_loader):
            frames = batch['frames'].to(device)
            texts = batch['text']
            
            tokens = tokenizer(texts, padding=True, truncation=True, return_tensors='pt', max_length=77).to(device)
            
            # We use c_video and c_text for retrieval as per paper
            c_video, c_text, _, _, _ = model(frames, tokens['input_ids'], tokens['attention_mask'])
            
            all_vid_embeds.append(c_video.cpu())
            all_txt_embeds.append(c_text.cpu())
            
            if batch_idx % 5 == 0:
                print(f"Processed {batch_idx+1}/{len(test_loader)} batches")
                
    all_vid_embeds = torch.cat(all_vid_embeds, dim=0) # (142, 256)
    all_txt_embeds = torch.cat(all_txt_embeds, dim=0) # (142, 256)
    
    print("Calculating similarity matrix...")
    similarity_matrix = all_txt_embeds @ all_vid_embeds.t() # (142, 142)
    
    print("Calculating metrics...")
    metrics = calculate_metrics(similarity_matrix.numpy(), num_descriptions=1)
    
    print("\n--- True MFGF Evaluation Results ---")
    print(f"Rank-1:  {metrics['R1']:.2f}%")
    print(f"Rank-5:  {metrics['R5']:.2f}%")
    print(f"Rank-10: {metrics['R10']:.2f}%")
    print(f"mAP:     {metrics['mAP']:.2f}%")

if __name__ == '__main__':
    main()
