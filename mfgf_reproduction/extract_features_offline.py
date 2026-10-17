import os
import sys
import torch
import argparse
import torchvision.models as models
import torchvision.models.video as video_models
from transformers import BertModel, BertTokenizer
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm

os.environ['TORCH_HOME'] = 'E:/BTP/.cache/torch'
os.environ['HF_HOME'] = 'E:/BTP/.cache/huggingface'

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from src.dataset.tvpreid import TVPReIDDataset

def extract_features(split, dataset_name):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    data_dir = f"../data/TVPReID/TVPReid-{dataset_name}"
    dataset = TVPReIDDataset(data_dir, split=split, num_frames=8)
    loader = DataLoader(dataset, batch_size=4, shuffle=False, num_workers=0)
    
    out_dir = f"data/{dataset_name}_features/{split}"
    os.makedirs(out_dir, exist_ok=True)
    
    # Dictionary to store features by video_id
    all_features = {}
    
    # 1. Extract Visual Features (ViT)
    print("\n--- Extracting Visual Features (ViT) ---")
    vit = models.vit_b_16(pretrained=True).to(device)
    vit.eval()
    with torch.no_grad():
        for batch in tqdm(loader, desc="ViT"):
            frames = batch['frames'].to(device) # (B, T, C, H, W)
            vid_ids = batch['video_id']
            
            B, T, C, H, W = frames.shape
            frames_flat = frames.view(B * T, C, H, W)
            visual_feat = vit._process_input(frames_flat)
            n = visual_feat.shape[0]
            batch_class_token = vit.class_token.expand(n, -1, -1)
            visual_feat = torch.cat([batch_class_token, visual_feat], dim=1)
            visual_feat = vit.encoder(visual_feat)
            visual_feat = visual_feat[:, 0] # (B*T, 768)
            visual_feat = visual_feat.view(B, T, -1).mean(dim=1) # (B, 768)
            
            for i, vid in enumerate(vid_ids):
                if vid not in all_features:
                    all_features[vid] = {}
                all_features[vid]['visual'] = visual_feat[i].cpu()
                all_features[vid]['text_raw'] = batch['text'][i]
    
    del vit
    torch.cuda.empty_cache()
    
    # 2. Extract Motion Features (S3D)
    print("\n--- Extracting Motion Features (S3D) ---")
    s3d = video_models.s3d(weights=video_models.S3D_Weights.KINETICS400_V1).to(device)
    s3d.eval()
    with torch.no_grad():
        for batch in tqdm(loader, desc="S3D"):
            frames = batch['frames'].to(device)
            vid_ids = batch['video_id']
            
            B = frames.shape[0]
            frames_s3d = frames.permute(0, 2, 1, 3, 4)
            motion_feat = s3d.features(frames_s3d) 
            motion_feat = F.adaptive_avg_pool3d(motion_feat, (1, 1, 1)).view(B, -1) # (B, 1024)
            
            for i, vid in enumerate(vid_ids):
                all_features[vid]['motion'] = motion_feat[i].cpu()
                
    del s3d
    torch.cuda.empty_cache()
    
    # 3. Extract Text Features (BERT)
    print("\n--- Extracting Text Features (BERT) ---")
    bert = BertModel.from_pretrained('bert-base-uncased').to(device)
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    bert.eval()
    with torch.no_grad():
        for vid in tqdm(all_features.keys(), desc="BERT"):
            text = all_features[vid]['text_raw']
            tokens = tokenizer([text], padding=True, truncation=True, return_tensors='pt', max_length=77).to(device)
            outputs = bert(**tokens)
            cls_token = outputs.last_hidden_state[:, 0, :]
            all_features[vid]['text'] = cls_token[0].cpu() # (768,)
            
    del bert
    torch.cuda.empty_cache()
    
    # Save the consolidated features
    print("\nSaving features...")
    torch.save(all_features, os.path.join(out_dir, 'extracted_features.pt'))
    print(f"Successfully saved {len(all_features)} items to {out_dir}/extracted_features.pt")

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, required=True, choices=['PRID', 'iLIDs', 'Duke'])
    args = parser.parse_args()
    extract_features('train', args.dataset)
    extract_features('test', args.dataset)
