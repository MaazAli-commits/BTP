import os
import argparse
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from transformers import DistilBertTokenizer
from src.dataset.tvpreid import TVPReIDDataset
from mfgf_reproduction.mfgf import MFGFModel

def compute_loss(vid_embeds, txt_embeds, logit_scale):
    logits_per_video = logit_scale * vid_embeds @ txt_embeds.t()
    logits_per_text = logits_per_video.t()
    labels = torch.arange(len(logits_per_video), device=vid_embeds.device)
    loss_v = nn.functional.cross_entropy(logits_per_video, labels)
    loss_t = nn.functional.cross_entropy(logits_per_text, labels)
    return (loss_v + loss_t) / 2

def main(args):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    os.makedirs(args.save_dir, exist_ok=True)
    
    model = MFGFModel(embed_dim=512).to(device)
    tokenizer = DistilBertTokenizer.from_pretrained('distilbert-base-uncased')
    
    train_dataset = TVPReIDDataset(data_dir=args.data_dir, split='train', num_frames=args.num_frames)
    val_dataset = TVPReIDDataset(data_dir=args.data_dir, split='val', num_frames=args.num_frames)
    
    def collate_fn(batch):
        frames = torch.stack([item['frames'] for item in batch])
        texts = [item['text'] for item in batch]
        video_ids = [item['video_id'] for item in batch]
        return frames, texts, video_ids
        
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, collate_fn=collate_fn)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, collate_fn=collate_fn)
    
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)
    scaler = torch.amp.GradScaler('cuda')
    
    best_val_loss = float('inf')
    patience_counter = 0
    patience = 3
    
    for epoch in range(1, args.epochs + 1):
        model.train()
        train_loss = 0.0
        for batch_idx, (frames, texts, _) in enumerate(train_loader):
            frames = frames.to(device)
            encoded_text = tokenizer(texts, return_tensors="pt", padding=True, truncation=True, max_length=77).to(device)
            
            optimizer.zero_grad()
            with torch.autocast(device_type='cuda', dtype=torch.float16):
                vid_embeds, txt_embeds, logit_scale = model(frames, encoded_text['input_ids'], encoded_text['attention_mask'])
                loss = compute_loss(vid_embeds, txt_embeds, logit_scale)
                
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            
            train_loss += loss.item()
            
        avg_train_loss = train_loss / len(train_loader)
        
        # Validation
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for frames, texts, _ in val_loader:
                frames = frames.to(device)
                encoded_text = tokenizer(texts, return_tensors="pt", padding=True, truncation=True, max_length=77).to(device)
                with torch.autocast(device_type='cuda', dtype=torch.float16):
                    vid_embeds, txt_embeds, logit_scale = model(frames, encoded_text['input_ids'], encoded_text['attention_mask'])
                    loss = compute_loss(vid_embeds, txt_embeds, logit_scale)
                val_loss += loss.item()
                
        avg_val_loss = val_loss / len(val_loader)
        print(f"Epoch {epoch}/{args.epochs} | Train Loss: {avg_train_loss:.4f} | Val Loss: {avg_val_loss:.4f}")
        
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            patience_counter = 0
            checkpoint_path = os.path.join(args.save_dir, "mfgf_best.pth")
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_loss': best_val_loss
            }, checkpoint_path)
            print(f"  -> Saved best model to {checkpoint_path}")
        else:
            patience_counter += 1
            print(f"  -> No improvement (patience {patience_counter}/{patience})")
            if patience_counter >= patience:
                print("Early stopping triggered!")
                break

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", type=str, default="data/TVPReID/TVPReid-PRID")
    parser.add_argument("--save_dir", type=str, default="experiments/checkpoints/mfgf_prid")
    parser.add_argument("--batch_size", type=int, default=2)
    parser.add_argument("--num_frames", type=int, default=8)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--epochs", type=int, default=15)
    args = parser.parse_args()
    main(args)
