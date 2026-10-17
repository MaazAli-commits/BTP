import os
import argparse
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, ConcatDataset
from transformers import CLIPModel, CLIPProcessor
from src.dataset.tvpreid import TVPReIDDataset
from src.models.vclip_adaptation import VCLIPVideoEncoder

def compute_contrastive_loss(video_embeds, text_embeds, logit_scale):
    video_embeds = video_embeds / video_embeds.norm(dim=-1, keepdim=True)
    text_embeds = text_embeds / text_embeds.norm(dim=-1, keepdim=True)
    logits_per_video = logit_scale * video_embeds @ text_embeds.t()
    logits_per_text = logits_per_video.t()
    labels = torch.arange(len(logits_per_video), device=video_embeds.device)
    loss_v = nn.functional.cross_entropy(logits_per_video, labels)
    loss_t = nn.functional.cross_entropy(logits_per_text, labels)
    return (loss_v + loss_t) / 2

def main(args):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    clip_model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32").to(device)
    clip_processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
    for param in clip_model.text_model.parameters():
        param.requires_grad = False
    for param in clip_model.vision_model.parameters():
        param.requires_grad = False
    clip_model.vision_model.gradient_checkpointing_enable()
    video_encoder = VCLIPVideoEncoder(clip_model=clip_model).to(device)
    
    # UNIFIED DATASET LOGIC
    datasets = ["data/TVPReID/TVPReid-Duke", "data/TVPReID/TVPReid-iLIDs", "data/TVPReID/TVPReid-PRID"]
    prefixes = ["Duke_", "iLIDS_", "PRID_"]
    
    train_dsets = []
    val_dsets = []
    for d_path, pfx in zip(datasets, prefixes):
        train_dsets.append(TVPReIDDataset(data_dir=d_path, split='train', num_frames=args.num_frames, prefix=pfx))
        val_dsets.append(TVPReIDDataset(data_dir=d_path, split='val', num_frames=args.num_frames, prefix=pfx))
        
    train_dataset = ConcatDataset(train_dsets)
    val_dataset = ConcatDataset(val_dsets)
    
    def collate_fn(batch):
        frames = torch.stack([item['frames'] for item in batch])
        texts = [item['text'] for item in batch]
        video_ids = [item['video_id'] for item in batch]
        return frames, texts, video_ids
        
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, 
                              num_workers=0, collate_fn=collate_fn)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, 
                            num_workers=0, collate_fn=collate_fn)
                              
    optimizer = torch.optim.AdamW(video_encoder.parameters(), lr=args.lr)
    scaler = torch.cuda.amp.GradScaler()
    
    start_epoch = 0
    if args.resume_from and os.path.exists(args.resume_from):
        print(f"Resuming from checkpoint: {args.resume_from}")
        checkpoint = torch.load(args.resume_from, map_location=device)
        video_encoder.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        start_epoch = checkpoint['epoch'] + 1
        
    os.makedirs(args.save_dir, exist_ok=True)
    logit_scale = nn.Parameter(torch.ones([]) * torch.tensor(1 / 0.07).log()).to(device)
    
    grad_accum_steps = args.grad_accum
    best_val_loss = float('inf')
    patience_counter = 0

    for epoch in range(start_epoch, args.epochs):
        video_encoder.train()
        total_loss = 0
        optimizer.zero_grad()
        
        for batch_idx, (frames, texts, _) in enumerate(train_loader):
            frames = frames.to(device)
            text_inputs = clip_processor(text=texts, return_tensors="pt", padding=True, truncation=True).to(device)
            with torch.no_grad():
                text_outputs = clip_model.text_model(**text_inputs)
                text_embeds = clip_model.text_projection(text_outputs.pooler_output)
                
            with torch.autocast(device_type='cuda', dtype=torch.float16):
                video_embeds = video_encoder(frames)
                loss = compute_contrastive_loss(video_embeds, text_embeds, logit_scale.exp())
                loss = loss / grad_accum_steps
            
            scaler.scale(loss).backward()
            
            if (batch_idx + 1) % grad_accum_steps == 0 or (batch_idx + 1) == len(train_loader):
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad()
            
            total_loss += loss.item() * grad_accum_steps
            
            if batch_idx % 10 == 0:
                print(f"Epoch {epoch} | Batch {batch_idx}/{len(train_loader)} | Train Loss: {loss.item() * grad_accum_steps:.4f}")
                
        avg_train_loss = total_loss / len(train_loader)
        print(f"==== Epoch {epoch} completed | Avg Train Loss: {avg_train_loss:.4f} ====")
        
        video_encoder.eval()
        val_loss = 0
        print(f"Starting validation for Epoch {epoch}...")
        
        with torch.no_grad():
            for frames, texts, _ in val_loader:
                frames = frames.to(device)
                text_inputs = clip_processor(text=texts, return_tensors="pt", padding=True, truncation=True).to(device)
                text_outputs = clip_model.text_model(**text_inputs)
                text_embeds = clip_model.text_projection(text_outputs.pooler_output)
                with torch.autocast(device_type='cuda', dtype=torch.float16):
                    video_embeds = video_encoder(frames)
                    loss = compute_contrastive_loss(video_embeds, text_embeds, logit_scale.exp())
                val_loss += loss.item()
                
        avg_val_loss = val_loss / len(val_loader)
        print(f"==== Epoch {epoch} Validation | Avg Val Loss: {avg_val_loss:.4f} ====", flush=True)
        
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            patience_counter = 0
            save_path = os.path.join(args.save_dir, "best_model.pth")
            torch.save({
                'epoch': epoch,
                'model_state_dict': video_encoder.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': avg_train_loss,
                'val_loss': avg_val_loss,
            }, save_path, _use_new_zipfile_serialization=False)
            print(f"*** Validation improved! Saved NEW BEST checkpoint to {save_path} ***", flush=True)
        else:
            patience_counter += 1
            print(f"!!! Validation loss did not improve. Patience: {patience_counter}/{args.patience} !!!")
            if patience_counter >= args.patience:
                print(f"\n>>> EARLY STOPPING TRIGGERED! Training aborted after {epoch+1} epochs. <<<")
                print(f">>> Best Validation Loss: {best_val_loss:.4f} <<<")
                break

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train TVPReID Unified Baseline (Model 4)")
    parser.add_argument("--save_dir", type=str, default="experiments/checkpoints/unified", help="Where to save models")
    parser.add_argument("--batch_size", type=int, default=8, help="Batch size")
    parser.add_argument("--num_frames", type=int, default=8, help="Frames per video")
    parser.add_argument("--epochs", type=int, default=30, help="Total epochs to train (Max 30)")
    parser.add_argument("--lr", type=float, default=1e-4, help="Learning rate")
    parser.add_argument("--resume_from", type=str, default=None, help="Path to checkpoint")
    parser.add_argument("--grad_accum", type=int, default=2, help="Gradient accumulation steps")
    parser.add_argument("--patience", type=int, default=3, help="Early stopping patience")
    args = parser.parse_args()
    main(args)
