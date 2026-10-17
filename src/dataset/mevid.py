import os
import json
import torch
from torch.utils.data import Dataset
from PIL import Image

class MEVIDDataset(Dataset):
    """
    Official MEVID Dataset loader.
    Expects structure:
    mevid/
    ├── bbox_train/
    ├── bbox_test/
    ├── train_name.txt
    ├── test_name.txt
    ├── track_train_info.txt
    ├── track_test_info.txt
    └── query_IDX.txt
    """
    def __init__(self, data_dir, transform=None, visual_corruptor=None, split='train', num_frames=8, role='all'):
        """
        split: 'train' or 'test'
        role: 'query', 'gallery', or 'all' (only applicable if split=='test')
        """
        self.data_dir = data_dir
        self.transform = transform
        self.visual_corruptor = visual_corruptor
        self.split = split
        self.num_frames = num_frames
        self.role = role
        
        self.tracklets = []
        self._load_data()
        
    def _load_data(self):
        if not os.path.exists(self.data_dir):
            print(f"Warning: Data directory {self.data_dir} not found.")
            return

        bbox_dir = os.path.join(self.data_dir, f"bbox_{self.split}")
        name_file = os.path.join(self.data_dir, f"{self.split}_name.txt")
        info_file = os.path.join(self.data_dir, f"track_{self.split}_info.txt")
        query_file = os.path.join(self.data_dir, "query_IDX.txt")
        
        if not (os.path.exists(bbox_dir) and os.path.exists(name_file) and os.path.exists(info_file)):
            print(f"Warning: Official MEVID files missing in {self.data_dir} for split {self.split}.")
            return

        with open(name_file, 'r') as f:
            frame_names = [line.strip() for line in f.readlines() if line.strip()]
            
        with open(info_file, 'r') as f:
            track_infos = [line.strip() for line in f.readlines() if line.strip()]
            
        query_idxs = set()
        if self.split == 'test' and os.path.exists(query_file):
            with open(query_file, 'r') as f:
                query_idxs = set([int(line.strip()) for line in f.readlines() if line.strip()])
                
        for idx, track_line in enumerate(track_infos):
            # Assumed format: pid tracklet_id start_idx length
            parts = track_line.split()
            if len(parts) >= 4:
                pid = parts[0]
                tracklet_id = parts[1]
                start_idx = int(parts[2])
                length = int(parts[3])
                
                # Filter by role for test set
                is_query = idx in query_idxs
                if self.split == 'test':
                    if self.role == 'query' and not is_query:
                        continue
                    if self.role == 'gallery' and is_query:
                        continue
                        
                frames = [os.path.join(bbox_dir, frame_names[start_idx + i]) for i in range(length)]
                
                self.tracklets.append({
                    'pid': pid,
                    'tracklet_id': tracklet_id,
                    'frames': frames,
                    'is_query': is_query
                })
                
    def __len__(self):
        return len(self.tracklets)
        
    def __getitem__(self, idx):
        tracklet_info = self.tracklets[idx]
        frame_paths = tracklet_info['frames']
        
        if len(frame_paths) >= self.num_frames:
            indices = torch.linspace(0, len(frame_paths) - 1, self.num_frames).long()
        else:
            indices = torch.cat([
                torch.arange(len(frame_paths)),
                torch.randint(0, len(frame_paths), (self.num_frames - len(frame_paths),))
            ])
            
        sampled_paths = [frame_paths[i] for i in indices]
        
        frames = []
        for path in sampled_paths:
            try:
                img = Image.open(path).convert('RGB')
                frames.append(img)
            except Exception as e:
                pass
                
        # Apply visual corruptions if provided
        if self.visual_corruptor is not None and len(frames) > 0:
            frames = self.visual_corruptor(frames)
            
        # Apply transforms
        processed_frames = []
        for img in frames:
            if self.transform is not None:
                processed_frames.append(self.transform(img))
            else:
                processed_frames.append(img)
                
        if len(processed_frames) == 0:
            processed_frames = [torch.zeros(3, 224, 224) for _ in range(self.num_frames)]
            
        if isinstance(processed_frames[0], torch.Tensor):
            frames_tensor = torch.stack(processed_frames) # (T, C, H, W)
        else:
            frames_tensor = processed_frames
            
        return {
            'frames': frames_tensor,
            'pid': tracklet_info['pid'],
            'tracklet_id': tracklet_info['tracklet_id'],
            'is_query': tracklet_info['is_query']
        }
