import os
import json
import numpy as np
from PIL import Image

def generate_dummy_mevid(output_dir, num_pids=5, tracklets_per_pid=2, frames_per_tracklet=8):
    os.makedirs(output_dir, exist_ok=True)
    
    attributes = {}
    
    for split in ['train', 'test']:
        bbox_dir = os.path.join(output_dir, f"bbox_{split}")
        os.makedirs(bbox_dir, exist_ok=True)
        
        name_file_lines = []
        track_info_lines = []
        
        current_frame_idx = 0
        
        # PIDs 0 to num_pids-1 for train, and same (or different) for test.
        # For a proper Re-ID dataset, train and test PIDs are usually disjoint.
        # Let's do pid 0,1,2 for train, 3,4 for test.
        if split == 'train':
            pid_range = range(0, num_pids // 2 + 1)
        else:
            pid_range = range(num_pids // 2 + 1, num_pids)
            
        for i in pid_range:
            pid = f"pid_{i:04d}"
            
            # Attributes for query stage 2
            if pid not in attributes:
                attributes[pid] = {
                    "gender": np.random.choice(["male", "female"]),
                    "top_color": np.random.choice(["red", "blue", "black", "white"]),
                    "bottom_color": np.random.choice(["blue", "black", "grey", "brown"])
                }
            
            for j in range(tracklets_per_pid):
                tracklet_id = f"tracklet_{j:03d}"
                start_idx = current_frame_idx
                
                base_color = np.random.randint(0, 255, size=3)
                
                for k in range(frames_per_tracklet):
                    # Flat structure inside bbox_train/test, or pid folders. Let's use pid folders for cleanliness
                    # even within bbox_train, or just flat. The prompt says bbox_train/ then image-chip filenames.
                    # We will use flat naming: pid_tracklet_frame.jpg
                    frame_name = f"{pid}_{tracklet_id}_f{k:03d}.jpg"
                    frame_path = os.path.join(bbox_dir, frame_name)
                    
                    noise = np.random.randint(-20, 20, size=3)
                    color = tuple(np.clip(base_color + noise, 0, 255).astype(int))
                    
                    img = Image.new('RGB', (224, 224), color=color)
                    img.save(frame_path)
                    
                    name_file_lines.append(frame_name)
                    current_frame_idx += 1
                
                # Format: pid tracklet_id start_idx length
                track_info_lines.append(f"{pid} {tracklet_id} {start_idx} {frames_per_tracklet}")
                
        with open(os.path.join(output_dir, f"{split}_name.txt"), 'w') as f:
            f.write('\n'.join(name_file_lines))
            
        with open(os.path.join(output_dir, f"track_{split}_info.txt"), 'w') as f:
            f.write('\n'.join(track_info_lines))
            
        if split == 'test':
            # Pick first tracklet of each PID as query, others as gallery
            query_idxs = []
            for idx, line in enumerate(track_info_lines):
                if "tracklet_000" in line:
                    query_idxs.append(str(idx))
            with open(os.path.join(output_dir, "query_IDX.txt"), 'w') as f:
                f.write('\n'.join(query_idxs))
                
    with open(os.path.join(output_dir, "annotations.json"), 'w') as f:
        json.dump(attributes, f, indent=4)
        
    print(f"Generated official-layout dummy MEVID dataset at {output_dir}")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--output_dir', type=str, default='e:/BTP/data/MEVID_dummy')
    args = parser.parse_args()
    generate_dummy_mevid(args.output_dir)
