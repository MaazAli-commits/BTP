import re
import numpy as np

log_path = r'C:\Users\hp\.gemini\antigravity-ide\brain\e06d28de-5769-4d52-a910-cfdc2f5669b3\.system_generated\tasks\task-3301.log'

with open(log_path, 'r') as f:
    content = f.read()

mean_r1, mean_r5, mean_r10, mean_map = [], [], [], []
weighted_r1, weighted_r5, weighted_r10, weighted_map = [], [], [], []

blocks = content.split('Evaluating ')

for block in blocks[1:]:
    if 'mean pooling' in block:
        is_mean = True
    elif 'weighted pooling' in block:
        is_mean = False
    else:
        continue
        
    r1 = float(re.search(r'Rank-1:\s+([\d\.]+)%', block).group(1))
    r5 = float(re.search(r'Rank-5:\s+([\d\.]+)%', block).group(1))
    r10 = float(re.search(r'Rank-10:\s+([\d\.]+)%', block).group(1))
    mAP = float(re.search(r'mAP:\s+([\d\.]+)%', block).group(1))
    
    if is_mean:
        mean_r1.append(r1)
        mean_r5.append(r5)
        mean_r10.append(r10)
        mean_map.append(mAP)
    else:
        weighted_r1.append(r1)
        weighted_r5.append(r5)
        weighted_r10.append(r10)
        weighted_map.append(mAP)

print("FINAL ABLATION RESULTS (Mean ± Std over 5 seeds)")
print("\n[MEAN POOLING]")
print(f"Rank-1:  {np.mean(mean_r1):.2f}% ± {np.std(mean_r1):.2f}%")
print(f"Rank-5:  {np.mean(mean_r5):.2f}% ± {np.std(mean_r5):.2f}%")
print(f"Rank-10: {np.mean(mean_r10):.2f}% ± {np.std(mean_r10):.2f}%")
print(f"mAP:     {np.mean(mean_map):.2f}% ± {np.std(mean_map):.2f}%")

print("\n[WEIGHTED POOLING]")
print(f"Rank-1:  {np.mean(weighted_r1):.2f}% ± {np.std(weighted_r1):.2f}%")
print(f"Rank-5:  {np.mean(weighted_r5):.2f}% ± {np.std(weighted_r5):.2f}%")
print(f"Rank-10: {np.mean(weighted_r10):.2f}% ± {np.std(weighted_r10):.2f}%")
print(f"mAP:     {np.mean(weighted_map):.2f}% ± {np.std(weighted_map):.2f}%")
