import os
import subprocess
import json
import numpy as np
import argparse

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, required=True, choices=['PRID', 'iLIDs', 'Duke'])
    parser.add_argument('--pools', type=str, nargs='+', default=['baseline'], choices=['baseline', 'mean', 'weighted'])
    args = parser.parse_args()
    seeds = [42, 100, 2024, 777, 999]
    pools = args.pools
    
    for pool in pools:
        print(f"\n{'='*50}")
        print(f"Running Ablation: {pool.upper()} POOLING")
        print(f"{'='*50}")
        
        for seed in seeds:
            print(f"\n--- Seed {seed} ---")
            
            # Train
            print("Training...")
            subprocess.run([
                r"e:\BTP\venv\Scripts\python.exe", 
                "train_mfgf_offline.py", 
                "--dataset", args.dataset,
                "--pool", pool, 
                "--seed", str(seed)
            ], check=True)
            
            # Evaluate
            print("Evaluating...")
            subprocess.run([
                r"e:\BTP\venv\Scripts\python.exe", 
                "evaluate_mfgf_offline.py", 
                "--dataset", args.dataset,
                "--pool", pool, 
                "--seed", str(seed)
            ], check=True)
            
    print("\n\n" + "*"*50)
    print("FINAL ABLATION RESULTS (Mean ± Std over 5 seeds)")
    print("*"*50)
    
    res_dir = "../experiments/results"
    
    for pool in pools:
        r1, r5, r10, map_ = [], [], [], []
        for seed in seeds:
            path = os.path.join(res_dir, f"{args.dataset}_{pool}_{seed}.json")
            with open(path, 'r') as f:
                metrics = json.load(f)
            r1.append(metrics['R1'])
            r5.append(metrics['R5'])
            r10.append(metrics['R10'])
            map_.append(metrics['mAP'])
            
        print(f"\n[{pool.upper()} POOLING]")
        print(f"Rank-1:  {np.mean(r1):.2f}% ± {np.std(r1):.2f}%")
        print(f"Rank-5:  {np.mean(r5):.2f}% ± {np.std(r5):.2f}%")
        print(f"Rank-10: {np.mean(r10):.2f}% ± {np.std(r10):.2f}%")
        print(f"mAP:     {np.mean(map_):.2f}% ± {np.std(map_):.2f}%")

if __name__ == '__main__':
    main()
