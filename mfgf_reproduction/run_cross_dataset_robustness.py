"""
Cross-Dataset q_j Robustness Evaluation
========================================
Runs the full quality-weighting robustness pipeline on iLIDS and Duke.

Steps per dataset:
1. Extract mean/weighted ViT features (if not already cached)
2. Train MFGF heads for 'mean' and 'weighted' pooling (5 seeds each)
3. Extract corrupted test features (blur latter half of frames)
4. Evaluate both pooling strategies on corrupted test data (5 seeds)
5. Report mean ± std results
"""

import os
import sys
import subprocess
import json
import numpy as np

PYTHON = r"e:\BTP\venv\Scripts\python.exe"
SEEDS = [42, 100, 2024, 777, 999]
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))


def run_cmd(args, desc=""):
    print(f"\n>>> {desc}")
    print(f"    CMD: {' '.join(args)}")
    result = subprocess.run(args, cwd=SCRIPT_DIR)
    if result.returncode != 0:
        print(f"    !!! FAILED (exit code {result.returncode})")
        return False
    return True


def step1_extract_features(dataset):
    """Extract mean/weighted ViT features for train and test splits."""
    train_path = f"data/{dataset}_features_ablation/train/mean_visual_features.pt"
    test_path = f"data/{dataset}_features_ablation/test/mean_visual_features.pt"
    
    if os.path.exists(os.path.join(SCRIPT_DIR, train_path)) and os.path.exists(os.path.join(SCRIPT_DIR, test_path)):
        print(f"\n[SKIP] {dataset} ablation features already exist.")
        return True
    
    return run_cmd(
        [PYTHON, "extract_weighted_vit.py", "--dataset", dataset],
        f"Extracting mean/weighted ViT features for {dataset}"
    )


def step2_train_heads(dataset):
    """Train MFGF heads for mean and weighted pooling (5 seeds each)."""
    for pool in ['mean', 'weighted']:
        for seed in SEEDS:
            ckpt_dir = f"../experiments/checkpoints/mfgf_{dataset.lower()}_offline_{pool}_{seed}"
            ckpt_path = os.path.join(SCRIPT_DIR, ckpt_dir, "offline_mfgf_final.pt")
            
            if os.path.exists(ckpt_path):
                print(f"\n[SKIP] {dataset} {pool} seed={seed} checkpoint already exists.")
                continue
            
            ok = run_cmd(
                [PYTHON, "train_mfgf_offline.py", "--dataset", dataset, "--pool", pool, "--seed", str(seed)],
                f"Training {dataset} {pool} pooling (seed={seed})"
            )
            if not ok:
                return False
            
            # Also evaluate on clean test data so we have the result JSON
            run_cmd(
                [PYTHON, "evaluate_mfgf_offline.py", "--dataset", dataset, "--pool", pool, "--seed", str(seed)],
                f"Evaluating {dataset} {pool} pooling (seed={seed}) on clean data"
            )
    return True


def step3_extract_corrupted_and_evaluate(dataset):
    """Extract corrupted test features and evaluate robustness."""
    corrupted_mean_path = os.path.join(SCRIPT_DIR, f"data/{dataset}_features_ablation/test/corrupted_mean.pt")
    
    if os.path.exists(corrupted_mean_path):
        print(f"\n[SKIP] {dataset} corrupted test features already exist.")
    
    # Run the 5-seed robustness ablation
    ok = run_cmd(
        [PYTHON, "run_robustness_ablation.py", "--dataset", dataset],
        f"Running 5-seed robustness ablation for {dataset}"
    )
    return ok


def collect_results(dataset):
    """Collect and print results from existing JSON files."""
    res_dir = os.path.join(SCRIPT_DIR, "..", "experiments", "results")
    
    print(f"\n{'='*60}")
    print(f"  {dataset} — CLEAN BASELINE RESULTS (5-seed mean ± std)")
    print(f"{'='*60}")
    
    for pool in ['mean', 'weighted']:
        r1, r5, r10, map_ = [], [], [], []
        for seed in SEEDS:
            path = os.path.join(res_dir, f"{dataset}_{pool}_{seed}.json")
            if os.path.exists(path):
                with open(path, 'r') as f:
                    metrics = json.load(f)
                r1.append(metrics['R1'])
                r5.append(metrics['R5'])
                r10.append(metrics['R10'])
                map_.append(metrics['mAP'])
            else:
                print(f"  [MISSING] {path}")
        
        if len(r1) == 5:
            print(f"\n  [{pool.upper()} POOLING] (Clean Test)")
            print(f"    Rank-1:  {np.mean(r1):.2f}% ± {np.std(r1):.2f}%")
            print(f"    Rank-5:  {np.mean(r5):.2f}% ± {np.std(r5):.2f}%")
            print(f"    Rank-10: {np.mean(r10):.2f}% ± {np.std(r10):.2f}%")
            print(f"    mAP:     {np.mean(map_):.2f}% ± {np.std(map_):.2f}%")
            
            # Also report best single seed
            best_r1_idx = np.argmax(r1)
            best_map_idx = np.argmax(map_)
            print(f"    Peak R1:  {max(r1):.2f}% (seed {SEEDS[best_r1_idx]})")
            print(f"    Peak mAP: {max(map_):.2f}% (seed {SEEDS[best_map_idx]})")
            same_seed = "YES" if best_r1_idx == best_map_idx else "NO"
            print(f"    Same seed for peak R1 & mAP: {same_seed}")


def main():
    datasets = ['iLIDs', 'Duke']
    
    for dataset in datasets:
        print(f"\n\n{'#'*60}")
        print(f"#  PROCESSING: {dataset}")
        print(f"{'#'*60}")
        
        # Step 1: Extract features
        if not step1_extract_features(dataset):
            print(f"FAILED at feature extraction for {dataset}. Stopping.")
            return
        
        # Step 2: Train heads
        if not step2_train_heads(dataset):
            print(f"FAILED at training for {dataset}. Stopping.")
            return
        
        # Step 3: Corrupted test + robustness evaluation
        if not step3_extract_corrupted_and_evaluate(dataset):
            print(f"FAILED at robustness evaluation for {dataset}. Stopping.")
            return
        
        # Collect and print results
        collect_results(dataset)
    
    print(f"\n\n{'*'*60}")
    print(f"  ALL DONE — Cross-dataset q_j robustness complete!")
    print(f"{'*'*60}")


if __name__ == '__main__':
    main()
