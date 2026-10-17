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

def main():
    print("=" * 70)
    print("UNIFIED MFGF EXPERIMENT PIPELINE (PRID + iLIDS + Duke)")
    print("5 Seeds: [42, 100, 2024, 777, 999]")
    print("=" * 70)

    # 1. Train heads for 'mean' and 'weighted' pooling across all 5 seeds
    print("\n[PHASE 1] Training Unified MFGF Heads...")
    for pool in ['mean', 'weighted']:
        for seed in SEEDS:
            ckpt_path = os.path.join(
                SCRIPT_DIR,
                f"../experiments/checkpoints/mfgf_unified_offline_{pool}_{seed}/offline_mfgf_final.pt"
            )
            if os.path.exists(ckpt_path):
                print(f"[SKIP] Checkpoint already exists for {pool} seed={seed}")
                continue
                
            ok = run_cmd(
                [PYTHON, "train_unified_mfgf.py", "--pool", pool, "--seed", str(seed)],
                f"Training Unified MFGF ({pool} pooling, seed={seed})"
            )
            if not ok:
                print(f"Error training {pool} seed={seed}")
                sys.exit(1)

    # 2. Evaluate Clean Test Split (5 seeds)
    print("\n[PHASE 2] Evaluating Clean Unified Test Split (820 videos)...")
    for pool in ['mean', 'weighted']:
        for seed in SEEDS:
            res_file = os.path.join(SCRIPT_DIR, f"../experiments/results/Unified_clean_{pool}_{seed}.json")
            if os.path.exists(res_file):
                print(f"[SKIP] Clean result already exists for {pool} seed={seed}")
                continue
            ok = run_cmd(
                [PYTHON, "evaluate_unified_mfgf.py", "--pool", pool, "--seed", str(seed), "--split", "clean"],
                f"Evaluating Clean Unified ({pool} pooling, seed={seed})"
            )
            if not ok:
                print(f"Error evaluating clean {pool} seed={seed}")
                sys.exit(1)

    # 3. Evaluate Corrupted Test Split (50% Gaussian blur, 5 seeds)
    print("\n[PHASE 3] Evaluating Corrupted Unified Test Split (820 videos, 50% blur)...")
    for pool in ['mean', 'weighted']:
        for seed in SEEDS:
            res_file = os.path.join(SCRIPT_DIR, f"../experiments/results/Unified_corrupted_{pool}_{seed}.json")
            if os.path.exists(res_file):
                print(f"[SKIP] Corrupted result already exists for {pool} seed={seed}")
                continue
            ok = run_cmd(
                [PYTHON, "evaluate_unified_mfgf.py", "--pool", pool, "--seed", str(seed), "--split", "corrupted"],
                f"Evaluating Corrupted Unified ({pool} pooling, seed={seed})"
            )
            if not ok:
                print(f"Error evaluating corrupted {pool} seed={seed}")
                sys.exit(1)

    # 4. Aggregate Results
    res_dir = os.path.join(SCRIPT_DIR, "..", "experiments", "results")
    
    summary = {}
    for split in ['clean', 'corrupted']:
        summary[split] = {}
        for pool in ['mean', 'weighted']:
            r1s, r5s, r10s, maps = [], [], [], []
            for seed in SEEDS:
                fpath = os.path.join(res_dir, f"Unified_{split}_{pool}_{seed}.json")
                if os.path.exists(fpath):
                    with open(fpath, 'r') as f:
                        data = json.load(f)
                    r1s.append(data['R1'])
                    r5s.append(data['R5'])
                    r10s.append(data['R10'])
                    maps.append(data['mAP'])
            summary[split][pool] = {
                'R1_mean': float(np.mean(r1s)), 'R1_std': float(np.std(r1s)),
                'R5_mean': float(np.mean(r5s)), 'R5_std': float(np.std(r5s)),
                'R10_mean': float(np.mean(r10s)), 'R10_std': float(np.std(r10s)),
                'mAP_mean': float(np.mean(maps)), 'mAP_std': float(np.std(maps)),
                'R1_raw': r1s, 'mAP_raw': maps
            }

    print("\n" + "=" * 75)
    print("           FINAL UNIFIED MFGF REPRODUCTION REPORT (820 TEST VIDEOS)")
    print("=" * 75)
    
    print("\n1. CLEAN TEST SET (5-Seed Mean ± Std):")
    print(f"{'Method':<20} | {'Rank-1 (%)':<15} | {'Rank-5 (%)':<15} | {'Rank-10 (%)':<15} | {'mAP (%)':<15}")
    print("-" * 88)
    for pool, label in [('mean', 'Uniform Mean Pool'), ('weighted', 'Quality-Weighted (q_j)')]:
        m = summary['clean'][pool]
        print(f"{label:<20} | {m['R1_mean']:.2f} ± {m['R1_std']:.2f}    | {m['R5_mean']:.2f} ± {m['R5_std']:.2f}    | {m['R10_mean']:.2f} ± {m['R10_std']:.2f}    | {m['mAP_mean']:.2f} ± {m['mAP_std']:.2f}")

    print("\n2. CORRUPTED TEST SET — 50% GAUSSIAN BLUR (5-Seed Mean ± Std):")
    print(f"{'Method':<20} | {'Rank-1 (%)':<15} | {'Rank-5 (%)':<15} | {'Rank-10 (%)':<15} | {'mAP (%)':<15}")
    print("-" * 88)
    for pool, label in [('mean', 'Uniform Mean Pool'), ('weighted', 'Quality-Weighted (q_j)')]:
        m = summary['corrupted'][pool]
        print(f"{label:<20} | {m['R1_mean']:.2f} ± {m['R1_std']:.2f}    | {m['R5_mean']:.2f} ± {m['R5_std']:.2f}    | {m['R10_mean']:.2f} ± {m['R10_std']:.2f}    | {m['mAP_mean']:.2f} ± {m['mAP_std']:.2f}")

    delta_clean = summary['clean']['weighted']['R1_mean'] - summary['clean']['mean']['R1_mean']
    delta_corr = summary['corrupted']['weighted']['R1_mean'] - summary['corrupted']['mean']['R1_mean']
    print("\n" + "-" * 75)
    print(f"Gain from Quality-Weighting (q_j) on Clean:     {delta_clean:+.2f}% Rank-1")
    print(f"Gain from Quality-Weighting (q_j) on Corrupted: {delta_corr:+.2f}% Rank-1")
    print("-" * 75)

    summary_file = os.path.join(res_dir, "Unified_summary_5seeds.json")
    with open(summary_file, 'w') as f:
        json.dump(summary, f, indent=2)
    print(f"\nConsolidated 5-seed summary saved to: {summary_file}")

if __name__ == '__main__':
    main()
