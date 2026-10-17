import numpy as np
from test_robustness import run_robustness_test

seeds = [42, 100, 2024, 777, 999]

mean_r1s = []
weighted_r1s = []

import argparse
parser = argparse.ArgumentParser()
parser.add_argument('--dataset', type=str, required=True, choices=['PRID', 'iLIDs', 'Duke'])
args = parser.parse_args()

for seed in seeds:
    print(f"\n{'='*40}\nEvaluating Seed: {seed}\n{'='*40}")
    metrics_mean, metrics_weighted = run_robustness_test(seed=seed, dataset_name=args.dataset)
    mean_r1s.append(metrics_mean['R1'])
    weighted_r1s.append(metrics_weighted['R1'])


print(f"\n{'='*40}\nFINAL ROBUSTNESS RESULTS (5 SEEDS)\n{'='*40}")
print("Mean Pooling R1s: ", mean_r1s)
print(f"Mean Pooling Avg Rank-1:     {np.mean(mean_r1s):.2f}% ± {np.std(mean_r1s):.2f}%")

print("Weighted Pooling R1s: ", weighted_r1s)
print(f"Weighted Pooling Avg Rank-1: {np.mean(weighted_r1s):.2f}% ± {np.std(weighted_r1s):.2f}%")
