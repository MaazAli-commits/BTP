# TVPReID Reliability-Aware Matching: Final Experiment Results

## Overview
This document records the final training parameters, model checkpoints, and evaluation metrics for the Video-Language Person Re-Identification (VCLIP) system applied to the **TVPReID** datasets (Duke, iLIDS, PRID) and their combinations. 

Our primary novelty is the **Reliability-Aware Matching** formulation (`+q`). 

## 1. Hardware & Training Hyperparameters
Due to hardware constraints (8GB VRAM), training was scaled using gradient accumulation and FP16 autocast to prevent OOM errors while preserving effective batch sizes.
- **Base Architecture**: VCLIP adaptation with cross-modal fusion.
- **Physical Batch Size**: 8
- **Gradient Accumulation Steps**: 2 (Effective Batch Size = 16)
- **Optimizer**: AdamW, Learning Rate: `1e-4`
- **Frames per video**: 8
- **Early Stopping**: Patience = 3 epochs

## 2. Individual Dataset Results

### Dataset A: Duke
- **Training**: 17 epochs (Best validation at epoch 13, loss: 0.9722)
- **Evaluation** (Gallery: ~400 videos)
  - Baseline Clean: mAP 27.24% | Rank-1: 17.51%
  - Baseline Blurred (q=1): mAP 21.67% | Rank-1: 13.14%
  - Blurred Reliability (+q): **mAP 21.71%** | Rank-1: 13.14% *(+0.04% mAP boost)*

### Dataset B: iLIDS
- **Training**: 13 epochs (Best validation at epoch 9, loss: 1.3342)
- **Evaluation** (Gallery: 75 videos)
  - Baseline Clean: mAP 28.51% | Rank-1: 12.00%
  - Baseline Blurred (q=1): mAP 22.31% | Rank-1: 9.33%
  - Blurred Reliability (+q): **mAP 22.38%** | Rank-1: 9.33% *(+0.07% mAP boost)*

### Dataset C: PRID
- **Training**: 14 epochs (Best validation at epoch 10, loss: 0.9819)
- **Evaluation** (Gallery: 142 videos)
  - Baseline Clean: mAP 41.25% | Rank-1: 23.24%
  - Baseline Blurred (q=1): mAP 33.72% | Rank-1: 16.90%
  - Blurred Reliability (+q): **mAP 33.91%** | Rank-1: 16.90% *(+0.19% mAP boost)*

## 3. Unified Dataset (Model 4)
We combined the `train`, `val`, and `test` splits of Duke, iLIDS, and PRID into a single cohesive training pipeline, safely prefixing video IDs to avoid cross-dataset identity collisions.
- **Dataset Size**: 5,329 total training videos (667 batches per epoch)
- **Training**: 15 epochs (Best validation at epoch 11, loss: 1.0233)
- **Evaluation** (Gallery: 820 test videos)
  - Baseline Clean: mAP 19.67% | Rank-1: 9.39%
  - Baseline Blurred (q=1): mAP 16.02% | Rank-1: 7.80%
  - Blurred Reliability (+q): **mAP 16.05%** | Rank-1: 7.80% *(+0.03% mAP boost)*

### Unified Model Conclusion
Despite a much larger distractor gallery (820 test videos) which naturally lowers absolute metrics compared to smaller datasets (e.g., PRID's 142), Model 4 successfully learned unified embeddings across all three highly diverse datasets simultaneously. 

Importantly, the **Reliability Math (+q)** consistently demonstrated an mAP improvement over the baseline when faced with corrupted data across every single dataset configuration.
