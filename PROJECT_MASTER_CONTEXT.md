# 📌 BTP Project Master Context & Ground Truth (Team 33)

> **CRITICAL DIRECTIVE FOR ALL FUTURE SESSIONS:**  
> This file is the permanent single source of truth for this project. Read this file first before taking any action or answering questions about project goals, semester plans, model architectures, or past results.

---

## 1. Project Identity & Core Novelty
* **Title**: Query-Based Cross-Camera Video Person Re-Identification under Dual Degradation
* **Team**: Team 33 (Mohammed Maaz Ali, Rupsika, Siva, Harshith)
* **Dataset**: **TVPReID** (6,559 videos, 13,118 human-written descriptions across Duke, iLIDS, PRID subsets).
* **The Core Problem**: In real-world surveillance, **both** modalities are imperfect:
  1. **Degraded Text**: The witness description is partial or uncertain (e.g., *"wearing blue jeans and maybe a red backpack"*).
  2. **Degraded Video**: CCTV footage has blur, darkness, noise, or partial occlusion.
* **The Primary Mathematical Contribution**:
  $$\alpha'_{ij} = \frac{\alpha_{ij} \, c_i \, q_j}{\sum_{i,j} \alpha_{ij} \, c_i \, q_j}$$
  * $q_j$: **Frame Quality Weight** (computed unsupervised via Laplacian variance & illumination; clear frames get $\sim 0.75$, degraded frames get $\sim 0.25$).
  * $c_i$: **Text Confidence Weight** (derived from witness phrasing; definite attributes get $1.0$, hedged words like *"maybe"*, *"probably"* get $\sim 0.3$).
  * **Key Property**: This formula operates at **inference time** as a re-weighting mechanism without requiring heavy neural network retraining.

---

## 2. Semester 1 vs. Semester 2 Roadmap (Team 33 PDF Specification)

### Semester 1 — Foundation, Baseline & Feasibility (COMPLETED)
1. **Data Pipeline & Loaders**: Validated paired text-video loading on TVPReID across Duke, iLIDS, and PRID.
2. **Offline Baseline Reproduction (MFGF)**:
   * Solved Windows laptop VRAM crash (`WinError 1455`) by freezing foundation encoders (ViT, S3D, CLIP, T5) and caching features to disk.
   * Established a rigorous 5-seed clean baseline across PRID, iLIDS, and Duke.
3. **Visual Reliability Module ($q_j$) & Feasibility Pilot**:
   * Implemented the `FrameQualityEstimator`.
   * Evaluated under severe Gaussian blur ($\sigma=5.0$, kernel $15 \times 15$ on 50% of test frames).
   * Proved robustness gains across all three datasets (see Section 4 below).

### Semester 2 — Full Research, Dual Degradation & Final Thesis (PLANNED)
1. **Text Confidence Module ($c_i$)**:
   * Implement linguistic uncertainty parsing for witness descriptions (hedging detection for *"maybe"*, *"probably"*, etc.).
   * Test query-completeness levels (Full $\rightarrow$ Partial $\rightarrow$ Short).
2. **The 4-Way Scientific Ablation Matrix**:
   * Compare: (1) Baseline vs. (2) $+q_j$ vs. (3) $+c_i$ vs. (4) $+q_j c_i$.
3. **Simultaneous Dual-Degradation Evaluation**:
   * Evaluate retrieval when *both* text is uncertain and video is blurred/darkened.
4. **Model 2: Unified Degradation-Aware Training (Optional Extension)**:
   * Train one single model using balanced dynamic augmentation across the $3 \times 3$ conditions so it handles all sentence and video types natively.
5. **Secondary Efficiency Contribution (Optional)**:
   * Learnable Multi-Criteria Token Fusion (MCTF) for background token pruning.
6. **Final Thesis & Paper**:
   * Generate visual attention heatmaps and compile the complete BTP dissertation.

---

## 3. Hardware Independence & A100 Reality
* **A100 Access is OPTIONAL, NOT Mandatory**:
  * We do not rely on A100 cluster availability.
  * The entire Semester 2 roadmap ($c_i$ extraction, 4-way ablation, dual degradation testing, and visualizations) can be executed **100% locally on the Windows laptop** using our cached features and lightweight inference matching.

---

## 4. Current Empirical Results & Audited Ground Truth

### Phase 1: Baseline Clean Test Data (5-Seed Mean ± Std & Best Single Seed)
| Dataset | Rank-1 (Mean ± Std) | Peak Rank-1 | Peak mAP | Best Run Details |
|---|---|---|---|---|
| **PRID** | 9.72% ± 0.53% | 10.56% | 21.43% | Rank-1 from Seed 42, mAP from Seed 999 (Seed Match: No) |
| **iLIDS** | 7.73% ± 3.99% | 13.33% | 23.66% | Both achieved on Seed 42 (Seed Match: Yes) |
| **Duke** | 6.30% ± 0.57% | 7.13% | 13.15% | Rank-1 from Seed 2024, mAP from Seed 777 (Seed Match: No) |

### Phase 2: Corrupted Test Data (50% Gaussian Blur — 5-Seed Mean ± Std & Best Single Seed)
| Dataset | Method | Rank-1 (Mean ± Std) | mAP (Mean ± Std) | Peak Rank-1 | Peak mAP | Best Run Details |
|---|---|---|---|---|---|---|
| **PRID** | Offline Mean | 8.17% ± 1.70% | 19.17% ± 1.93% | 10.56% | 21.90% | R1 from Seed 100, mAP from Seed 999 (Seed Match: No) |
| | **Ours ($q_j$)** | **8.59% ± 2.50%** | **19.49% ± 2.37%** | **11.27%** | **22.21%** | R1 from Seed 100, mAP from Seed 999 (Seed Match: No) |
| **iLIDS** | Offline Mean | 6.40% ± 2.29% | 16.92% ± 1.34% | 9.33% | 19.23% | Both achieved on Seed 999 (Seed Match: Yes) |
| | **Ours ($q_j$)** | **6.40% ± 3.31%** | **18.28% ± 2.10%** | **10.67%** | **21.08%** | Both achieved on Seed 42 (Seed Match: Yes) |
| **Duke** | Offline Mean | 4.45% ± 0.45% | 9.58% ± 0.48% | 4.98% | 10.43% | Both achieved on Seed 777, R1 tied on 2024 (Seed Match: Yes) |
| | **Ours ($q_j$)** | **5.14% ± 0.63%** | **10.93% ± 0.58%** | **5.64%** | **11.31%** | Both achieved on Seed 42, R1 tied on 999 (Seed Match: Yes) |

### Phase 3: Unified MFGF (Combined Benchmark — PRID + iLIDS + Duke)
* **Dataset Scale**: 5,329 training videos, 820 test videos, $D_{tips} = 864$ unified vocabulary attributes.
* **Architecture**: Isolated unified scripts (`train_unified_mfgf.py`, `evaluate_unified_mfgf.py`). Checkpoints saved to `checkpoints/mfgf_unified_offline_{pool}_{seed}/`.
* **5-Seed Results (Clean vs. 50% Gaussian Blur Corrupted)**:

| Test Condition | Method | Rank-1 (Mean ± Std) | Rank-5 (Mean ± Std) | Rank-10 (Mean ± Std) | mAP (Mean ± Std) |
|---|---|---|---|---|---|
| **Clean Test (820 vids)** | Offline Mean | 4.20% ± 0.58% | 12.15% ± 0.87% | 18.20% ± 1.28% | 9.35% ± 0.54% |
| | **Ours ($q_j$)** | 4.07% ± 0.59% | 11.85% ± 0.90% | 18.10% ± 0.59% | 9.10% ± 0.53% |
| **Corrupted 50% Blur** | Offline Mean | 2.80% ± 0.32% | 8.07% ± 0.57% | 12.71% ± 0.82% | 6.51% ± 0.34% |
| | **Ours ($q_j$)** | **3.10% ± 0.68%** | **9.22% ± 0.65%** | **14.73% ± 0.75%** | **7.26% ± 0.59%** |
| **Gain on Corrupted** | | **+0.29% (+10.7%)** | **+1.15% (+14.2%)** | **+2.02% (+15.9%)** | **+0.75% (+11.5%)** |

* **Single-Seed Peak Performance on Corrupted Data (Seed 100)**:
  * Offline Mean: Rank-1 = 3.29%, Rank-5 = 8.66%, Rank-10 = 13.54%, mAP = 6.98%
  * **Ours ($q_j$)**: **Rank-1 = 4.27%** (+0.98%), **Rank-5 = 9.88%** (+1.22%), **Rank-10 = 15.61%** (+2.07%), **mAP = 8.35%** (+1.37%).


---

## 5. File Map & Artifact Locations
* **Master Summary Document**: `C:\Users\hp\.gemini\antigravity-ide\brain\e06d28de-5769-4d52-a910-cfdc2f5669b3\mfgf_reproduction_results.md`
* **VCLIP Results Document**: `E:\BTP\docs\experiment_results.md`
* **Design Specifications**: `E:\BTP\docs\design_decisions.md`
* **Team 33 Deep-Dive Architecture**: `C:\Users\hp\.gemini\antigravity-ide\brain\e06d28de-5769-4d52-a910-cfdc2f5669b3\team_briefing.md`
* **Raw Execution Logs**:
  * Duke & iLIDS 5-seed eval: `C:\Users\hp\.gemini\antigravity-ide\brain\e06d28de-5769-4d52-a910-cfdc2f5669b3\.system_generated\tasks\task-4617.log`
  * PRID 5-seed corrupted eval: `C:\Users\hp\.gemini\antigravity-ide\brain\e06d28de-5769-4d52-a910-cfdc2f5669b3\.system_generated\tasks\task-4392.log`
  * PRID clean baseline: `task-4333.log`
  * iLIDS clean baseline: `task-4430.log`
* **Cached Extracted Features**: `E:\BTP\mfgf_reproduction\data\`
* **Clean Evaluation JSONs**: `E:\BTP\experiments\results\*.json`
