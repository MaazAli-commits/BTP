# Query-Based Cross-Camera Video Person Re-Identification under Dual Degradation

[![Python 3.10+](https://img.shields.io/badge/Python-3.10%2B-blue.svg)](https://www.python.org/)
[![PyTorch 2.0+](https://img.shields.io/badge/PyTorch-2.0%2B-ee4c2c.svg)](https://pytorch.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

**Major Project (BTP) — Team 33**  
**Department of Computer Science and Engineering, Indian Institute of Information Technology Kottayam (2026–2027)**  
**Authors:** Mohammed Maaz Ali, Rupsika, Siva, Harshith  

---

## 📌 Overview

In surveillance investigations, locating a suspect across non-overlapping CCTV cameras typically relies on natural language witness reports (Text-to-Video Person Retrieval, TVPR). While state-of-the-art models perform well on clean academic benchmarks, real-world deployment faces **Dual Degradation**:
1. **Visual Degradation:** CCTV footage is frequently degraded by pedestrian motion blur, camera defocus, and poor lighting.
2. **Textual Degradation:** Eyewitness testimonies are inherently partial, ambiguous, or linguistically hedged (*"maybe wearing dark pants"*).

Standard architectures aggregate video frames using **Uniform Mean Pooling** ($w_j = 1/T$), causing blurred frames to corrupt the global video representation and causing true targets to drop out of top retrieval ranks.

### Our Solution (Phase 1)
We introduce an **Unsupervised Frame Reliability Engine ($q_j$)**:
- Computes spatial high-frequency energy using an isotropic 2D Laplacian operator over boundary-eliminated grayscale frames.
- Incorporates a bounded illumination prior to penalize extreme over/under-exposure.
- Maps clarity to a normalized weight $w_j = q_j / \sum q_k$ via a calibrated sigmoid—requiring **zero manual blur labels** and running in **$\sim$1.8 ms on CPU**.
- Evaluated on a decoupled reproduction of the **Multielement Feature Guided Fragments Learning (MFGF)** architecture (ACM Multimedia 2024).

---

## 🔬 Empirical 5-Seed Benchmark Results (Unified 820 Gallery)

Evaluated across 5 independent random seeds under **50% synthetic Gaussian motion blur** ($\sigma=5.0$, $15\times15$ kernel) on the unified TVPReID benchmark (DukeMTMC, PRID-2011, iLIDS-VID):

| Metric | Baseline Mean Pooling | Ours ($q_j$ Weighted) | Net Improvement | Seed Consistency |
| :--- | :---: | :---: | :---: | :---: |
| **mAP** | 6.51% ± 0.34% | **7.26% ± 0.59%** | **+0.75 percentage points (+11.5% rel)** | **100% (5/5 seeds)** |
| **Rank-10** | 12.71% ± 0.82% | **14.73% ± 0.75%** | **+2.02 percentage points (+15.9% rel)** | **100% (5/5 seeds)** |
| **Rank-5** | 8.07% ± 0.57% | **9.22% ± 0.65%** | **+1.15 percentage points (+14.2% rel)** | **80% (4/5 seeds)** |
| **Rank-1** | 2.80% ± 0.32% | **3.10% ± 0.68%** | **+0.30 percentage points (+10.7% rel)** | **Peak: 4.27% (Seed 100)** |

*On the large-scale DukeMTMC benchmark, our quality-weighted pipeline achieved a **+15.5% relative recovery in Rank-1** ($4.45\% \to 5.14\%$).*

---

## 📂 Repository Structure

```
├── configs/                     # Experiment configurations
├── docs/                        # Complete technical documentation & manuals
│   ├── TECHNICAL_DEEP_DIVE_MANUAL.md  # 22-page master technical & defense manual
│   ├── design_decisions.md      # Architectural pivot logs
│   └── experiment_results.md    # Raw experiment records
├── experiments/                 # 5-seed evaluation logs and metric JSONs
│   └── results/                 # Per-seed JSON results across Duke, PRID, iLIDS, Unified
├── mfgf_reproduction/           # MFGF reproduction scripts
│   ├── real_mfgf.py             # Decoupled ViT-B/16 + S3D + BERT dual-space architecture
│   ├── extract_features_offline.py # Feature extraction caching
│   ├── extract_weighted_vit.py  # q_j weighted representation generation
│   ├── train_unified_mfgf.py    # Head training on unified gallery
│   └── evaluate_unified_mfgf.py # 5-seed evaluation runner
├── src/                         # Core foundational modules
│   ├── dataset/tvpreid.py       # Dataset loaders
│   ├── models/quality_estimator.py # Laplacian variance q_j math
│   └── models/reliability_matching.py # Cross-attention reweighting
├── static/                      # Modern glassmorphism web interface
│   ├── app.js                   # Client state management
│   ├── index.html               # Semantic HTML5 layout
│   └── style.css                # Dark-mode CSS design system
├── demo_server.py               # Live Flask demonstration backend
├── demo_data.json               # Audited scenario presets & metadata
└── requirements.txt             # Environment dependencies
```

---

## 🚀 Quickstart & Reproduction

### 1. Environment Setup
```bash
git clone https://github.com/MaazAli-commits/BTP.git
cd BTP
python -m venv venv
.\venv\Scripts\activate
pip install torch torchvision transformers flask tqdm pillow numpy
```

### 2. Launching the Interactive Demonstration
```bash
python demo_server.py
```
Open `http://localhost:5000` in your browser to interact with the neural retrieval engine over the complete 820-video gallery with real-time BERT text encoding and H.264 video streaming.

### 3. Running 5-Seed Evaluation Under Gaussian Blur
```bash
python mfgf_reproduction/evaluate_unified_mfgf.py --split corrupted --pool weighted --seed 42
```

---

## 📖 Complete Technical Manual
For the complete mathematical derivations, architectural diagrams, worked arithmetic examples, and viva defense answers, see the [Master Technical Deep-Dive Manual](docs/TECHNICAL_DEEP_DIVE_MANUAL.md).
