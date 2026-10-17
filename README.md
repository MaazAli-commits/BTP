# Query-Based Cross-Camera Video Person Re-Identification under Dual Degradation

[![Python 3.10+](https://img.shields.io/badge/Python-3.10%2B-blue.svg)](https://www.python.org/)
[![PyTorch 2.0+](https://img.shields.io/badge/PyTorch-2.0%2B-ee4c2c.svg)](https://pytorch.org/)
[![Transformers 4.30+](https://img.shields.io/badge/Transformers-4.30%2B-orange.svg)](https://huggingface.co/transformers/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

**B.Tech Major Project (BTP) — Team 33**  
**Department of Computer Science and Engineering, Indian Institute of Information Technology Kottayam (2026–2027)**  
**Authors:** Mohammed Maaz Ali (`2023BCS0202`), Rupsika, Siva, Harshith  
**Project Guide:** Faculty of Computer Science & Engineering, IIIT Kottayam  

---

## 📌 1. Executive Summary & Problem Formulation

In modern surveillance forensics, locating a suspect across non-overlapping CCTV camera networks typically relies on natural language eyewitness descriptions—a task formalized as **Text-to-Video Person Retrieval (TVPR)**. While state-of-the-art vision-language models achieve impressive scores on curated academic benchmarks, real-world deployment collapses due to **Dual Degradation**:

1. **Visual Degradation (Camera Domain):** CCTV feeds are plagued by pedestrian rapid motion blur, optical defocus, heavy compression artifacts, and poor illumination.
2. **Textual Degradation (Query Domain):** Eyewitness testimonies are partial, temporally scrambled, linguistically hedged (*"maybe dark pants"*), or subject to memory decay.

```
       WITNESS QUERY: "A tall man in a dark hoodie walking past the entrance"
                                   │
                                   ▼
                      [BERT-base Text Encoder]
                                   │
                                   ▼
                            t ∈ ℝ^512
                                   │
             Cosine Similarity S(t, v) = (t · v) / (||t|| ||v||)
                                   ▲
                                   │
                                   ▼
            [Dynamic Frame Reliability Engine (q_j)]
                                   ▲
                                   │
  Frame 1: Crisp  ───► q_1 = 0.94 (Weight: 0.28)  ──┐
  Frame 2: Blurred ──► q_2 = 0.12 (Weight: 0.03)  ──┼──► v_weighted = ∑ w_j v_j ∈ ℝ^512
  Frame 3: Occluded ─► q_3 = 0.35 (Weight: 0.09)  ──┤
  Frame 4: Crisp  ───► q_4 = 0.89 (Weight: 0.26)  ──┘
```

### Why Existing Methods Fail
Standard video Re-ID backbones aggregate frame-level feature embeddings $\{v_j\}_{j=1}^T$ using **Uniform Mean Pooling**:

$$
v_{\text{mean}} = \frac{1}{T} \sum_{j=1}^T v_j
$$

When 3 out of 8 frames in a pedestrian tracklet suffer severe motion blur, uniform pooling treats blurry artifacts and clear discriminative cues identically ($w_j = 0.125$). The degraded frames dilute the identity signature, dragging true target candidates out of the top retrieval ranks (e.g., dropping from Rank-1 to Rank-35).

---

## 💡 2. Our Methodology: Unsupervised Frame Reliability Pooling ($q_j$)

To resolve visual degradation without requiring expensive frame-level blur annotations, we introduce an **Unsupervised Frame Reliability Engine ($q_j$)**:

### Step 1: Spatial High-Frequency Energy
For each frame $j$, we convert RGB to grayscale ($Y = 0.2989R + 0.5870G + 0.1140B$), crop a 10-pixel border margin to prevent padding artifacts from dominating the variance, and compute the discrete isotropic 2D Laplacian operator $\Delta I$ using kernel $K$:

$$
K = \begin{bmatrix} 
0 & 1 & 0 \\ 
1 & -4 & 1 \\ 
0 & 1 & 0 
\end{bmatrix}
$$

The response map $\Delta I$ and spatial sharpness variance $\sigma_L^2$ across cropped pixels $\Omega$ are given by:

$$
\Delta I(x, y) = (I * K)(x, y)
$$

$$
\sigma_L^2 = \text{Var}_{(x,y) \in \Omega}(\Delta I(x, y))
$$

### Step 2: Bounded Illumination Prior
Extreme darkness or severe sensor over-exposure suppresses edge contrast artificially. We compute the mean grayscale brightness $\mu_{\text{gray}} \in [0, 1]$ and apply a parabolic illumination penalty factor $P_{\text{illum}}$ centered at mid-brightness ($0.5$):

$$
P_{\text{illum}} = 1.0 - 4.0 \left( \mu_{\text{gray}} - 0.5 \right)^2 \in [0, 1]
$$

### Step 3: Calibrated Sigmoid Mapping & Quality Weighting
We map sharpness $\sigma_L^2$ through a calibrated logistic function centered at variance 50 with scale 20:

$$
S_{\text{sharp}} = \frac{1}{1 + \exp\left(-\frac{\sigma_L^2 - 50.0}{20.0}\right)} \in (0, 1)
$$

The composite frame quality score $q_j$ combines sharpness with the illumination penalty:

$$
q_j = 0.8 \cdot S_{\text{sharp}} + 0.2 \cdot P_{\text{illum}}
$$

The normalized frame weights $w_j$ and the aggregated video representation $v_{\text{weighted}}$ are given by:

$$
w_j = \frac{q_j}{\sum_{k=1}^T q_k + 10^{-6}}, \qquad v_{\text{weighted}} = \sum_{j=1}^T w_j v_j
$$

* **Zero Supervision:** Requires zero manual blur bounding boxes or class labels.
* **Ultra-Low Latency:** Runs in **~1.8 ms per frame on CPU** and <0.3 ms on GPU.

---

## 🔬 3. 5-Seed Empirical Benchmark Results (Unified 820 Gallery)

We conducted a multi-seed evaluation across **5 independent random seeds** (`[42, 100, 2024, 777, 999]`) on the unified TVPReID benchmark (DukeMTMC, PRID-2011, iLIDS-VID) comprising **820 gallery video tracklets** subjected to **50% synthetic Gaussian motion blur** ($\sigma=5.0$, $15\times15$ kernel):

### Paired 5-Seed Comparison Table

| Seed | Baseline mAP | Ours mAP | Net $\Delta$ mAP | Baseline R@1 | Ours R@1 | Baseline R@5 | Ours R@5 | Baseline R@10 | Ours R@10 |
| :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **42** | 5.93% | **7.38%** | **+1.45 pp** | 2.32% | **3.41%** | 7.44% | **8.90%** | 11.46% | **14.51%** |
| **100** | 6.98% | **8.35%** | **+1.37 pp** | 3.29% | **4.27%** | 8.41% | **10.00%** | 12.44% | **15.61%** |
| **2024** | 6.66% | **6.99%** | **+0.33 pp** | 2.68% | **2.80%** | 9.02% | **8.54%** | 14.02% | **15.61%** |
| **777** | 6.48% | **6.68%** | **+0.20 pp** | 2.93% | **2.68%** | 7.80% | **8.66%** | 12.80% | **14.02%** |
| **999** | 6.52% | **6.91%** | **+0.39 pp** | 2.80% | **2.32%** | 7.68% | **10.00%** | 12.80% | **13.90%** |
| **Mean ± Std** | **6.51% ± 0.34%** | **7.26% ± 0.59%** | **+0.75 pp (+11.5% rel)** | **2.80% ± 0.32%** | **3.10% ± 0.68%** | **8.07% ± 0.57%** | **9.22% ± 0.65%** | **12.71% ± 0.82%** | **14.73% ± 0.75%** |

### Statistical Significance
- **Directional Consistency:** mAP improved in **5 out of 5 seeds (100%)**; Rank-10 improved in **5 out of 5 seeds (100%)**.
- **Paired $t$-Test on mAP:**
  - One-sided $p$-value: **$p = 0.0257$** (Statistically significant at $\alpha = 0.05$).
  - Two-sided $p$-value: **$p = 0.0513$** ($t = 2.76$, $df = 4$).
- **Large-Scale DukeMTMC Gallery:** Rank-1 improved from $4.45\%$ to $5.14\%$ (**+15.5% relative recovery** under severe motion corruption).

---

## 📂 4. Repository Structure

```
BTP/
├── configs/                             # Experiment & architecture YAML configurations
│   ├── stage1.yaml                      # Sanity check baseline configs
│   ├── stage2.yaml                      # Cross-modal alignment settings
│   ├── stage3.yaml                      # VCLIP adaptation config
│   ├── stage4.yaml                      # Dynamic reliability pooling config
│   └── stage5.yaml                      # Full evaluation & metric logging settings
├── docs/                                # Comprehensive technical documentation
│   ├── TECHNICAL_DEEP_DIVE_MANUAL.md    # 22-page master architecture & viva defense manual
│   ├── design_decisions.md              # Architectural pivot log & rationale
│   └── experiment_results.md            # Raw per-seed experiment metric tables
├── experiments/                         # Reproducible metric outputs
│   └── results/                         # 5-seed JSON metric dumps (Duke, PRID, iLIDS, Unified)
│       ├── Unified_summary_5seeds.json  # Aggregated 5-seed benchmark summary
│       ├── Unified_corrupted_weighted_*.json
│       └── Unified_corrupted_mean_*.json
├── mfgf_reproduction/                   # Decoupled MFGF reproduction pipeline
│   ├── real_mfgf.py                     # ViT-B/16 + S3D + BERT dual-space architecture
│   ├── preprocess_tips.py               # Dataset-specific NLP vocabulary & TIPS extractor
│   ├── unified_preprocess_tips.py       # Unified multi-dataset vocabulary extractor
│   ├── extract_features_offline.py      # Offline backbone feature caching
│   ├── extract_weighted_vit.py          # Frame-level q_j weighting pipeline
│   ├── unified_prepare_features.py      # Merges dataset caches into unified 820 splits
│   ├── train_unified_mfgf.py            # Train projection & cross-attention heads
│   ├── evaluate_unified_mfgf.py         # Multi-metric rank evaluator (mAP, R@1/5/10)
│   └── run_unified_experiments.py       # Master automated 5-seed training & test runner
├── scripts/                             # Utility & evaluation execution scripts
│   ├── find_cases.py                    # Demo case finder
│   ├── pretranscode_videos.py           # Video transcoding helper
│   ├── train_unified.py                 # Alternate training runner
│   └── evaluate_unified.py              # Alternate eval runner
├── src/                                 # Core foundational modules
│   ├── dataset/
│   │   ├── tvpreid.py                   # PyTorch dataset loader for video tracklets
│   │   ├── mevid.py                     # MEVID surveillance video loader
│   │   └── corruption.py                # Gaussian blur & text corruption engine
│   └── models/
│       ├── quality_estimator.py         # Laplacian variance + illumination penalty math
│       ├── reliability_matching.py      # Dynamic cross-attention matching head
│       └── vclip_adaptation.py          # Temporal adaptation module
├── static/                              # Demonstration Web UI (Vanilla CSS + HTML5)
│   ├── app.js                           # State machine, neural search client, video player
│   ├── index.html                       # Semantic layout with glassmorphic cards
│   └── style.css                        # Modern responsive dark-mode styling
├── tests/                               # Verification & unit tests
│   ├── test_metrics.py                  # Evaluation metrics assertions
│   ├── test_neural_search.py            # Real-time search testing
│   ├── test_search.py                   # Metadata retrieval testing
│   └── test_server_api.py               # Flask demo server API tests
├── demo_server.py                       # Live Flask demonstration backend with real-time BERT
├── demo_data.json                       # 820-gallery metadata, captions, and preset scenarios
├── generate_demo_data.py                # Precomputes gallery representations
├── run_demo.bat                         # Quick launcher for local demo
├── requirements.txt                     # Pinned environment dependencies
└── README.md                            # Project documentation
```

---

## ⚡ 5. Complete Step-by-Step Reproduction Guide

Follow these steps to replicate our exact experimental results from scratch.

### 5.1 System Prerequisites & Installation
* **OS:** Windows 10/11 or Ubuntu 20.04/22.04 LTS
* **Python:** 3.10 or 3.11
* **Hardware:** NVIDIA GPU with $\ge$ 4 GB VRAM (e.g., RTX 3050/4060) or CPU (inference/eval supported).

```bash
# Clone the repository
git clone https://github.com/MaazAli-commits/BTP.git
cd BTP

# Create and activate virtual environment
python -m venv venv

# Windows:
.\venv\Scripts\activate
# Linux/macOS:
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

### 5.2 Dataset Organization
Organize the TVPReID dataset under `data/TVPReID/` matching the following directory structure:
```
data/
└── TVPReID/
    ├── TVPReid-Duke/
    │   ├── videos/              # .mp4 video tracklets
    │   └── captions/
    │       ├── TVPReid-Duke.json
    │       ├── train.csv
    │       └── test.csv
    ├── TVPReid-PRID/
    │   ├── videos/
    │   └── captions/
    │       ├── TVPReid-PRID.json
    │       ├── train.csv
    │       └── test.csv
    └── TVPReid-iLIDS/
        ├── videos/
        └── captions/
            ├── TVPReid-iLIDS.json
            ├── train.csv
            └── test.csv
```

### 5.3 Step 1: Generate TIPS Vocabulary Bank
Extract NLTK POS-tagged keywords (nouns, verbs, adjectives) and construct the binary multi-hot TIPS semantic vectors:
```bash
python mfgf_reproduction/unified_preprocess_tips.py
```

### 5.4 Step 2: Extract Frozen Backbone Features
Extract spatial features via ViT-B/16 (768-d), motion features via S3D (1024-d), and text queries via BERT-base (768-d) for each dataset partition, then merge into the unified 820 benchmark splits:
```bash
python mfgf_reproduction/extract_features_offline.py --dataset PRID
python mfgf_reproduction/extract_features_offline.py --dataset iLIDS
python mfgf_reproduction/extract_features_offline.py --dataset Duke
python mfgf_reproduction/unified_prepare_features.py
```

### 5.5 Step 3: Compute $q_j$ Blur Reliability Weights
Compute Laplacian energy and apply the calibrated sigmoid mapping to generate quality-weighted video representations:
```bash
python mfgf_reproduction/extract_weighted_vit.py
```

### 5.6 Step 4: Train Decoupled Projection Heads (5 Seeds)
Train the decoupled projection heads under InfoNCE contrastive alignment and TIPS classification:
```bash
# Train baseline (uniform mean pooling)
python mfgf_reproduction/train_unified_mfgf.py --pool mean --seed 42

# Train proposed (dynamic q_j reliability pooling)
python mfgf_reproduction/train_unified_mfgf.py --pool weighted --seed 42
```
*(Repeat for seeds `100`, `2024`, `777`, `999`)*.

### 5.7 Step 5: Evaluate Under 50% Motion Blur
Evaluate Rank-1, Rank-5, Rank-10, and mAP over the 820 test tracklets:
```bash
# Baseline evaluation
python mfgf_reproduction/evaluate_unified_mfgf.py --pool mean --seed 42 --split corrupted

# Proposed q_j evaluation
python mfgf_reproduction/evaluate_unified_mfgf.py --pool weighted --seed 42 --split corrupted
```

### 🚀 One-Click Master Pipeline
To automatically execute training, testing, and metric aggregation across all 5 seeds in a single unattended execution:
```bash
python mfgf_reproduction/run_unified_experiments.py
```
This script will produce the final JSON summaries in `experiments/results/Unified_summary_5seeds.json`.

---

## 🖥️ 6. Interactive Neural Demonstration Engine

We provide a real-time web application to visually inspect the impact of frame degradation and $q_j$ reliability pooling:

```bash
python demo_server.py
```
Then navigate to `http://localhost:5000` in your browser.

### Key Demo Features:
* **Interactive Natural Language Search:** Type arbitrary queries (*"man in black coat"*) encoded live by BERT-base into the joint cross-modal latent space.
* **Dual Degradation Simulation:** Dynamically apply synthetic Gaussian motion blur or textual hedges to observe retrieval rank shifts.
* **Per-Frame Quality Visualizer:** Live display of $q_j$ reliability curves, Laplacian energy bars, and illumination penalty scores.
* **Side-by-Side Comparison:** Direct visual comparison showing how mean pooling fails vs. how reliability pooling recovers true targets.

---

## ⚙️ 7. Key Hyperparameter Reference

| Hyperparameter | Value | Description |
| :--- | :---: | :--- |
| **Laplacian Kernel** | $3 \times 3$ isotropic | Standard 4-connectivity discrete Laplacian |
| **Boundary Crop Margin** | $10$ pixels | Excluded from edge variance calculation |
| **Sigmoid Center & Scale** | $x_0 = 50.0, s = 20.0$ | Calibrated slope for high-frequency energy |
| **Quality Combination** | $0.8 \cdot S_{\text{sharp}} + 0.2 \cdot P_{\text{illum}}$ | Balances sharpness and illumination |
| **Video Backbone** | ViT-B/16 + S3D | 768-d spatial + 1024-d spatio-temporal features (frozen) |
| **Text Backbone** | BERT-base-uncased | 768-d contextual token embeddings (frozen) |
| **Common Projection Dim** | $256$ | Normalized joint cross-modal embedding dimension |
| **TIPS Attribute Dim** | $864$ | Multi-hot attribute vocabulary dimension ($D_{\text{tips}}$) |
| **Loss Formulation** | InfoNCE + BCE | Common space alignment + D² attribute supervision |
| **Temperature Scale ($s_0$)** | Init $\ln(1/0.05) \approx 2.9957$ | Learnable temperature parameter ($\tau = 0.05$) |
| **Optimizer** | AdamW | Initial learning rate $1 \times 10^{-4}$ |
| **Batch Size & Epochs** | $64$, $120$ epochs | Cosine annealing schedule |

---

## 📚 8. Citation & References

If you find this codebase or our research useful, please cite our project:

```bibtex
@misc{ali2026tvpr_dual_degradation,
  author    = {Mohammed Maaz Ali and Rupsika and Siva and Harshith},
  title     = {Query-Based Cross-Camera Video Person Re-Identification under Dual Degradation},
  year      = {2026},
  publisher = {Department of Computer Science and Engineering, IIIT Kottayam},
  howpublished = {\url{https://github.com/MaazAli-commits/BTP}}
}
```

### Reference Literature
1. **MFGF (Baseline):** *Multielement Feature Guided Fragments Learning for Text-Based Person Retrieval*, ACM Multimedia (ACM MM), 2024.
2. **VCLIP:** *Video-Conditioned Cross-Modal Pre-training for Video Retrieval*, IEEE TIP, 2023.
3. **TVPReID Benchmarks:** DukeMTMC-VideoReID, PRID-2011, and iLIDS-VID datasets.

---

## 📄 License
This project is open-sourced under the **MIT License**. See [LICENSE](LICENSE) for details.
