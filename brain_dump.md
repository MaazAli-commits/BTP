# Brain Dump & Project Context

> **READ `E:\BTP\PROJECT_MASTER_CONTEXT.md` FIRST!**  
> That file contains the complete, permanent ground truth of Team 33's BTP project, the PDF roadmap, the Semester 1 vs Semester 2 division, and all verified results.

## Quick Summary
1. **Team 33 Goal**: Query-Based Cross-Camera Video Person Re-Identification under Dual Degradation (Degraded Text + Degraded Video).
2. **Core Formula**: $\alpha'_{ij} = \frac{\alpha_{ij} c_i q_j}{\sum \alpha_{ij} c_i q_j}$.
3. **Semester 1 (COMPLETED)**:
   - TVPReID dataset pipeline established.
   - Offline MFGF clean baseline reproduced across PRID, iLIDS, Duke over 5 seeds.
   - Frame quality reliability mechanism ($q_j$) proven under blur across PRID, iLIDS, Duke over 5 seeds.
   - Clean tables and audited results are documented in `PROJECT_MASTER_CONTEXT.md` and `C:\Users\hp\.gemini\antigravity-ide\brain\e06d28de-5769-4d52-a910-cfdc2f5669b3\mfgf_reproduction_results.md`.
4. **Semester 2 (PLANNED)**:
   - Text confidence ($c_i$) for witness uncertainty ("maybe", "probably").
   - Full 4-way ablation (Baseline, $+q_j$, $+c_i$, $+q_j c_i$).
   - Dual-degradation evaluation.
   - Model 2 (optional degradation-aware dynamic augmentation).
   - Thesis writing.
5. **Compute**:
   - Zero dependence on A100. The entire project is runnable locally on the Windows laptop using cached features.
