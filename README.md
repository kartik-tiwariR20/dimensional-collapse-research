# Predicting Dimensional Collapse in Self-Supervised Learning

> Target: **NeurIPS 2026 Workshop — NeurReps** (Symmetry and Geometry in Neural Representations), Proceedings Track (archival, PMLR)

---

## 1. The Research Problem

Self-supervised learning (SSL) models sometimes suffer **dimensional collapse** -> the learned embeddings occupy a lower-dimensional subspace than the full embedding dimension, wasting representational capacity.

**Our question:** Given augmentation strength (α), projector width, and dataset structure — can we *predict* the effective rank at convergence, and does the relationship between effective rank and downstream performance *transfer* from vision (where it's been studied) to time-series data?

**Why it matters:**
- Collapse was long treated as a pure failure mode; recent work (Cosentino et al. 2022) shows it can sometimes *help* generalization — but this nuance is established almost entirely in vision-based SSL (SimCLR, BYOL, Barlow Twins, VICReg)
- No existing study tests whether the vision-derived collapse dynamics hold across genuinely different data structures (channel count, sequence length, sample size, class balance)
- If early-training signals can predict final representation quality, that's a real compute-saving diagnostic tool

**What makes this novel:**
1. First systematic test of the augmentation-strength → collapse relationship outside vision, across 5 structurally distinct time-series datasets
2. Tests whether the "collapse can help" effect (Cosentino et al.) is specific to *how* you evaluate — linear probe vs. fine-tuned — not just whether collapse occurred
3. Demonstrates that **how you measure effective rank changes your conclusion about which direction the effect runs** — measuring on augmented projector output vs. a clean, held-out encoder representation produces different, sometimes opposite, correlation signs on the same data (Section 5b). This is a methodological finding in its own right, not just a robustness check.
4. Tests early-epoch predictability (H2) against **both** accuracy regimes, not just one — across every dataset where the effect appears at all, early rank significantly predicts final linear-probe accuracy but never fine-tune accuracy (Section 5c)

---

## 2. Theoretical Framework

### Core Metric: Effective Rank

Given a batch of embeddings `{z_1, ..., z_N}`, compute the covariance matrix:
```
C = (1/N) Σ_i (z_i - z̄)(z_i - z̄)ᵀ
```
Eigendecompose to get `λ_1 ≥ λ_2 ≥ ... ≥ λ_d`, normalize into a distribution `p_i = λ_i / Σ_j λ_j`, then:
```
erank(C) = exp( -Σ_i p_i log p_i )
```
`erank → 1` means near-total collapse; `erank → d` means the embedding space is fully utilized. (Roy & Vetterli, 2007)

**Measurement matters as much as the definition.** Effective rank is reported in two ways throughout this study:
- **`erank(z)`, augmented view (legacy/naive)** — computed on the projector output during a normal training step, i.e. on an *augmented* input. This is what most prior collapse studies report, and what our own earlier draft pipeline used.
- **`erank(h)`, clean batch (corrected, primary)** — computed on the *encoder* output (`h`, not the projector output `z`), on a fixed, un-augmented held-out batch, evaluated once per epoch outside the training loop. This isolates what the model actually learned from what augmentation strength (α) does to the input distribution directly, and measures the representation space (`h`) that downstream probes actually consume, not the projector's (`z`).

Both are logged for every run; Section 5b shows why the distinction changes the paper's conclusions.

### SSL Loss: VICReg

Chosen because its variance term is an explicit, tunable anti-collapse regularizer, making the collapse mechanism directly inspectable:
```
Loss = λ·invariance(z1, z2) + μ·variance(z1, z2) + ν·covariance(z1, z2)
```
All claims in this study are scoped to VICReg specifically, not SSL broadly — collapse dynamics are not assumed to transfer to contrastive or other non-contrastive methods without separate verification.

### Why encoder-space collapse might help linear-probe accuracy (mechanistic sketch, not a proven claim)

The corrected finding in Section 5b — less effective rank in `h` predicts *higher* linear-probe accuracy — is an empirical pattern, and this study does not derive it from first principles. But it is consistent with two existing theoretical threads, worth citing rather than treating as an unexplained coincidence:

- **Cosentino et al. (2022)** show geometrically that a *moderate* degree of collapse can concentrate representations around task-relevant directions, effectively acting as implicit dimensionality reduction that a linear probe can exploit directly — too little collapse leaves the probe fighting noise dimensions, too much removes signal entirely.
- **VICReg's own loss structure (Bardes et al. 2022)** only regularizes variance and covariance in the *projector*'s output space, not the encoder's. Nothing in the loss directly constrains `erank(h)` — so encoder-space collapse under VICReg is a downstream, emergent consequence of how the projector's constraints propagate backward through training, not a directly optimized quantity. That the encoder collapses at all, and that its collapse correlates with linear separability, is itself worth flagging as a secondary finding.

This is offered as plausible framing for the Discussion section, not a mechanism this study tests directly — the paper's contribution is the measurement-dependent empirical result, not a new theory of why it happens.

### Hypotheses

- **H1 (transfer):** The augmentation-strength → effective-rank relationship established in vision transfers, at least directionally, across structurally different time-series datasets.
- **H2 (predictability):** Effective rank and downstream accuracy at convergence can be predicted from early-training signals (e.g., effective rank at epoch 5), without training to completion.

---

## 3. Experimental Setup

### Shared Pipeline (identical across every dataset — this is what makes cross-dataset comparison fair)

- **Encoder:** small 1D-CNN, fixed capacity (64-dim output), channel count read automatically from data
- **Projector:** MLP head, width is the swept structural variable (64 vs. 512)
- **Augmentations:** jitter, scaling, time-masking, random-crop-resize, all parametrized by a single strength scalar α ∈ [0, 1]
- **Sweep grid:** 5 α values × 2 projector widths × **5 seeds = 50 runs per dataset, 250 runs total** (increased from an earlier 3-seed draft for statistical power)
- **Splits:** every dataset uses its **official, canonical train/test split** (not a re-pooled random split) — this makes results directly comparable to published benchmarks on these datasets, and for UCI HAR specifically means the split is subject-disjoint by construction (no near-duplicate windows from the same person on both sides)
- **Pretraining pool:** official train split only — the official test split is **never** seen during SSL pretraining, only at final downstream evaluation
- **Downstream eval:** both linear probe (frozen encoder) and full fine-tune, per run
- **Baselines:** majority-class accuracy and a random-initialized (never pretrained) encoder, evaluated with the identical probe protocol, per dataset (Section 5f)

### Compute

molab (marimo cloud notebooks), 4 CPU / 32 GiB / RTX Pro 6000. All 5 datasets, all 250 runs, completed under this single unified protocol — no dataset required a reduced-epoch or scoped-down variant.

---

## 4. Dataset Inventory

| Dataset | Channels | Seq. Length | Classes | Samples | Domain | Status |
|---|---|---|---|---|---|---|
| **ECG5000** | 1 | 140 | 5 (imbalanced) | 5,000 | Medical/ECG | ✅ Done, fully analyzed |
| **UCI HAR** | 9 | 128 | 6 (balanced) | 10,299 | Motion/wearable | ✅ Done, fully analyzed |
| **FordA** | 1 | 500 | 2 (balanced) | 4,921 | Mechanical/sensor | ✅ Done, fully analyzed |
| **Spoken Arabic Digits** | 13 | 93 (zero-padded from variable 4–93) | 10 (balanced) | 8,800 | Speech/MFCC | ✅ Done, fully analyzed |
| **EigenWorms** | 6 | 17,984 | 5 (imbalanced) | 259 | Behavioral genetics | ✅ Done, fully analyzed — H1 clean; H2/accuracy-correlation null and explained (Section 5g) |

This spread deliberately varies channel count (1→13), sequence length (128→17,984), sample size (259→10,299), and class balance — the axes needed to say something precise about *where* the vision-derived relationship holds vs. breaks down, rather than a single pass/fail verdict.

---

## 5. Consolidated Results (all 5 datasets, full writeup)

### 5a. H1 — Does augmentation strength predict effective rank?

Effective rank measured the corrected way: `erank(h)`, clean held-out batch, projector width=64.

| Dataset | Channels | Length | Capacity (ch×len) | Mean erank at α=0 | Mean erank at α=1 | % collapse | Direction confirmed? |
|---|---|---|---|---|---|---|---|
| FordA | 1 | 500 | 500 | 12.44 | 2.21 | **82.2%** | ✅ Strong, monotonic |
| ECG5000 | 1 | 140 | 140 | 13.80 | 3.45 | **75.0%** | ✅ Strong, monotonic |
| Spoken Arabic Digits | 13 | 93 | 1,209 | 29.55 | 17.56 | **40.6%** | ✅ Monotonic |
| UCI HAR | 9 | 128 | 1,152 | 34.26 | 20.50 | **40.2%** | ✅ Monotonic |
| EigenWorms | 6 | 17,984 | 107,904 | 13.51 | 10.07 | **25.4%** | ✅ Monotonic |

> [!IMPORTANT]
> **The direction of H1 replicates cleanly in all five datasets** — more augmentation → more collapse, at every projector width tested, no exceptions. On magnitude: EigenWorms (by far the largest raw input capacity) does show the smallest collapse, which is *suggestive* of a capacity-related trend — but with only 5 datasets, the full ordering is **not** clean (FordA collapses more than ECG5000 despite having ~3.6x its capacity; HAR and Spoken Arabic Digits are within 0.4 points of each other and effectively tied). Spearman rank correlation between capacity and % collapse is ρ = −0.80, **p = 0.10 — not significant at n=5**. This is reported as a descriptive, exploratory observation only, not a validated predictor; a claim of "predicts almost perfectly" would not survive scrutiny with this sample size and is explicitly not made here.

Projector width effect replicates cleanly in every dataset: wider projector (512) → consistently higher effective rank than narrower (64), at every α tested — matching Garrido et al.'s theory.

### 5b. Effective rank → downstream accuracy — naive vs. corrected measurement

This is the section where measurement choice changes the conclusion. Two versions of the same statistic, side by side: **naive** (`erank(z)`, augmented view — matches most prior literature and an earlier draft of this pipeline) vs. **corrected** (`erank(h)`, clean held-out batch). Both are shown pooled (raw correlation, ignoring α) and **partial** (controlling for α and projector width — see Section 5b footnote on why pooling alone is misleading).

**Linear probe accuracy — pooled correlation:**

| Dataset | naive `erank(z)` r | p | corrected `erank(h)` r | p |
|---|---|---|---|---|
| ECG5000 | −0.855 | <0.001 | **−0.952** | <0.001 |
| FordA | −0.152 | 0.292 (ns) | **−0.633** | <0.001 |
| UCI HAR | −0.522 | <0.001 | **−0.794** | <0.001 |
| Spoken Arabic Digits | −0.481 | <0.001 | **−0.910** | <0.001 |
| EigenWorms | +0.047 | 0.744 (ns) | +0.042 | 0.772 (ns) |

**Linear probe accuracy — partial correlation, controlling for α and projector width:**

| Dataset | naive `erank(z)` partial r | p | corrected `erank(h)` partial r | p |
|---|---|---|---|---|
| ECG5000 | −0.155 | 0.282 (ns) | **−0.305** | 0.031 |
| FordA | −0.543 | <0.001 | **−0.562** | <0.001 |
| UCI HAR | **+0.009** | 0.949 (ns, ~zero) | **−0.577** | <0.001 |
| Spoken Arabic Digits | **+0.421** | 0.002 (sign flip) | **−0.928** | <0.001 |
| EigenWorms | −0.088 | 0.546 (ns) | +0.026 | 0.860 (ns) |

> [!IMPORTANT]
> **This is the headline methodological finding of the paper.** Under the naive measurement, controlling for α produces an inconsistent picture: ECG5000's effect disappears, HAR's collapses to exactly zero, and Spoken Arabic Digits **reverses sign** — naively suggesting collapse *hurts* accuracy in that dataset, the opposite of ECG5000/FordA. This looked, at an earlier stage of this project, like real cross-dataset heterogeneity worth reporting as a finding on its own.
>
> **It wasn't.** Once effective rank is measured correctly — clean held-out batch, encoder space `h` rather than the augmented projector space `z` — **all four non-EigenWorms datasets agree in direction**: lower effective rank (more collapse) predicts *higher* linear-probe accuracy, and this holds after controlling for α and projector width, with partial correlations ranging from −0.305 (ECG5000) to −0.928 (Spoken Arabic Digits), all statistically significant. The apparent sign flip in HAR and Spoken Arabic Digits was a measurement artifact: augmentation directly perturbs input variance, which mechanically shrinks projector-space effective rank on top of whatever the model actually learned, and that contamination doesn't cancel out evenly across datasets when you control for α. The corrected metric removes it.
>
> This is confirmed a third way: config-level correlation (averaging the 5 seeds per configuration down to n=10 points, the more conservative test given seed non-independence) agrees with the partial-correlation table above in every dataset — ECG5000 r=−0.958 (p<0.001), FordA r=−0.669 (p=0.034), HAR r=−0.841 (p=0.002), Spoken Arabic Digits r=−0.925 (p<0.001), EigenWorms r=+0.056 (p=0.879, still null).

**Fine-tune accuracy** shows no reliable relationship with effective rank under either measurement, pooled or partial, in any dataset (all |r| < 0.25, all p > 0.13 except EigenWorms' pooled naive-metric fine-tune correlation at p=0.026, which does not survive the corrected metric or partial control). This null result is itself a clean, replicating finding — see Section 5c.

### 5c. H2 — Early-epoch predictability

Regressing final linear-probe accuracy on effective rank at epoch 5 (`early_erank_5ep`), corrected `erank(h)` metric, both pooled and controlling for α and projector width.

| Dataset | Pooled R² | Pooled p | Partial r (controlling α, width) | Partial p |
|---|---|---|---|---|
| ECG5000 | 0.838 | <0.001 | −0.397 | 0.004 |
| UCI HAR | 0.661 | <0.001 | −0.616 | <0.001 |
| Spoken Arabic Digits | 0.805 | <0.001 | **−0.920** | <0.001 |
| FordA | 0.324 | <0.001 | −0.563 | <0.001 |
| EigenWorms | 0.000 | 0.915 (ns) | −0.095 | 0.513 (ns) |

> [!IMPORTANT]
> **Four of five datasets show a strong, significant relationship that survives controlling for α** — this is a materially stronger and more defensible result than an earlier draft's uncontrolled version, which would not have distinguished "early rank predicts accuracy" from "early rank and accuracy both just track α." With the confound controlled for, the practical claim holds: **you can estimate final linear-probe representation quality from effective rank at epoch 5, without training to convergence, in 4 of 5 datasets tested — but this relationship disappears for fine-tune accuracy in every dataset** (all fine-tune R² ≤ 0.058, all p > 0.09; not tabulated above for space, available in `results/*/results.csv`). EigenWorms remains the one consistent exception, for reasons established directly in Section 5g below — not left as an open question.

### 5d. Baselines — did SSL pretraining actually do anything?

| Dataset | Majority-class acc. | Random-init encoder, linear probe | SSL-pretrained, linear probe (mean across all 50 runs) | SSL-pretrained, linear probe (best run) |
|---|---|---|---|---|
| ECG5000 | 0.584 | 0.666 | **0.819** | 0.890 |
| FordA | 0.516 | 0.516 (= majority exactly) | **0.852** | 0.898 |
| UCI HAR | 0.182 | 0.447 | **0.830** | 0.867 |
| Spoken Arabic Digits | 0.100 | 0.457 | **0.922** | 0.970 |
| EigenWorms | 0.420 | 0.420 (= majority exactly) | **0.413** | 0.496 |

> [!IMPORTANT]
> SSL pretraining clears both baselines by a wide margin in four of five datasets — mean linear-probe accuracy across the entire 50-run sweep beats a random-init encoder by 15–47 points. **EigenWorms is the exception, and this table makes the reason unambiguous rather than merely suspected**: its random-init encoder ties the majority-class baseline *exactly* (0.420 both), and SSL pretraining's mean linear-probe accuracy (0.413) does not clear either one. Whatever representation VICReg pretraining produces on this dataset, it is not usable by a frozen linear probe under this evaluation protocol — independent of collapse, independent of α. This directly explains the null results in Sections 5b and 5c for EigenWorms: there's no accuracy signal to correlate with rank in the first place.
>
> Notably, EigenWorms' fine-tune baseline (random-init encoder, then fully fine-tuned: 0.779) is well above both the majority baseline and the frozen linear-probe numbers — the representation *is* learnable given enough adaptation capacity, it's specifically the frozen-linear-probe protocol combined with a small (155-sample), imbalanced, 5-class training set that's the bottleneck. This is a dataset/protocol limitation, not evidence that collapse dynamics don't apply to long, high-capacity sequences.

### 5e. Fine-tune null — a second, independently replicating result

Across every dataset and every measurement variant in Sections 5b and 5c, fine-tune accuracy is essentially uncorrelated with effective rank (naive or corrected, pooled or partial, final or early-epoch). This null result replicates cleanly across 4 structurally different, fully-powered datasets (n=50 each) and is not attributable to any of the measurement issues that affected the linear-probe numbers — fine-tuning updates the encoder itself, so whatever representational structure effective rank captures at the start gets overwritten during adaptation. This is a legitimate, citable finding on its own: **the erank↔accuracy relationship is a linear-probe-specific phenomenon, not a general property of the learned representation.**

### 5f. Capacity ordering — descriptive only (see Section 5a)

Not repeated as a separate table; see the callout in Section 5a. The n=5 sample size means this observation is reported honestly as suggestive, not predictive.

### 5g. EigenWorms — why it's null, established directly rather than inferred

Earlier analysis of this dataset (an earlier draft, reduced-epoch protocol) inferred a probable floor effect from indirect evidence (most linear-probe accuracies landing exactly on the majority-class fraction). **The current, fully-completed run (same 50-epoch protocol as every other dataset, no scoped-down settings) confirms this directly**, via the baseline comparison in Section 5d: SSL-pretrained linear-probe accuracy (mean 0.413) is statistically indistinguishable from a random, never-trained encoder (0.420) on this dataset. The floor is a property of the frozen-linear-probe evaluation combined with EigenWorms' 155-sample, 5-class, imbalanced training set — not an artifact of a shortened evaluation budget (there wasn't one, this time), and not evidence against H1 (EigenWorms' collapse-magnitude result in Section 5a used only pretraining-time embeddings, unaffected by the downstream evaluation issue).

**Recommended framing for the paper:** EigenWorms is a fully valid data point for H1 (Section 5a) and for the baseline/floor-effect finding (Section 5d), and should be explicitly flagged as excluded from the H2/accuracy-correlation claims (Sections 5b, 5c) — not because it broke the pipeline, but because its own baseline comparison shows there's no learnable linear-probe signal to correlate with anything, under this specific protocol. This is a stronger, more defensible way to handle the exception than the previous draft's inferred explanation.

---

## 6. Pipeline Architecture

```
5 preprocessed .pkl files (official train/test split, per-series normalized, one per dataset)
        ↓
Single notebook (main.ipynb / notebook.py) — every stage below is one shared, unmodified
code path used identically across all 5 datasets:
        ↓
Official-split loader → TimeSeriesDataset
        ↓
Augmentation (strength α) → Encoder (channels auto-detected) → Projector (width swept)
        ↓
VICReg loss ← effective_rank() logged every epoch, BOTH on a clean held-out batch (h and z)
              AND on the augmented training view (z only, legacy/naive, kept for comparison)
        ↓
Downstream eval: linear probe AND fine-tune, both logged; random-init-encoder and
majority-class baselines computed once per dataset, independent of the α/width sweep
        ↓
results/<dataset>/runs/*.json (resumable — completed configs are skipped on re-run)
        ↓
Cross-dataset analysis cells: naive-vs-corrected comparison, pooled + partial + config-level
correlation, H2 early-prediction (pooled + partial), baseline table, capacity observation
```

Adding a new dataset touches exactly one thing: one new preprocessed `.pkl` file matching the shared schema (`X`, `y`, `official_train_idx`, `official_test_idx`, `num_classes`, `in_channels`). Everything else — model, augmentations, loss, training loop, measurement, analysis — is shared, unmodified, identical across all 5 datasets. This is what keeps the cross-dataset comparison methodologically fair.

---

## 7. Known Limitations & Honest Assessment

1. **Correlation, not causation** — the erank↔accuracy relationship is observational; no mechanistic claim is made about *why* collapse helps linear-probe accuracy specifically, beyond what VICReg's own loss terms suggest
2. **Partial correlation controls for α and projector width as observed factors**, not a full causal graph — residual confounding from unmeasured factors (e.g. learning-rate/α interactions) is not ruled out
3. **Single SSL method** — every claim here is about VICReg specifically; collapse dynamics differ across contrastive/non-contrastive methods and are not assumed to transfer without separate testing
4. **5 datasets total** — enough for genuine structural spread (channels, length, balance, sample size all vary), but not exhaustive; results should be read as "where we've tested so far," not a universal claim. The capacity-ordering observation (Section 5a) explicitly does not reach significance at this n and is reported as descriptive only
5. **EigenWorms' accuracy-correlation results are excluded from the pooled H2/5b claims** for the specific, evidenced reason in Section 5g, not omitted silently — it remains a full, valid data point for H1 and the baseline comparison
6. **Fixed probe train/val split** (seed 42, decoupled from the run seed) means seed variance in the reported statistics reflects model initialization and minibatch order only, not resampling of which examples are labeled — a deliberate choice for comparability across seeds, but worth stating explicitly as a design decision

---

## 8. Naive vs. Corrected Measurement — Compute Note

No dataset required a reduced-epoch or scoped-down protocol in this run; all 250 runs (5 datasets × 5 α × 2 widths × 5 seeds) completed under the identical 50-epoch protocol, including EigenWorms despite its 17,984-timestep sequences, using a GPU-backed molab Studio. The two effective-rank measurements (naive `erank(z)` on the augmented training view, corrected `erank(h)`/`erank(z)` on a clean held-out batch) add negligible overhead — the corrected measurement is a single extra forward pass per epoch, no backward pass, no augmentation — so there was no compute tradeoff involved in adding it.

---

## 9. Pre-Writing Checklist — Are We Ready to Start the Paper?

**Yes — start writing now.** Every planned dataset has completed the full, corrected, 50-run sweep, and the cross-dataset analysis (naive-vs-corrected comparison, partial correlation, config-level correlation, baselines) is done and in this document.

| # | Item | Status | Blocking? |
|---|---|---|---|
| 1 | ECG5000 sweep (50 runs) + full analysis | ✅ Done | — |
| 2 | UCI HAR sweep (50 runs) + full analysis | ✅ Done | — |
| 3 | FordA sweep (50 runs) + full analysis | ✅ Done | — |
| 4 | Spoken Arabic Digits sweep (50 runs) + full analysis | ✅ Done | — |
| 5 | EigenWorms sweep (50 runs) + full analysis, including baseline-confirmed floor-effect explanation | ✅ Done | — |
| 6 | Naive-vs-corrected effective-rank comparison across all 5 datasets | ✅ Done | — |
| 7 | Partial correlation (H1/5b and H2/5c) controlling for α and width | ✅ Done | — |
| 8 | Baseline table (random-init encoder, majority class) | ✅ Done | — |
| 9 | Cross-dataset comparison figure | ✅ Done (`figures/cross_dataset_alpha_vs_erank.png`) | — |

**Recommended order of operations from here, given tomorrow's deadline:**
1. **Write the Results section around the two headline findings**, in this order of strength: (a) the naive-vs-corrected measurement finding (Section 5b) — this is your most original, most defensible contribution, and doubles as your methods contribution; (b) the 4-for-4 linear-probe-only H2 predictability pattern, surviving partial correlation (Section 5c).
2. **Write the Method section's effective-rank subsection carefully** — explicitly define both measurements (naive/corrected) since Section 5b's finding depends on the reader understanding the distinction before seeing the result.
3. **Write the Discussion/Limitations section using Sections 5d, 5e, 5g, and 7 nearly verbatim** — they're already written in a form suitable for a paper's discussion and limitations subsections. The mechanistic sketch in Section 2 is safe to fold into the Discussion's "why might this be true" paragraph, clearly flagged as post-hoc framing rather than a tested mechanism.
4. **Introduction and Background can be drafted in parallel** — nothing in them depends on anything above.
5. **Figures to pull directly**: make **`figures/<dataset>/naive_vs_corrected.png` your Figure 1** — not a supplementary figure buried mid-paper. It's the single clearest visual evidence for your strongest claim: put Spoken Arabic Digits' or HAR's version front and center, since those are the two datasets where the naive/corrected panels visibly disagree (the left panel is noisy and weakly negative, the right panel is tight and strongly negative) — that contrast *is* the paper's core argument, shown rather than told. `figures/cross_dataset_alpha_vs_erank.png` works well as a secondary figure for the H1 summary.

**One thing to flag:** this figure did not actually exist in the repo as of this checklist being written — it was regenerated from the real run data and is now present at `figures/<dataset>/naive_vs_corrected.png` for all 5 datasets. Confirm it's committed to the repo before citing it in the paper.

Nothing on this list is still pending. You are not blocked on compute, analysis, or missing datasets.

---

## 10. References

### Foundational SSL methods

1. Chen, T., Kornblith, S., Norouzi, M., & Hinton, G. (2020). A Simple Framework for Contrastive Learning of Visual Representations (SimCLR). *ICML*.
2. He, K., Fan, H., Wu, Y., Xie, S., & Girshick, R. (2020). Momentum Contrast for Unsupervised Visual Representation Learning (MoCo). *CVPR*.
3. Grill, J.-B., Strub, F., Altché, F., et al. (2020). Bootstrap Your Own Latent: A New Approach to Self-Supervised Learning (BYOL). *NeurIPS*.
4. Caron, M., Misra, I., Mairal, J., et al. (2020). Unsupervised Learning of Visual Features by Contrasting Cluster Assignments (SwAV). *NeurIPS*.
5. Caron, M., Touvron, H., Misra, I., et al. (2021). Emerging Properties in Self-Supervised Vision Transformers (DINO). *ICCV*.
6. Zbontar, J., Jing, L., Misra, I., LeCun, Y., & Deny, S. (2021). Barlow Twins: Self-Supervised Learning via Redundancy Reduction. *ICML*.
7. Bardes, A., Ponce, J., & LeCun, Y. (2022). VICReg: Variance-Invariance-Covariance Regularization for Self-Supervised Learning. *ICLR*.
8. Chen, X., & He, K. (2021). Exploring Simple Siamese Representation Learning (SimSiam). *CVPR*.
9. Ermolov, A., Siarohin, A., Sangineto, E., & Sebe, N. (2021). Whitening for Self-Supervised Representation Learning (W-MSE). *ICML*.
10. Oquab, M., Darcet, T., Moutakanni, T., et al. (2023). DINOv2: Learning Robust Visual Features without Supervision. *arXiv:2304.07193*.

### Dimensional collapse — theory and mechanisms

11. Jing, L., Vincent, P., LeCun, Y., & Tian, Y. (2022). Understanding Dimensional Collapse in Contrastive Self-Supervised Learning. *ICLR*.
12. Hua, T., Wang, W., Xue, Z., Ren, S., Wang, Y., & Zhao, H. (2021). On Feature Decorrelation in Self-Supervised Learning. *ICCV*.
13. Tian, Y., Chen, X., & Ganguli, S. (2021). Understanding Self-Supervised Learning Dynamics without Contrastive Pairs. *ICML*.
14. Ziyin, L., Lubana, E. S., Ueda, M., & Tanaka, H. (2023). What Shapes the Loss Landscape of Self-Supervised Learning? *ICLR*.
15. Garrido, Q., Chen, Y., Bardes, A., Najman, L., & LeCun, Y. (2023). On the Duality Between Contrastive and Non-Contrastive Self-Supervised Learning. *ICLR*.
16. He, B., & Ozay, M. (2022). Exploring the Gap Between Collapsed & Whitened Features in Self-Supervised Learning. *ICML*.
17. Pokle, A., Tian, J., Li, Y., & Risteski, A. (2022). Contrasting the Landscape of Contrastive and Non-Contrastive Learning. *AISTATS*.
18. Weng, X., Ni, L., & Wang, L. (2024). OrthoReg: Robust Anti-Collapse Regularization via Orthogonalization for Self-Supervised Learning. *NeurIPS*.
19. Cosentino, R., Sengupta, A., Avestimehr, S., et al. (2022). Toward a Geometrical Understanding of Self-Supervised Contrastive Learning. *arXiv:2205.06926*.

### Downstream evaluation and predictive diagnostics

20. Zhang, C., Zhang, K., Zhang, C., et al. (2022). How Does SimSiam Avoid Collapse Without Negative Samples? A Unified Understanding with Self-Supervised Contrastive Learning. *ICLR*.
21. Roy, O., & Vetterli, M. (2007). The Effective Rank: A Measure of Effective Dimensionality. *EUSIPCO*.
22. Wang, T., & Isola, P. (2020). Understanding Contrastive Representation Learning through Alignment and Uniformity on the Hypersphere. *ICML*.
23. Nozawa, K., & Sato, I. (2021). Understanding Negative Samples in Instance Discriminative Self-Supervised Representation Learning. *NeurIPS*.

### Time-series self-supervised learning (application domain)

24. Yue, Z., Wang, Y., Duan, J., et al. (2022). TS2Vec: Towards Universal Representation of Time Series. *AAAI*.
25. Eldele, E., Ragab, M., Chen, Z., et al. (2021). Time-Series Representation Learning via Temporal and Contextual Contrasting (TS-TCC). *IJCAI*.
26. Franceschi, J.-Y., Dieuleveut, A., & Jaggi, M. (2019). Unsupervised Scalable Representation Learning for Multivariate Time Series. *NeurIPS*.
27. Tonekaboni, S., Eytan, D., & Goldenberg, A. (2021). Unsupervised Representation Learning for Time Series with Temporal Neighborhood Coding. *ICLR*.

### Datasets used

28. Dau, H. A., Bagnall, A., Kamgar, K., et al. (2019). The UCR Time Series Archive. *IEEE/CAA Journal of Automatica Sinica*.
29. Bagnall, A., Dau, H. A., Lines, J., et al. (2018). The UEA Multivariate Time Series Classification Archive. *arXiv:1811.00075*.
30. Anguita, D., Ghio, A., Oneto, L., Parra, X., & Reyes-Ortiz, J. L. (2013). A Public Domain Dataset for Human Activity Recognition Using Smartphones. *ESANN*.

---

## 11. Repository Structure

```
dimensional-collapse-research/
├── README.md                  (this file)
├── main.ipynb / notebook.py    (single notebook — every pipeline stage, all 5 datasets)
├── pkl/
│   └── ecg5000.pkl / forda.pkl / har.pkl / spokenarabicdigits.pkl / eigenworms.pkl
│       (preprocessed: official train/test split, per-series normalized)
├── results/
│   ├── baselines.csv           (majority-class + random-encoder baselines, per dataset)
│   ├── all_runs.csv
│   └── <dataset_name>/
│       ├── results.csv
│       └── runs/                (per-run JSON logs, resumable sweep)
└── figures/
    ├── cross_dataset_alpha_vs_erank.png
    ├── alpha_vs_erank_per_dataset.png
    ├── erank_vs_accuracy.png
    ├── early_prediction.png
    └── <dataset_name>/
        ├── alpha_vs_erank.png
        ├── erank_vs_accuracy.png
        ├── naive_vs_corrected.png
        └── early_prediction.png
```
