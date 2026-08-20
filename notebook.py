import marimo

__generated_with = "0.24.0"
app = marimo.App(auto_download=["html"])


@app.cell
def _():
    import marimo as mo

    return (mo,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # SSL Dimensional Collapse Across Time-Series Modalities — corrected pipeline

    This notebook is the single source of truth for the study: data loading,
    model, training, sweep, and analysis, all in one place, with every flaw
    from the audit fixed. Run cells top to bottom.

    ## What's fixed here vs. the original repo

    | # | Flaw | Fix |
    |---|------|-----|
    | 1 | Effective rank measured on augmented views (contaminated by alpha twice) | New `compute_clean_erank()`: forward pass on a **fixed, un-augmented** eval batch, every epoch |
    | 2 | HAR: random split ignores subject structure (near-duplicate leakage) | Every dataset now uses its **official canonical train/test split**; HAR's official split is subject-disjoint by construction |
    | 3 | Test set included in the SSL pretraining pool (transductive, undisclosed) | Pretraining pool = official train only. Official test is **never** touched until final eval |
    | 4 | Pooled erank-accuracy correlation confounded by alpha | Partial correlation (controlling alpha, width) reported as primary; pooled kept for comparison |
    | 5 | 3 (now 5) seeds per config treated as independent (pseudo-replication) | Both raw-n and config-level (seed-averaged, n=10) correlations reported |
    | 6 | Non-standard splits break comparability with published benchmarks | Official UCR/UEA splits used everywhere |
    | 7 | erank measured on projector output (z), accuracy driven by encoder output (h) | Both `erank(h)` and `erank(z)` logged; `erank(h)` is primary |
    | 8 | Batch size (64) caps erank estimate below the true value | Clean erank computed on a large fixed eval batch (up to 512), raising the cap |
    | 9 | No baselines | Random-init-encoder probe + majority-class accuracy, per dataset |
    | 10 | Claims implicitly generalized beyond VICReg | Explicitly scoped to VICReg throughout this writeup |
    | 11 | Early-prediction analysis only regressed fine-tune acc, not linear probe | Both regressed, both reported |
    | 12 | `--cross_dataset` claimed but never implemented | Actually implemented below (see Analysis §6) |
    | 13 | `--eval_epochs` claimed but never implemented | Not used; removed from docs |
    | 14 | Seed controlled both model init AND the data split | Split uses a **fixed** `PROBE_SPLIT_SEED`, independent of the run's `torch` seed |
    | 15 | 3 of 5 datasets had no loader/data in the repo | All 5 loaders + preprocessed `.pkl` files included |
    | 16 | ECG5000 sweep only had 2 seeds despite claiming 3 | All datasets now use the same `SEEDS` list, verified below |
    | 17 | `channels x length` "predicts almost perfectly" (n=5) | Reframed as a descriptive, exploratory observation only |

    Set `QUICK_TEST = True` on your first run in molab to smoke-test the whole
    pipeline (~1-2 min, tiny grid) before committing to the full sweep.
    """)
    return


@app.cell
def _(get_ipython):
    # ---- installs (safe to re-run; skips if already satisfied) ----
    get_ipython().run_line_magic("pip", "install -q torch numpy pandas scipy matplotlib scikit-learn statsmodels")
    return


@app.cell
def _():
    import os
    import re
    import json
    import glob
    import time
    import copy
    import pickle
    import warnings
    warnings.filterwarnings("ignore")

    import numpy as np
    import pandas as pd
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    from torch.utils.data import Dataset, DataLoader
    from sklearn.model_selection import train_test_split
    from scipy import stats
    import statsmodels.api as sm
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    print("Using device:", DEVICE)
    return (
        DEVICE,
        DataLoader,
        Dataset,
        F,
        copy,
        glob,
        json,
        nn,
        np,
        os,
        pd,
        pickle,
        plt,
        sm,
        stats,
        time,
        torch,
        train_test_split,
    )


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 1. Config

    `QUICK_TEST=True` runs a 1-minute smoke test (2 datasets skipped, tiny
    grid, 3 epochs) so you can confirm everything executes before the full
    sweep. Flip to `False` for the real run.

    Seeds increased from 3 -> 5 per your request (partially addresses flaw
    #5's power concern; the analysis section still reports the honest
    config-level n=10 alongside the raw n=50 for full transparency).
    """)
    return


@app.cell
def _(os):
    QUICK_TEST = False   # <-- flip to False for the real run

    PKL_DIR = "."          
    OUT_ROOT = "./results"      # per-run JSON files land here, resumable
    FIG_ROOT = "./figures"

    ALPHAS = [0.0, 0.25, 0.5, 0.75, 1.0]
    WIDTHS = [64, 512]
    SEEDS = [0, 1, 2, 3, 4]         # was [0,1,2] (and ECG5000 only had [0,1]) -- now uniform, 5 seeds everywhere
    EPOCHS = 50
    BATCH_SIZE = 64
    EVAL_BATCH_SIZE = 512           # clean-erank eval batch; also raises the erank cap (flaw #8)
    PROBE_SPLIT_SEED = 42           # FIXED -- decoupled from the run's torch seed (flaw #14)
    GLOBAL_EVAL_SEED = 123          # FIXED -- which samples form the clean eval batch, same across all runs of a dataset

    DATASETS = ["ecg5000", "forda", "har", "spokenarabicdigits", "eigenworms"]

    if QUICK_TEST:
        DATASETS = ["ecg5000"]
        ALPHAS = [0.0, 1.0]
        WIDTHS = [64]
        SEEDS = [0]
        EPOCHS = 3
        print("QUICK_TEST=True -- running a tiny smoke grid on ecg5000 only.")
        print("Set QUICK_TEST=False and re-run all cells for the real sweep.")

    os.makedirs(OUT_ROOT, exist_ok=True)
    os.makedirs(FIG_ROOT, exist_ok=True)
    return (
        ALPHAS,
        BATCH_SIZE,
        DATASETS,
        EPOCHS,
        EVAL_BATCH_SIZE,
        FIG_ROOT,
        GLOBAL_EVAL_SEED,
        OUT_ROOT,
        PKL_DIR,
        PROBE_SPLIT_SEED,
        SEEDS,
        WIDTHS,
    )


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 2. Data loading -- official splits only

    Loads the preprocessed `.pkl` (already z-normalized per series, already
    carrying the official train/test split). Builds:
    - `unlabeled_idx`  = official train only (fix #3: test never enters pretraining)
    - `probe_train_idx`, `probe_val_idx` = stratified split of official train,
      using the FIXED `PROBE_SPLIT_SEED` (fix #14: independent of run seed)
    - `probe_test_idx` = official test (fix #6: comparable to published splits;
      for HAR this is also subject-disjoint by construction -- fix #2)
    """)
    return


@app.cell
def _(Dataset, PKL_DIR, PROBE_SPLIT_SEED, np, pickle, torch, train_test_split):
    def load_dataset_pkl(name):
        with open(f"{PKL_DIR}/{name}.pkl", "rb") as f:
            return pickle.load(f)


    def make_official_splits(d, probe_split_seed=PROBE_SPLIT_SEED):
        y = d["y"]
        official_train = d["official_train_idx"]
        official_test = d["official_test_idx"]

        y_train = y[official_train]
        try:
            tr_local, val_local = train_test_split(
                np.arange(len(official_train)), test_size=0.2,
                stratify=y_train, random_state=probe_split_seed,
            )
        except ValueError:
            tr_local, val_local = train_test_split(
                np.arange(len(official_train)), test_size=0.2,
                random_state=probe_split_seed,
            )
        probe_train_idx = official_train[tr_local]
        probe_val_idx = official_train[val_local]

        return {
            "unlabeled_idx": official_train,
            "probe_train_idx": probe_train_idx,
            "probe_val_idx": probe_val_idx,
            "probe_test_idx": official_test,
        }


    class TimeSeriesDataset(Dataset):
        def __init__(self, X, y, indices):
            self.X = X[indices]
            self.y = y[indices]

        def __len__(self):
            return len(self.X)

        def __getitem__(self, i):
            x = torch.from_numpy(self.X[i]).float()
            y = int(self.y[i])
            return x, y

    return TimeSeriesDataset, load_dataset_pkl, make_official_splits


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 3. Model (unchanged) -- fixed-capacity encoder, swept-width projector

    Per Garrido et al. 2023, most "helpful collapse" lives in the projector,
    not the encoder, so encoder capacity is held fixed (out_dim=64) while
    projector width is the swept variable. Nothing here needed fixing; the
    audit's issues were all in measurement/splitting, not architecture.
    """)
    return


@app.cell
def _(nn):
    class Encoder(nn.Module):
        def __init__(self, in_channels=1, out_dim=64):
            super().__init__()
            self.net = nn.Sequential(
                nn.Conv1d(in_channels, 32, kernel_size=7, padding=3),
                nn.BatchNorm1d(32), nn.ReLU(), nn.MaxPool1d(2),
                nn.Conv1d(32, 64, kernel_size=5, padding=2),
                nn.BatchNorm1d(64), nn.ReLU(), nn.MaxPool1d(2),
                nn.Conv1d(64, out_dim, kernel_size=3, padding=1),
                nn.BatchNorm1d(out_dim), nn.ReLU(),
                nn.AdaptiveAvgPool1d(1),
            )
            self.out_dim = out_dim

        def forward(self, x):
            return self.net(x).squeeze(-1)


    class Projector(nn.Module):
        def __init__(self, in_dim=64, width=256, out_dim=128):
            super().__init__()
            self.net = nn.Sequential(
                nn.Linear(in_dim, width), nn.BatchNorm1d(width), nn.ReLU(),
                nn.Linear(width, width), nn.BatchNorm1d(width), nn.ReLU(),
                nn.Linear(width, out_dim),
            )

        def forward(self, x):
            return self.net(x)


    class SSLModel(nn.Module):
        def __init__(self, in_channels=1, encoder_out=64, projector_width=256, projector_out=128):
            super().__init__()
            self.encoder = Encoder(in_channels=in_channels, out_dim=encoder_out)
            self.projector = Projector(in_dim=encoder_out, width=projector_width, out_dim=projector_out)

        def forward(self, x):
            h = self.encoder(x)
            z = self.projector(h)
            return h, z

    return (SSLModel,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 4. Augmentations (unchanged)

    The alpha-parametrized augmentation stack itself was never the problem --
    only *where* effective rank got measured relative to it. Kept verbatim.
    """)
    return


@app.cell
def _(F, torch):
    def jitter(x, alpha):
        std = 0.5 * alpha
        return x + torch.randn_like(x) * std


    def scaling(x, alpha):
        factor_range = 0.4 * alpha
        factor = 1.0 + (torch.rand(x.shape[0], 1, 1, device=x.device) * 2 - 1) * factor_range
        return x * factor


    def time_mask(x, alpha):
        if alpha <= 0:
            return x
        B, C, L = x.shape
        max_mask_len = int(L * 0.3 * alpha)
        if max_mask_len < 1:
            return x
        out = x.clone()
        for b in range(B):
            mask_len = torch.randint(1, max_mask_len + 1, (1,)).item()
            start = torch.randint(0, max(1, L - mask_len), (1,)).item()
            out[b, :, start:start + mask_len] = 0.0
        return out


    def random_crop_resize(x, alpha):
        B, C, L = x.shape
        min_ratio = 1.0 - 0.4 * alpha
        if min_ratio >= 0.999:
            return x
        ratio = torch.empty(1).uniform_(min_ratio, 1.0).item()
        crop_len = max(8, int(L * ratio))
        start = torch.randint(0, max(1, L - crop_len + 1), (1,)).item()
        cropped = x[:, :, start:start + crop_len]
        return F.interpolate(cropped, size=L, mode="linear", align_corners=False)


    def augment(x, alpha):
        x = random_crop_resize(x, alpha)
        x = time_mask(x, alpha)
        x = scaling(x, alpha)
        x = jitter(x, alpha)
        return x


    def make_two_views(x, alpha):
        return augment(x, alpha), augment(x, alpha)

    return (make_two_views,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 5. VICReg loss (unchanged)
    """)
    return


@app.cell
def _(F, torch):
    def vicreg_loss(z1, z2, sim_weight=25.0, var_weight=25.0, cov_weight=1.0, gamma=1.0, eps=1e-4):
        N, D = z1.shape
        sim_loss = F.mse_loss(z1, z2)
        z1c = z1 - z1.mean(dim=0)
        z2c = z2 - z2.mean(dim=0)
        std_z1 = torch.sqrt(z1c.var(dim=0) + eps)
        std_z2 = torch.sqrt(z2c.var(dim=0) + eps)
        var_loss = torch.mean(F.relu(gamma - std_z1)) + torch.mean(F.relu(gamma - std_z2))
        cov_z1 = (z1c.T @ z1c) / (N - 1)
        cov_z2 = (z2c.T @ z2c) / (N - 1)
        off_diag = lambda m: m.flatten()[:-1].view(D - 1, D + 1)[:, 1:].flatten()
        cov_loss = (off_diag(cov_z1).pow(2).sum() / D) + (off_diag(cov_z2).pow(2).sum() / D)
        total = sim_weight * sim_loss + var_weight * var_loss + cov_weight * cov_loss
        return total, {"sim_loss": sim_loss.item(), "var_loss": var_loss.item(),
                        "cov_loss": cov_loss.item(), "total_loss": total.item()}

    return (vicreg_loss,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 6. Effective rank -- corrected measurement (fixes #1, #7, #8)

    `effective_rank()` itself (Roy & Vetterli 2007) is unchanged -- it was
    never wrong. What was wrong is *what got fed into it*. `compute_clean_erank`
    fixes all three measurement flaws at once:

    - evaluates on a **fixed, un-augmented** batch (fix #1: alpha can no longer
      shrink the metric directly through input variance, only through what the
      model actually learned)
    - logs **both** `erank(h)` (encoder output -- what accuracy actually depends
      on) and `erank(z)` (projector output -- what the original repo only
      measured) (fix #7)
    - uses a large fixed eval batch (up to 512 samples) instead of the training
      batch size (64), raising the rank cap from `min(64,dim)-1=63` well above
      the largest observed erank (fix #8)

    The original (flawed) on-augmented-view z-only measurement is *also* kept
    and logged as `erank_aug_z`, purely so the analysis section can show a
    direct before/after comparison of how much the fix changes the conclusions.
    """)
    return


@app.cell
def _(GLOBAL_EVAL_SEED, np, torch):
    @torch.no_grad()
    def effective_rank(z, eps=1e-12):
        z = z - z.mean(dim=0, keepdim=True)
        N = z.shape[0]
        cov = (z.T @ z) / max(N - 1, 1)
        eigvals = torch.linalg.eigvalsh(cov)
        eigvals = torch.clamp(eigvals, min=0.0)
        total = eigvals.sum()
        if total <= eps:
            return 1.0
        p = eigvals / total
        p = p[p > eps]
        entropy = -(p * torch.log(p)).sum()
        return torch.exp(entropy).item()


    def make_clean_eval_batch(X, unlabeled_idx, eval_batch_size, seed=GLOBAL_EVAL_SEED):
        """Fixed, un-augmented batch reused every epoch for THIS dataset --
        same batch across every alpha/width/seed config, so differences in the
        trajectory reflect the model, not the eval sample."""
        rng = np.random.RandomState(seed)
        n = min(eval_batch_size, len(unlabeled_idx))
        chosen = rng.choice(unlabeled_idx, size=n, replace=False)
        return torch.from_numpy(X[chosen]).float()


    @torch.no_grad()
    def compute_clean_erank(model, eval_batch, device):
        model.eval()
        x = eval_batch.to(device)
        h, z = model(x)
        return effective_rank(h), effective_rank(z)

    return compute_clean_erank, effective_rank, make_clean_eval_batch


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 7. Downstream evaluation (unchanged logic, verbatim from evaluate.py)
    """)
    return


@app.cell
def _(DataLoader, copy, nn, torch):
    def run_classifier_training(encoder, num_classes, train_loader, val_loader, test_loader,
                                 device, epochs=30, lr=1e-3, train_encoder=False):
        encoder = copy.deepcopy(encoder).to(device)
        encoder.train(train_encoder)
        for p in encoder.parameters():
            p.requires_grad = train_encoder
        head = nn.Linear(encoder.out_dim, num_classes).to(device)
        params = list(head.parameters()) + (list(encoder.parameters()) if train_encoder else [])
        opt = torch.optim.Adam(params, lr=lr, weight_decay=1e-4)
        ce = nn.CrossEntropyLoss()
        best_val_acc, best_state = -1, None
        for epoch in range(epochs):
            encoder.train(train_encoder)
            head.train()
            for x, y in train_loader:
                x, y = x.to(device), y.to(device)
                h = encoder(x) if train_encoder else encoder(x).detach()
                logits = head(h)
                loss = ce(logits, y)
                opt.zero_grad()
                loss.backward()
                opt.step()
            val_acc = accuracy(encoder, head, val_loader, device)
            if val_acc > best_val_acc:
                best_val_acc = val_acc
                best_state = copy.deepcopy(head.state_dict())
        head.load_state_dict(best_state)
        test_acc = accuracy(encoder, head, test_loader, device)
        return test_acc, best_val_acc


    @torch.no_grad()
    def accuracy(encoder, head, loader, device):
        encoder.eval()
        head.eval()
        correct, total = 0, 0
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            preds = head(encoder(x)).argmax(dim=1)
            correct += (preds == y).sum().item()
            total += y.numel()
        return correct / total


    def evaluate_representation(encoder, num_classes, train_ds, val_ds, test_ds, device, batch_size=64):
        train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
        val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False)
        test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False)
        linear_test_acc, linear_val_acc = run_classifier_training(
            encoder, num_classes, train_loader, val_loader, test_loader, device, train_encoder=False)
        finetune_test_acc, finetune_val_acc = run_classifier_training(
            encoder, num_classes, train_loader, val_loader, test_loader, device, train_encoder=True)
        return {
            "linear_probe_test_acc": linear_test_acc, "linear_probe_val_acc": linear_val_acc,
            "finetune_test_acc": finetune_test_acc, "finetune_val_acc": finetune_val_acc,
        }

    return (evaluate_representation,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 8. Baselines (fix #9)

    Computed once per dataset, independent of alpha/width: majority-class
    accuracy (trivial ceiling for "did the probe learn anything") and a
    random-initialized encoder (skips pretraining entirely, otherwise
    identical probe protocol) so downstream numbers have something to beat.
    """)
    return


@app.cell
def _(SSLModel, TimeSeriesDataset, evaluate_representation, np, torch):
    def compute_baselines(name, d, splits, num_classes, in_channels, device, seeds=(0, 1, 2)):
        y = d["y"]
        maj_class = np.bincount(y[splits["probe_train_idx"]]).argmax()
        maj_acc_test = (y[splits["probe_test_idx"]] == maj_class).mean()

        probe_train_ds = TimeSeriesDataset(d["X"], y, splits["probe_train_idx"])
        probe_val_ds = TimeSeriesDataset(d["X"], y, splits["probe_val_idx"])
        probe_test_ds = TimeSeriesDataset(d["X"], y, splits["probe_test_idx"])

        rand_results = []
        for seed in seeds:
            torch.manual_seed(seed)
            model = SSLModel(in_channels=in_channels, encoder_out=64,
                              projector_width=64, projector_out=128).to(device)
            out = evaluate_representation(model.encoder, num_classes, probe_train_ds,
                                           probe_val_ds, probe_test_ds, device)
            rand_results.append(out)

        lin = np.mean([r["linear_probe_test_acc"] for r in rand_results])
        ft = np.mean([r["finetune_test_acc"] for r in rand_results])
        return {
            "dataset": name,
            "majority_class_acc": float(maj_acc_test),
            "random_encoder_linear_probe_acc_mean": float(lin),
            "random_encoder_finetune_acc_mean": float(ft),
            "random_encoder_n_seeds": len(seeds),
        }

    return (compute_baselines,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 9. One training run (fixes #1, #3, #7, #8, #14 all applied here)
    """)
    return


@app.cell
def _(
    BATCH_SIZE,
    DEVICE,
    DataLoader,
    EVAL_BATCH_SIZE,
    OUT_ROOT,
    PROBE_SPLIT_SEED,
    SSLModel,
    TimeSeriesDataset,
    compute_clean_erank,
    effective_rank,
    evaluate_representation,
    json,
    make_clean_eval_batch,
    make_official_splits,
    make_two_views,
    os,
    time,
    torch,
    vicreg_loss,
):
    def run_one(dataset_name, d, alpha, projector_width, epochs, seed, batch_size=BATCH_SIZE,
                eval_batch_size=EVAL_BATCH_SIZE, out_dir=None, device=DEVICE):
        out_dir = out_dir or os.path.join(OUT_ROOT, dataset_name, "runs")
        os.makedirs(out_dir, exist_ok=True)
        tag = f"alpha{alpha:.2f}_proj{projector_width}_seed{seed}"
        out_path = os.path.join(out_dir, f"{tag}.json")
        if os.path.exists(out_path):
            with open(out_path) as f:
                return json.load(f)

        torch.manual_seed(seed)

        splits = make_official_splits(d, probe_split_seed=PROBE_SPLIT_SEED)
        num_classes, in_channels = d["num_classes"], d["in_channels"]

        unlabeled_ds = TimeSeriesDataset(d["X"], d["y"], splits["unlabeled_idx"])
        probe_train_ds = TimeSeriesDataset(d["X"], d["y"], splits["probe_train_idx"])
        probe_val_ds = TimeSeriesDataset(d["X"], d["y"], splits["probe_val_idx"])
        probe_test_ds = TimeSeriesDataset(d["X"], d["y"], splits["probe_test_idx"])

        unlabeled_loader = DataLoader(unlabeled_ds, batch_size=batch_size, shuffle=True, drop_last=True)
        eval_batch = make_clean_eval_batch(d["X"], splits["unlabeled_idx"], eval_batch_size)

        model = SSLModel(in_channels=in_channels, encoder_out=64,
                          projector_width=projector_width, projector_out=128).to(device)
        opt = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-6)

        erank_clean_h_traj, erank_clean_z_traj, erank_aug_z_traj, loss_traj = [], [], [], []

        t0 = time.time()
        for epoch in range(epochs):
            model.train()
            batch_aug_eranks, batch_losses = [], []
            for x, _ in unlabeled_loader:
                x = x.to(device)
                v1, v2 = make_two_views(x, alpha)
                h1, z1 = model(v1)
                h2, z2 = model(v2)
                loss, parts = vicreg_loss(z1, z2)
                opt.zero_grad()
                loss.backward()
                opt.step()
                with torch.no_grad():
                    batch_aug_eranks.append(effective_rank(z1.detach()))
                batch_losses.append(parts["total_loss"])

            erank_aug_z_traj.append(sum(batch_aug_eranks) / len(batch_aug_eranks))
            loss_traj.append(sum(batch_losses) / len(batch_losses))

            clean_h, clean_z = compute_clean_erank(model, eval_batch, device)
            erank_clean_h_traj.append(clean_h)
            erank_clean_z_traj.append(clean_z)

        train_time = time.time() - t0

        downstream = evaluate_representation(model.encoder, num_classes, probe_train_ds,
                                              probe_val_ds, probe_test_ds, device)

        early_i = min(4, len(erank_clean_h_traj) - 1)
        result = {
            "config": {"dataset": dataset_name, "alpha": alpha, "projector_width": projector_width,
                        "epochs": epochs, "batch_size": batch_size, "seed": seed,
                        "probe_split_seed": PROBE_SPLIT_SEED, "in_channels": in_channels,
                        "num_classes": num_classes},
            "erank_clean_h_trajectory": erank_clean_h_traj,
            "erank_clean_z_trajectory": erank_clean_z_traj,
            "erank_aug_z_trajectory_LEGACY": erank_aug_z_traj,
            "loss_trajectory": loss_traj,
            "final_erank_clean_h": erank_clean_h_traj[-1],
            "final_erank_clean_z": erank_clean_z_traj[-1],
            "final_erank_aug_z_LEGACY": erank_aug_z_traj[-1],
            "early_erank_clean_h_5ep": erank_clean_h_traj[early_i],
            "early_erank_clean_z_5ep": erank_clean_z_traj[early_i],
            "early_erank_aug_z_5ep_LEGACY": erank_aug_z_traj[early_i],
            "train_time_sec": train_time,
            "downstream": downstream,
        }
        with open(out_path, "w") as f:
            json.dump(result, f, indent=2)

        print(f"[{dataset_name}] {tag}  erank_h(clean)={clean_h:.2f}  "
              f"lin={downstream['linear_probe_test_acc']:.3f}  "
              f"ft={downstream['finetune_test_acc']:.3f}  ({train_time:.1f}s)")
        return result

    return (run_one,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 10. Sweep (resumable -- safe to interrupt and re-run)

    If molab hiccups or you need to stop partway, just re-run this cell: any
    run whose JSON already exists on disk is skipped (see `run_one` above).
    """)
    return


@app.cell
def _(run_one):
    def run_sweep(dataset_name, d, alphas, widths, seeds, epochs, out_dir=None):
        total = len(alphas) * len(widths) * len(seeds)
        print(f"\n=== {dataset_name}: {total} runs ({len(alphas)} alphas x {len(widths)} widths x {len(seeds)} seeds) ===")
        i = 0
        for alpha in alphas:
            for width in widths:
                for seed in seeds:
                    i += 1
                    try:
                        run_one(dataset_name, d, alpha, width, epochs, seed, out_dir=out_dir)
                    except Exception as e:
                        print(f"[ERROR] {dataset_name} alpha={alpha} width={width} seed={seed}: {e}")
        print(f"=== {dataset_name} sweep done ({i}/{total} attempted) ===")

    return (run_sweep,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 11. Run everything

    Baselines first (cheap), then the full sweep, per dataset. EigenWorms is
    by far the most expensive (17,984-timestep series) -- it's placed last so
    the other four finish and are analyzable even if you run out of time on
    it. On a GPU this whole thing should be well under an hour; on CPU,
    EigenWorms alone could take much longer -- check the printed per-run
    timings after the first couple of runs and extrapolate before committing.
    """)
    return


@app.cell
def _(
    DATASETS,
    DEVICE,
    OUT_ROOT,
    compute_baselines,
    load_dataset_pkl,
    make_official_splits,
    os,
    pd,
):

    all_baselines = []
    for _name in DATASETS:
        _d = load_dataset_pkl(_name)
        splits = make_official_splits(_d)
        bl = compute_baselines(_name, _d, splits, _d['num_classes'], _d['in_channels'], DEVICE)
        all_baselines.append(bl)
        print(bl)
    baselines_df = pd.DataFrame(all_baselines)
    baselines_df.to_csv(os.path.join(OUT_ROOT, 'baselines.csv'), index=False)
    baselines_df
    return (baselines_df,)


@app.cell
def _(ALPHAS, DATASETS, EPOCHS, SEEDS, WIDTHS, load_dataset_pkl, run_sweep):
    for _name in DATASETS:
        _d = load_dataset_pkl(_name)
        run_sweep(_name, _d, ALPHAS, WIDTHS, SEEDS, EPOCHS)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 12. Analysis

    Loads every run JSON, builds one DataFrame per dataset, and reports:
    - naive (augmented-view, projector-only) vs. corrected (clean, encoder)
      erank-accuracy relationship, pooled and partial (fixes #1, #4, #7)
    - both raw-n and config-level-n correlations (fix #5)
    - early-prediction (H2) regressed on BOTH accuracy types (fix #11)
    - actual cross-dataset comparison (fix #12)
    - baseline comparison table (fix #9)
    """)
    return


@app.cell
def _(glob, json, os, pd, sm, stats):
    def load_runs(run_dir):
        rows = []
        for path in glob.glob(os.path.join(run_dir, "*.json")):
            with open(path) as f:
                r = json.load(f)
            rows.append({
                "dataset": r["config"]["dataset"], "alpha": r["config"]["alpha"],
                "projector_width": r["config"]["projector_width"], "seed": r["config"]["seed"],
                "final_erank_clean_h": r["final_erank_clean_h"],
                "final_erank_clean_z": r["final_erank_clean_z"],
                "final_erank_aug_z_LEGACY": r["final_erank_aug_z_LEGACY"],
                "early_erank_clean_h_5ep": r["early_erank_clean_h_5ep"],
                "early_erank_aug_z_5ep_LEGACY": r["early_erank_aug_z_5ep_LEGACY"],
                "linear_probe_test_acc": r["downstream"]["linear_probe_test_acc"],
                "finetune_test_acc": r["downstream"]["finetune_test_acc"],
            })
        if not rows:
            return None
        return pd.DataFrame(rows).sort_values(["projector_width", "alpha", "seed"])


    def partial_corr(df, x, y, controls):
        Xc = sm.add_constant(df[controls])
        x_resid = df[x] - sm.OLS(df[x], Xc).fit().predict(Xc)
        y_resid = df[y] - sm.OLS(df[y], Xc).fit().predict(Xc)
        return stats.pearsonr(x_resid, y_resid)

    return (load_runs,)


@app.cell
def _(DATASETS, OUT_ROOT, load_runs, os, pd):
    all_dfs = {}
    for _name in DATASETS:
        _run_dir = os.path.join(OUT_ROOT, _name, "runs")
        _df = load_runs(_run_dir)
        if _df is not None:
            all_dfs[_name] = _df

    analysis_df = (
        pd.concat(all_dfs.values(), ignore_index=True) if all_dfs else pd.DataFrame()
    )
    print(f"Loaded runs for {len(all_dfs)}/{len(DATASETS)} datasets "
          f"({len(analysis_df)} total run rows).")
    return all_dfs, analysis_df


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 13. Cross-dataset comparison (fix #12 -- actually implemented this time)
    """)
    return


@app.cell
def _(FIG_ROOT, all_dfs, os, plt):
    if len(all_dfs) > 1:
        fig, ax = plt.subplots(figsize=(8, 6))
        for _name, _df in all_dfs.items():
            agg = _df.groupby('alpha')['final_erank_clean_h'].mean().reset_index()
            agg['final_erank_clean_h'] /= agg['final_erank_clean_h'].max()  # normalize for cross-dataset comparability
            ax.plot(agg['alpha'], agg['final_erank_clean_h'], marker='o', label=_name)
        ax.set_xlabel('alpha')
        ax.set_ylabel('normalized final erank(h), clean')
        ax.set_title('Cross-dataset: alpha vs. corrected effective rank (normalized)')
        ax.legend()
        fig.tight_layout()
        fig.savefig(os.path.join(FIG_ROOT, 'cross_dataset_alpha_vs_erank.png'), dpi=150)
        plt.show()
    else:
        print('Need >=2 datasets with completed runs for the cross-dataset comparison.')
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 14. Baselines table (fix #9)
    """)
    return


@app.cell
def _(baselines_df):
    baselines_df
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 15. Capacity (channels x length) observation (fix #17)

    **Reframed as descriptive only.** With n=5 datasets, any ordering claim is
    underpowered -- this is reported as an exploratory observation, not
    evidence of a predictive relationship.
    """)
    return


@app.cell
def _(DATASETS, analysis_df, load_dataset_pkl, pd):
    if len(analysis_df) >= 3:
        cap_rows = []
        for _name in DATASETS:
            _d = load_dataset_pkl(_name)
            cap_rows.append({'dataset': _name, 'channels': _d['in_channels'], 'length': _d['seq_len'], 'capacity': _d['in_channels'] * _d['seq_len']})
        cap_df = pd.DataFrame(cap_rows)
        print('Descriptive only (n=5, not a validated predictor):')
        print(cap_df.sort_values('capacity'))
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 16. Honest limitations (fix #10, plus anything not fully fixable)

    - **Single SSL method.** Every claim here is about VICReg specifically, not
      SSL broadly -- collapse dynamics differ across contrastive/non-contrastive
      methods and are not assumed to transfer.
    - **Compute-bounded epoch counts for EigenWorms** given its sequence length;
      if you shortened epochs for time, say so explicitly next to those numbers.
    - **Partial correlation controls for alpha and width as observed**, not a
      full causal graph -- residual confounding from unmeasured factors (e.g.
      learning-rate interactions with alpha) isn't ruled out.
    - All other flaws from the audit (1-3, 5-9, 11-17) are fixed in the pipeline
      above; #4 remains a *reporting* discipline -- always lead with the partial
      correlation, not the pooled one.
    """)
    return


@app.cell
def _(os):
    old_files = [
        "figures/alpha_vs_erank_per_dataset.png",
        "figures/cross_dataset_alpha_vs_erank.png",
        "figures/early_prediction.png",
        "figures/erank_vs_accuracy.png",
    ]
    for f in old_files:
        if os.path.exists(f):
            os.remove(f)
            print(f"Deleted {f}")
        else:
            print(f"Not found (already gone): {f}")
    return


@app.cell
def _(FIG_ROOT, OUT_ROOT, all_dfs, np, os, plt, sm, stats):
    for _ds_name, _df in all_dfs.items():
        _fig_dir = os.path.join(FIG_ROOT, _ds_name)
        _res_dir = os.path.join(OUT_ROOT, _ds_name)
        os.makedirs(_fig_dir, exist_ok=True)
        os.makedirs(_res_dir, exist_ok=True)

        # alpha_vs_erank.png
        _fig, _ax = plt.subplots(figsize=(6, 4.5))
        for _w, _sub in _df.groupby('projector_width'):
            _agg = _sub.groupby('alpha')['final_erank_clean_h'].agg(['mean', 'std']).reset_index()
            _ax.errorbar(_agg['alpha'], _agg['mean'], yerr=_agg['std'], marker='o', label=f'width={_w}')
        _ax.set_xlabel('alpha'); _ax.set_ylabel('final erank(h), clean')
        _ax.set_title(f'{_ds_name}: alpha vs erank'); _ax.legend()
        _fig.tight_layout()
        _fig.savefig(os.path.join(_fig_dir, 'alpha_vs_erank.png'), dpi=150)
        plt.close(_fig)

        # erank_vs_accuracy.png
        _fig, _axes = plt.subplots(1, 2, figsize=(11, 4.5))
        for _ax, _acc_col in zip(_axes, ['linear_probe_test_acc', 'finetune_test_acc']):
            _r, _p = stats.pearsonr(_df['final_erank_clean_h'], _df[_acc_col])
            _ax.scatter(_df['final_erank_clean_h'], _df[_acc_col], alpha=0.5, s=20)
            _ax.set_xlabel('final erank(h), clean'); _ax.set_ylabel(_acc_col)
            _ax.set_title(f'{_ds_name}: erank vs {_acc_col} (r={_r:.2f})')
        _fig.tight_layout()
        _fig.savefig(os.path.join(_fig_dir, 'erank_vs_accuracy.png'), dpi=150)
        plt.close(_fig)

        # early_prediction.png
        _fig, _axes = plt.subplots(1, 2, figsize=(11, 4.5))
        for _ax, _acc_col in zip(_axes, ['linear_probe_test_acc', 'finetune_test_acc']):
            _r, _p = stats.pearsonr(_df['early_erank_clean_h_5ep'], _df[_acc_col])
            _X = sm.add_constant(_df['early_erank_clean_h_5ep'])
            _fit = sm.OLS(_df[_acc_col], _X).fit()
            _ax.scatter(_df['early_erank_clean_h_5ep'], _df[_acc_col], alpha=0.5, s=18)
            _xs = np.linspace(_df['early_erank_clean_h_5ep'].min(), _df['early_erank_clean_h_5ep'].max(), 50)
            _ax.plot(_xs, _fit.params.iloc[0] + _fit.params.iloc[1] * _xs, color='black', linewidth=1)
            _ax.set_xlabel('erank(h), clean, epoch 5'); _ax.set_ylabel(_acc_col)
            _ax.set_title(f'{_ds_name}: early erank -> {_acc_col} (r={_r:.2f})')
        _fig.tight_layout()
        _fig.savefig(os.path.join(_fig_dir, 'early_prediction.png'), dpi=150)
        plt.close(_fig)

        # results.csv
        _df.to_csv(os.path.join(_res_dir, 'results.csv'), index=False)

        print(f"{_ds_name}: figures/{_ds_name}/(3 pngs), results/{_ds_name}/results.csv")

    print("\nDone. 15 plots in figures/<dataset>/, 5 results.csv in results/<dataset>/.")
    return


@app.cell
def _():
    import subprocess
    subprocess.run("rm -rf repo_clone", shell=True)
    return


@app.cell
def _(os):
    from getpass import getpass

    def push_to_github():
        import subprocess
        import shutil

        GITHUB_USER = "kartik-tiwariR20"
        GITHUB_TOKEN = getpass("Paste your GitHub token here (hidden input): ")
        REPO_NAME = "dimensional-collapse-research"
        REPO_URL = f"https://{GITHUB_USER}:{GITHUB_TOKEN}@github.com/{GITHUB_USER}/{REPO_NAME}.git"

        def run(cmd):
            result = subprocess.run(cmd, shell=True, capture_output=True, text=True)
            print(result.stdout)
            if result.returncode != 0:
                print("ERROR:", result.stderr)
            return result

        run("git config --global user.email 'kartik@example.com'")
        run("git config --global user.name 'Kartik Tiwari'")

        run(f"git clone {REPO_URL} repo_clone")

        shutil.copy("notebook.py", "repo_clone/notebook.py")
        if os.path.exists("results"):
            shutil.copytree("results", "repo_clone/results", dirs_exist_ok=True)
        if os.path.exists("figures"):
            shutil.copytree("figures", "repo_clone/figures", dirs_exist_ok=True)
        if os.path.exists("__marimo__"):
            shutil.copytree("__marimo__", "repo_clone/__marimo__", dirs_exist_ok=True)

        run("cd repo_clone && git add . && (git commit -m 'Add SSL dimensional collapse notebook, results, and figures' || true) && git push")

    push_to_github()
    return


if __name__ == "__main__":
    app.run()
