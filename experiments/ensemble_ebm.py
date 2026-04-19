#!/usr/bin/env python
"""
Ensemble EBM: fit, save, compare, and evaluate performance.

Subcommands:
    fit      - Create K EBMs with different neg distances, save to disk
    compare  - Load saved EBMs, compute energy mean/variance at eval points
    evaluate - Generate augmented data via ensemble, measure downstream accuracy

Usage:
    # 1. Fit and save 4 EBMs
    python experiments/ensemble_ebm.py fit \
        --dataset biodeg --target_class 0 --n_real 200 \
        --neg_distances 2 5 10 15 \
        --save_dir experiments/ebms/biodeg_c0

    # 2. Compare energies (mean/variance analysis)
    python experiments/ensemble_ebm.py compare \
        --ebm_dir experiments/ebms/biodeg_c0 --gpu 0

    # 3. Evaluate augmentation performance
    python experiments/ensemble_ebm.py evaluate \
        --ebm_dir experiments/ebms/biodeg_c0 --gpu 0
"""

import argparse
import json
import os
import time
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import balanced_accuracy_score
from sklearn.preprocessing import LabelEncoder, StandardScaler

warnings.filterwarnings("ignore")

DATASET_IDS = {
    # Paper primary 8 (OpenML)
    "protein": 40966, "fourier": 14, "biodeg": 1494, "steel": 1504,
    "stock": 841, "energy": 1472, "collins": 40971, "texture": 40499,
    # Paper extra (run_experiment.py DATASET_REGISTRY)
    "clinical": 43898, "support2": 43897, "mushroom": 24,
    "auction": 43896, "abalone": 183, "statlog": 31,
}


# ===========================================================================
# Shared helpers
# ===========================================================================
def load_and_preprocess(dataset_name, n_real, seed=42, standardize=True, impute=True,
                         encode_categorical=True, return_cat_idx=False):
    """Load dataset, ordinal-encode categoricals, subsample, optionally impute + z-score.

    .. deprecated::
        standardize=True / impute=True (full-data stats, split 전) 는 test leakage 위험.
        Canonical path: standardize=False, impute=False + fit_preprocessor/apply_preprocessor.
    """
    import warnings
    if standardize or impute:
        warnings.warn(
            "load_and_preprocess(standardize=True, impute=True) uses full-data statistics "
            "before splitting → potential test leakage. For paper B.3 protocol, "
            "pass standardize=False, impute=False and use fit_preprocessor + apply_preprocessor "
            "on the per-split training set only.",
            DeprecationWarning, stacklevel=2,
        )
    return _load_and_preprocess_impl(dataset_name, n_real, seed, standardize, impute,
                                       encode_categorical, return_cat_idx)


def _load_and_preprocess_impl(dataset_name, n_real, seed, standardize, impute,
                                encode_categorical, return_cat_idx):
    """Original loading logic (called by load_and_preprocess).

    standardize:
        True  → fit StandardScaler on full (subsampled) X — legacy (test leakage 가능).
        False → raw X. 호출자가 split 후 train-only StandardScaler fit.
    impute:
        True  → mean imputation on full (subsampled) X — legacy.
        False → NaN intact. 호출자가 split 후 train-only imputer fit.
    encode_categorical:
        True  → paper run_experiment.py 처럼 OrdinalEncoder(handle_unknown='use_encoded_value',
                unknown_value=-1) 로 categorical column 을 global 하게 변환.
        False → 모든 column 을 float 으로 강제 cast (구 동작).

    paper B.3 protocol (paper 재현): standardize=False, impute=False, encode_categorical=True.
    호출자는 split 후 fit_preprocessor/apply_preprocessor 로 numeric 만 별도 처리.

    Returns:
        X, y                 if return_cat_idx=False (default)
        X, y, cat_idx        if return_cat_idx=True  (cat_idx: ordinal-encoded 된 column idx 리스트)
    """
    import openml
    ds = openml.datasets.get_dataset(DATASET_IDS[dataset_name])
    X_df, y_raw, cat_indicator, _ = ds.get_data(target=ds.default_target_attribute)

    # cat_indicator: list[bool], length = n_features
    cat_idx = [i for i, is_cat in enumerate(cat_indicator or [])
               if is_cat]

    if encode_categorical and cat_idx:
        from sklearn.preprocessing import OrdinalEncoder
        enc = OrdinalEncoder(handle_unknown='use_encoded_value', unknown_value=-1)
        cat_cols = [X_df.columns[i] for i in cat_idx]
        # object astype keeps NaN behavior for OrdinalEncoder
        X_df[cat_cols] = enc.fit_transform(X_df[cat_cols].astype('object'))

    X = X_df.to_numpy().astype(np.float64) if hasattr(X_df, "to_numpy") else np.array(X_df, dtype=np.float64)
    y = LabelEncoder().fit_transform(y_raw)

    if impute:
        nan_mask = np.isnan(X)
        if nan_mask.any():
            for col in range(X.shape[1]):
                X[nan_mask[:, col], col] = np.nanmean(X[:, col])

    rng = np.random.RandomState(seed)
    if n_real < len(X):
        idx = rng.choice(len(X), n_real, replace=False)
        X, y = X[idx], y[idx]

    y = LabelEncoder().fit_transform(y)
    if standardize:
        scaler = StandardScaler()
        X = scaler.fit_transform(X)

    if return_cat_idx:
        return X, y, cat_idx
    return X, y


def fit_preprocessor(X_train):
    """Fit train-only imputer (column nan-mean) + scaler (column mean/std).

    Paper B.3: preprocessing statistics must come from the training set only.
    Returns dict with imp_mean / scl_mean / scl_std (1-D arrays, length D).
    """
    imp_mean = np.nanmean(X_train, axis=0)
    imp_mean = np.where(np.isnan(imp_mean), 0.0, imp_mean)
    X_imp = np.where(np.isnan(X_train), imp_mean, X_train)
    scl_mean = X_imp.mean(axis=0)
    scl_std = X_imp.std(axis=0)
    scl_std = np.where(scl_std == 0, 1.0, scl_std)
    return dict(
        imp_mean=imp_mean.astype(np.float64),
        scl_mean=scl_mean.astype(np.float64),
        scl_std=scl_std.astype(np.float64),
    )


def apply_preprocessor(X, params):
    """Apply a train-fit preprocessor (from fit_preprocessor) to X."""
    X_imp = np.where(np.isnan(X), params['imp_mean'], X)
    return ((X_imp - params['scl_mean']) / params['scl_std']).astype(np.float64)


def split_preprocessor_from_npz(sp, split_i):
    """Pull per-split preprocessor params out of a splits.npz mapping."""
    return {
        'imp_mean': sp[f'imp_mean_{split_i}'],
        'scl_mean': sp[f'scl_mean_{split_i}'],
        'scl_std':  sp[f'scl_std_{split_i}'],
    }


def rebuild_ebm(ebm_path, gpu=0):
    """Load saved surrogate data and re-fit TabPFN. Returns (tabebm_instance, config)."""
    from tabebm.TabEBM import TabEBM

    data = np.load(ebm_path / "surrogate_data.npz")
    config = json.loads((ebm_path / "config.json").read_text())

    device = f"cuda:{gpu}"
    tabebm = TabEBM(device=device)

    X_ebm = torch.from_numpy(data["X_ebm"]).float().to(device)
    y_ebm = torch.from_numpy(data["y_ebm"]).long().to(device)

    batch = tabebm._prepare_tabpfn_batch_data(X_ebm, y_ebm)
    tabebm.model.fit_from_preprocessed(
        [x.to(device) for x in batch["X_train"]],
        [y.to(device) for y in batch["y_train"]],
        cat_ix=batch["cat_ixs"], configs=batch["confs"],
    )
    return tabebm, config


def evaluate_energy(tabebm, X_eval, gpu=0):
    """
    Compute energy and gradients at X_eval points using a fitted TabEBM.

    TabPFN is an in-context learner: the training data is the "context" and
    eval points are "test" data. We pass X_eval as test data to the already-
    fitted model (which holds surrogate data internally as context).
    """
    from tabebm.TabEBM import TabEBM as TabEBMClass

    device = f"cuda:{gpu}"
    X_t = torch.from_numpy(X_eval).float().to(device)
    # TabPFN forward expects list of 3D tensors: [batch=1, n_samples, n_features]
    X_t_3d = X_t.unsqueeze(0).requires_grad_(True)
    X_list = [X_t_3d]

    logits = tabebm.model.forward(X_list, return_logits=True)
    # logits shape from TabPFN: (n_estimators, n_samples, n_classes) or (n_samples, n_classes)
    # Flatten to (n_samples, n_classes) for energy computation
    if logits.dim() == 3:
        logits = logits.squeeze(0)  # remove estimator dim
    if logits.shape[0] == 2 and logits.shape[1] != 2:
        logits = logits.T  # (n_classes, n_samples) -> (n_samples, n_classes)
    energy = TabEBMClass.compute_energy(logits)

    total_energy = energy.sum() / X_eval.shape[1]  # same normalization as TabEBM.py:493
    total_energy.backward()

    grads = X_t_3d.grad.detach().squeeze(0)  # back to (N, D)
    return (
        energy.detach().cpu().float().numpy().flatten(),
        grads.cpu().float().numpy().reshape(len(X_eval), -1),
    )


# ===========================================================================
# FIT: create and save K EBMs
# ===========================================================================
def cmd_fit(args):
    from tabebm.TabEBM import TabEBM, seed_everything

    save_dir = Path(args.save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    method = args.method
    n_ebms = args.n_ebms

    X, y = load_and_preprocess(args.dataset, args.n_real, args.seed)
    X_class = X[y == args.target_class]

    print(f"\n{'='*60}")
    print(f"FIT: {n_ebms} EBMs for {args.dataset} class={args.target_class}")
    print(f"  Method: {method}")
    if method == "distance":
        print(f"  Distances: {args.neg_distances}")
    elif method == "subsample":
        print(f"  Subsample ratio: {args.subsample_ratio}, neg_distance: {args.neg_distance}")
    print(f"  Class {args.target_class}: {len(X_class)} samples, {X_class.shape[1]} features")
    print(f"  Save to: {save_dir}")
    print(f"{'='*60}\n")

    np.savez(save_dir / "class_data.npz", X_class=X_class, X_all=X, y_all=y)

    neg_distances_used = []

    for i in range(n_ebms):
        seed_everything(args.seed + i)
        rng = np.random.RandomState(args.seed + i)
        ebm_dir = save_dir / f"ebm_{i}"
        ebm_dir.mkdir(exist_ok=True)

        if method == "distance":
            # Each EBM uses all data but different negative distance
            dist = args.neg_distances[i % len(args.neg_distances)]
            X_pos = X_class
            desc = f"distance={dist}, all {len(X_pos)} samples"

        elif method == "subsample":
            # Each EBM uses a random subset of positive data, same neg distance
            dist = args.neg_distance
            n_sub = max(2, int(len(X_class) * args.subsample_ratio))
            sub_idx = rng.choice(len(X_class), n_sub, replace=False)
            X_pos = X_class[sub_idx]
            desc = f"subsample {n_sub}/{len(X_class)} samples, distance={dist}"

        else:
            raise ValueError(f"Unknown method: {method}")

        neg_distances_used.append(dist)

        X_ebm_np, y_ebm_np = TabEBM.add_surrogate_negative_samples(
            X_pos, distance_negative_class=dist
        )

        np.savez(ebm_dir / "surrogate_data.npz", X_ebm=X_ebm_np, y_ebm=y_ebm_np)

        config = {
            "ebm_idx": i,
            "method": method,
            "neg_distance": dist,
            "n_positive": len(X_pos),
            "n_positive_total": len(X_class),
            "subsample_ratio": args.subsample_ratio if method == "subsample" else None,
            "n_negative": len(y_ebm_np) - len(X_pos),
            "n_features": X_class.shape[1],
            "seed": args.seed + i,
            "dataset": args.dataset,
            "target_class": args.target_class,
            "n_real": args.n_real,
        }
        (ebm_dir / "config.json").write_text(json.dumps(config, indent=2))
        print(f"  Saved EBM {i}: {desc}")

    meta = {
        "dataset": args.dataset,
        "target_class": args.target_class,
        "n_real": args.n_real,
        "n_ebms": n_ebms,
        "method": method,
        "neg_distances": neg_distances_used,
        "subsample_ratio": args.subsample_ratio if method == "subsample" else None,
        "seed": args.seed,
        "n_class_samples": len(X_class),
        "n_features": X_class.shape[1],
    }
    (save_dir / "meta.json").write_text(json.dumps(meta, indent=2))
    print(f"\n  Done. {n_ebms} EBMs saved to {save_dir}/")


# ===========================================================================
# COMPARE: load EBMs, evaluate energy, compute mean/variance
# ===========================================================================
def cmd_compare(args):
    ebm_dir = Path(args.ebm_dir)
    meta = json.loads((ebm_dir / "meta.json").read_text())
    K = meta["n_ebms"]

    print(f"\n{'='*60}")
    print(f"COMPARE: {K} EBMs from {ebm_dir}")
    print(f"  Distances: {meta['neg_distances']}")
    print(f"{'='*60}\n")

    # Load class data
    class_data = np.load(ebm_dir / "class_data.npz")
    X_class = class_data["X_class"]

    # Create eval points
    rng = np.random.RandomState(42)
    d = X_class.shape[1]
    X_near = X_class[rng.choice(len(X_class), 200, replace=True)] + rng.randn(200, d) * 0.5
    X_far = X_class.mean(axis=0) + rng.randn(200, d) * 3.0
    X_eval = np.concatenate([X_class, X_near, X_far])
    regions = ["real"] * len(X_class) + ["near"] * 200 + ["far"] * 200

    # Evaluate each EBM
    all_energies = []
    all_grads = []
    energy_rows = []

    BATCH_SIZE = 64  # TabPFN can be unstable with large batches
    for i in range(K):
        path = ebm_dir / f"ebm_{i}"
        tabebm, cfg = rebuild_ebm(path, gpu=args.gpu)

        # Evaluate in batches to avoid TabPFN shape issues
        e_parts, g_parts = [], []
        for start in range(0, len(X_eval), BATCH_SIZE):
            end = min(start + BATCH_SIZE, len(X_eval))
            e_batch, g_batch = evaluate_energy(tabebm, X_eval[start:end], gpu=args.gpu)
            e_parts.append(e_batch)
            g_parts.append(g_batch)
        energies = np.concatenate(e_parts)
        grads = np.concatenate(g_parts)

        all_energies.append(energies)
        all_grads.append(grads)

        dist = cfg["neg_distance"]
        print(f"  EBM {i} (d={dist}): energy [{energies.min():.2f}, {energies.max():.2f}]")

        for j in range(len(X_eval)):
            energy_rows.append({
                "ebm_idx": i, "neg_distance": dist,
                "eval_idx": j, "region": regions[j],
                "energy": round(float(energies[j]), 6),
            })

    # Stack for metrics
    E = np.stack(all_energies)  # (K, N)
    G = np.stack(all_grads)     # (K, N, D)

    # Per-point stats
    e_mean = E.mean(axis=0)
    e_std = E.std(axis=0)
    e_range = E.ptp(axis=0)

    # GAR
    mean_g = G.mean(axis=0)
    mean_g_norm_sq = (mean_g ** 2).sum(axis=-1)
    ind_norm_sq = (G ** 2).sum(axis=-1).mean(axis=0)
    gar = np.where(ind_norm_sq > 1e-12, mean_g_norm_sq / ind_norm_sq, 1.0)

    # Grad variance (VP-SGLD preconditioner)
    grad_var = G.var(axis=0).mean(axis=-1)

    # Save per-EBM energies
    out_dir = ebm_dir / "compare"
    out_dir.mkdir(exist_ok=True)
    pd.DataFrame(energy_rows).to_csv(out_dir / "energy_per_ebm.csv", index=False)

    # Save per-point stats
    stats = pd.DataFrame({
        "eval_idx": range(len(X_eval)),
        "region": regions,
        "energy_mean": np.round(e_mean, 6),
        "energy_std": np.round(e_std, 6),
        "energy_range": np.round(e_range, 6),
        "gar": np.round(gar, 6),
        "grad_var": np.round(grad_var, 6),
    })
    stats.to_csv(out_dir / "ensemble_stats.csv", index=False)

    # Summary by region
    summary = stats.groupby("region")[
        ["energy_std", "energy_range", "gar", "grad_var"]
    ].agg(["mean", "median", "count"]).round(4)
    summary.columns = [f"{col}_{stat}" for col, stat in summary.columns]
    summary = summary.rename(columns={c: c for c in summary.columns})
    summary.to_csv(out_dir / "diversity_summary.csv")

    # Pairwise correlation
    corr_rows = []
    for i in range(K):
        for j in range(i + 1, K):
            r = float(np.corrcoef(all_energies[i], all_energies[j])[0, 1])
            corr_rows.append({
                "ebm_i": i, "dist_i": meta["neg_distances"][i],
                "ebm_j": j, "dist_j": meta["neg_distances"][j],
                "pearson_r": round(r, 4),
            })
    pd.DataFrame(corr_rows).to_csv(out_dir / "pairwise_correlation.csv", index=False)

    # --- Real data vs random points comparison ---
    # Random points: uniform within feature range of full dataset
    X_all = class_data["X_all"]
    rng_rand = np.random.RandomState(123)
    n_rand = 100
    X_rand = np.zeros((n_rand, X_class.shape[1]))
    for col in range(X_class.shape[1]):
        lo, hi = X_all[:, col].min(), X_all[:, col].max()
        X_rand[:, col] = rng_rand.uniform(lo, hi, n_rand)

    rand_energies = []
    for i in range(K):
        path = ebm_dir / f"ebm_{i}"
        tabebm_r, _ = rebuild_ebm(path, gpu=args.gpu)
        e_parts = []
        for s in range(0, n_rand, 64):
            e_b, _ = evaluate_energy(tabebm_r, X_rand[s:s+64], gpu=args.gpu)
            e_parts.append(e_b)
        rand_energies.append(np.concatenate(e_parts))

    RE = np.stack(rand_energies)  # (K, n_rand)

    # Save real data per-EBM energy table
    real_mask = np.array(regions) == "real"
    real_indices = np.where(real_mask)[0]
    real_rows = []
    for j_idx, j in enumerate(real_indices):
        row = {"point_idx": j_idx, "type": "real"}
        for col_name in [f"f{fi}" for fi in range(X_class.shape[1])]:
            row[col_name] = round(float(X_eval[j, int(col_name[1:])]), 4)
        for ki in range(K):
            row[f"ebm_{ki}_d{meta['neg_distances'][ki]}"] = round(float(all_energies[ki][j]), 4)
        e_arr = np.array([all_energies[ki][j] for ki in range(K)])
        row["energy_mean"] = round(float(e_arr.mean()), 4)
        row["energy_var"] = round(float(e_arr.var()), 4)
        real_rows.append(row)

    # Save random points per-EBM energy table
    for j in range(n_rand):
        row = {"point_idx": len(real_indices) + j, "type": "random"}
        for fi in range(X_rand.shape[1]):
            row[f"f{fi}"] = round(float(X_rand[j, fi]), 4)
        for ki in range(K):
            row[f"ebm_{ki}_d{meta['neg_distances'][ki]}"] = round(float(RE[ki, j]), 4)
        e_arr = RE[:, j]
        row["energy_mean"] = round(float(e_arr.mean()), 4)
        row["energy_var"] = round(float(e_arr.var()), 4)
        real_rows.append(row)

    df_points = pd.DataFrame(real_rows)
    df_points.to_csv(out_dir / "ebm_comparison_per_point.csv", index=False)

    # Save variance comparison summary
    real_E = E[:, real_mask]  # (K, n_real)
    real_var_mean = float(real_E.var(axis=0).mean())
    rand_var_mean = float(RE.var(axis=0).mean())
    var_summary = pd.DataFrame([
        {"group": "real_data", "n_points": int(real_mask.sum()),
         "variance_mean": round(real_var_mean, 6),
         "variance_median": round(float(np.median(real_E.var(axis=0))), 6),
         "variance_max": round(float(real_E.var(axis=0).max()), 6),
         "variance_min": round(float(real_E.var(axis=0).min()), 6)},
        {"group": "random_points", "n_points": n_rand,
         "variance_mean": round(rand_var_mean, 6),
         "variance_median": round(float(np.median(RE.var(axis=0))), 6),
         "variance_max": round(float(RE.var(axis=0).max()), 6),
         "variance_min": round(float(RE.var(axis=0).min()), 6)},
    ])
    var_summary.to_csv(out_dir / "variance_real_vs_random.csv", index=False)

    # Print
    print(f"\n{'='*60}")
    print("DIVERSITY SUMMARY (by region)")
    print(f"{'='*60}")
    print(summary.to_string())

    ideal = summary.loc["real", "energy_std_mean"] < summary.loc["far", "energy_std_mean"]  if "energy_std_mean" in summary.columns else False
    print(f"\n  Ideal pattern (real_std < far_std): {'YES' if ideal else 'NO'}")

    print(f"\n  Pairwise correlation:")
    for row in corr_rows:
        print(f"    d={row['dist_i']} vs d={row['dist_j']}: r={row['pearson_r']}")

    print(f"\n  Real vs Random variance:")
    print(f"    real data  variance mean: {real_var_mean:.4f}")
    print(f"    random pts variance mean: {rand_var_mean:.4f}")
    print(f"    ratio (random/real): {rand_var_mean/max(real_var_mean, 1e-9):.1f}x")

    print(f"\n  Saved CSVs:")
    print(f"    {out_dir}/ebm_comparison_per_point.csv  (각 포인트의 x좌표 + 각 EBM energy + mean/var)")
    print(f"    {out_dir}/variance_real_vs_random.csv    (real vs random variance 요약)")
    print(f"    {out_dir}/energy_per_ebm.csv")
    print(f"    {out_dir}/ensemble_stats.csv")
    print(f"    {out_dir}/pairwise_correlation.csv")


# ===========================================================================
# EVALUATE: use ensemble for augmentation, measure downstream performance
# ===========================================================================
def cmd_evaluate(args):
    from tabebm.TabEBM import TabEBM, seed_everything

    ebm_dir = Path(args.ebm_dir)
    meta = json.loads((ebm_dir / "meta.json").read_text())
    K = meta["n_ebms"]

    print(f"\n{'='*60}")
    print(f"EVALUATE: ensemble augmentation performance")
    print(f"  {K} EBMs from {ebm_dir}")
    print(f"{'='*60}\n")

    # Load full data (need train/test split)
    class_data = np.load(ebm_dir / "class_data.npz")
    X_all, y_all = class_data["X_all"], class_data["y_all"]

    # Train/test split
    from sklearn.model_selection import StratifiedShuffleSplit
    n_test = min(len(X_all) // 2, 500)
    results = []

    for split_i in range(args.n_splits):
        seed = args.seed + split_i
        sss = StratifiedShuffleSplit(n_splits=1, test_size=n_test, random_state=seed)
        train_idx, test_idx = next(sss.split(X_all, y_all))
        X_train, y_train = X_all[train_idx], y_all[train_idx]
        X_test, y_test = X_all[test_idx], y_all[test_idx]

        n_classes = len(np.unique(y_train))
        samples_per_class = max(1, args.n_syn // n_classes)

        # --- Method A: Single EBM (baseline TabEBM, distance=5) ---
        seed_everything(seed)
        single_ebm = TabEBM(device=f"cuda:{args.gpu}")
        single_aug = single_ebm.generate(
            X_train, y_train, num_samples=samples_per_class,
            distance_negative_class=5.0, seed=seed,
        )

        # --- Method B: Ensemble mean (average of K EBMs' synthetic data) ---
        # Generate from each EBM separately, then combine
        ensemble_syn_X, ensemble_syn_y = [], []
        for c in range(n_classes):
            X_c = X_train[y_train == c]
            if len(X_c) < 2:
                continue
            per_ebm_samples = max(1, samples_per_class // K)
            class_syns = []
            for ebm_i in range(K):
                seed_everything(seed + ebm_i * 100 + c)
                dist = meta["neg_distances"][ebm_i]
                ebm_inst = TabEBM(device=f"cuda:{args.gpu}")
                y_c = np.zeros(len(X_c), dtype=int)
                aug = ebm_inst.generate(
                    X_c, y_c, num_samples=per_ebm_samples,
                    distance_negative_class=dist, seed=seed + ebm_i,
                )
                if "class_0" in aug:
                    class_syns.append(aug["class_0"])

            if class_syns:
                combined = np.concatenate(class_syns)
                # Take samples_per_class from the combined pool
                if len(combined) > samples_per_class:
                    idx = np.random.RandomState(seed).choice(len(combined), samples_per_class, replace=False)
                    combined = combined[idx]
                ensemble_syn_X.append(combined)
                ensemble_syn_y.append(np.full(len(combined), c))

        # --- Method C: No augmentation (baseline) ---
        # (just train on X_train)

        # --- Evaluate each method ---
        from sklearn.neighbors import KNeighborsClassifier
        from sklearn.ensemble import RandomForestClassifier

        classifiers = {
            "knn": KNeighborsClassifier(n_jobs=-1),
            "rf": RandomForestClassifier(n_jobs=-1),
        }

        for clf_name, clf_factory in classifiers.items():
            # Baseline (no aug)
            clf = KNeighborsClassifier(n_jobs=-1) if clf_name == "knn" else RandomForestClassifier(n_jobs=-1)
            clf.fit(X_train, y_train)
            acc_base = balanced_accuracy_score(y_test, clf.predict(X_test)) * 100

            # Single EBM (d=5)
            X_single = [single_aug[f"class_{c}"] for c in range(n_classes) if f"class_{c}" in single_aug]
            y_single = [np.full(len(single_aug[f"class_{c}"]), c) for c in range(n_classes) if f"class_{c}" in single_aug]
            if X_single:
                X_s = np.concatenate([X_train] + X_single)
                y_s = np.concatenate([y_train] + y_single)
                clf = KNeighborsClassifier(n_jobs=-1) if clf_name == "knn" else RandomForestClassifier(n_jobs=-1)
                clf.fit(X_s, y_s)
                acc_single = balanced_accuracy_score(y_test, clf.predict(X_test)) * 100
            else:
                acc_single = acc_base

            # Ensemble mean
            if ensemble_syn_X:
                X_e = np.concatenate([X_train] + ensemble_syn_X)
                y_e = np.concatenate([y_train] + ensemble_syn_y)
                clf = KNeighborsClassifier(n_jobs=-1) if clf_name == "knn" else RandomForestClassifier(n_jobs=-1)
                clf.fit(X_e, y_e)
                acc_ensemble = balanced_accuracy_score(y_test, clf.predict(X_test)) * 100
            else:
                acc_ensemble = acc_base

            results.append({"split": split_i, "classifier": clf_name,
                            "baseline": round(acc_base, 2),
                            "single_ebm_d5": round(acc_single, 2),
                            "ensemble_mean": round(acc_ensemble, 2)})

            print(f"  split={split_i} {clf_name}: baseline={acc_base:.1f}, "
                  f"single={acc_single:.1f}, ensemble={acc_ensemble:.1f}")

    # Save and summarize
    df = pd.DataFrame(results)
    out_dir = ebm_dir / "evaluate"
    out_dir.mkdir(exist_ok=True)
    df.to_csv(out_dir / "performance.csv", index=False)

    print(f"\n{'='*60}")
    print("PERFORMANCE SUMMARY (mean across splits)")
    print(f"{'='*60}")
    summary = df.groupby("classifier")[["baseline", "single_ebm_d5", "ensemble_mean"]].agg(["mean", "std"])
    print(summary.round(2).to_string())

    # Improvement
    print(f"\n  Improvement over baseline (pp):")
    for clf in df["classifier"].unique():
        sub = df[df["classifier"] == clf]
        single_imp = sub["single_ebm_d5"].mean() - sub["baseline"].mean()
        ens_imp = sub["ensemble_mean"].mean() - sub["baseline"].mean()
        print(f"    {clf}: single={single_imp:+.2f}, ensemble={ens_imp:+.2f}")

    summary.to_csv(out_dir / "performance_summary.csv")
    print(f"\n  Output: {out_dir}/")


# ===========================================================================
# Main
# ===========================================================================
def main():
    parser = argparse.ArgumentParser(
        description="Ensemble EBM: fit, compare, evaluate",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    sub = parser.add_subparsers(dest="cmd", required=True)

    # fit
    p = sub.add_parser("fit", help="Create and save K EBMs")
    p.add_argument("--dataset", required=True, choices=list(DATASET_IDS.keys()))
    p.add_argument("--target_class", type=int, default=0)
    p.add_argument("--n_real", type=int, default=200)
    p.add_argument("--method", choices=["distance", "subsample"], default="distance",
                   help="distance: vary neg distance. subsample: random half of data.")
    p.add_argument("--n_ebms", type=int, default=4, help="Number of EBMs to create")
    # distance method args
    p.add_argument("--neg_distances", nargs="+", type=float, default=[2.0, 5.0, 10.0, 15.0],
                   help="(distance method) Negative distances per EBM")
    # subsample method args
    p.add_argument("--subsample_ratio", type=float, default=0.5,
                   help="(subsample method) Fraction of data each EBM sees")
    p.add_argument("--neg_distance", type=float, default=5.0,
                   help="(subsample method) Fixed negative distance for all EBMs")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--save_dir", required=True)

    # compare
    p = sub.add_parser("compare", help="Load EBMs, compute energy mean/variance")
    p.add_argument("--ebm_dir", required=True)
    p.add_argument("--gpu", type=int, default=0)

    # evaluate
    p = sub.add_parser("evaluate", help="Augment with ensemble, measure performance")
    p.add_argument("--ebm_dir", required=True)
    p.add_argument("--gpu", type=int, default=0)
    p.add_argument("--n_splits", type=int, default=5)
    p.add_argument("--n_syn", type=int, default=500)
    p.add_argument("--seed", type=int, default=42)

    args = parser.parse_args()
    {"fit": cmd_fit, "compare": cmd_compare, "evaluate": cmd_evaluate}[args.cmd](args)


if __name__ == "__main__":
    main()
