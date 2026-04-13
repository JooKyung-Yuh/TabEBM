#!/usr/bin/env python
"""
TabEBM Reproduction Experiment Runner (v3 — parallel)

Key optimizations over v2:
  1. TabEBM class-parallel generation across multiple GPUs
  2. Split-level parallelism via --single_split
  3. XGBoost/RF use all CPU cores (n_jobs=-1)

Usage:
    # Use all 4 GPUs for TabEBM class parallelism
    python run_experiment.py --dataset fourier --n_real 100 --gpus 0 1 2 3

    # Run single split (for external parallelism)
    python run_experiment.py --dataset fourier --n_real 100 --single_split 3 --gpus 0

    # Full help
    python run_experiment.py --help
"""

import argparse
import json
import multiprocessing as mp
import os
import platform
import sys
import time
import traceback
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import balanced_accuracy_score, recall_score
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import LabelEncoder, StandardScaler

warnings.filterwarnings("ignore")

# ---------------------------------------------------------------------------
# Dataset registry
# ---------------------------------------------------------------------------
DATASET_REGISTRY = {
    "protein": {"openml_id": 40966},
    "fourier": {"openml_id": 14},
    "biodeg": {"openml_id": 1494},
    "steel": {"openml_id": 1504},
    "stock": {"openml_id": 841},
    "energy": {"openml_id": 1472},
    "collins": {"openml_id": 40971},
    "texture": {"openml_id": 40499},
    "clinical": {"openml_id": 43898},
    "support2": {"openml_id": 43897},
    "mushroom": {"openml_id": 24},
    "auction": {"openml_id": 43896},
    "abalone": {"openml_id": 183},
    "statlog": {"openml_id": 31},
}

CLASSIFIER_REGISTRY = {
    "lr": "LogisticRegression",
    "knn": "KNeighborsClassifier",
    "mlp": "MLPClassifier",
    "rf": "RandomForestClassifier",
    "xgboost": "XGBClassifier",
    "tabpfn": "TabPFNClassifier",
}

AUGMENT_REGISTRY = {}  # populated after function defs


# ---------------------------------------------------------------------------
# Environment info
# ---------------------------------------------------------------------------
def collect_env_info(gpus: list) -> dict:
    import sklearn
    env = {
        "python": platform.python_version(),
        "numpy": np.__version__,
        "pandas": pd.__version__,
        "sklearn": sklearn.__version__,
        "platform": platform.platform(),
    }
    try:
        import torch
        env["torch"] = torch.__version__
        env["cuda_available"] = torch.cuda.is_available()
        env["gpu_count"] = torch.cuda.device_count()
        if torch.cuda.is_available():
            env["gpus_used"] = gpus
            for g in gpus:
                if g < torch.cuda.device_count():
                    env[f"gpu{g}_name"] = torch.cuda.get_device_name(g)
    except ImportError:
        pass
    try:
        import tabpfn; env["tabpfn"] = tabpfn.__version__
    except (ImportError, AttributeError):
        pass
    try:
        import xgboost; env["xgboost"] = xgboost.__version__
    except ImportError:
        pass
    try:
        import imblearn; env["imbalanced_learn"] = imblearn.__version__
    except ImportError:
        pass
    return env


# ---------------------------------------------------------------------------
# Data loading & preprocessing
# ---------------------------------------------------------------------------
def load_dataset(dataset_name: str, cache_dir: str = None) -> tuple:
    import openml
    if cache_dir:
        openml.config.set_root_cache_directory(cache_dir)

    info = DATASET_REGISTRY[dataset_name]
    dataset = openml.datasets.get_dataset(info["openml_id"])
    X, y, _, _ = dataset.get_data(target=dataset.default_target_attribute)

    meta = {
        "openml_id": info["openml_id"],
        "openml_version": dataset.version,
        "target_attribute": dataset.default_target_attribute,
        "original_n_samples": X.shape[0],
        "n_features": X.shape[1],
    }

    if hasattr(X, "to_numpy"):
        X_df = X.copy()
    else:
        X_df = pd.DataFrame(X)

    cat_cols = X_df.select_dtypes(include=["object", "category"]).columns.tolist()
    if cat_cols:
        from sklearn.preprocessing import OrdinalEncoder
        oe = OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1)
        X_df[cat_cols] = oe.fit_transform(X_df[cat_cols].astype(str))
        meta["n_categorical_features"] = len(cat_cols)
    else:
        meta["n_categorical_features"] = 0

    X = X_df.to_numpy().astype(np.float64)
    le = LabelEncoder()
    y = le.fit_transform(y)
    meta["original_n_classes"] = len(np.unique(y))

    class_counts = np.bincount(y)
    valid_classes = np.where(class_counts >= 10)[0]
    classes_removed = int(len(class_counts) - len(valid_classes))
    if classes_removed > 0:
        mask = np.isin(y, valid_classes)
        X, y = X[mask], y[mask]
        y = LabelEncoder().fit_transform(y)
        print(f"  Removed {classes_removed} classes with <10 samples")
    meta["classes_removed"] = classes_removed

    nan_mask = np.isnan(X)
    n_imputed = int(nan_mask.sum())
    if n_imputed > 0:
        col_means = np.nanmean(X, axis=0)
        for col in range(X.shape[1]):
            X[nan_mask[:, col], col] = col_means[col]
    meta["n_imputed"] = n_imputed
    meta["final_n_samples"] = X.shape[0]
    meta["final_n_classes"] = int(len(np.unique(y)))

    print(f"  Loaded {dataset_name}: {X.shape[0]} samples, {X.shape[1]} features, {meta['final_n_classes']} classes")
    return X, y, meta


def prepare_splits(X, y, n_real, n_splits, base_seed):
    N = X.shape[0]
    n_test = min(N // 2, 500)
    splits = []

    for i in range(n_splits):
        seed = base_seed + i
        sss_test = StratifiedShuffleSplit(n_splits=1, test_size=n_test, random_state=seed)
        pool_idx, test_idx = next(sss_test.split(X, y))
        X_pool, y_pool = X[pool_idx], y[pool_idx]
        X_test, y_test = X[test_idx], y[test_idx]

        n_classes_actual = len(np.unique(y_pool))
        if n_real >= len(X_pool):
            train_idx = np.arange(len(X_pool))
        elif n_real < n_classes_actual:
            train_idx = np.random.RandomState(seed).choice(len(X_pool), size=n_real, replace=False)
        else:
            sss_train = StratifiedShuffleSplit(n_splits=1, train_size=n_real, random_state=seed)
            train_idx, _ = next(sss_train.split(X_pool, y_pool))

        X_train, y_train = X_pool[train_idx], y_pool[train_idx]

        # Re-encode labels to 0..n-1 based on classes present in TRAIN set.
        # This fixes non-contiguous labels when n_real < n_classes (e.g. energy, collins).
        # Test labels not in train set get mapped to -1 and those samples are excluded.
        train_classes = np.unique(y_train)
        class_map = {old: new for new, old in enumerate(train_classes)}
        y_train = np.array([class_map[c] for c in y_train])

        test_mask = np.isin(y_test, train_classes)
        X_test_filtered = X_test[test_mask]
        y_test_filtered = np.array([class_map[c] for c in y_test[test_mask]])

        scaler = StandardScaler()
        X_train = scaler.fit_transform(X_train)
        X_test_filtered = scaler.transform(X_test_filtered)

        n_classes_in_split = len(train_classes)
        train_counts = np.bincount(y_train, minlength=n_classes_in_split).tolist()
        test_counts = np.bincount(y_test_filtered, minlength=n_classes_in_split).tolist()

        splits.append({
            "X_train": X_train, "y_train": y_train,
            "X_test": X_test_filtered, "y_test": y_test_filtered,
            "split_idx": i, "seed": seed,
            "n_train": len(y_train), "n_test": len(y_test_filtered), "n_pool": len(y_pool),
            "n_classes_in_split": n_classes_in_split,
            "n_test_excluded": int((~test_mask).sum()),
            "train_class_counts": train_counts, "test_class_counts": test_counts,
        })
    return splits, n_test


# ---------------------------------------------------------------------------
# Augmentation methods
# ---------------------------------------------------------------------------
def augment_baseline(X_train, y_train, **kwargs):
    return X_train, y_train


def augment_smote(X_train, y_train, n_syn=500, seed=42, **kwargs):
    from imblearn.over_sampling import SMOTE
    n_classes = len(np.unique(y_train))
    class_counts = np.bincount(y_train)
    class_ratios = class_counts / class_counts.sum()
    syn_per_class = np.round(class_ratios * n_syn).astype(int)
    target_counts = {c: int(class_counts[c] + syn_per_class[c]) for c in range(n_classes)}
    min_samples = class_counts.min()
    k_neighbors = min(5, min_samples - 1) if min_samples > 1 else 1
    if k_neighbors < 1:
        return X_train, y_train
    try:
        smote = SMOTE(sampling_strategy=target_counts, k_neighbors=k_neighbors, random_state=seed)
        return smote.fit_resample(X_train, y_train)
    except ValueError:
        return X_train, y_train


# --- TabEBM: multiprocessing worker for class-parallel generation ---
def _tabebm_class_worker(args):
    """Generate synthetic data for ONE class on a specific GPU. Runs in a spawned process."""
    class_idx, X_class, gpu_id, samples_per_class, sgld_steps, sgld_step_size, sgld_noise_std, distance_neg, seed = args
    try:
        import torch
        from tabebm.TabEBM import TabEBM
        device = f"cuda:{gpu_id}" if gpu_id >= 0 else "cpu"
        tabebm_inst = TabEBM(device=device)
        y_dummy = np.zeros(len(X_class), dtype=int)
        result = tabebm_inst.generate(
            X_class, y_dummy,
            num_samples=samples_per_class,
            sgld_steps=sgld_steps, sgld_step_size=sgld_step_size,
            sgld_noise_std=sgld_noise_std, distance_negative_class=distance_neg,
            seed=seed + class_idx,
        )
        return class_idx, result["class_0"], None
    except Exception as e:
        import traceback as tb
        return class_idx, None, f"class {class_idx} on gpu {gpu_id}: {tb.format_exc()}"


def augment_tabebm(X_train, y_train, n_syn=500, seed=42, device="cuda:0",
                   gpus=None, sgld_steps=200, sgld_step_size=0.1,
                   sgld_noise_std=0.01, distance_negative_class=5.0, **kwargs):
    """TabEBM augmentation. Uses multi-GPU class parallelism when len(gpus) > 1."""
    n_classes = len(np.unique(y_train))
    samples_per_class = max(1, n_syn // n_classes)

    if gpus is None:
        gpus = [0]

    use_parallel = len(gpus) > 1 and n_classes > 1

    if use_parallel:
        # --- Multi-GPU class-parallel generation ---
        worker_args = []
        for c in range(n_classes):
            X_c = X_train[y_train == c].copy()
            gpu_id = gpus[c % len(gpus)]
            worker_args.append((
                c, X_c, gpu_id, samples_per_class,
                sgld_steps, sgld_step_size, sgld_noise_std,
                distance_negative_class, seed
            ))

        n_workers = min(len(gpus), n_classes)
        ctx = mp.get_context("spawn")
        with ctx.Pool(processes=n_workers) as pool:
            results = pool.map(_tabebm_class_worker, worker_args)

        X_syn_list, y_syn_list = [], []
        failed_classes = []
        for class_idx, X_syn_c, err_msg in sorted(results, key=lambda x: x[0]):
            if X_syn_c is not None:
                X_syn_list.append(X_syn_c)
                y_syn_list.append(np.full(len(X_syn_c), class_idx))
            else:
                failed_classes.append((class_idx, err_msg))
        if failed_classes:
            # Hard fail: do not return partial synthetic data (reproducibility risk)
            n_ok = len(X_syn_list)
            n_fail = len(failed_classes)
            msg = f"TabEBM class-parallel: {n_fail}/{n_ok+n_fail} classes failed. "
            msg += "Failures: " + "; ".join(
                f"class {ci}: {em.splitlines()[-1] if em else 'unknown'}"
                for ci, em in failed_classes
            )
            raise RuntimeError(msg)

    else:
        # --- Single-GPU sequential generation (original) ---
        from tabebm.TabEBM import TabEBM
        tabebm_inst = TabEBM(device=device)
        augmented = tabebm_inst.generate(
            X_train, y_train,
            num_samples=samples_per_class,
            sgld_steps=sgld_steps, sgld_step_size=sgld_step_size,
            sgld_noise_std=sgld_noise_std,
            distance_negative_class=distance_negative_class, seed=seed,
        )
        X_syn_list, y_syn_list = [], []
        for c in range(n_classes):
            key = f"class_{c}"
            if key in augmented:
                X_syn_list.append(augmented[key])
                y_syn_list.append(np.full(len(augmented[key]), c))

    if X_syn_list:
        X_syn = np.concatenate(X_syn_list)
        y_syn = np.concatenate(y_syn_list)
        return np.concatenate([X_train, X_syn]), np.concatenate([y_train, y_syn])
    return X_train, y_train


def augment_tvae(X_train, y_train, n_syn=500, seed=42, **kwargs):
    from ctgan import TVAE as TVAEModel
    n_classes = len(np.unique(y_train))
    class_counts = np.bincount(y_train)
    class_ratios = class_counts / class_counts.sum()
    syn_per_class = np.maximum(np.round(class_ratios * n_syn).astype(int), 1)
    columns = [f"f{i}" for i in range(X_train.shape[1])]
    X_syn_list, y_syn_list = [], []
    for c in range(n_classes):
        X_c = X_train[y_train == c]
        if len(X_c) < 2:
            continue
        try:
            model = TVAEModel(epochs=300, batch_size=min(500, len(X_c)))
            model.fit(pd.DataFrame(X_c, columns=columns))
            syn_df = model.sample(int(syn_per_class[c]))
            X_syn_list.append(syn_df.to_numpy())
            y_syn_list.append(np.full(int(syn_per_class[c]), c))
        except Exception:
            continue
    if X_syn_list:
        return np.concatenate([X_train] + X_syn_list), np.concatenate([y_train] + y_syn_list)
    return X_train, y_train


def augment_ctgan(X_train, y_train, n_syn=500, seed=42, **kwargs):
    from ctgan import CTGAN as CTGANModel
    n_classes = len(np.unique(y_train))
    class_counts = np.bincount(y_train)
    class_ratios = class_counts / class_counts.sum()
    syn_per_class = np.maximum(np.round(class_ratios * n_syn).astype(int), 1)
    columns = [f"f{i}" for i in range(X_train.shape[1])]
    X_syn_list, y_syn_list = [], []
    for c in range(n_classes):
        X_c = X_train[y_train == c]
        if len(X_c) < 2:
            continue
        try:
            model = CTGANModel(epochs=300, batch_size=min(500, len(X_c)))
            model.fit(pd.DataFrame(X_c, columns=columns))
            syn_df = model.sample(int(syn_per_class[c]))
            X_syn_list.append(syn_df.to_numpy())
            y_syn_list.append(np.full(int(syn_per_class[c]), c))
        except Exception:
            continue
    if X_syn_list:
        return np.concatenate([X_train] + X_syn_list), np.concatenate([y_train] + y_syn_list)
    return X_train, y_train


AUGMENT_REGISTRY = {
    "baseline": augment_baseline,
    "smote": augment_smote,
    "tabebm": augment_tabebm,
    "tvae": augment_tvae,
    "ctgan": augment_ctgan,
}


# ---------------------------------------------------------------------------
# Downstream classifiers (n_jobs=-1 for CPU-parallel)
# ---------------------------------------------------------------------------
def get_classifier(name: str, device: str = "cuda:0", cpu_jobs: int = -1):
    if name == "lr":
        return LogisticRegression(max_iter=1000, n_jobs=cpu_jobs)
    elif name == "knn":
        return KNeighborsClassifier(n_jobs=cpu_jobs)
    elif name == "mlp":
        return MLPClassifier(max_iter=500)
    elif name == "rf":
        return RandomForestClassifier(n_jobs=cpu_jobs)
    elif name == "xgboost":
        import xgboost as xgb
        return xgb.XGBClassifier(
            eval_metric="mlogloss", verbosity=0,
            use_label_encoder=False, n_jobs=cpu_jobs,
        )
    elif name == "tabpfn":
        from tabpfn import TabPFNClassifier
        from tabpfn.config import ModelInterfaceConfig, PreprocessorConfig
        no_preprocess = ModelInterfaceConfig(
            FINGERPRINT_FEATURE=False, FEATURE_SHIFT_METHOD=None,
            CLASS_SHIFT_METHOD=None,
            PREPROCESS_TRANSFORMS=[PreprocessorConfig(name="none")],
        )
        # TabPFN supports max 10 classes. Caller should check n_classes before using.
        return TabPFNClassifier(n_estimators=1, device=device, inference_config=no_preprocess)
    else:
        raise ValueError(f"Unknown classifier: {name}")


# ---------------------------------------------------------------------------
# Fidelity metrics
# ---------------------------------------------------------------------------
def compute_fidelity(X_real, X_syn):
    from scipy.stats import ks_2samp
    if X_syn.shape[0] == 0:
        return {"ks_median_pvalue": None, "ks_mean_pvalue": None}
    p_values = [ks_2samp(X_real[:, c], X_syn[:, c])[1] for c in range(X_real.shape[1])]
    return {
        "ks_median_pvalue": round(float(np.median(p_values)), 6),
        "ks_mean_pvalue": round(float(np.mean(p_values)), 6),
    }


# ---------------------------------------------------------------------------
# Main experiment
# ---------------------------------------------------------------------------
def run_experiment(args):
    run_start = time.time()
    gpus = args.gpus
    primary_device = f"cuda:{gpus[0]}" if gpus[0] >= 0 else "cpu"
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    err_suffix = f"_s{args.single_split}" if args.single_split is not None else ""
    err_path = output_dir / f"{args.dataset}_n{args.n_real}{err_suffix}.err"

    print(f"\n{'='*60}")
    print(f"Dataset: {args.dataset}, N_real: {args.n_real}, GPUs: {gpus}")
    print(f"Methods: {args.methods}, Classifiers: {args.classifiers}")
    split_desc = f"split={args.single_split}" if args.single_split is not None else f"{args.n_splits} splits"
    print(f"Splits: {split_desc}, N_syn: {args.n_syn}, Seed: {args.seed}")
    print(f"{'='*60}\n")

    env_info = collect_env_info(gpus)
    print(f"[env] tabpfn={env_info.get('tabpfn','?')}, torch={env_info.get('torch','?')}, "
          f"gpus={gpus}, cpu_jobs={args.cpu_jobs}")

    print("\n[1/4] Loading dataset...")
    X, y, dataset_meta = load_dataset(args.dataset, cache_dir=args.cache_dir)

    print(f"[2/4] Preparing splits...")
    all_splits, n_test = prepare_splits(X, y, args.n_real, args.n_splits, args.seed)

    # Filter to single split if requested
    if args.single_split is not None:
        all_splits = [s for s in all_splits if s["split_idx"] == args.single_split]
        if not all_splits:
            print(f"  Split {args.single_split} not found, skipping.")
            return None

    print(f"[3/4] Running experiments ({len(all_splits)} splits)...")
    results = []
    errors = []
    total_runs = len(all_splits) * len(args.methods) * len(args.classifiers)
    run_count = 0

    for split in all_splits:
        si = split["split_idx"]
        split_seed = split["seed"]

        for method_name in args.methods:
            aug_fn = AUGMENT_REGISTRY[method_name]
            aug_kwargs = {
                "n_syn": args.n_syn, "seed": split_seed,
                "device": primary_device, "gpus": gpus,
                "sgld_steps": args.sgld_steps, "sgld_step_size": args.sgld_step_size,
                "sgld_noise_std": args.sgld_noise_std,
                "distance_negative_class": args.distance_negative_class,
            }

            t0 = time.time()
            try:
                X_aug, y_aug = aug_fn(split["X_train"].copy(), split["y_train"].copy(), **aug_kwargs)
                aug_status = "ok"
            except Exception as e:
                X_aug, y_aug = split["X_train"].copy(), split["y_train"].copy()
                aug_status = f"aug_error: {type(e).__name__}"
                errors.append(f"[split={si} method={method_name}] AUG ERROR:\n{traceback.format_exc()}")
            aug_time = time.time() - t0

            n_synthetic = len(X_aug) - len(split["X_train"])
            n_classes_split = split.get("n_classes_in_split", len(np.unique(split["y_train"])))
            if n_synthetic > 0:
                syn_y = y_aug[len(split["y_train"]):]
                syn_counts = np.bincount(syn_y.astype(int), minlength=n_classes_split).tolist()
            else:
                syn_counts = [0] * n_classes_split

            fidelity = {"ks_median_pvalue": None, "ks_mean_pvalue": None}
            if n_synthetic > 0 and method_name != "baseline":
                fidelity = compute_fidelity(split["X_train"], X_aug[len(split["X_train"]):])

            for clf_name in args.classifiers:
                run_count += 1
                t1 = time.time()
                acc = np.nan
                per_class_recall = None
                status = aug_status

                # Skip TabPFN for >10 classes (TabPFN v2 limit)
                if clf_name == "tabpfn" and n_classes_split > 10:
                    status = "skipped: tabpfn max 10 classes"
                    eval_time = 0.0
                    result = {
                        "dataset": args.dataset, "n_real": args.n_real,
                        "split": si, "method": method_name, "classifier": clf_name,
                        "balanced_accuracy": None,
                        "split_seed": split_seed,
                        "n_train": split["n_train"], "n_test": split["n_test"], "n_pool": split["n_pool"],
                    "n_classes_in_split": split.get("n_classes_in_split"),
                    "n_test_excluded": split.get("n_test_excluded", 0),
                        "train_class_counts": json.dumps(split["train_class_counts"]),
                        "test_class_counts": json.dumps(split["test_class_counts"]),
                        "n_syn_generated": n_synthetic,
                        "syn_class_counts": json.dumps(syn_counts),
                        "aug_time_sec": round(aug_time, 2),
                        "per_class_recall": None,
                        "ks_median_pvalue": fidelity["ks_median_pvalue"],
                        "ks_mean_pvalue": fidelity["ks_mean_pvalue"],
                        "eval_time_sec": 0.0,
                        "status": status,
                    }
                    results.append(result)
                    print(f"  [{run_count}/{total_runs}] split={si} {method_name:>8s} + {clf_name:>8s} -> SKIP (>{10} classes)")
                    continue

                try:
                    clf = get_classifier(clf_name, device=primary_device, cpu_jobs=args.cpu_jobs)
                    clf.fit(X_aug, y_aug)
                    y_pred = clf.predict(split["X_test"])
                    acc = balanced_accuracy_score(split["y_test"], y_pred) * 100
                    recalls = recall_score(split["y_test"], y_pred, average=None, zero_division=0)
                    per_class_recall = [round(float(r), 4) for r in recalls]
                    if aug_status == "ok":
                        status = "ok"
                except Exception as e:
                    status = f"clf_error: {type(e).__name__}"
                    errors.append(f"[split={si} method={method_name} clf={clf_name}] CLF ERROR:\n{traceback.format_exc()}")

                eval_time = time.time() - t1
                result = {
                    "dataset": args.dataset, "n_real": args.n_real,
                    "split": si, "method": method_name, "classifier": clf_name,
                    "balanced_accuracy": round(acc, 4) if not np.isnan(acc) else None,
                    "split_seed": split_seed,
                    "n_train": split["n_train"], "n_test": split["n_test"], "n_pool": split["n_pool"],
                    "n_classes_in_split": split.get("n_classes_in_split"),
                    "n_test_excluded": split.get("n_test_excluded", 0),
                    "train_class_counts": json.dumps(split["train_class_counts"]),
                    "test_class_counts": json.dumps(split["test_class_counts"]),
                    "n_syn_generated": n_synthetic,
                    "syn_class_counts": json.dumps(syn_counts),
                    "aug_time_sec": round(aug_time, 2),
                    "per_class_recall": json.dumps(per_class_recall) if per_class_recall else None,
                    "ks_median_pvalue": fidelity["ks_median_pvalue"],
                    "ks_mean_pvalue": fidelity["ks_mean_pvalue"],
                    "eval_time_sec": round(eval_time, 2),
                    "status": status,
                }
                results.append(result)

                acc_str = f"{acc:.2f}%" if not np.isnan(acc) else "FAILED"
                print(f"  [{run_count}/{total_runs}] split={si} {method_name:>8s} + {clf_name:>8s} -> {acc_str} ({eval_time:.1f}s)")

    # --- Save results ---
    run_end = time.time()
    print(f"\n[4/4] Saving results...")

    df = pd.DataFrame(results)
    suffix = f"_s{args.single_split}" if args.single_split is not None else ""
    csv_path = output_dir / f"{args.dataset}_n{args.n_real}{suffix}.csv"
    df.to_csv(csv_path, index=False)
    print(f"  Results: {csv_path}")

    if args.single_split is None and len(df) > 0:
        # Filter to valid runs for summary
        df_valid = df[df["balanced_accuracy"].notna() & (df["status"] == "ok")]

        # Summary
        if len(df_valid) > 0:
            summary = df_valid.pivot_table(values="balanced_accuracy", index="method", columns="classifier", aggfunc=["mean", "std"])
            print(f"\n{'='*60}")
            print("SUMMARY")
            print(f"{'='*60}")
            print(summary.round(2).to_string())
            summary.to_csv(output_dir / f"{args.dataset}_n{args.n_real}_summary.csv")

        # Ranks (only valid runs)
        rank_rows = []
        for (si2, clf), group in df_valid.groupby(["split", "classifier"]):
            if len(group) < 2:
                continue
            for rank, (_, row) in enumerate(group.sort_values("balanced_accuracy", ascending=False).iterrows(), 1):
                rank_rows.append({"split": si2, "classifier": clf, "method": row["method"], "rank": rank})
        rank_df = pd.DataFrame(rank_rows)
        rank_pivot = rank_df.pivot_table(values="rank", index="method", columns="classifier", aggfunc="mean")
        rank_pivot["AVG_RANK"] = rank_pivot.mean(axis=1)
        rank_pivot.sort_values("AVG_RANK").to_csv(output_dir / f"{args.dataset}_n{args.n_real}_ranks.csv")

        # Improvement (valid rows only, not oil-tainted by aug_error rows)
        if len(df_valid) > 0:
            mean_acc = df_valid.pivot_table(values="balanced_accuracy", index="method", columns="classifier", aggfunc="mean")
            if "baseline" in mean_acc.index:
                improvement = mean_acc.subtract(mean_acc.loc["baseline"], axis=1)
                improvement["MEAN"] = improvement.mean(axis=1)
                improvement.to_csv(output_dir / f"{args.dataset}_n{args.n_real}_improvement.csv")

    # Config
    config = {
        "args": {k: v for k, v in vars(args).items()},
        "dataset_meta": dataset_meta,
        "split_protocol": {"n_test": n_test, "base_seed": args.seed, "normalization": "z-score"},
        "parallelism": {
            "tabebm_class_parallel": len(gpus) > 1,
            "n_gpus": len(gpus), "gpus": gpus,
            "cpu_jobs": args.cpu_jobs,
            "mode": "class_parallel" if len(gpus) > 1 else "single_gpu",
            "single_split": args.single_split,
        },
        "environment": env_info,
        "timing": {
            "start_time": time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(run_start)),
            "end_time": time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(run_end)),
            "total_time_sec": round(run_end - run_start, 1),
        },
    }
    config_path = output_dir / f"{args.dataset}_n{args.n_real}{suffix}_config.json"
    with open(config_path, "w") as f:
        json.dump(config, f, indent=2)

    if errors:
        with open(err_path, "a") as f:
            for err in errors:
                f.write(err + "\n" + "-" * 40 + "\n")
        print(f"  Errors ({len(errors)}): {err_path}")

    total_min = (run_end - run_start) / 60
    print(f"\n  Done in {total_min:.1f} min")
    return df


# ---------------------------------------------------------------------------
# Argument parser
# ---------------------------------------------------------------------------
def parse_args():
    parser = argparse.ArgumentParser(
        description="TabEBM Reproduction Experiment Runner (v3 parallel)",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--dataset", type=str, required=True, choices=list(DATASET_REGISTRY.keys()))
    parser.add_argument("--n_real", type=int, default=100)
    parser.add_argument("--n_syn", type=int, default=500)
    parser.add_argument("--n_splits", type=int, default=10)
    parser.add_argument("--single_split", type=int, default=None,
                        help="Run only this split index (for external parallelism)")
    parser.add_argument("--methods", nargs="+", default=["baseline", "smote", "tabebm"],
                        choices=list(AUGMENT_REGISTRY.keys()))
    parser.add_argument("--classifiers", nargs="+", default=["knn", "rf", "tabpfn"],
                        choices=list(CLASSIFIER_REGISTRY.keys()))
    parser.add_argument("--sgld_steps", type=int, default=200)
    parser.add_argument("--sgld_step_size", type=float, default=0.1)
    parser.add_argument("--sgld_noise_std", type=float, default=0.01)
    parser.add_argument("--distance_negative_class", type=float, default=5.0)
    parser.add_argument("--gpus", nargs="+", type=int, default=[0],
                        help="GPU indices for TabEBM class-parallel generation")
    parser.add_argument("--gpu", type=int, default=None,
                        help="(compat) Single GPU index. Overridden by --gpus if both given.")
    parser.add_argument("--cpu_jobs", type=int, default=-1,
                        help="n_jobs for CPU classifiers (LR/KNN/RF/XGBoost). -1=all cores.")
    parser.add_argument("--output_dir", type=str, default="experiments/results")
    parser.add_argument("--cache_dir", type=str, default=None)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    # --gpu (old) -> --gpus (new) compatibility
    if args.gpu is not None and args.gpus == [0]:
        args.gpus = [args.gpu]
    run_experiment(args)
