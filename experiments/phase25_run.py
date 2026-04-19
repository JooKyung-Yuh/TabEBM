#!/usr/bin/env python
"""
Phase 2.5 runner: SGLD from a saved EBM + augmentation evaluation.

Loads one saved ensemble member, runs SGLD with the given hyperparameters,
then evaluates downstream classifiers (knn/lr/rf/xgboost/mlp) with and
without the synthetic samples. Logs per-split rows to CSV and optionally
to Weights & Biases.

Usage:
    python experiments/phase25_run.py \
        --ebm_dir experiments/ebms/stock_distance \
        --ebm_idx 1 --target_class 0 \
        --output_dir experiments/results/phase25
"""
import argparse
import json
import os
import sys
import time
from pathlib import Path


# =============================================================================
# WANDB credential handling — MUST run before TabPFN import.
# TabPFN's pydantic 2 strict TabPFNSettings raises "Extra inputs are not
# permitted" if any WANDB_* env var is present. So we:
#   1) pop WANDB_* from os.environ BEFORE importing anything TabPFN-related
#   2) read .env manually into a module-private _WANDB_KEY
#   3) pass _WANDB_KEY directly to wandb.login(key=...) later
# =============================================================================
REPO_ROOT = Path(__file__).resolve().parent.parent
_WANDB_KEY = None


def _stash_wandb_env():
    """Pop WANDB_* vars from os.environ; remember WANDB_API_KEY privately."""
    global _WANDB_KEY
    for k in list(os.environ.keys()):
        if k.startswith("WANDB_"):
            if k == "WANDB_API_KEY" and _WANDB_KEY is None:
                _WANDB_KEY = os.environ[k]
            del os.environ[k]


def _load_env_file(env_path: Path):
    """Read .env; stash WANDB_API_KEY privately (no os.environ mutation)."""
    global _WANDB_KEY
    if not env_path.exists():
        return
    for raw in env_path.read_text().splitlines():
        line = raw.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        k, v = line.split("=", 1)
        k = k.strip()
        v = v.strip().strip('"').strip("'")
        if k == "WANDB_API_KEY" and _WANDB_KEY is None:
            _WANDB_KEY = v


# Note: file name is .env.local (not .env). TabPFN uses pydantic-settings
# which auto-loads any ".env" in cwd and crashes on WANDB_* keys. Using a
# different filename keeps our secrets readable only by us, not TabPFN.
_stash_wandb_env()
_load_env_file(REPO_ROOT / ".env.local")


# =============================================================================
# Now safe to import TabPFN-dependent and heavy deps
# =============================================================================
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import torch  # noqa: E402
from sklearn.ensemble import RandomForestClassifier  # noqa: E402
from sklearn.linear_model import LogisticRegression  # noqa: E402
from sklearn.metrics import balanced_accuracy_score  # noqa: E402
from sklearn.model_selection import StratifiedShuffleSplit  # noqa: E402
from sklearn.neighbors import KNeighborsClassifier  # noqa: E402
from sklearn.neural_network import MLPClassifier  # noqa: E402

try:
    import xgboost as xgb
    HAS_XGB = True
except ImportError:
    HAS_XGB = False

try:
    import wandb
    HAS_WANDB = True
except ImportError:
    HAS_WANDB = False


sys.path.insert(0, str(REPO_ROOT / "experiments"))
from ensemble_ebm import rebuild_ebm  # noqa: E402


# ============================================================
# Classifier factory
# ============================================================
def get_classifier(name, seed=0):
    if name == "knn":
        return KNeighborsClassifier(n_jobs=-1)
    if name == "lr":
        return LogisticRegression(max_iter=1000, n_jobs=-1, random_state=seed)
    if name == "rf":
        return RandomForestClassifier(n_jobs=-1, random_state=seed)
    if name == "xgboost":
        if not HAS_XGB:
            raise ImportError("xgboost not installed")
        return xgb.XGBClassifier(
            n_jobs=-1, eval_metric="logloss",
            use_label_encoder=False, random_state=seed,
        )
    if name == "mlp":
        return MLPClassifier(max_iter=300, random_state=seed)
    raise ValueError(f"Unknown classifier: {name}")


# ============================================================
# SGLD from a saved EBM (reuses rebuild_ebm + TabEBM internals)
# ============================================================
def generate_from_saved_ebm(
    ebm_member_dir,
    num_samples,
    sgld_step_size,
    sgld_noise_std,
    sgld_steps,
    starting_point_noise_std,
    seed,
    gpu=0,
):
    """Rebuild TabPFN on the saved surrogate + run SGLD chains."""
    from tabebm.TabEBM import seed_everything

    tabebm, cfg = rebuild_ebm(Path(ebm_member_dir), gpu=gpu)

    data = np.load(Path(ebm_member_dir) / "surrogate_data.npz")
    X_ebm = torch.from_numpy(data["X_ebm"]).float().to(tabebm.device)
    y_ebm = torch.from_numpy(data["y_ebm"]).long().to(tabebm.device)

    seed_everything(seed)
    start_dict = tabebm._initialize_sgld_starting_points(
        X_ebm, y_ebm, num_samples, starting_point_noise_std, seed
    )
    X_start = start_dict["X_start"]
    y_start = start_dict["y_start"]

    batch_dict = tabebm._prepare_tabpfn_batch_data(X_start, y_start)
    X_sgld_tensor = batch_dict["X_train"][0].to(tabebm.device).requires_grad_(True)

    noise_shape = (sgld_steps, num_samples, X_start.shape[1])
    noise_tensor = torch.randn(noise_shape, device=tabebm.device, dtype=X_sgld_tensor.dtype)

    X_sgld_final = tabebm._perform_sgld_sampling(
        X_sgld_tensor, noise_tensor, sgld_step_size, sgld_noise_std, sgld_steps, debug=False
    )
    return X_sgld_final.detach().cpu().squeeze(0).numpy(), cfg


# ============================================================
# Downstream evaluation
# ============================================================
def evaluate_one(X_train, y_train, X_test, y_test, clf_name, seed=0):
    clf = get_classifier(clf_name, seed=seed)
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)
    return balanced_accuracy_score(y_test, y_pred) * 100.0


# ============================================================
# Main
# ============================================================
def main():
    parser = argparse.ArgumentParser(
        description="Phase 2.5: SGLD from saved EBM + augmentation eval",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--ebm_dir", required=True,
                        help="Saved ensemble dir (contains ebm_i/ subdirs)")
    parser.add_argument("--ebm_idx", type=int, required=True,
                        help="Which member to use (0..K-1)")
    parser.add_argument("--target_class", type=int, required=True,
                        help="Class label that synthetic samples take")
    parser.add_argument("--num_samples", type=int, default=500)
    parser.add_argument("--sgld_step_size", type=float, default=0.1)
    parser.add_argument("--sgld_noise_std", type=float, default=0.01)
    parser.add_argument("--sgld_steps", type=int, default=200)
    parser.add_argument("--starting_point_noise_std", type=float, default=0.01)
    parser.add_argument("--classifiers", nargs="+",
                        default=["knn", "lr", "rf", "xgboost", "mlp"])
    parser.add_argument("--n_splits", type=int, default=10)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--output_dir", default="experiments/results/phase25")
    parser.add_argument("--wandb_project", default="tabebm-ensemble-phase25")
    parser.add_argument("--wandb_tags", nargs="*", default=[])
    parser.add_argument("--wandb_group", default=None)
    parser.add_argument("--no_wandb", action="store_true")
    parser.add_argument("--run_tag", default=None,
                        help="Optional tag appended to output filename")
    args = parser.parse_args()

    ebm_dir = Path(args.ebm_dir)
    ebm_member_dir = ebm_dir / f"ebm_{args.ebm_idx}"
    meta = json.loads((ebm_dir / "meta.json").read_text())
    method = meta["method"]
    neg_dist = meta["neg_distances"][args.ebm_idx]
    dataset_name = meta.get("dataset", "unknown")
    n_real = meta.get("n_real", -1)
    ebm_target_class = meta.get("target_class", -1)

    # sanity: the EBM we load was trained for a specific class;
    # args.target_class must match what we want to augment.
    if ebm_target_class != args.target_class:
        print(f"[warn] ebm meta target_class={ebm_target_class} but "
              f"--target_class={args.target_class}. Labels will use --target_class.")

    config_dict = {
        "dataset": dataset_name,
        "n_real": n_real,
        "method": method,
        "ebm_idx": args.ebm_idx,
        "neg_distance": neg_dist,
        "target_class": args.target_class,
        "num_samples": args.num_samples,
        "sgld_step_size": args.sgld_step_size,
        "sgld_noise_std": args.sgld_noise_std,
        "sgld_steps": args.sgld_steps,
        "starting_point_noise_std": args.starting_point_noise_std,
        "seed": args.seed,
        "n_splits": args.n_splits,
    }

    run_name = f"{method}_ebm{args.ebm_idx}_c{args.target_class}"
    if args.run_tag:
        run_name = f"{run_name}_{args.run_tag}"
    print(f"\n=== Phase 2.5 run: {run_name} ===")
    for k, v in config_dict.items():
        print(f"  {k}: {v}")

    Path(args.output_dir).mkdir(parents=True, exist_ok=True)

    # ---------- wandb init ----------
    use_wandb = HAS_WANDB and not args.no_wandb
    wandb_run = None
    if use_wandb:
        try:
            if _WANDB_KEY:
                # Pass key explicitly; do NOT pollute os.environ.
                wandb.login(key=_WANDB_KEY, verify=False)
            wandb_run = wandb.init(
                project=args.wandb_project,
                name=run_name,
                group=args.wandb_group,
                tags=args.wandb_tags,
                config=config_dict,
                reinit=True,
            )
        except Exception as e:
            print(f"[warn] wandb.init failed: {e}. Continuing without wandb.")
            use_wandb = False

    # ---------- 1) Generate synthetic ----------
    print(f"\n[1/3] Generating {args.num_samples} synthetic samples from saved EBM...")
    t0 = time.time()
    synthetic, ebm_cfg = generate_from_saved_ebm(
        ebm_member_dir,
        num_samples=args.num_samples,
        sgld_step_size=args.sgld_step_size,
        sgld_noise_std=args.sgld_noise_std,
        sgld_steps=args.sgld_steps,
        starting_point_noise_std=args.starting_point_noise_std,
        seed=args.seed,
        gpu=args.gpu,
    )
    gen_time = time.time() - t0
    print(f"  synthetic shape: {synthetic.shape}  time: {gen_time:.1f}s")

    # ---------- 2) Load original data, run splits ----------
    class_data = np.load(ebm_dir / "class_data.npz")
    X_all = class_data["X_all"]
    y_all = class_data["y_all"]

    n_classes_all = len(np.unique(y_all))
    print(f"\n  X_all shape: {X_all.shape}  n_classes: {n_classes_all}  "
          f"class_counts: {np.bincount(y_all).tolist()}")

    n_test = min(len(X_all) // 2, 500)
    results = []

    print(f"\n[2/3] Running {args.n_splits} splits × {len(args.classifiers)} classifiers...")
    for split_i in range(args.n_splits):
        sss = StratifiedShuffleSplit(
            n_splits=1, test_size=n_test, random_state=args.seed + split_i
        )
        train_idx, test_idx = next(sss.split(X_all, y_all))
        X_train, y_train = X_all[train_idx], y_all[train_idx]
        X_test, y_test = X_all[test_idx], y_all[test_idx]

        # Augmented: all synthetic labeled as target_class
        X_train_aug = np.concatenate([X_train, synthetic])
        y_train_aug = np.concatenate(
            [y_train, np.full(len(synthetic), args.target_class, dtype=y_all.dtype)]
        )

        for clf_name in args.classifiers:
            try:
                acc_base = evaluate_one(X_train, y_train, X_test, y_test,
                                         clf_name, seed=args.seed + split_i)
                acc_aug = evaluate_one(X_train_aug, y_train_aug, X_test, y_test,
                                        clf_name, seed=args.seed + split_i)
                delta = acc_aug - acc_base
                status = "ok"
                err = ""
            except Exception as e:
                acc_base = float("nan")
                acc_aug = float("nan")
                delta = float("nan")
                status = "error"
                err = str(e)[:200]

            row = {
                **config_dict,
                "split": split_i,
                "classifier": clf_name,
                "bal_acc_baseline": round(acc_base, 3) if status == "ok" else acc_base,
                "bal_acc_augmented": round(acc_aug, 3) if status == "ok" else acc_aug,
                "delta_pp": round(delta, 3) if status == "ok" else delta,
                "status": status,
                "error": err,
            }
            results.append(row)

            print(f"  split={split_i} {clf_name:8s}  base={acc_base:.2f}  "
                  f"aug={acc_aug:.2f}  Δ={delta:+.2f}  [{status}]")

    # ---------- 3) Save CSV + wandb summary ----------
    df = pd.DataFrame(results)
    csv_name = (
        f"{run_name}_ss{args.sgld_step_size}_ns{args.sgld_noise_std}"
        f"_T{args.sgld_steps}_sp{args.starting_point_noise_std}"
        f"_N{args.num_samples}.csv"
    )
    csv_path = Path(args.output_dir) / csv_name
    df.to_csv(csv_path, index=False)
    print(f"\n[3/3] Saved {csv_path}  ({len(df)} rows)")

    df_ok = df[df["status"] == "ok"]
    if len(df_ok) > 0:
        summary = df_ok.groupby("classifier")[
            ["bal_acc_baseline", "bal_acc_augmented", "delta_pp"]
        ].agg(["mean", "std"]).round(2)
        print("\n=== Summary (mean ± std across splits) ===")
        print(summary.to_string())

        if use_wandb:
            # 1) Raw rows as a single W&B table.
            wandb.log({"results": wandb.Table(dataframe=df_ok)})

            # 2) Run-level summary scalars (one value per run) so dashboards can
            #    filter / sort / scatter runs at a glance.
            wandb.run.summary["gen_time_sec"] = round(gen_time, 2)
            wandb.run.summary["n_ok_rows"] = int(len(df_ok))
            wandb.run.summary["n_error_rows"] = int((df["status"] == "error").sum())

            wandb.run.summary["mean_bal_acc_baseline_all"] = round(
                float(df_ok["bal_acc_baseline"].mean()), 3)
            wandb.run.summary["mean_bal_acc_augmented_all"] = round(
                float(df_ok["bal_acc_augmented"].mean()), 3)
            wandb.run.summary["mean_delta_pp_all"] = round(
                float(df_ok["delta_pp"].mean()), 3)
            wandb.run.summary["std_delta_pp_all"] = round(
                float(df_ok["delta_pp"].std()), 3)
            wandb.run.summary["best_delta_pp"] = round(
                float(df_ok["delta_pp"].max()), 3)
            wandb.run.summary["worst_delta_pp"] = round(
                float(df_ok["delta_pp"].min()), 3)

            # Per-classifier scalars
            for clf_name in df_ok["classifier"].unique():
                sub = df_ok[df_ok["classifier"] == clf_name]
                wandb.run.summary[f"{clf_name}_delta_pp_mean"] = round(
                    float(sub["delta_pp"].mean()), 3)
                wandb.run.summary[f"{clf_name}_delta_pp_std"] = round(
                    float(sub["delta_pp"].std()), 3)
                wandb.run.summary[f"{clf_name}_bal_acc_augmented_mean"] = round(
                    float(sub["bal_acc_augmented"].mean()), 3)

    if use_wandb:
        wandb.finish()


if __name__ == "__main__":
    main()
