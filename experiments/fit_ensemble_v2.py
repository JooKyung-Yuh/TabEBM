#!/usr/bin/env python
"""Fit a TabEBM ensemble using one or more randomization methods.

Unlike the older `ensemble_ebm.py fit` which supports a single method
(distance OR subsample) with fixed parameters, this script:
- Accepts any combination of methods registered in `ensemble_methods.py`
  (Subsample, Distance, CornerNoise, NumFakeCorners, ...).
- Randomizes each active method's parameter per member (seeded).
- Saves per-member `config.json` containing all resolved values so the
  ensemble is bit-exact reproducible.
- Fits both class 0 and class 1 ensembles under a single timestamped root:
      experiments/ebms/{YYYYMMDD_HHMMSS}_{methods-joined}_EBM/
        ├── meta.json                  # run-level: methods, K, seed, classes
        ├── c0/                        # class-0 ensemble (standard layout)
        │   ├── meta.json
        │   ├── class_data.npz
        │   └── ebm_{0..K-1}/{config.json, surrogate_data.npz}
        └── c1/                        # class-1 ensemble

Usage:
    python experiments/fit_ensemble_v2.py \
        --dataset stock --n_real 100 --n_ebms 10 \
        --methods CornerNoise Distance NumFakeCorners \
        --seed 42
"""
from __future__ import annotations

import argparse
import datetime as dt
import json
import sys
import time
from pathlib import Path

import numpy as np

# Local imports (experiments/ must be on sys.path or run from repo root)
sys.path.insert(0, str(Path(__file__).parent))
from ensemble_ebm import load_and_preprocess  # dataset loader reused
from ensemble_methods import METHODS, build_surrogate_data, sample_member_configs


def fit_class_ensemble(
    class_dir: Path,
    X_class: np.ndarray,
    X_all: np.ndarray,
    y_all: np.ndarray,
    *,
    methods: list[str],
    K: int,
    seed: int,
    method_params: dict | None,
    dataset: str,
    target_class: int,
    n_real: int,
    shared_corners: bool = True,
) -> None:
    class_dir.mkdir(parents=True, exist_ok=True)
    np.savez(class_dir / "class_data.npz", X_class=X_class, X_all=X_all, y_all=y_all)

    configs = sample_member_configs(methods, K=K, seed=seed, X_class=X_class,
                                     method_params=method_params,
                                     shared_corners=shared_corners)

    print(f"  [class {target_class}]  K={K}  d={X_class.shape[1]}  n_pos(full)={len(X_class)}")
    all_negs = []          # 멤버별 negative 누적 (union 저장용)
    all_alphas = []        # 각 member 의 α (또는 None)
    for k, cfg in enumerate(configs):
        ebm_dir = class_dir / f"ebm_{k}"
        ebm_dir.mkdir(exist_ok=True)
        X_ebm, y_ebm = build_surrogate_data(X_class, cfg)
        np.savez(ebm_dir / "surrogate_data.npz", X_ebm=X_ebm, y_ebm=y_ebm)

        # --- 명시적 negatives 파일 (재탐색 편의) ---
        X_neg = X_ebm[y_ebm == 1]
        alpha = cfg.get("method_distance", {}).get("neg_distance")
        noise_std = cfg.get("method_corner_noise", {}).get("noise_std", 0.0)
        n_corners_cfg = cfg.get("method_num_fake_corners", {}).get("n_corners")
        np.savez(
            ebm_dir / "negatives.npz",
            X_neg=X_neg.astype(np.float64),
            alpha=np.array(alpha if alpha is not None else float("nan")),
            noise_std=np.array(float(noise_std)),
            n_corners=np.array(int(n_corners_cfg) if n_corners_cfg is not None else len(X_neg)),
            member_idx=np.array(int(cfg["member_idx"])),
            corner_seed=np.array(int(cfg.get("corner_seed", cfg["seed"]))),
        )
        all_negs.append(X_neg)
        all_alphas.append(alpha)

        (ebm_dir / "config.json").write_text(json.dumps(cfg, indent=2))
        n_pos = int((y_ebm == 0).sum())
        n_neg = int((y_ebm == 1).sum())
        noise = cfg.get("method_corner_noise", {}).get("noise_std")
        nc = cfg.get("method_num_fake_corners", {}).get("n_corners")
        bits = []
        if alpha is not None: bits.append(f"d={alpha:.2f}")
        if noise is not None: bits.append(f"noise={noise:.2f}")
        if nc is not None: bits.append(f"corners={nc}")
        if "method_subsample" in cfg:
            bits.append(f"sub={cfg['method_subsample']['ratio']:.2f}")
        tag = ", ".join(bits) if bits else "default"
        print(f"    ebm_{k:<2d}  pos={n_pos:<4d} neg={n_neg:<3d}  [{tag}]")

    # --- class 레벨 union 저장 (빠른 로드용) ---
    X_neg_union = np.vstack(all_negs)
    member_of = np.concatenate([np.full(len(X), i, dtype=np.int64) for i, X in enumerate(all_negs)])
    alphas_arr = np.array([a if a is not None else float("nan") for a in all_alphas], dtype=np.float64)
    np.savez(
        class_dir / "negatives_union.npz",
        X_neg=X_neg_union,
        member_idx=member_of,
        alpha_per_member=alphas_arr,
        K=np.array(K),
    )

    meta = {
        "dataset": dataset,
        "target_class": target_class,
        "n_real": n_real,
        "n_ebms": K,
        "method": "+".join(methods) if methods else "default",
        "methods_list": methods,
        "seed": seed,
        "n_class_samples": int(len(X_class)),
        "n_features": int(X_class.shape[1]),
    }
    (class_dir / "meta.json").write_text(json.dumps(meta, indent=2))


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                  formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dataset", default="stock")
    ap.add_argument("--n_real", type=int, default=100)
    ap.add_argument("--n_ebms", "-K", type=int, required=True)
    ap.add_argument("--methods", nargs="+", required=True,
                    metavar="METHOD",
                    help=f"Active methods. Available: {list(METHODS)}")
    ap.add_argument("--classes", type=int, nargs="+", default=[0, 1],
                    help="Target classes to fit. Default: 0 1")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--save_root", default="experiments/ebms")
    ap.add_argument("--run_name", default=None,
                    help="Override auto-generated run dir name (default: "
                         "{YYYYMMDD_HHMMSS}_{methods}_EBM)")
    ap.add_argument("--method_params", default=None,
                    help="JSON override for per-method sampler kwargs. "
                         "E.g. '{\"Distance\": {\"dist_range\": [2, 20]}}'")
    ap.add_argument("--shared-corners", dest="shared_corners",
                    action="store_true", default=True,
                    help="K 멤버가 같은 corner vertex 공유 (기본, Subsample/Distance 로 diversity)")
    ap.add_argument("--no-shared-corners", dest="shared_corners",
                    action="store_false",
                    help="멤버마다 다른 corner vertex (d>2 에서 자연 diversity)")
    args = ap.parse_args()

    for m in args.methods:
        if m not in METHODS:
            raise SystemExit(f"Unknown method: {m!r}. Available: {list(METHODS)}")

    method_params = json.loads(args.method_params) if args.method_params else None

    if args.run_name is None:
        ts = dt.datetime.now().strftime("%Y%m%d_%H%M%S")
        methods_str = "-".join(args.methods)
        run_name = f"{ts}_{methods_str}_EBM"
    else:
        run_name = args.run_name

    root = Path(args.save_root) / run_name
    if root.exists():
        raise SystemExit(f"Run dir already exists: {root}. Pick a different --run_name.")
    root.mkdir(parents=True)

    print(f"\nRun root: {root}")
    print(f"Methods : {args.methods}")
    print(f"K       : {args.n_ebms}   seed: {args.seed}   classes: {args.classes}")
    print()

    X, y = load_and_preprocess(args.dataset, args.n_real, args.seed)

    t0 = time.time()
    for c in args.classes:
        X_c = X[y == c]
        if len(X_c) == 0:
            raise SystemExit(f"No samples for class {c} in {args.dataset}.")
        fit_class_ensemble(
            class_dir=root / f"c{c}",
            X_class=X_c, X_all=X, y_all=y,
            methods=args.methods, K=args.n_ebms, seed=args.seed + c * 1000,
            method_params=method_params,
            dataset=args.dataset, target_class=c, n_real=args.n_real,
            shared_corners=args.shared_corners,
        )
        print()

    run_meta = {
        "run_name": run_name,
        "created_at": dt.datetime.now().isoformat(timespec="seconds"),
        "dataset": args.dataset,
        "n_real": args.n_real,
        "n_ebms": args.n_ebms,
        "methods": args.methods,
        "classes": args.classes,
        "seed": args.seed,
        "method_params_override": method_params,
        "elapsed_sec": round(time.time() - t0, 2),
    }
    (root / "meta.json").write_text(json.dumps(run_meta, indent=2))
    print(f"Done in {time.time()-t0:.1f}s. Run root:\n  {root}")


if __name__ == "__main__":
    main()
