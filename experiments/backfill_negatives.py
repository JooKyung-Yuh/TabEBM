#!/usr/bin/env python
"""기존 ensemble 들에 명시적 negatives 파일 (ebm_{k}/negatives.npz, c{c}/negatives_union.npz)
을 backfill. `ebm_{k}/surrogate_data.npz` + `config.json` 만 있으면 됨.

사용:
    python experiments/backfill_negatives.py                         # experiments/ebms/ 전체
    python experiments/backfill_negatives.py --root experiments/ebms/20260415_...
    python experiments/backfill_negatives.py --force                  # 이미 있어도 덮어쓰기
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


def backfill_class(class_dir: Path, force: bool = False) -> tuple[int, int]:
    """한 class 폴더를 처리. return: (per-member written, union written [0|1])"""
    meta = json.loads((class_dir / "meta.json").read_text())
    K = meta["n_ebms"]
    all_negs = []
    all_alphas = []
    wrote_member = 0
    for k in range(K):
        ebm_dir = class_dir / f"ebm_{k}"
        sd_path = ebm_dir / "surrogate_data.npz"
        cfg_path = ebm_dir / "config.json"
        out_path = ebm_dir / "negatives.npz"
        if not sd_path.exists():
            raise FileNotFoundError(f"Missing {sd_path}")
        cfg = json.loads(cfg_path.read_text()) if cfg_path.exists() else {}
        sd = np.load(sd_path)
        X_neg = sd["X_ebm"][sd["y_ebm"] == 1]
        alpha = cfg.get("method_distance", {}).get("neg_distance")
        noise_std = cfg.get("method_corner_noise", {}).get("noise_std", 0.0)
        n_corners_cfg = cfg.get("method_num_fake_corners", {}).get("n_corners")
        if out_path.exists() and not force:
            pass   # skip write but still collect for union
        else:
            np.savez(
                out_path,
                X_neg=X_neg.astype(np.float64),
                alpha=np.array(alpha if alpha is not None else float("nan")),
                noise_std=np.array(float(noise_std)),
                n_corners=np.array(int(n_corners_cfg) if n_corners_cfg is not None else len(X_neg)),
                member_idx=np.array(int(cfg.get("member_idx", k))),
                corner_seed=np.array(int(cfg.get("corner_seed", cfg.get("seed", -1)))),
            )
            wrote_member += 1
        all_negs.append(X_neg)
        all_alphas.append(alpha)

    union_path = class_dir / "negatives_union.npz"
    wrote_union = 0
    if not union_path.exists() or force:
        X_neg_union = np.vstack(all_negs)
        member_of = np.concatenate([np.full(len(X), i, dtype=np.int64) for i, X in enumerate(all_negs)])
        alphas_arr = np.array([a if a is not None else float("nan") for a in all_alphas], dtype=np.float64)
        np.savez(
            union_path,
            X_neg=X_neg_union, member_idx=member_of,
            alpha_per_member=alphas_arr, K=np.array(K),
        )
        wrote_union = 1
    return wrote_member, wrote_union


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                  formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--root", type=Path, default=None,
                    help="특정 ensemble 폴더 (생략 시 experiments/ebms/*_EBM 전체)")
    ap.add_argument("--force", action="store_true")
    args = ap.parse_args()

    if args.root is not None:
        roots = [args.root]
    else:
        roots = sorted(p for p in Path("experiments/ebms").iterdir()
                        if p.is_dir() and p.name.endswith("_EBM"))

    total_m = total_u = 0
    for root in roots:
        class_dirs = sorted(p for p in root.iterdir()
                             if p.is_dir() and p.name.startswith("c")
                             and p.name[1:].isdigit())
        for cd in class_dirs:
            m, u = backfill_class(cd, force=args.force)
            total_m += m; total_u += u
            print(f"{cd}: +{m} per-member, +{u} union")
    print(f"\nDone. per-member files: {total_m}  union files: {total_u}")


if __name__ == "__main__":
    main()
