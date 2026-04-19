#!/usr/bin/env python
"""Step 1/3: per-split ensemble fit.

Usage:
    python experiments/fair_01_fit_ensembles.py \
        --ensemble_root experiments/ebms/20260415_214026_Distance_EBM \
        --methods Distance \
        --method_params '{"Distance":{"mode":"fixed","value":2.0}}' \
        --no-shared-corners -K 10 --seed 42 --n_splits 10

Output:
    {eval_dir}/
      config.json        # 전체 설정 (step 2, 3 에서 읽음)
      data.npz           # X_all, y_all
      splits.npz         # tr_0, te_0, ...
      ensembles/         # per-split ensemble dirs
        split_0/c0/, split_0/c1/, ...
"""
from __future__ import annotations

import argparse
import datetime as dt
import json
import sys
import time
from pathlib import Path

import numpy as np
from sklearn.model_selection import StratifiedShuffleSplit

sys.path.insert(0, str(Path(__file__).parent))


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                  formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--ensemble_root', required=True,
                    help='기존 ensemble root (dataset 로드용)')
    ap.add_argument('--eval_dir', default=None,
                    help='출력 디렉터리 (default: experiments/fair_eval/{timestamp})')
    ap.add_argument('--methods', nargs='+', default=['Distance'])
    ap.add_argument('--method_params', default=None,
                    help='JSON, e.g. \'{"Distance":{"mode":"fixed","value":2.0}}\'')
    ap.add_argument('--shared-corners', dest='shared_corners',
                    action='store_true', default=False)
    ap.add_argument('--no-shared-corners', dest='shared_corners',
                    action='store_false')
    ap.add_argument('-K', type=int, default=10)
    ap.add_argument('--seed', type=int, default=42)
    ap.add_argument('--n_splits', type=int, default=10)
    ap.add_argument('--workers', type=int, default=10)
    args = ap.parse_args()

    ens_root = Path(args.ensemble_root).resolve()
    assert (ens_root / 'c0').exists(), f'{ens_root}/c0 not found'

    data = np.load(ens_root / 'c0' / 'class_data.npz')
    X_all, y_all = data['X_all'], data['y_all']
    classes = sorted(np.unique(y_all).tolist())

    n_test = min(len(X_all) // 2, 500)
    splits = [next(StratifiedShuffleSplit(
        n_splits=1, test_size=n_test, random_state=args.seed + i
    ).split(X_all, y_all)) for i in range(args.n_splits)]

    if args.eval_dir is None:
        ts = dt.datetime.now().strftime('%Y%m%d_%H%M%S')
        eval_dir = Path('experiments/fair_eval') / ts
    else:
        eval_dir = Path(args.eval_dir)
    eval_dir = eval_dir.resolve()
    eval_dir.mkdir(parents=True, exist_ok=True)

    np.savez(eval_dir / 'data.npz', X_all=X_all, y_all=y_all)
    split_kv = {}
    for i, (tr, te) in enumerate(splits):
        split_kv[f'tr_{i}'] = tr
        split_kv[f'te_{i}'] = te
    np.savez(eval_dir / 'splits.npz', **split_kv)

    method_params = json.loads(args.method_params) if args.method_params else None
    config = {
        'ensemble_root': str(ens_root),
        'methods': args.methods,
        'method_params': method_params,
        'shared_corners': args.shared_corners,
        'K': args.K,
        'seed': args.seed,
        'n_splits': args.n_splits,
        'n_test': n_test,
        'classes': classes,
        'n_samples': len(X_all),
        'n_features': int(X_all.shape[1]),
        'created_at': dt.datetime.now().isoformat(timespec='seconds'),
    }
    (eval_dir / 'config.json').write_text(json.dumps(config, indent=2))

    print(f'eval_dir:  {eval_dir}')
    print(f'dataset:   {X_all.shape}, classes={classes}')
    print(f'splits:    {args.n_splits} × train={len(splits[0][0])}/test={n_test}')
    print(f'ensemble:  methods={args.methods}, K={args.K}, '
          f'shared_corners={args.shared_corners}')
    print()

    from concurrent.futures import ProcessPoolExecutor, as_completed
    from fair_eval_worker import fit_one_split_ensemble

    ens_dir = eval_dir / 'ensembles'
    ens_dir.mkdir(exist_ok=True)

    tasks = [(i, tr, X_all, y_all, classes,
              args.methods, method_params, args.shared_corners,
              args.K, args.seed, str(ens_dir))
             for i, (tr, _) in enumerate(splits)]

    t0 = time.time()
    with ProcessPoolExecutor(max_workers=args.workers) as ex:
        futs = {ex.submit(fit_one_split_ensemble, t): t[0] for t in tasks}
        for f in as_completed(futs):
            r = f.result()
            tag = '(cached)' if r.get('cached') else f'{r["dt"]:.1f}s'
            print(f'  split {r["split_i"]:>2d}  {tag}', flush=True)

    elapsed = time.time() - t0
    print(f'\nStep 1 done — {elapsed:.1f}s')
    print(f'  {eval_dir}')


if __name__ == '__main__':
    import multiprocessing as mp
    try:
        mp.set_start_method('spawn', force=True)
    except RuntimeError:
        pass
    main()
