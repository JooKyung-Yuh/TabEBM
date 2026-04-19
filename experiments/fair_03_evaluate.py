#!/usr/bin/env python
"""Step 3/3: classifier evaluation on augmented data.

Usage:
    python experiments/fair_03_evaluate.py \
        --eval_dir experiments/fair_eval/20260417_210000

    # classifier 선택
    python experiments/fair_03_evaluate.py \
        --eval_dir experiments/fair_eval/20260417_210000 \
        --classifiers rf xgboost

Step 1, 2 결과물만 있으면 수초 내 완료 (CPU only).

Output:
    {eval_dir}/results/
      splits_raw.csv
      mean_bacc.csv, std_bacc.csv
      delta_vs_baseline.csv
      vp_config_summary.csv
      bar_bacc.png
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import balanced_accuracy_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
import xgboost as xgb

CLASSIFIER_BUILDERS = {
    'knn':     lambda s: KNeighborsClassifier(n_jobs=-1),
    'lr':      lambda s: LogisticRegression(max_iter=1000, n_jobs=-1, random_state=s),
    'rf':      lambda s: RandomForestClassifier(n_jobs=-1, random_state=s),
    'xgboost': lambda s: xgb.XGBClassifier(n_jobs=-1, eval_metric='logloss',
                                            use_label_encoder=False, random_state=s),
    'mlp':     lambda s: MLPClassifier(max_iter=300, random_state=s),
}


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                  formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--eval_dir', required=True)
    ap.add_argument('--classifiers', nargs='+',
                    default=list(CLASSIFIER_BUILDERS.keys()),
                    help=f'사용할 classifier (default: 전부). '
                         f'Available: {list(CLASSIFIER_BUILDERS.keys())}')
    args = ap.parse_args()

    eval_dir = Path(args.eval_dir).resolve()
    config = json.loads((eval_dir / 'config.json').read_text())
    sample_config = json.loads((eval_dir / 'sample_config.json').read_text())

    data = np.load(eval_dir / 'data.npz')
    X_all, y_all = data['X_all'], data['y_all']
    sp = np.load(eval_dir / 'splits.npz')
    splits = [(sp[f'tr_{i}'], sp[f'te_{i}'])
              for i in range(config['n_splits'])]
    classes = config['classes']

    vp_configs = sample_config['vp_configs']
    vp_names = [c['name'] for c in vp_configs]
    include_single = sample_config.get('include_single', True)
    settings = ['baseline']
    if include_single:
        settings.append('tabebm_single')
    settings.extend(vp_names)

    samples_dir = eval_dir / 'samples'
    results_dir = eval_dir / 'results'
    results_dir.mkdir(exist_ok=True)

    print(f'eval_dir:    {eval_dir}')
    print(f'classifiers: {args.classifiers}')
    print(f'settings:    {len(settings)} ({len(vp_configs)} VP'
          f'{" + single" if include_single else ""})')
    print()

    t0 = time.time()
    all_rows = []

    for split_i, (tr, te) in enumerate(splits):
        X_tr, y_tr = X_all[tr], y_all[tr]
        X_te, y_te = X_all[te], y_all[te]
        seed_i = config['seed'] + split_i

        base_scores = {}
        for clf_name in args.classifiers:
            clf = CLASSIFIER_BUILDERS[clf_name](seed_i)
            clf.fit(X_tr, y_tr)
            base_scores[clf_name] = balanced_accuracy_score(
                y_te, clf.predict(X_te)) * 100

        aug = {}
        if include_single:
            single = {c: np.load(
                samples_dir / f'split_{split_i}' / 'tabebm_single' / f'c{c}.npy')
                for c in classes}
            X_s = np.vstack([X_tr] + [single[c] for c in classes])
            y_s = np.concatenate(
                [y_tr] + [np.full(len(single[c]), c) for c in classes])
            aug['tabebm_single'] = (X_s, y_s)

        for cfg in vp_configs:
            vp = {c: np.load(
                samples_dir / f'split_{split_i}' / cfg['name'] / f'c{c}.npy')
                for c in classes}
            X_v = np.vstack([X_tr] + [vp[c] for c in classes])
            y_v = np.concatenate(
                [y_tr] + [np.full(len(vp[c]), c) for c in classes])
            aug[cfg['name']] = (X_v, y_v)

        for clf_name in args.classifiers:
            row = {'split': split_i, 'classifier': clf_name,
                   'baseline': base_scores[clf_name]}
            for sname, (X_aug, y_aug) in aug.items():
                clf = CLASSIFIER_BUILDERS[clf_name](seed_i)
                clf.fit(X_aug, y_aug)
                row[sname] = balanced_accuracy_score(
                    y_te, clf.predict(X_te)) * 100
            all_rows.append(row)

        bl = base_scores.get('rf', list(base_scores.values())[0])
        print(f'  split {split_i:>2d}  baseline(rf)={bl:.1f}%', flush=True)

    df = pd.DataFrame(all_rows).sort_values(
        ['split', 'classifier']).reset_index(drop=True)
    df.to_csv(results_dir / 'splits_raw.csv', index=False)

    mean_df = df.groupby('classifier')[settings].mean()
    std_df = df.groupby('classifier')[settings].std()
    mean_df.to_csv(results_dir / 'mean_bacc.csv')
    std_df.to_csv(results_dir / 'std_bacc.csv')

    delta = df.copy()
    for s in settings[1:]:
        delta[f'Δ_{s}'] = delta[s] - delta['baseline']
    delta_cols = [f'Δ_{s}' for s in settings[1:]]
    delta_agg = delta.groupby('classifier')[delta_cols].agg(['mean', 'std'])
    delta_agg.to_csv(results_dir / 'delta_vs_baseline.csv')

    summary = []
    for ci, cfg in enumerate(vp_configs):
        col = f'Δ_{vp_names[ci]}'
        avg_delta = float(delta[col].mean())
        robust = int((delta.groupby('classifier')[col].mean() > 0).sum())
        summary.append({
            'name': vp_names[ci],
            'beta': cfg['beta'], 'eta': cfg['eta'], 'tau': cfg['tau'],
            'ignore_variance': cfg.get('ignore_variance', False),
            'mean_Δ_pp': round(avg_delta, 3),
            'classifiers_beating_baseline': f'{robust}/{len(args.classifiers)}',
        })
    summary_df = pd.DataFrame(summary).sort_values('mean_Δ_pp', ascending=False)
    summary_df.to_csv(results_dir / 'vp_config_summary.csv', index=False)

    # bar chart
    n_set = len(settings)
    fig, ax = plt.subplots(
        figsize=(max(8, 1.2 * len(args.classifiers) * n_set / 3), 5))
    x = np.arange(len(args.classifiers))
    w = 0.8 / n_set
    for i, s in enumerate(settings):
        means = [mean_df.loc[c, s] for c in args.classifiers]
        stds = [std_df.loc[c, s] for c in args.classifiers]
        ax.bar(x + (i - (n_set - 1) / 2) * w, means, w,
               yerr=stds, label=s, capsize=2)
    ax.set_xticks(x)
    ax.set_xticklabels(args.classifiers)
    ax.set_ylabel('balanced accuracy (%)')
    ax.set_title(
        f'Fair Evaluation (per-split refit)\n'
        f'N_syn={sample_config["n_syn_per_class"]}/class, '
        f'K={config["K"]}, {config["n_splits"]} splits')
    ax.legend(ncol=min(n_set, 4), fontsize=8)
    ax.grid(axis='y', alpha=0.3)
    plt.tight_layout()
    fig.savefig(results_dir / 'bar_bacc.png', dpi=120, bbox_inches='tight')
    plt.close()

    dt = time.time() - t0
    print(f'\nStep 3 done — {dt:.1f}s  ({len(df)} rows)')
    print(f'\nMean balanced acc (%):')
    print(mean_df.to_string())
    print(f'\nΔ vs baseline:')
    print(delta_agg.to_string())
    print(f'\n  {results_dir}')


if __name__ == '__main__':
    main()
