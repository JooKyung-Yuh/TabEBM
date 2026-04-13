#!/usr/bin/env python
"""
Unified analysis and visualization tool for TabEBM experiments.

Implements all three paper research questions (Q1 augmentation performance,
Q2 statistical fidelity, Q3 privacy) plus plots and report generation.

Subcommands:
    summary    - Quick mean/std table + coverage
    q1         - Paper Q1: per-dataset table, ADTM, Average Rank, CD diagram
    q2         - Paper Q2: KS/KL/Chi2 fidelity metrics
    q3         - Paper Q3: DCR, delta-presence, TSTR privacy
    plots      - Scaling curves, comparison heatmaps, ranking, boxplot
    report     - Markdown + LaTeX paper-style report

Usage:
    python experiments/analyze.py summary --results_dir experiments/results/full_v3
    python experiments/analyze.py q1 --results_dir experiments/results/full_v3 --cd_diagram --save
    python experiments/analyze.py q2 --results_dir experiments/results/full_v3 --save
    python experiments/analyze.py q3 --dataset biodeg --n_real 100 --save
    python experiments/analyze.py plots --results_dir experiments/results/full_v3
    python experiments/analyze.py report --results_dir experiments/results/full_v3
"""

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

REQUIRED_COLS = {"dataset", "n_real", "split", "method", "classifier", "balanced_accuracy", "status"}

METHOD_COLORS = {
    "baseline": "#888888", "smote": "#F28E2B",
    "tabebm": "#4E79A7", "tvae": "#59A14F", "ctgan": "#E15759",
}
METHOD_MARKERS = {"baseline": "o", "smote": "s", "tabebm": "D", "tvae": "^", "ctgan": "v"}


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------
def load_raw_results(results_dir: str) -> pd.DataFrame:
    """Load all raw result CSVs, filtering out summary/ranks/improvement/config."""
    results_dir = Path(results_dir)
    skip = ["summary", "ranks", "improvement", "config", "q1_", "q2_", "q3_"]
    files = [f for f in sorted(results_dir.glob("*.csv")) if not any(s in f.name for s in skip)]
    if not files:
        print(f"No result files found in {results_dir}")
        sys.exit(1)
    dfs = []
    for f in files:
        df = pd.read_csv(f)
        if REQUIRED_COLS.issubset(df.columns):
            dfs.append(df)
    return pd.concat(dfs, ignore_index=True) if dfs else pd.DataFrame()


def load_valid_results(results_dir: str) -> pd.DataFrame:
    """Load raw results and filter to status=ok with non-null accuracy."""
    df = load_raw_results(results_dir)
    valid = df[df["balanced_accuracy"].notna() & (df["status"] == "ok")]
    print(f"Loaded {len(df)} rows, {len(valid)} valid")
    return valid


# ---------------------------------------------------------------------------
# SUMMARY
# ---------------------------------------------------------------------------
def cmd_summary(args):
    df_all = load_raw_results(args.results_dir)
    df = df_all[df_all["balanced_accuracy"].notna() & (df_all["status"] == "ok")]
    print(f"Total: {len(df_all)} rows, {len(df)} valid")

    print("\n" + "=" * 70)
    print("COVERAGE (ok / skipped / failed)")
    print("=" * 70)
    for ds in sorted(df_all["dataset"].unique()):
        sub = df_all[df_all["dataset"] == ds]
        total = len(sub)
        ok = len(sub[sub["status"] == "ok"])
        skipped = len(sub[sub["status"].str.startswith("skipped", na=False)])
        failed = total - ok - skipped
        print(f"  {ds:>10}: {ok}/{total} ok, {skipped} skipped, {failed} failed")

    print("\n" + "=" * 70)
    print("MEAN BALANCED ACCURACY per (dataset, n_real)")
    print("=" * 70)
    for ds in sorted(df["dataset"].unique()):
        for nr in sorted(df[df["dataset"] == ds]["n_real"].unique()):
            sub = df[(df["dataset"] == ds) & (df["n_real"] == nr)]
            if sub.empty:
                continue
            print(f"\n--- {ds} (N_real={int(nr)}) ---")
            pm = sub.pivot_table(values="balanced_accuracy", index="method", columns="classifier", aggfunc="mean")
            ps = sub.pivot_table(values="balanced_accuracy", index="method", columns="classifier", aggfunc="std")
            pm["MEAN"] = pm.mean(axis=1)
            ps["MEAN"] = ps.mean(axis=1)
            fmt = pm.copy()
            for c in pm.columns:
                fmt[c] = pm[c].round(2).astype(str) + " ± " + ps[c].round(2).astype(str)
            print(fmt.to_string())


# ---------------------------------------------------------------------------
# Q1: Augmentation performance (paper Table 1)
# ---------------------------------------------------------------------------
def _compute_adtm(df: pd.DataFrame) -> pd.DataFrame:
    """ADTM via per-condition affine normalization."""
    rows = []
    for _, g in df.groupby(["dataset", "n_real", "classifier", "split"]):
        lo, hi = g["balanced_accuracy"].min(), g["balanced_accuracy"].max()
        if hi - lo < 1e-9:
            continue
        g = g.assign(norm_acc=(g["balanced_accuracy"] - lo) / (hi - lo))
        rows.append(g)
    if not rows:
        return pd.DataFrame()
    normalized = pd.concat(rows, ignore_index=True)
    out = normalized.groupby("method")["norm_acc"].agg(["mean", "std"]).reset_index()
    return out.rename(columns={"mean": "ADTM", "std": "ADTM_std"}).sort_values("ADTM", ascending=False)


def _compute_average_rank(df: pd.DataFrame) -> pd.DataFrame:
    """Average rank across all valid conditions."""
    rows = []
    for _, g in df.groupby(["dataset", "n_real", "classifier", "split"]):
        if len(g) < 2:
            continue
        g = g.copy()
        g["rank"] = g["balanced_accuracy"].rank(method="average", ascending=False)
        rows.append(g)
    if not rows:
        return pd.DataFrame()
    ranked = pd.concat(rows, ignore_index=True)
    out = ranked.groupby("method")["rank"].agg(["mean", "std"]).reset_index()
    return out.rename(columns={"mean": "AvgRank", "std": "AvgRank_std"}).sort_values("AvgRank")


def _per_dataset_table(df: pd.DataFrame) -> pd.DataFrame:
    """Paper Table 1: method × dataset mean ± std."""
    g = df.groupby(["dataset", "method"])["balanced_accuracy"].agg(["mean", "std"])
    wide_mean = g["mean"].unstack("dataset")
    wide_std = g["std"].unstack("dataset")
    fmt = wide_mean.copy()
    for col in wide_mean.columns:
        fmt[col] = wide_mean[col].round(2).astype(str) + " ± " + wide_std[col].round(2).astype(str)
    return fmt


def _cd_diagram(df: pd.DataFrame, output: Path):
    """Nemenyi critical difference diagram."""
    import matplotlib.pyplot as plt
    from scipy.stats import friedmanchisquare, rankdata

    pivot = df.pivot_table(
        values="balanced_accuracy",
        index=["dataset", "n_real", "classifier", "split"],
        columns="method", aggfunc="first",
    ).dropna()
    if pivot.empty:
        print("  CD diagram skipped (not enough data)")
        return

    n_datasets = len(pivot)
    n_methods = pivot.shape[1]
    ranks = pd.DataFrame(index=pivot.index, columns=pivot.columns, dtype=float)
    for idx, row in pivot.iterrows():
        ranks.loc[idx] = rankdata(-row.values, method="average")
    avg_ranks = ranks.mean(axis=0).sort_values()

    try:
        stat, p = friedmanchisquare(*[pivot[c].values for c in pivot.columns])
        info = f"Friedman χ²={stat:.2f}, p={p:.4f}, n={n_datasets}"
    except Exception:
        info = f"n={n_datasets}"

    q_alpha = {2: 1.960, 3: 2.343, 4: 2.569, 5: 2.728, 6: 2.850, 7: 2.949, 8: 3.031}
    q = q_alpha.get(n_methods, 3.0)
    cd = q * np.sqrt(n_methods * (n_methods + 1) / (6.0 * n_datasets))

    fig, ax = plt.subplots(figsize=(10, max(3, 0.5 * n_methods + 2)))
    y = np.arange(n_methods)
    ax.scatter(avg_ranks.values, y, s=100, zorder=3, color="#4E79A7")
    for i, (m, r) in enumerate(avg_ranks.items()):
        ax.text(r, i + 0.2, f"  {m}", va="bottom", ha="center", fontsize=10)
    ax.axvline(x=avg_ranks.values[0] + cd, color="red", linestyle="--",
               alpha=0.5, label=f"CD = {cd:.2f}")
    ax.set_xlabel("Average Rank (lower is better)")
    ax.set_yticks([])
    ax.set_title(f"Critical Difference Diagram ({info})")
    ax.legend()
    ax.invert_xaxis()
    plt.tight_layout()
    fig.savefig(output, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  saved: {output}")


def cmd_q1(args):
    df = load_valid_results(args.results_dir)

    print("\n" + "=" * 70)
    print("Q1.1: MEAN BALANCED ACCURACY PER DATASET (Paper Table 1)")
    print("=" * 70)
    tbl = _per_dataset_table(df)
    print(tbl.to_string())

    print("\n" + "=" * 70)
    print("Q1.2: ADTM (higher = better, affine-normalized)")
    print("=" * 70)
    adtm = _compute_adtm(df)
    print(adtm.round(4).to_string(index=False))

    print("\n" + "=" * 70)
    print("Q1.3: AVERAGE RANK (lower = better)")
    print("=" * 70)
    rank = _compute_average_rank(df)
    print(rank.round(3).to_string(index=False))

    if args.save:
        out = Path(args.results_dir)
        tbl.to_csv(out / "q1_table1_per_dataset.csv")
        adtm.to_csv(out / "q1_adtm.csv", index=False)
        rank.to_csv(out / "q1_average_rank.csv", index=False)
        print(f"\nSaved to {out}/")

    if args.cd_diagram:
        _cd_diagram(df, Path(args.results_dir) / "q1_cd_diagram.png")


# ---------------------------------------------------------------------------
# Q2: Statistical fidelity
# ---------------------------------------------------------------------------
def _inverse_kl(X_real, X_syn, n_bins=20):
    kls = []
    for col in range(X_real.shape[1]):
        r, s = X_real[:, col], X_syn[:, col]
        lo, hi = min(r.min(), s.min()), max(r.max(), s.max())
        if hi - lo < 1e-9:
            continue
        bins = np.linspace(lo, hi, n_bins + 1)
        p, _ = np.histogram(r, bins=bins, density=True)
        q, _ = np.histogram(s, bins=bins, density=True)
        p = (p + 1e-6) / (p.sum() + n_bins * 1e-6)
        q = (q + 1e-6) / (q.sum() + n_bins * 1e-6)
        kls.append(1.0 / (1.0 + float((p * np.log(p / q)).sum())))
    return float(np.mean(kls)) if kls else 0.0


def _chi_squared(X_real, X_syn, n_bins=10):
    from scipy.stats import chi2_contingency
    ps = []
    for col in range(X_real.shape[1]):
        r, s = X_real[:, col], X_syn[:, col]
        lo, hi = min(r.min(), s.min()), max(r.max(), s.max())
        if hi - lo < 1e-9:
            continue
        bins = np.linspace(lo, hi, n_bins + 1)
        rh, _ = np.histogram(r, bins=bins)
        sh, _ = np.histogram(s, bins=bins)
        try:
            _, p, _, _ = chi2_contingency(np.array([rh, sh]) + 1)
            ps.append(p)
        except Exception:
            continue
    return float(np.median(ps)) if ps else 0.0


def cmd_q2(args):
    df = load_raw_results(args.results_dir)
    print("=" * 70)
    print("Q2.1: KS p-values from experiment CSVs (already recorded)")
    print("=" * 70)
    fidelity = df[df["ks_median_pvalue"].notna()]
    if fidelity.empty:
        print("No KS data in results.")
    else:
        agg = fidelity.groupby(["method", "dataset"])[["ks_median_pvalue", "ks_mean_pvalue"]].median()
        print(agg.round(4).to_string())
        if args.save:
            agg.to_csv(Path(args.results_dir) / "q2_ks_summary.csv")

    if args.regenerate:
        print("\n" + "=" * 70)
        print(f"Q2.2: Full fidelity (regenerate, {args.dataset} n={args.n_real})")
        print("=" * 70)
        sys.path.insert(0, str(Path(__file__).parent))
        from run_experiment import load_dataset, prepare_splits, AUGMENT_REGISTRY
        from scipy.stats import ks_2samp

        X, y, _ = load_dataset(args.dataset)
        splits, _ = prepare_splits(X, y, args.n_real, n_splits=3, base_seed=42)
        rows = []
        for split in splits:
            for method in args.methods:
                try:
                    X_aug, y_aug = AUGMENT_REGISTRY[method](
                        split["X_train"].copy(), split["y_train"].copy(),
                        n_syn=500, seed=split["seed"], device=f"cuda:{args.gpu}", gpus=[args.gpu],
                        sgld_steps=200, sgld_step_size=0.1, sgld_noise_std=0.01,
                        distance_negative_class=5.0,
                    )
                except Exception as e:
                    print(f"  {method}: {type(e).__name__}: {e}")
                    continue
                X_syn = X_aug[len(split["X_train"]):]
                if len(X_syn) == 0:
                    continue
                ks_ps = [ks_2samp(split["X_train"][:, c], X_syn[:, c])[1] for c in range(X_syn.shape[1])]
                rows.append({
                    "dataset": args.dataset, "n_real": args.n_real,
                    "split": split["split_idx"], "method": method,
                    "inverse_kl": round(_inverse_kl(split["X_train"], X_syn), 4),
                    "ks_median": round(float(np.median(ks_ps)), 4),
                    "chi2_median": round(_chi_squared(split["X_train"], X_syn), 4),
                })
        if rows:
            full = pd.DataFrame(rows)
            agg = full.groupby("method")[["inverse_kl", "ks_median", "chi2_median"]].mean()
            print(agg.round(4).to_string())
            if args.save:
                full.to_csv(Path(args.results_dir) / f"q2_fidelity_{args.dataset}_n{args.n_real}.csv", index=False)


# ---------------------------------------------------------------------------
# Q3: Privacy
# ---------------------------------------------------------------------------
def _dcr(X_real, X_syn):
    from scipy.spatial.distance import cdist
    if len(X_syn) == 0 or len(X_real) == 0:
        return {"dcr_median": None, "dcr_mean": None}
    d = cdist(X_syn, X_real).min(axis=1)
    return {"dcr_median": round(float(np.median(d)), 4), "dcr_mean": round(float(np.mean(d)), 4)}


def _delta_presence(X_real, X_syn, delta=0.1):
    from scipy.spatial.distance import cdist
    if len(X_syn) == 0 or len(X_real) == 0:
        return {"delta_presence": None}
    ref = X_real[np.random.RandomState(42).choice(len(X_real), min(200, len(X_real)), replace=False)]
    d = cdist(ref, ref)
    scale = float(np.median(d[np.triu_indices_from(d, k=1)])) if len(ref) > 1 else 1.0
    thr = delta * scale
    nn = cdist(X_real, X_syn).min(axis=1)
    return {"delta_presence": round(float((nn < thr).sum() / len(X_real)), 4)}


def _tstr(X_syn, y_syn, X_test, y_test):
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.metrics import balanced_accuracy_score
    if len(X_syn) == 0 or len(np.unique(y_syn)) < 2:
        return {"tstr_acc": None}
    try:
        clf = RandomForestClassifier(n_jobs=-1, random_state=42)
        clf.fit(X_syn, y_syn)
        return {"tstr_acc": round(float(balanced_accuracy_score(y_test, clf.predict(X_test)) * 100), 2)}
    except Exception:
        return {"tstr_acc": None}


def cmd_q3(args):
    sys.path.insert(0, str(Path(__file__).parent))
    from run_experiment import load_dataset, prepare_splits, AUGMENT_REGISTRY

    print(f"Privacy analysis: {args.dataset} n_real={args.n_real}")
    X, y, _ = load_dataset(args.dataset)
    splits, _ = prepare_splits(X, y, args.n_real, n_splits=args.n_splits, base_seed=42)

    rows = []
    for split in splits:
        for method in args.methods:
            try:
                X_aug, y_aug = AUGMENT_REGISTRY[method](
                    split["X_train"].copy(), split["y_train"].copy(),
                    n_syn=500, seed=split["seed"], device=f"cuda:{args.gpu}", gpus=[args.gpu],
                    sgld_steps=200, sgld_step_size=0.1, sgld_noise_std=0.01,
                    distance_negative_class=5.0,
                )
            except Exception as e:
                print(f"  split {split['split_idx']} {method}: {type(e).__name__}")
                continue
            X_syn = X_aug[len(split["X_train"]):]
            y_syn = y_aug[len(split["y_train"]):]
            row = {"dataset": args.dataset, "n_real": args.n_real,
                   "split": split["split_idx"], "method": method, "n_syn": len(X_syn)}
            row.update(_dcr(split["X_train"], X_syn))
            row.update(_delta_presence(split["X_train"], X_syn))
            row.update(_tstr(X_syn, y_syn, split["X_test"], split["y_test"]))
            rows.append(row)
            print(f"  split={split['split_idx']} {method}: "
                  f"DCR={row['dcr_median']}, δ={row['delta_presence']}, TSTR={row['tstr_acc']}")

    if rows:
        df = pd.DataFrame(rows)
        print("\n" + "=" * 70)
        print("Q3: PRIVACY METRICS (mean across splits)")
        print("=" * 70)
        print("  DCR median     = synthetic's distance to nearest real (higher=safer)")
        print("  delta_presence = fraction of real exposed by synthetic (lower=safer)")
        print("  TSTR acc       = train-synthetic test-real balanced accuracy")
        print()
        agg = df.groupby("method")[["dcr_median", "dcr_mean", "delta_presence", "tstr_acc"]].mean()
        print(agg.round(3).to_string())
        if args.save:
            out = Path(args.results_dir)
            out.mkdir(parents=True, exist_ok=True)
            df.to_csv(out / f"q3_privacy_{args.dataset}_n{args.n_real}.csv", index=False)


# ---------------------------------------------------------------------------
# PLOTS
# ---------------------------------------------------------------------------
def cmd_plots(args):
    import matplotlib.pyplot as plt
    df = load_valid_results(args.results_dir)
    out = Path(args.results_dir)

    # 1. Scaling curves per dataset
    datasets = sorted(df["dataset"].unique())
    n_cols = 4
    n_rows = int(np.ceil(len(datasets) / n_cols))
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(4 * n_cols, 3 * n_rows), squeeze=False)
    for idx, ds in enumerate(datasets):
        ax = axes[idx // n_cols, idx % n_cols]
        sub = df[df["dataset"] == ds]
        for method in sorted(sub["method"].unique()):
            m = sub[sub["method"] == method]
            agg = m.groupby("n_real")["balanced_accuracy"].agg(["mean", "std"]).reset_index()
            ax.errorbar(agg["n_real"], agg["mean"], yerr=agg["std"],
                        label=method, marker=METHOD_MARKERS.get(method, "o"),
                        color=METHOD_COLORS.get(method, "black"),
                        markersize=6, capsize=3, linewidth=1.5, alpha=0.85)
        ax.set_title(ds, fontsize=11, fontweight="bold")
        ax.set_xlabel("N_real")
        ax.set_ylabel("Balanced Accuracy (%)")
        ax.set_xscale("log")
        ax.grid(alpha=0.3)
        if idx == 0:
            ax.legend(fontsize=8)
    for idx in range(len(datasets), n_rows * n_cols):
        axes[idx // n_cols, idx % n_cols].axis("off")
    fig.suptitle("Scaling: Balanced Accuracy vs N_real", fontsize=14, fontweight="bold")
    plt.tight_layout()
    fig.savefig(out / "plot_scaling.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  saved: {out}/plot_scaling.png")

    # 2. Improvement heatmap (baseline 대비 향상/하락, diverging colormap)
    pv = df.pivot_table(values="balanced_accuracy",
                        index=["dataset", "n_real", "classifier", "split"],
                        columns="method", aggfunc="first")
    if "baseline" in pv.columns:
        imp = pv.subtract(pv["baseline"], axis=0).drop(columns=["baseline"]).reset_index()
        long = imp.melt(id_vars=["dataset", "n_real", "classifier", "split"],
                        var_name="method", value_name="imp").dropna()
        mat = long.pivot_table(values="imp", index="method", columns="dataset", aggfunc="mean")
        mat = mat.loc[sorted(mat.index), datasets]
        fig, ax = plt.subplots(figsize=(max(6, len(datasets) * 0.9), max(3, len(mat) * 0.6)))
        vmax = max(abs(np.nanmin(mat.values)), abs(np.nanmax(mat.values)))
        im = ax.imshow(mat.values, cmap="RdBu_r", aspect="auto", vmin=-vmax, vmax=vmax)
        ax.set_xticks(np.arange(len(datasets)))
        ax.set_yticks(np.arange(len(mat)))
        ax.set_xticklabels(datasets, rotation=30, ha="right")
        ax.set_yticklabels(mat.index)
        for i in range(len(mat)):
            for j in range(len(datasets)):
                v = mat.iloc[i, j]
                if not np.isnan(v):
                    ax.text(j, i, f"{v:+.1f}", ha="center", va="center", fontsize=9, fontweight="bold")
        plt.colorbar(im, ax=ax, label="Improvement (pp)")
        ax.set_title("Improvement over Baseline: Method × Dataset")
        plt.tight_layout()
        fig.savefig(out / "plot_improvement.png", dpi=150, bbox_inches="tight")
        plt.close(fig)
        print(f"  saved: {out}/plot_improvement.png")

    # 4. Ranking plot
    rows = []
    for _, g in df.groupby(["dataset", "n_real", "classifier", "split"]):
        if len(g) < 2:
            continue
        g = g.copy()
        g["rank"] = g["balanced_accuracy"].rank(method="average", ascending=False)
        rows.append(g)
    if rows:
        ranked = pd.concat(rows, ignore_index=True)
        agg = ranked.groupby("method")["rank"].agg(["mean", "std", "count"]).reset_index()
        agg["ci95"] = 1.96 * agg["std"] / np.sqrt(agg["count"])
        agg = agg.sort_values("mean")
        fig, ax = plt.subplots(figsize=(7, max(3, len(agg) * 0.6)))
        y = np.arange(len(agg))
        ax.errorbar(agg["mean"], y, xerr=agg["ci95"], fmt="o", markersize=10,
                    capsize=5, linewidth=2, color="#4E79A7")
        ax.set_yticks(y)
        ax.set_yticklabels(agg["method"])
        ax.set_xlabel("Average Rank (lower is better)")
        ax.set_title("Method Ranking across all conditions")
        ax.grid(axis="x", alpha=0.3)
        ax.invert_yaxis()
        plt.tight_layout()
        fig.savefig(out / "plot_ranking.png", dpi=150, bbox_inches="tight")
        plt.close(fig)
        print(f"  saved: {out}/plot_ranking.png")


# ---------------------------------------------------------------------------
# REPORT
# ---------------------------------------------------------------------------
def cmd_report(args):
    df_all = load_raw_results(args.results_dir)
    df = df_all[df_all["balanced_accuracy"].notna() & (df_all["status"] == "ok")]
    out = Path(args.results_dir)

    lines = []
    lines.append("# TabEBM Reproduction Report\n")
    lines.append(f"**Results dir**: `{args.results_dir}`  ")
    lines.append(f"**Valid rows**: {len(df)}  ")
    lines.append(f"**Datasets**: {', '.join(sorted(df['dataset'].unique()))}  ")
    lines.append(f"**Methods**: {', '.join(sorted(df['method'].unique()))}  ")
    lines.append(f"**Classifiers**: {', '.join(sorted(df['classifier'].unique()))}\n")

    # Table 1
    lines.append("\n## Table 1: Balanced Accuracy (%) per Dataset\n")
    g = df.groupby(["dataset", "method"])["balanced_accuracy"].agg(["mean", "std"])
    datasets = sorted(df["dataset"].unique())
    methods = sorted(df["method"].unique())
    header = "| Method | " + " | ".join(datasets) + " | **Avg** |"
    sep = "|" + "---|" * (len(datasets) + 2)
    lines += [header, sep]
    for m in methods:
        row = [f"**{m}**"]
        accs = []
        for d in datasets:
            if (d, m) in g.index:
                mean, std = g.loc[(d, m), "mean"], g.loc[(d, m), "std"]
                row.append(f"{mean:.2f} ± {std:.2f}")
                accs.append(mean)
            else:
                row.append("—")
        avg = np.mean(accs) if accs else float("nan")
        row.append(f"**{avg:.2f}**" if not np.isnan(avg) else "—")
        lines.append("| " + " | ".join(row) + " |")

    # Rank + ADTM
    rank = _compute_average_rank(df)
    adtm = _compute_adtm(df)
    lines.append("\n## Average Rank (lower = better)\n")
    lines.append("| Method | Rank ± std |")
    lines.append("|---|---|")
    for _, row in rank.iterrows():
        lines.append(f"| **{row['method']}** | {row['AvgRank']:.3f} ± {row['AvgRank_std']:.3f} |")

    lines.append("\n## ADTM (higher = better, affine-normalized)\n")
    lines.append("| Method | ADTM |")
    lines.append("|---|---|")
    for _, row in adtm.iterrows():
        lines.append(f"| **{row['method']}** | {row['ADTM']:.4f} |")

    # Coverage
    lines.append("\n## Coverage\n")
    lines.append("| Dataset | OK | Skipped | Failed |")
    lines.append("|---|---|---|---|")
    for ds in sorted(df_all["dataset"].unique()):
        sub = df_all[df_all["dataset"] == ds]
        ok = len(sub[sub["status"] == "ok"])
        sk = len(sub[sub["status"].str.startswith("skipped", na=False)])
        fa = len(sub) - ok - sk
        lines.append(f"| {ds} | {ok} | {sk} | {fa} |")

    (out / "report.md").write_text("\n".join(lines))
    print(f"  saved: {out}/report.md")

    # LaTeX table
    tex = ["\\begin{table}[t]", "\\centering",
           "\\caption{Balanced accuracy (\\%) averaged over splits, classifiers, N\\_real.}",
           "\\label{tab:main}",
           "\\begin{tabular}{l" + "c" * len(datasets) + "c}",
           "\\toprule",
           "Method & " + " & ".join(f"\\textsc{{{d}}}" for d in datasets) + " & Avg \\\\",
           "\\midrule"]
    for m in methods:
        row = [m]
        accs = []
        for d in datasets:
            if (d, m) in g.index:
                mean, std = g.loc[(d, m), "mean"], g.loc[(d, m), "std"]
                row.append(f"${mean:.1f}_{{\\pm {std:.1f}}}$")
                accs.append(mean)
            else:
                row.append("—")
        avg = np.mean(accs) if accs else float("nan")
        row.append(f"${avg:.1f}$" if not np.isnan(avg) else "—")
        tex.append(" & ".join(row) + " \\\\")
    tex += ["\\bottomrule", "\\end{tabular}", "\\end{table}"]
    (out / "report_table1.tex").write_text("\n".join(tex))
    print(f"  saved: {out}/report_table1.tex")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(
        description="TabEBM experiment analysis (paper Q1/Q2/Q3 + plots + report)",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    sub = parser.add_subparsers(dest="cmd", required=True)

    # summary
    p = sub.add_parser("summary", help="Quick table + coverage")
    p.add_argument("--results_dir", required=True)

    # q1
    p = sub.add_parser("q1", help="Paper Q1: augmentation performance")
    p.add_argument("--results_dir", required=True)
    p.add_argument("--save", action="store_true")
    p.add_argument("--cd_diagram", action="store_true")

    # q2
    p = sub.add_parser("q2", help="Paper Q2: statistical fidelity")
    p.add_argument("--results_dir", required=True)
    p.add_argument("--save", action="store_true")
    p.add_argument("--regenerate", action="store_true")
    p.add_argument("--dataset", default="biodeg")
    p.add_argument("--n_real", type=int, default=100)
    p.add_argument("--methods", nargs="+", default=["tabebm", "smote", "tvae", "ctgan"])
    p.add_argument("--gpu", type=int, default=0, help="GPU for regeneration")

    # q3
    p = sub.add_parser("q3", help="Paper Q3: privacy")
    p.add_argument("--dataset", default="biodeg")
    p.add_argument("--n_real", type=int, default=100)
    p.add_argument("--methods", nargs="+", default=["tabebm", "smote", "tvae", "ctgan"])
    p.add_argument("--n_splits", type=int, default=3)
    p.add_argument("--save", action="store_true")
    p.add_argument("--results_dir", default="experiments/results/full_v3")
    p.add_argument("--gpu", type=int, default=0, help="GPU for generation")

    # plots
    p = sub.add_parser("plots", help="Scaling/heatmap/ranking plots")
    p.add_argument("--results_dir", required=True)

    # report
    p = sub.add_parser("report", help="Markdown + LaTeX report")
    p.add_argument("--results_dir", required=True)

    args = parser.parse_args()
    {"summary": cmd_summary, "q1": cmd_q1, "q2": cmd_q2,
     "q3": cmd_q3, "plots": cmd_plots, "report": cmd_report}[args.cmd](args)


if __name__ == "__main__":
    main()
