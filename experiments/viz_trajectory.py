#!/usr/bin/env python
"""VP-SGLD trajectory 시각화 (plotly HTML + optional wandb).

ensemble std heatmap 을 배경으로 깔고 그 위에 K 체인의 SGLD 경로를 점+선으로
그림. 하단 슬라이더로 step 을 scrubbing 해서 "step t 에 모든 체인이 어디에 있나"
확인 가능. 배경이 있어서 "체인이 high-variance 지역을 피하는지 / 끌려가는지"
눈으로 튜닝 가능 — β / η / τ / sigma_start 바꿔가며 즉시 비교.

사용:
    python experiments/viz_trajectory.py \\
        --ensemble-root experiments/ebms/20260415_210502_Distance_EBM \\
        --class 0 \\
        --n-samples 500 --n-steps 50 \\
        --beta 1.0 --eta 0.05 --tau 1.0 --sigma-start 0.1 --auto-beta \\
        --n-chains-show 30
        [--wandb-project my-tabebm]     # 원하면 wandb 에도 업로드
        [--out path.html]               # 기본: {ensemble_root}/viz/trajectory_c{C}_{tag}.html
"""
from __future__ import annotations

import json
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Optional

import numpy as np
import torch
from sklearn.decomposition import PCA

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))
sys.path.insert(0, str(Path(__file__).parent))

from tabebm.vp_sgld import vp_sgld_from_ensemble


def _load_negatives_pca(class_dir: Path, n_members: int, pca: PCA) -> np.ndarray:
    """Union of all K members' surrogate negatives, projected to PCA(2).

    우선순위: `c{c}/negatives_union.npz` (backfill_negatives.py 가 만든 합본) →
    없으면 각 `ebm_{k}/surrogate_data.npz` 에서 y_ebm==1 추출.
    """
    class_dir = Path(class_dir)
    union = class_dir / "negatives_union.npz"
    if union.exists():
        X_neg = np.load(union)["X_neg"]
        return pca.transform(X_neg) if len(X_neg) else np.zeros((0, 2))

    negs = []
    for k in range(n_members):
        # per-member negatives.npz 우선, 없으면 surrogate_data.npz 에서 추출
        neg_file = class_dir / f"ebm_{k}" / "negatives.npz"
        if neg_file.exists():
            X_neg = np.load(neg_file)["X_neg"]
        else:
            sd = np.load(class_dir / f"ebm_{k}" / "surrogate_data.npz")
            X_neg = sd["X_ebm"][sd["y_ebm"] == 1]
        if len(X_neg):
            negs.append(pca.transform(X_neg))
    return np.vstack(negs) if negs else np.zeros((0, 2))


def _plan_pad_for_bounds(
    class_dir: Path, pca: PCA,
    Z_extra_list: list[np.ndarray],
    base_pad: float = 2.0,
    margin: float = 1.0,
) -> float:
    """현재 real + pad 범위를 벗어나는 Z 점들이 있으면 그만큼 pad 를 확장."""
    X_all = np.load(class_dir / "class_data.npz")["X_all"]
    Z_all = pca.transform(X_all)
    x_lo, x_hi = Z_all[:, 0].min(), Z_all[:, 0].max()
    y_lo, y_hi = Z_all[:, 1].min(), Z_all[:, 1].max()
    extras = [z for z in Z_extra_list if z is not None and len(z)]
    if not extras:
        return base_pad
    Z_extra = np.vstack(extras).reshape(-1, 2)
    pad_needed = max(
        x_lo - Z_extra[:, 0].min(),
        Z_extra[:, 0].max() - x_hi,
        y_lo - Z_extra[:, 1].min(),
        Z_extra[:, 1].max() - y_hi,
        base_pad,
    )
    return float(round(pad_needed + margin, 1))


# in-memory cache: (class_dir_abspath, pad, h, gpu) → viz_context dict
_VIZ_CONTEXT_CACHE: dict = {}


def clear_viz_context_cache() -> None:
    """viz context (heatmap / PCA / neg) 메모리 캐시 비우기."""
    global _VIZ_CONTEXT_CACHE
    _VIZ_CONTEXT_CACHE = {}


def precompute_class_viz_context(
    class_dir: Path | str,
    *,
    pad: float = 3.0,
    h: float = 0.2,
    gpu: int = 0,
    use_cache: bool = True,
) -> dict:
    """한 class 의 viz 공통 재료 (PCA, heatmap, bounds, real/neg) 를 **한 번만**
    계산해서 모든 snapshot 이 재사용. 메모리 캐시 + 디스크 heatmap 캐시 둘 다.

    반환 dict keys:
      pca, Z_real, Z_neg, hm (ZZ1/ZZ2/std/mean), xlim, ylim, pad, h
    """
    class_dir = Path(class_dir)
    key = (str(class_dir.resolve()), float(pad), float(h), int(gpu))
    if use_cache and key in _VIZ_CONTEXT_CACHE:
        return _VIZ_CONTEXT_CACHE[key]

    real = np.load(class_dir / "class_data.npz")["X_class"]
    pca = PCA(n_components=2).fit(real)
    Z_real = pca.transform(real)
    meta = json.loads((class_dir / "meta.json").read_text())
    Z_neg = _load_negatives_pca(class_dir, meta["n_ebms"], pca)

    hm = _compute_heatmap_for_pad(class_dir, pad=pad, h=h, gpu=gpu)
    xlim = (float(hm["ZZ1"].min()), float(hm["ZZ1"].max()))
    ylim = (float(hm["ZZ2"].min()), float(hm["ZZ2"].max()))

    ctx = dict(pca=pca, Z_real=Z_real, Z_neg=Z_neg, hm=hm,
                xlim=xlim, ylim=ylim, pad=pad, h=h)
    if use_cache:
        _VIZ_CONTEXT_CACHE[key] = ctx
    return ctx


def _compute_heatmap_for_pad(
    class_dir: Path, pad: float, h: float = 0.2, gpu: int = 0, force: bool = False,
) -> dict:
    """pad 에 해당하는 heatmap cache 를 읽거나 신규 계산 (nb 01 의 것과 호환).

    파일명: `heatmap_pad{pad}_h{h}.npz` — nb 01 과 **같은 형식** 으로 저장하므로
    다음에 nb 01 실행 시에도 재사용됨.
    """
    from ensemble_ebm import rebuild_ebm, evaluate_energy

    class_dir = Path(class_dir)
    cache = class_dir / f"heatmap_pad{pad}_h{h}.npz"
    meta = json.loads((class_dir / "meta.json").read_text())
    class_data = np.load(class_dir / "class_data.npz")
    X_class, X_all = class_data["X_class"], class_data["X_all"]
    pca = PCA(n_components=2).fit(X_class)
    Z_class = pca.transform(X_class)
    Z_all = pca.transform(X_all)
    z1 = np.arange(Z_all[:, 0].min() - pad, Z_all[:, 0].max() + pad, h)
    z2 = np.arange(Z_all[:, 1].min() - pad, Z_all[:, 1].max() + pad, h)
    ZZ1, ZZ2 = np.meshgrid(z1, z2)

    if not force and cache.exists():
        c = np.load(cache)
        if int(c["K"]) == meta["n_ebms"] and c["ZZ1"].shape == ZZ1.shape:
            all_E = [c[f"E_{k}"] for k in range(int(c["K"]))]
            E_stack = np.stack(all_E, axis=0)
            K = E_stack.shape[0]
            return dict(ZZ1=ZZ1, ZZ2=ZZ2, Z_class=Z_class,
                         std=E_stack.std(axis=0, ddof=1 if K > 1 else 0),
                         mean=E_stack.mean(axis=0))

    grid_z = np.c_[ZZ1.ravel(), ZZ2.ravel()]
    grid_x = pca.inverse_transform(grid_z).astype(np.float64)
    K = meta["n_ebms"]
    all_E = []
    print(f"  [extend heatmap] pad={pad} h={h} grid={ZZ1.shape} K={K} — evaluating...")
    t0 = time.time()
    for i in range(K):
        tabebm, _ = rebuild_ebm(class_dir / f"ebm_{i}", gpu=gpu)
        es = []
        for s in range(0, len(grid_x), 64):
            e, _ = evaluate_energy(tabebm, grid_x[s:s + 64], gpu=gpu)
            es.append(e)
        all_E.append(np.concatenate(es).reshape(ZZ1.shape))
    print(f"     done in {time.time()-t0:.1f}s")

    # save cache in nb 01 compatible format
    save = {f"E_{k}": E for k, E in enumerate(all_E)}
    save.update(dict(ZZ1=ZZ1, ZZ2=ZZ2, Z_class=Z_class, K=K,
                     var_ratio=float(pca.explained_variance_ratio_.sum())))
    np.savez(cache, **save)

    E_stack = np.stack(all_E, axis=0)
    return dict(ZZ1=ZZ1, ZZ2=ZZ2, Z_class=Z_class,
                 std=E_stack.std(axis=0, ddof=1 if K > 1 else 0),
                 mean=E_stack.mean(axis=0))


# ---------------------------------------------------------------------------
# 배경 — nb 01 이 캐시한 heatmap 로드 (없으면 None, plot 에서 생략)
# ---------------------------------------------------------------------------
def load_std_heatmap(class_dir: Path, pad: float = 2.0, h: float = 0.2) -> Optional[dict]:
    cache = class_dir / f"heatmap_pad{pad}_h{h}.npz"
    if not cache.exists():
        return None
    c = np.load(cache)
    K = int(c["K"])
    E = np.stack([c[f"E_{k}"] for k in range(K)], axis=0)
    std = E.std(axis=0, ddof=1 if K > 1 else 0)
    mean = E.mean(axis=0)
    return dict(
        ZZ1=c["ZZ1"], ZZ2=c["ZZ2"], std=std, mean=mean, Z_class=c["Z_class"]
    )


# ---------------------------------------------------------------------------
# Inline matplotlib version — notebook 안에서 바로 플롯 (HTML 없이)
# ---------------------------------------------------------------------------
def plot_trajectory_summary_mpl(
    ensemble_root: Path | str,
    class_c: int,
    *,
    n_samples: int = 500,
    n_steps: int = 50,
    beta: float = 1.0,
    eta: float = 0.05,
    tau: float = 1.0,
    sigma_start: float = 0.1,
    auto_beta: bool = True,
    restart: bool = False,
    kappa_sigma: Optional[float] = None,
    kappa_mu: Optional[float] = None,
    ignore_variance: bool = False,
    seed: int = 0,
    gpu: int = 0,
    n_chains_show: int = 30,
    background: str = "std",
    bg_cmap: str = "magma",
    traj_cmap: str = "turbo",
    real_color: str = "cyan",
    neg_color: str = "red",
    neg_marker: str = "+",
    final_color: str = "0.15",
    figsize: tuple = (9.5, 7.5),
    ax=None,
    title: Optional[str] = None,
    auto_extend_heatmap: bool = True,
    heatmap_base_pad: float = 2.0,
    heatmap_margin: float = 1.0,
    heatmap_h: float = 0.2,
    show_trajectory_lines: bool = True,
    show_trajectory_arrows: bool = True,
    verbose: bool = True,
):
    """SGLD 완료 후 정적 matplotlib 시각화 — notebook inline 전용.

    같은 PCA(2) 공간에:
      1) ensemble std/mean contourf (nb 01 캐시)
      2) real positives (scatter)
      3) surrogate negatives (x marker)
      4) final generated samples (모든 B 체인의 마지막 위치)
      5) n_chains_show 체인의 trajectory 선 + **이동방향 arrow** (x_0 → x_T)

    Returns: (fig, ax, final_samples_np)
    """
    import matplotlib.pyplot as plt

    ensemble_root = Path(ensemble_root)
    class_dir = ensemble_root / f"c{class_c}"

    if verbose:
        print(f"[mpl viz] VP-SGLD on {class_dir}  "
              f"(N={n_samples}, T={n_steps}, β={beta}, η={eta}, τ={tau}, "
              f"ignore_variance={ignore_variance})")
    t0 = time.time()
    _, _, traj = vp_sgld_from_ensemble(
        class_dir,
        n_samples=n_samples, n_steps=n_steps,
        beta=beta, eta=eta, tau=tau,
        sigma_start=sigma_start, auto_beta=auto_beta,
        restart=restart, kappa_sigma=kappa_sigma, kappa_mu=kappa_mu,
        ignore_variance=ignore_variance,
        seed=seed, gpu=gpu,
        return_diagnostics=True, return_trajectory=True,
    )
    traj_np = traj.numpy()     # (T+1, B, d)
    if verbose:
        print(f"  trajectory: {traj_np.shape}  ({time.time()-t0:.1f}s)")

    real = np.load(class_dir / "class_data.npz")["X_class"]
    pca = PCA(n_components=2).fit(real)
    Z_real = pca.transform(real)
    meta = json.loads((class_dir / "meta.json").read_text())
    Z_neg = _load_negatives_pca(class_dir, meta["n_ebms"], pca)

    T_eff, B, d = traj_np.shape
    Z_traj = pca.transform(traj_np.reshape(-1, d)).reshape(T_eff, B, 2)

    # heatmap — 필요 시 neg/trajectory 를 덮는 더 큰 pad 로 자동 재계산
    if auto_extend_heatmap:
        pad = _plan_pad_for_bounds(
            class_dir, pca,
            Z_extra_list=[Z_neg, Z_traj.reshape(-1, 2)],
            base_pad=heatmap_base_pad, margin=heatmap_margin,
        )
        if verbose and pad != heatmap_base_pad:
            print(f"  heatmap pad extended {heatmap_base_pad} → {pad} "
                  f"(to cover negatives / trajectory)")
        hm = _compute_heatmap_for_pad(class_dir, pad=pad, h=heatmap_h, gpu=gpu)
    else:
        hm = load_std_heatmap(class_dir)

    rng = np.random.default_rng(seed)
    show_idx = rng.choice(B, size=min(n_chains_show, B), replace=False)
    show_idx.sort()

    # --- plot ---
    if ax is None:
        fig, ax = plt.subplots(figsize=figsize)
    else:
        fig = ax.figure

    if hm is not None:
        Zb = hm[background]
        cs = ax.contourf(hm["ZZ1"], hm["ZZ2"], Zb, levels=24,
                          cmap=bg_cmap, alpha=0.85)
        cbar = fig.colorbar(cs, ax=ax, shrink=0.82)
        cbar.set_label(f"E_{background}", fontsize=9)

    # real
    ax.scatter(Z_real[:, 0], Z_real[:, 1], s=28, c=real_color,
                edgecolors="black", linewidths=0.5, label="real", alpha=0.85, zorder=4)
    # negatives (red +)
    if len(Z_neg):
        ax.scatter(Z_neg[:, 0], Z_neg[:, 1], s=80, c=neg_color,
                    marker=neg_marker, linewidths=1.8,
                    label=f"neg (∪ K, N={len(Z_neg)})", zorder=5)
    # final samples (all B)
    Z_final = Z_traj[-1, :, :]
    ax.scatter(Z_final[:, 0], Z_final[:, 1], s=14, c=final_color,
                edgecolors="white", linewidths=0.4,
                label=f"final (N={B})", alpha=0.75, zorder=6)

    # trajectory lines + direction arrows (둘 다 독립 토글)
    if show_trajectory_lines or show_trajectory_arrows:
        cmap = plt.get_cmap(traj_cmap)
        for j, i in enumerate(show_idx):
            col = cmap(j / max(1, len(show_idx) - 1))
            x = Z_traj[:, int(i), 0]; y = Z_traj[:, int(i), 1]
            if show_trajectory_lines:
                ax.plot(x, y, color=col, linewidth=1.1, alpha=0.65, zorder=7)
            if show_trajectory_arrows:
                dx = x[-1] - x[0]; dy = y[-1] - y[0]
                if dx * dx + dy * dy > 1e-10:
                    ax.annotate(
                        "", xy=(x[-1], y[-1]), xytext=(x[0], y[0]),
                        arrowprops=dict(arrowstyle="->", color=col, lw=1.2, alpha=0.8,
                                        shrinkA=0, shrinkB=0),
                        zorder=8,
                    )

    ax.set_xlabel("PC1"); ax.set_ylabel("PC2")
    if title is None:
        title = (f"VP-SGLD  ·  class {class_c}  ·  {ensemble_root.name}\n"
                 f"β={beta}, η={eta}, τ={tau}, σ_start={sigma_start}, "
                 f"T={n_steps}, N={n_samples}, ignore_variance={ignore_variance}, "
                 f"chains_shown={len(show_idx)}, "
                 f"negatives={len(Z_neg)}")
    ax.set_title(title, fontsize=10)
    ax.legend(loc="upper right", fontsize=8, framealpha=0.85)
    ax.grid(alpha=0.2)
    return fig, ax, Z_final


# ---------------------------------------------------------------------------
# 논문 figure style C — per-step chain density evolution (multi-panel)
# ---------------------------------------------------------------------------
def plot_trajectory_evolution_mpl(
    traj: np.ndarray,                          # (T+1, N, d) original feature space
    ensemble_root: Path | str,
    class_c: int,
    *,
    steps_to_show: Optional[list[int]] = None,  # default: [0, T/3, 2T/3, T]
    background: str = "std",
    bg_cmap: str = "Greys",                    # 배경은 차분하게
    density_cmap: str = "Blues",               # chain density overlay
    density_alpha: float = 0.75,
    real_color: str = "#00a8c8",               # 조금 진한 cyan
    neg_color: str = "red",
    neg_marker: str = "+",
    figsize: Optional[tuple] = None,
    title: Optional[str] = None,
    auto_extend_heatmap: bool = True,
    heatmap_base_pad: float = 2.0,
    heatmap_margin: float = 1.0,
    heatmap_h: float = 0.2,
    gpu: int = 0,
    density_method: str = "kde",               # 'kde' | 'hist2d'
    density_grid: int = 100,
    density_levels: int = 8,
    show_real: bool = True,
    show_neg: bool = True,
    show_chain_density: bool = True,
    show_chain_scatter: bool = False,
    chain_scatter_size: float = 8.0,
    chain_scatter_alpha: float = 0.35,
    verbose: bool = True,
):
    """논문 figure C: 4-panel per-step chain density evolution.

    각 panel:
      - 배경: ensemble std (또는 mean) contour — 차분한 grayscale/cmap 으로
      - overlay: 해당 step 에서 체인들의 **2D density** (KDE 또는 hist2d) — 짙은 색
      - real positives + surrogate negatives marker 로 anchor

    panel 간 density scale 공통 (fair comparison).

    Toggles:
      - `show_chain_density`: 체인 density contour 끌지 (기본 on)
      - `show_chain_scatter`: 체인 개별 점 scatter 도 함께 (기본 off, density 와 중복)
    """
    import matplotlib.pyplot as plt
    try:
        from scipy.stats import gaussian_kde
    except ImportError:
        if density_method == "kde":
            raise RuntimeError("scipy 필요: `pip install scipy`")

    traj = np.asarray(traj)
    T_eff, B, d = traj.shape
    if steps_to_show is None:
        steps_to_show = [0, (T_eff - 1) // 3, 2 * (T_eff - 1) // 3, T_eff - 1]
    steps_to_show = [int(s) for s in steps_to_show]
    assert all(0 <= s < T_eff for s in steps_to_show), \
        f"step 범위 [0, {T_eff-1}] 벗어남: {steps_to_show}"

    class_dir = Path(ensemble_root) / f"c{class_c}"
    real = np.load(class_dir / "class_data.npz")["X_class"]
    pca = PCA(n_components=2).fit(real)
    Z_real = pca.transform(real)
    meta = json.loads((class_dir / "meta.json").read_text())
    Z_neg = _load_negatives_pca(class_dir, meta["n_ebms"], pca)
    Z_traj = pca.transform(traj.reshape(-1, d)).reshape(T_eff, B, 2)

    if auto_extend_heatmap:
        pad = _plan_pad_for_bounds(
            class_dir, pca,
            Z_extra_list=[Z_neg, Z_traj.reshape(-1, 2)],
            base_pad=heatmap_base_pad, margin=heatmap_margin,
        )
        hm = _compute_heatmap_for_pad(class_dir, pad=pad, h=heatmap_h, gpu=gpu)
    else:
        hm = load_std_heatmap(class_dir)

    if hm is not None:
        xlim = (float(hm["ZZ1"].min()), float(hm["ZZ1"].max()))
        ylim = (float(hm["ZZ2"].min()), float(hm["ZZ2"].max()))
    else:
        all_z = np.vstack([Z_real, Z_neg, Z_traj.reshape(-1, 2)])
        xlim = (float(all_z[:, 0].min() - 0.5), float(all_z[:, 0].max() + 0.5))
        ylim = (float(all_z[:, 1].min() - 0.5), float(all_z[:, 1].max() + 0.5))

    # chain density grid (panel 공통)
    gx = np.linspace(xlim[0], xlim[1], density_grid)
    gy = np.linspace(ylim[0], ylim[1], density_grid)
    GX, GY = np.meshgrid(gx, gy)

    densities = []
    for t in steps_to_show:
        pts = Z_traj[t, :, :].T   # (2, N)
        if density_method == "kde":
            try:
                kde = gaussian_kde(pts)
                Z_dens = kde(np.vstack([GX.ravel(), GY.ravel()])).reshape(GX.shape)
            except np.linalg.LinAlgError:
                # KDE 실패 (체인이 한 점으로 collapse) — hist2d 폴백
                H, _, _ = np.histogram2d(pts[0], pts[1], bins=[gx, gy], density=True)
                Z_dens = np.zeros_like(GX)
                Z_dens[:-1, :-1] = H.T
        else:   # hist2d
            H, _, _ = np.histogram2d(pts[0], pts[1], bins=[gx, gy], density=True)
            Z_dens = np.zeros_like(GX)
            Z_dens[:-1, :-1] = H.T
        densities.append(Z_dens)

    dens_vmax = max(d.max() for d in densities)
    dens_vmin = 0.0

    n_panels = len(steps_to_show)
    if figsize is None:
        figsize = (3.3 * n_panels, 3.6)
    fig, axes = plt.subplots(1, n_panels, figsize=figsize,
                              sharex=True, sharey=True)
    if n_panels == 1:
        axes = [axes]

    cs_d = None
    for i, (ax, t, dens) in enumerate(zip(axes, steps_to_show, densities)):
        if hm is not None:
            ax.contourf(hm["ZZ1"], hm["ZZ2"], hm[background],
                         levels=18, cmap=bg_cmap, alpha=0.9)
        if show_chain_density:
            cs_d = ax.contourf(GX, GY, dens, levels=density_levels,
                                cmap=density_cmap, alpha=density_alpha,
                                vmin=dens_vmin, vmax=dens_vmax)
        if show_chain_scatter:
            ax.scatter(Z_traj[t, :, 0], Z_traj[t, :, 1],
                        s=chain_scatter_size, c="#1f4e79",
                        alpha=chain_scatter_alpha,
                        edgecolors="white", linewidths=0.2, zorder=3)
        if show_real:
            ax.scatter(Z_real[:, 0], Z_real[:, 1], s=18, c=real_color,
                        edgecolors="black", linewidths=0.4, zorder=4)
        if show_neg and len(Z_neg):
            ax.scatter(Z_neg[:, 0], Z_neg[:, 1], s=55, c=neg_color,
                        marker=neg_marker, linewidths=1.5, zorder=5)
        ax.set_xlim(xlim); ax.set_ylim(ylim)
        ax.set_title(f"step {t}  /  T={T_eff-1}", fontsize=11)
        ax.set_xlabel("PC1", fontsize=9)
        if i == 0:
            ax.set_ylabel("PC2", fontsize=9)
        ax.tick_params(labelsize=8)
        ax.grid(alpha=0.15, linewidth=0.3)

    # density colorbar (density 꺼져 있으면 생략)
    if cs_d is not None:
        cbar = fig.colorbar(cs_d, ax=axes, shrink=0.85, pad=0.02, aspect=25)
        cbar.set_label(f"chain density (N={B}, same scale across panels)", fontsize=8)

    if title:
        fig.suptitle(title, fontsize=11, y=1.03)
    return fig, axes


# ---------------------------------------------------------------------------
# Inline matplotlib — 저장된 trajectory npz 에서 바로 플롯 (SGLD 재실행 없이)
# ---------------------------------------------------------------------------
def plot_trajectory_from_saved_mpl(
    samples_npz_path: Path | str,
    ensemble_root: Path | str,
    class_c: int,
    *,
    n_chains_show: int = 30,
    background: str = "std",
    bg_cmap: str = "magma",
    traj_cmap: str = "turbo",
    real_color: str = "cyan",
    neg_color: str = "red",
    neg_marker: str = "+",
    final_color: str = "0.15",
    figsize: tuple = (9.5, 7.5),
    ax=None,
    title: Optional[str] = None,
    auto_extend_heatmap: bool = True,
    heatmap_base_pad: float = 2.0,
    heatmap_margin: float = 1.0,
    heatmap_h: float = 0.2,
    show_trajectory_lines: bool = True,
    show_trajectory_arrows: bool = True,
    seed: int = 0,
    gpu: int = 0,
    verbose: bool = True,
):
    """저장된 `samples/<tag>.npz` 의 `traj_c{class_c}` 를 읽어 SGLD 재실행 없이 시각화.

    nb 02 의 `samples/{tag}.npz` (trajectory 항상 포함) 를 입력으로 받음. npz 에
    `traj_c{c}` 가 없으면 (레거시 run) `X_c{c}` 만 있는
    final 시각화로 폴백.
    """
    import matplotlib.pyplot as plt

    samples_npz_path = Path(samples_npz_path)
    ensemble_root = Path(ensemble_root)
    class_dir = ensemble_root / f"c{class_c}"
    data = np.load(samples_npz_path, allow_pickle=True)
    traj_key = f"traj_c{class_c}"
    fallback = traj_key not in data.files
    if fallback:
        if verbose:
            print(f"  [warn] '{traj_key}' not in {samples_npz_path.name} — "
                  f"trajectory 없음, final 만 플롯")
        traj_np = data[f"X_c{class_c}"][None, :, :]    # (1, N, d)
    else:
        traj_np = data[traj_key]                         # (T+1, N, d)
    if verbose:
        print(f"[from saved] {samples_npz_path.name}  class={class_c}  "
              f"traj shape={traj_np.shape}  (fallback={fallback})")

    real = np.load(class_dir / "class_data.npz")["X_class"]
    pca = PCA(n_components=2).fit(real)
    Z_real = pca.transform(real)
    meta = json.loads((class_dir / "meta.json").read_text())
    Z_neg = _load_negatives_pca(class_dir, meta["n_ebms"], pca)

    T_eff, B, d = traj_np.shape
    Z_traj = pca.transform(traj_np.reshape(-1, d)).reshape(T_eff, B, 2)

    if auto_extend_heatmap:
        pad = _plan_pad_for_bounds(
            class_dir, pca,
            Z_extra_list=[Z_neg, Z_traj.reshape(-1, 2)],
            base_pad=heatmap_base_pad, margin=heatmap_margin,
        )
        if verbose and pad != heatmap_base_pad:
            print(f"  heatmap pad extended {heatmap_base_pad} → {pad}")
        hm = _compute_heatmap_for_pad(class_dir, pad=pad, h=heatmap_h, gpu=gpu)
    else:
        hm = load_std_heatmap(class_dir)

    rng = np.random.default_rng(seed)
    show_idx = rng.choice(B, size=min(n_chains_show, B), replace=False)
    show_idx.sort()

    if ax is None:
        fig, ax = plt.subplots(figsize=figsize)
    else:
        fig = ax.figure

    if hm is not None:
        Zb = hm[background]
        cs = ax.contourf(hm["ZZ1"], hm["ZZ2"], Zb, levels=24,
                          cmap=bg_cmap, alpha=0.85)
        cbar = fig.colorbar(cs, ax=ax, shrink=0.82)
        cbar.set_label(f"E_{background}", fontsize=9)

    ax.scatter(Z_real[:, 0], Z_real[:, 1], s=28, c=real_color,
                edgecolors="black", linewidths=0.5, label="real", alpha=0.85, zorder=4)
    if len(Z_neg):
        ax.scatter(Z_neg[:, 0], Z_neg[:, 1], s=80, c=neg_color,
                    marker=neg_marker, linewidths=1.8,
                    label=f"neg (∪ K, N={len(Z_neg)})", zorder=5)
    Z_final = Z_traj[-1, :, :]
    ax.scatter(Z_final[:, 0], Z_final[:, 1], s=14, c=final_color,
                edgecolors="white", linewidths=0.4,
                label=f"final (N={B})", alpha=0.75, zorder=6)

    if (show_trajectory_lines or show_trajectory_arrows) and T_eff > 1:
        cmap = plt.get_cmap(traj_cmap)
        for j, i in enumerate(show_idx):
            col = cmap(j / max(1, len(show_idx) - 1))
            x = Z_traj[:, int(i), 0]; y = Z_traj[:, int(i), 1]
            if show_trajectory_lines:
                ax.plot(x, y, color=col, linewidth=1.1, alpha=0.65, zorder=7)
            if show_trajectory_arrows:
                dx = x[-1] - x[0]; dy = y[-1] - y[0]
                if dx * dx + dy * dy > 1e-10:
                    ax.annotate(
                        "", xy=(x[-1], y[-1]), xytext=(x[0], y[0]),
                        arrowprops=dict(arrowstyle="->", color=col, lw=1.2, alpha=0.8,
                                        shrinkA=0, shrinkB=0),
                        zorder=8,
                    )

    ax.set_xlabel("PC1"); ax.set_ylabel("PC2")
    if title is None:
        title = (f"[from saved] class {class_c}  ·  {samples_npz_path.name}\n"
                 f"T+1={T_eff}, N={B}, chains_shown={len(show_idx)}, negatives={len(Z_neg)}")
    ax.set_title(title, fontsize=10)
    ax.legend(loc="upper right", fontsize=8, framealpha=0.85)
    ax.grid(alpha=0.2)
    return fig, ax, Z_final


