"""
Variance-Preconditioned SGLD (VP-SGLD) — paper §3.3 update with Monte Carlo
approximation of the score-PFN posterior moments (paper §2.2, §3.2).

Update rule (§3.3):

    x_{t+1} = x_t + η_t · M(x_t) · μ(x_t)
              + sqrt(2 η_t τ_0) · M(x_t)^{1/2} · ξ_t,    ξ_t ~ N(0, I)
    M(x)   = (I + β · diag(Σ(x)))^{-1}                   ← diagonal preconditioner

Interpretation: an ensemble of K TabEBMs is viewed as K approximate samples
from the posterior p(ϕ | D_c, c) over class-conditional generators. Each
member's score `s_k(x) = -∇E_k(x)` approximates `∇log p_{ϕ_k}(x | c)`. Under
this view the Monte Carlo estimators

    μ̂(x) = (1/K) Σ_k s_k(x)
    Σ̂(x) = (1/(K-1)) Σ_k (s_k(x) - μ̂(x))²    ← Bessel-corrected

are **unbiased estimators** of the paper's score-PFN outputs

    μ_θ(x, D_c, c) = E_{ϕ | D_c, c}[g_ϕ(x)]
    Σ_θ(x, D_c, c) = Var_{ϕ | D_c, c}[g_ϕ(x)]                  (per-coord diag)

so running VP-SGLD with (μ̂, Σ̂) plugged in is a Monte Carlo approximation of
what the paper's trained score-PFN would compute — no new PFN pretrain
needed. Ensemble members stay separate; μ̂ and Σ̂ are approximations of
integrals over ϕ, not a "combined EBM".

API:
    score_var_fn(x) → (μ, Σ)     : user-supplied callable
    ensemble_score_var_fn(dir)   : builder that constructs one from a saved
                                   ensemble (distance/subsample v2 etc.)
    vp_sgld_sample(...)           : the sampler

Per-class semantics: one sampler run per class, backed by that class's
ensemble (e.g. stock_subsample_v2 for class 0, stock_subsample_v2_c1 for
class 1).
"""
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Optional

import numpy as np
import torch

from tabebm.TabEBM import TabEBM


ScoreVarFn = Callable[[torch.Tensor], tuple[torch.Tensor, torch.Tensor]]
"""A callable x of shape (B, d) → (μ: (B, d), Σ: (B, d) diagonal variance)."""


# ---------------------------------------------------------------------------
# Config + core update
# ---------------------------------------------------------------------------
@dataclass
class VPSGLDConfig:
    eta: float = 0.05                   # step size η_t (constant)
    beta: float = 1.0                   # preconditioner strength
    tau: float = 1.0                    # noise temperature τ_0
    sigma_start: float = 0.1            # init perturbation around real seeds
    eps_var: float = 1e-8               # Σ floor (K members agreeing → Σ=0)
    kappa_sigma: Optional[float] = None  # §3.3 restart: tr(Σ) threshold
    kappa_mu: Optional[float] = None     # §3.3 restart: μᵀMμ threshold
    ignore_variance: bool = False        # True → M=I (variance 무시, plain-SGLD-like)


def vp_sgld_step(
    x: torch.Tensor,
    score_var_fn: ScoreVarFn,
    cfg: VPSGLDConfig,
) -> tuple[torch.Tensor, dict]:
    """One VP-SGLD update on a batch x of shape (B, d). Returns (x_new, diag).

    ignore_variance=True → M = ones (preconditioner off) — update 식이
        x_{t+1} = x_t + η · μ + sqrt(2 η τ) · ξ
    로 축소돼 β 의 영향 사라짐. 단, μ/Σ 는 여전히 계산돼 diagnostic 에 들어감.
    """
    mu, var = score_var_fn(x)
    var = var.clamp_min(cfg.eps_var)
    if cfg.ignore_variance:
        M = torch.ones_like(var)
    else:
        M = 1.0 / (1.0 + cfg.beta * var)
    
    noise = torch.randn_like(x)
    drift_term = cfg.eta * M * mu
    noise_term = torch.sqrt(2.0 * cfg.eta * cfg.tau * M) * noise
    x_new = x + drift_term + noise_term

    drift_mag = drift_term.norm(dim=1).mean().item()
    noise_mag = noise_term.norm(dim=1).mean().item()
    diag = {
        # ensemble (μ, Σ) & preconditioner
        "score_norm": mu.norm(dim=1).mean().item(),      # ||μ(x)||
        "var_mean":   var.mean().item(),                 # E[Σ] over (B, d)
        "var_median": var.median().item(),               # median Σ
        "var_max":    var.max().item(),
        "M_mean":     M.mean().item(),                   # E[M] over (B, d)
        "M_min":      M.min().item(),
        "M_max":      M.max().item(),
        # 실제 update 에 기여하는 텀 크기
        "drift_norm":           drift_mag,                 # ||η M μ||_mean
        "noise_norm":           noise_mag,                 # ||√(2ητM) ξ||_mean
        "drift_over_noise":     drift_mag / max(noise_mag, 1e-12),  # drift-dominated vs noise-dominated
        # chain state 자체
        "x_mean_norm":          x.norm(dim=1).mean().item(),        # 현재 체인들의 평균 ||x||
        "x_new_mean_norm":      x_new.norm(dim=1).mean().item(),    # 다음 step 의 평균 ||x||
    }
    if cfg.kappa_sigma is not None and cfg.kappa_mu is not None:
        tr_sigma = var.sum(dim=1)
        muMmu = (mu * M * mu).sum(dim=1)
        diag["restart_mask"] = (tr_sigma > cfg.kappa_sigma) & (muMmu < cfg.kappa_mu)
    else:
        diag["restart_mask"] = torch.zeros(x.shape[0], dtype=torch.bool, device=x.device)
    return x_new, diag


def vp_sgld_sample(
    x_init: torch.Tensor,
    score_var_fn: ScoreVarFn,
    n_steps: int,
    cfg: Optional[VPSGLDConfig] = None,
    real_seeds: Optional[torch.Tensor] = None,
    return_diagnostics: bool = False,
    return_trajectory: bool = False,
):
    """Run n_steps of VP-SGLD starting from x_init (shape (B, d)).

    If `return_trajectory=True`, also collect x at every step including the
    initial state, yielding a `(n_steps+1, B, d)` tensor on CPU (fp32).
    """
    cfg = cfg or VPSGLDConfig()
    x = x_init
    diagnostics: list[dict] = []
    traj: list[torch.Tensor] = [x.detach().cpu()] if return_trajectory else []
    for t in range(n_steps):
        x, diag = vp_sgld_step(x, score_var_fn, cfg)
        mask = diag["restart_mask"]
        if real_seeds is not None and mask.any():
            n_r = int(mask.sum())
            idx = torch.randint(0, len(real_seeds), (n_r,), device=x.device)
            fresh = real_seeds[idx] + cfg.sigma_start * torch.randn(
                n_r, x.shape[1], device=x.device
            )
            x = x.clone()
            x[mask] = fresh
            diag["n_restarted"] = n_r
        if return_diagnostics:
            diag["step"] = t
            diagnostics.append(diag)
        if return_trajectory:
            traj.append(x.detach().cpu())

    if return_trajectory:
        traj_tensor = torch.stack(traj, dim=0)  # (T+1, B, d)
        if return_diagnostics:
            return x, diagnostics, traj_tensor
        return x, traj_tensor
    return (x, diagnostics) if return_diagnostics else x


def init_from_real(
    real_data: torch.Tensor,
    n_samples: int,
    sigma_start: float = 0.1,
    seed: Optional[int] = None,
) -> torch.Tensor:
    """§3.3 init: x_0 = x_real + ε,  ε ~ N(0, sigma_start² I)."""
    if seed is not None:
        torch.manual_seed(seed)
    idx = torch.randint(0, len(real_data), (n_samples,), device=real_data.device)
    return real_data[idx] + sigma_start * torch.randn(
        n_samples, real_data.shape[1], device=real_data.device
    )


# ---------------------------------------------------------------------------
# Ensemble → score_var_fn (MC estimator of score-PFN moments)
# ---------------------------------------------------------------------------
def _member_score(tabebm: TabEBM, x: torch.Tensor) -> torch.Tensor:
    """s_k(x) = -∇E_k(x) on a batch. Energy = -logsumexp(logits). Autograd.

    TabPFN 내부 gradient checkpointing 을 임시 비활성화해서
    작은 데이터 (per-split fit 30 samples 등) 에서의 CheckpointError 방지.
    """
    import torch.utils.checkpoint as _ckpt
    _orig = _ckpt.checkpoint
    _ckpt.checkpoint = lambda fn, *a, use_reentrant=True, **kw: fn(*a)
    try:
        x_in = x.detach().clone().requires_grad_(True)
        x_3d = x_in.unsqueeze(0)
        logits = tabebm.model.forward([x_3d], return_logits=True)
        if logits.dim() == 3:
            logits = logits.squeeze(0)
        if logits.shape[0] == 2 and logits.shape[1] != 2:
            logits = logits.T
        energy = -torch.logsumexp(logits, dim=1)
        total = energy.sum() / x_in.shape[0]
        grad = torch.autograd.grad(total, x_in)[0]
        return -grad.detach()
    finally:
        _ckpt.checkpoint = _orig


def load_ensemble_members(ensemble_dir: str | Path, gpu: int = 0) -> tuple[list[TabEBM], dict]:
    """Rebuild all K members of a saved ensemble. Returns (members, meta)."""
    try:
        from experiments.ensemble_ebm import rebuild_ebm
    except ModuleNotFoundError:
        from ensemble_ebm import rebuild_ebm

    ensemble_dir = Path(ensemble_dir)
    meta = json.loads((ensemble_dir / "meta.json").read_text())
    members = [rebuild_ebm(ensemble_dir / f"ebm_{k}", gpu=gpu)[0] for k in range(meta["n_ebms"])]
    return members, meta


# Module-level cache — (class_dir abspath, gpu) → (score_var_fn, K)
# 동일 ensemble 에 대한 반복 호출 (sweep / viz) 시 K TabPFN state_dict 재로드 방지.
_ENSEMBLE_SVF_CACHE: dict = {}


def clear_ensemble_cache() -> None:
    """GPU 메모리 회수하려면 호출. 파이썬 종료 시엔 자동."""
    global _ENSEMBLE_SVF_CACHE
    _ENSEMBLE_SVF_CACHE = {}


def ensemble_score_var_fn(
    ensemble_dir: str | Path, gpu: int = 0, use_cache: bool = True,
) -> ScoreVarFn:
    """Build score_var_fn that MC-estimates the score-PFN posterior moments.

    For each member k (≈ sample from p(ϕ | D_c, c)):
        s_k(x) = -∇E_k(x)   via autograd on TabPFN energy

    Returns (μ̂, Σ̂) with:
        μ̂(x) = mean_k  s_k(x)
        Σ̂(x) = var_k   s_k(x)   (unbiased, Bessel-corrected)

    Both are unbiased Monte Carlo estimators of the paper's (μ_θ, Σ_θ) under
    the deep-ensembles-approximate-posterior view.

    `use_cache=True` (default) 이면 `(class_dir, gpu)` 조합당 한 번만 K 멤버
    를 GPU 에 로드하고 이후 호출은 같은 함수 객체를 돌려줌 — sweep/viz 같은
    다회 호출 시나리오에서 필수적.
    """
    key = (str(Path(ensemble_dir).resolve()), int(gpu))
    if use_cache and key in _ENSEMBLE_SVF_CACHE:
        return _ENSEMBLE_SVF_CACHE[key]

    members, _ = load_ensemble_members(ensemble_dir, gpu=gpu)
    K = len(members)
    if K < 2:
        raise ValueError(f"Ensemble at {ensemble_dir} has K={K}; need K≥2 for variance estimate.")

    def fn(x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        scores = torch.stack([_member_score(m, x) for m in members], dim=0)  # (K, B, d)
        return scores.mean(dim=0), scores.var(dim=0, unbiased=True)

    if use_cache:
        _ENSEMBLE_SVF_CACHE[key] = fn
    return fn


def compute_beta_scale(score_var_fn: ScoreVarFn, ref_points: torch.Tensor) -> float:
    """β auto-scale: returns 1 / median(Σ(ref)) so β=1 means 'M=I/2 at median Σ'.

    Use as: `cfg.beta = user_beta * compute_beta_scale(fn, real_class_data)`.
    Makes β dimensionless and transferable across datasets / ensemble sizes.
    """
    _, var = score_var_fn(ref_points)
    med = var.detach().median().item()
    return 1.0 / max(med, 1e-12)


def vp_sgld_from_ensemble(
    ensemble_dir: str | Path,
    n_samples: int,
    n_steps: int = 50,
    beta: float = 1.0,
    eta: float = 0.05,
    tau: float = 1.0,
    sigma_start: float = 0.1,
    auto_beta: bool = True,
    restart: bool = False,
    kappa_sigma: Optional[float] = None,
    kappa_mu: Optional[float] = None,
    seed: int = 0,
    gpu: int = 0,
    return_diagnostics: bool = False,
    return_trajectory: bool = False,
    ignore_variance: bool = False,
):
    """Convenience: build ensemble MC score_var_fn + init from real + sample.

    Args:
        ensemble_dir: e.g. `experiments/ebms/stock_subsample_v2` (class-specific).
        auto_beta: if True, β is scaled by 1/median(Σ(real)) so that the
                   supplied `beta` is in dimensionless units (β=1 → M=I/2 at
                   median Σ across real class data).
        restart:   enable §3.3 restart rule using (kappa_sigma, kappa_mu).
        return_trajectory: if True, also return the per-step states as a
                   `(n_steps+1, n_samples, d)` CPU tensor.

    Returns (samples, [diagnostics], [trajectory]) depending on flags.
    """
    score_var_fn = ensemble_score_var_fn(ensemble_dir, gpu=gpu)

    X_real_np = np.load(Path(ensemble_dir) / "class_data.npz")["X_class"]
    real_tensor = torch.from_numpy(X_real_np).float().to(f"cuda:{gpu}")

    if auto_beta:
        beta = beta * compute_beta_scale(score_var_fn, real_tensor)

    cfg = VPSGLDConfig(
        eta=eta, beta=beta, tau=tau, sigma_start=sigma_start,
        kappa_sigma=kappa_sigma if restart else None,
        kappa_mu=kappa_mu if restart else None,
        ignore_variance=ignore_variance,
    )

    torch.manual_seed(seed)
    x_init = init_from_real(real_tensor, n_samples=n_samples, sigma_start=sigma_start, seed=seed)

    seeds_for_restart = real_tensor if restart else None
    result = vp_sgld_sample(
        x_init, score_var_fn, n_steps=n_steps, cfg=cfg,
        real_seeds=seeds_for_restart,
        return_diagnostics=return_diagnostics,
        return_trajectory=return_trajectory,
    )
    if return_diagnostics and return_trajectory:
        samples, diagnostics, traj = result
        return samples.detach().cpu(), diagnostics, traj
    if return_trajectory:
        samples, traj = result
        return samples.detach().cpu(), traj
    if return_diagnostics:
        samples, diagnostics = result
        return samples.detach().cpu(), diagnostics
    return result.detach().cpu()
