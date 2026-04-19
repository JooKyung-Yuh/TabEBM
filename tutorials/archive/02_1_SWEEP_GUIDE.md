# VP-SGLD Variance Preconditioner Sweep — 실행 가이드

VP-SGLD 의 `M = (I + β·Σ)⁻¹` 가 **유효한 regime 를 찾기 위한** 3 가지 sweep 실험.
각 실험은 **(a) ipynb 편집** 또는 **(b) CLI 스크립트** 둘 다로 가능.

---

## 0. 사전 점검 — 어느 ensemble 을 쓸 것인가

| ensemble | 멤버 다양성 | 적합도 |
|---|---|---|
| `20260415_214026_Distance_EBM` | α=2.0 동일 (NO diversity) | ❌ Σ 가 fake (FP 노이즈) — sweep 의미 없음 |
| `20260415_210238_Subsample-Distance_EBM` | Subsample 25% + α∈[8.9, 24.9] | ✅ 넓은 sweep, 추천 |
| `20260415_205627_Subsample-Distance_EBM` | Subsample + α∈[1, 10] | ✅ 좁은 α |
| `20260415_200422_Distance_EBM` | α∈[3.73, 29.3], no Subsample | △ Subsample 없어 source 단일 |

**기본 선택**: `20260415_210238_Subsample-Distance_EBM`

확인:
```bash
ls /home/work/JooKyung/TabEBM/experiments/ebms/20260415_210238_Subsample-Distance_EBM/c0
```

만약 새 ensemble 학습이 필요하면:
```bash
cd /home/work/JooKyung/TabEBM
/home/work/miniconda3/envs/TabEBM/bin/python experiments/fit_ensemble_v2.py \
  --dataset stock --n_real 100 --n_ebms 10 \
  --methods Subsample Distance \
  --classes 0 1 --seed 42 \
  --method_params '{"Subsample": {"ratio": 0.25}, "Distance": {"mode": "sweep", "dist_range": [1.0, 10.0]}}'
```

---

## 실험 1 — ensemble 교체만으로 즉시 변화 확인

**목적**: 동일 하이퍼파라미터로 진짜 diversity 있는 ensemble 에서 `ignore_variance` 토글이 의미있게 갈리는지 확인.

### (a) ipynb 방식

`tutorials/02_1_variance_ablation.ipynb` 열고 **cell 2 (setup)** 의 `ARGS_STR` 편집:

```python
ARGS_STR = '''
    --ensemble-root experiments/ebms/20260415_210238_Subsample-Distance_EBM   # ← 주석 풀고 경로 수정

    --classes 0 1
    --n-low 10
    --n-high 500
    --n-steps 50
    --beta 1.0 --eta 0.05 --tau 1.0 --sigma-start 0.1
    --auto-beta
    --seed 0 --gpu 0
'''
```

→ 변경한 줄: `--ensemble-root` 한 줄 (주석 `#` 제거 + 경로).

**실행**: cell 2 → cell 4 (SGLD 4 runs) → cell 6 (M envelope plot) → cell 10 (요약 표).

**예상 시간**: ~6 분 (4 runs × 2 class × ~44s)

### (b) CLI 방식

```bash
cd /home/work/JooKyung/TabEBM
/home/work/miniconda3/envs/TabEBM/bin/python experiments/compare_ignore_variance.py \
  --ensemble-root experiments/ebms/20260415_210238_Subsample-Distance_EBM \
  --classes 0 1 \
  --n-low 10 --n-high 500 --n-steps 50 \
  --beta 1.0 --eta 0.05 --tau 1.0 --sigma-start 0.1 \
  --gpu 0
```

→ 결과: `experiments/ebms/20260415_210238_Subsample-Distance_EBM/comparisons/{ts}_ignore_variance_ablation/`

### 무엇을 확인할 것인가

- `summary_M_analysis.csv` 의 **`M_mean_init`** 열 — `ignore_variance=False` 행에서 0.5 미만이면 Σ 가 진짜 작동 중
- `M_envelope_c0.png` — 파랑(ignore=False) 띠가 1 에서 의미있게 떨어져 있어야 함
- `ignore on/off` final sample L2 차이가 이전 (~2.15) 보다 **10 배 이상** 커지면 성공

---

## 실험 2 — β sweep (auto_beta OFF, 수동 β 폭 변화)

**목적**: $\beta$ 가 M 분포에 미치는 영향을 직접 측정. auto_beta 의 median 정합 한계를 분리.

### (a) ipynb 방식

cell 2 의 `ARGS_STR`:

```python
ARGS_STR = '''
    --ensemble-root experiments/ebms/20260415_210238_Subsample-Distance_EBM

    --classes 0 1
    --n-low 10
    --n-high 500
    --n-steps 50
    --beta 1.0 --eta 0.05 --tau 1.0 --sigma-start 0.1
    --no-auto-beta             # ← auto-beta 끄기 (manual β 사용)
    --seed 0 --gpu 0
'''
```

cell 4 (SGLD loop) 위에 다음 **새 셀 추가**:

```python
# β manual sweep — auto_beta 꺼져있고 --beta 가 그대로 사용됨
BETA_SWEEP = [1e-2, 1.0, 1e2, 1e4, 1e6]

beta_results = {c: {} for c in args.classes}
for beta_val in BETA_SWEEP:
    args.beta = beta_val   # in-place override
    print(f'\n========== β = {beta_val:.1e} ==========')
    for c in args.classes:
        class_dir = args.ensemble_root / f'c{c}'
        for ig in [False, True]:
            arr, keys, _ = run_one(class_dir, n_samples=args.n_high,
                                     ignore_variance=ig, save_trajectory=False)
            beta_results[c].setdefault(ig, {})[beta_val] = arr

# plot: x=β (log), y=M_mean(t=T), 곡선 = ignore on/off
fig, axes = plt.subplots(1, len(args.classes), figsize=(7*len(args.classes), 4))
axes = np.atleast_1d(axes)
for ax, c in zip(axes, args.classes):
    keys = list(beta_results[c][False].values())[0]
    # ... idx_M = keys index lookup, 위 cell 4 패턴과 동일
    for ig in [False, True]:
        Ms = [beta_results[c][ig][b][-1, 4] for b in BETA_SWEEP]   # M_mean is col 4
        ax.plot(BETA_SWEEP, Ms, marker='o', label=f'ignore={ig}')
    ax.set_xscale('log'); ax.set_xlabel('β'); ax.set_ylabel('M_mean (final)')
    ax.set_title(f'class {c}'); ax.legend(); ax.grid(alpha=0.3)
plt.tight_layout()
fig.savefig(OUT_DIR / 'beta_sweep.png', dpi=140, bbox_inches='tight')
```

### (b) CLI 방식 — bash loop

```bash
cd /home/work/JooKyung/TabEBM
ENS=experiments/ebms/20260415_210238_Subsample-Distance_EBM

for BETA in 0.01 1.0 100.0 10000.0 1000000.0; do
    /home/work/miniconda3/envs/TabEBM/bin/python experiments/compare_ignore_variance.py \
      --ensemble-root $ENS \
      --classes 0 \
      --n-low 10 --n-high 500 --n-steps 50 \
      --beta $BETA --eta 0.05 --tau 1.0 --sigma-start 0.1 \
      --gpu 0
done
```

→ 각 β 마다 새 `comparisons/{ts}_ignore_variance_ablation/` 폴더 생성. β 별 결과 비교는 후처리:

```bash
for d in $ENS/comparisons/*_ignore_variance_ablation/; do
    BETA=$(jq -r .beta $d/run_config.json)
    M=$(awk -F, 'NR==2 {print $5}' $d/diag_raw_c0.npz 2>/dev/null || echo "load npz")
    echo "β=$BETA → $d"
done
```

### 변경하는 값과 의미

| 인자 | 의미 | sweep 범위 권장 |
|---|---|---|
| `--beta` | preconditioner 강도. 클수록 M 이 Σ 에 민감 | `[0.01, 1, 100, 1e4, 1e6]` (5 점) |
| `--no-auto-beta` | manual β 활성화 (필수) | flag |
| `--auto-beta` | β 를 1/median(Σ) 로 스케일 (실험 2 에선 OFF) | flag |

### 예상 결과

- β=0.01: M ≈ 1 (preconditioner off) — ignore 토글 무의미
- β=1: 약간 M < 1
- β=100~1e4: 본격적 preconditioner — ignore 토글 효과 가시화
- β=1e6: M 대부분 0 — 체인 거의 안 움직임 (변동 없는 init 분포 유지)

**최적 β** 는 `M_mean ≈ 0.3 ~ 0.7` 인 지점 — 너무 0 면 movement 죽고, 너무 1 이면 preconditioner 무의미.

---

## 실험 3 — σ_start sweep (초기 위치 범위 확장)

**목적**: 체인이 high-Σ 영역을 통과하도록 강제. preconditioner 가 능동적으로 작동하는지 직접 관찰.

### (a) ipynb 방식

cell 2 의 `ARGS_STR` 에서 `--sigma-start` 한 줄만 변경하며 cell 4 → cell 6 반복.

또는 cell 4 위에 새 sweep 셀 (실험 2 와 동일 패턴):

```python
SIGMA_SWEEP = [0.1, 0.5, 2.0, 5.0]

sigma_results = {c: {} for c in args.classes}
for sigma_val in SIGMA_SWEEP:
    args.sigma_start = sigma_val
    print(f'\n========== σ_start = {sigma_val} ==========')
    for c in args.classes:
        class_dir = args.ensemble_root / f'c{c}'
        for ig in [False, True]:
            arr, keys, traj = run_one(class_dir, n_samples=args.n_high,
                                         ignore_variance=ig, save_trajectory=True)
            sigma_results[c].setdefault(ig, {})[sigma_val] = (arr, traj)
```

### (b) CLI 방식

```bash
ENS=experiments/ebms/20260415_210238_Subsample-Distance_EBM

for SIG in 0.1 0.5 2.0 5.0; do
    /home/work/miniconda3/envs/TabEBM/bin/python experiments/compare_ignore_variance.py \
      --ensemble-root $ENS \
      --classes 0 \
      --n-low 10 --n-high 500 --n-steps 50 \
      --beta 1.0 --eta 0.05 --tau 1.0 --sigma-start $SIG \
      --auto-beta --gpu 0
done
```

### 확인 포인트

- `M_envelope_c0.png` 의 **시간에 따른 M 변화**: σ_start 클수록 초기에 M 이 더 낮게 시작해야 함 (체인이 hot spot 가까이 init)
- σ_start 가 클수록 체인이 random 위치에서 시작해 ensemble 이 disagree 하는 영역을 더 잘 cover
- `drift_over_noise` 가 σ_start 에 따라 어떻게 변하는지 (초기에는 noise 가 큼)

---

## 결과 확인 위치 (자동 생성)

```
{ENSEMBLE_ROOT}/
└── comparisons/
    └── {ts}_M_term_analysis/                  # nb 02.1 결과
        ├── run_config.json
        ├── diag_raw_c0.npz                    # T×n_keys 매 step diag (M, Σ, drift, noise, ratio, ...)
        ├── diag_raw_c1.npz
        ├── M_envelope_c0.png                   # M_mean ± [min, max] 띠
        ├── M_envelope_c1.png
        ├── full_dashboard_c0.png               # Σ + drift + noise + ratio 6-panel
        ├── full_dashboard_c1.png
        ├── summary_M_analysis.csv
        ├── M_vs_N_scan.csv                     # cell 12 결과 (선택)
        └── M_vs_N_scan.png

    └── {ts}_ignore_variance_ablation/         # CLI compare 스크립트 결과
        ├── run_config.json
        ├── diag_raw_c0.npz
        └── ablation_M_comparison_c0.png        # 3 metrics × 2 N 패널
```

각 npz 의 `diag_raw_c{c}.npz` keys:
- `diag_cols`: 컬럼 이름 (`step, M_mean, M_min, M_max, var_*, drift_norm, noise_norm, drift_over_noise, x_mean_norm, x_new_mean_norm`)
- `ignore0_N10`, `ignore1_N10`, `ignore0_N500`, `ignore1_N500`: 각 (T, n_keys) float
- `traj_ignore{0,1}_N{N}`: (T+1, N, d) float32 — **항상 저장**

---

## 빠른 점검 명령어 모음

### 어떤 ensemble 들이 있나
```bash
ls /home/work/JooKyung/TabEBM/experiments/ebms/ | grep '_EBM$'
```

### 특정 ensemble 의 멤버 α / Subsample 확인
```bash
/home/work/miniconda3/envs/TabEBM/bin/python -c "
import json
from pathlib import Path
for k in range(10):
    cfg = json.loads(Path(f'experiments/ebms/20260415_210238_Subsample-Distance_EBM/c0/ebm_{k}/config.json').read_text())
    print(k, cfg.get('method_distance', {}).get('neg_distance'),
          cfg.get('method_subsample', {}).get('ratio'))
"
```

### 특정 sweep 결과의 M 빠르게 보기
```bash
COMP=experiments/ebms/20260415_210238_Subsample-Distance_EBM/comparisons
ls -t $COMP | head -3   # 최신 3 개

cat $COMP/$(ls -t $COMP | head -1)/summary_M_analysis.csv
```

### npz 안 들여다보기
```bash
/home/work/miniconda3/envs/TabEBM/bin/python -c "
import numpy as np
d = np.load('experiments/ebms/20260415_210238_Subsample-Distance_EBM/comparisons/<TS>/diag_raw_c0.npz', allow_pickle=True)
print('keys:', list(d.keys()))
print('cols:', list(d['diag_cols']))
print('ignore0_N500 shape:', d['ignore0_N500'].shape)
print('first/last step M_mean:')
arr = d['ignore0_N500']
cols = list(d['diag_cols'])
i = cols.index('M_mean')
print(f'  t=0:   {arr[0, i]:.4f}')
print(f'  t=T:   {arr[-1, i]:.4f}')
"
```

---

## Troubleshooting

| 증상 | 원인 | 해결 |
|---|---|---|
| `M_mean ≈ 1` 어떤 β 든 | ensemble Σ ≈ 0 (fake variance) | ensemble 교체 (실험 1) |
| OOM / GPU memory | N 너무 큼 | `--n-high 500` → `200` 으로 |
| TabPFN 모델 로드 매번 느림 | 다른 process | nb 재시작 (캐시 활용) |
| `traj_c{c}` 없음 경고 | 옛 npz | nb 02.1 cell 4 다시 실행 (trajectory 항상 저장됨) |
| `ignore_variance` 결과가 같음 | 실제 같음 OR cfg 누락 | `summary.csv` 의 `ignore_variance` 컬럼 확인 |

---

## 권장 실행 순서

1. **실험 1 (ensemble 교체)** — 5 분, 즉시 효과 확인
2. 결과 OK 면 → **실험 2 (β sweep)** — 15 분, M 의 β-반응 곡선
3. 추가로 **실험 3 (σ_start sweep)** — 10 분, 체인 동역학 확인
4. 셋 다 결과 모이면 thesis 의 **"VP-SGLD 의 유효 regime"** 섹션 작성 가능

---

## 한 번에 다 돌리는 통합 스크립트 (옵션)

```bash
#!/bin/bash
cd /home/work/JooKyung/TabEBM
ENS=experiments/ebms/20260415_210238_Subsample-Distance_EBM
PY=/home/work/miniconda3/envs/TabEBM/bin/python

echo "===== 실험 1: ensemble 교체 ====="
$PY experiments/compare_ignore_variance.py \
  --ensemble-root $ENS \
  --classes 0 1 --n-low 10 --n-high 500 --n-steps 50 \
  --beta 1.0 --eta 0.05 --tau 1.0 --sigma-start 0.1 --gpu 0

echo "===== 실험 2: β sweep ====="
for BETA in 0.01 1.0 100.0 10000.0 1000000.0; do
    echo "--- β = $BETA ---"
    $PY experiments/compare_ignore_variance.py \
      --ensemble-root $ENS \
      --classes 0 --n-low 10 --n-high 500 --n-steps 50 \
      --beta $BETA --eta 0.05 --tau 1.0 --sigma-start 0.1 --gpu 0
done

echo "===== 실험 3: σ_start sweep ====="
for SIG in 0.1 0.5 2.0 5.0; do
    echo "--- σ_start = $SIG ---"
    $PY experiments/compare_ignore_variance.py \
      --ensemble-root $ENS \
      --classes 0 --n-low 10 --n-high 500 --n-steps 50 \
      --beta 1.0 --eta 0.05 --tau 1.0 --sigma-start $SIG --gpu 0
done

echo "===== 완료. 결과: $ENS/comparisons/ ====="
ls -lt $ENS/comparisons/ | head -15
```

저장 위치: `tutorials/run_all_sweeps.sh` (만들어 두면 재사용 편리)
