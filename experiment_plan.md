# Bayesian TabEBM 실험 계획서

## 연구 목표

Deterministic EBM(Energy-Based Model)의 한계를 극복하기 위해 ensemble 기반 Bayesian 접근법을 적용하고,
variance-preconditioned SGLD를 통해 synthetic data 생성 품질을 향상시킨다.

---

## Phase 1: TabEBM Reproduce

### 1.1 목적

- TabEBM 원논문 결과 재현
- Augmentation 유무에 따른 성능 차이 확인 (baseline vs TabEBM)
- 이후 실험의 기준선(baseline) 확보

### 1.2 논문의 정확한 실험 프로토콜 (Table 1, Q1)

#### 1.2.1 데이터셋

**OpenML 8개 (Main)**

| Dataset  | OpenML ID | Samples | Features | Classes | Categorical | Missing |
|----------|-----------|---------|----------|---------|-------------|---------|
| protein  | 40966     | 1,080   | 77       | 8       | 없음        | 없음    |
| fourier  | 14        | 2,000   | 76       | 10      | 없음        | 없음    |
| biodeg   | 1494      | 1,055   | 41       | 2       | 없음        | 없음    |
| steel    | 1504      | 1,941   | 33       | 2       | 없음        | 없음    |
| stock    | 841       | 950     | 9        | 2       | 없음        | 없음    |
| energy   | 1472      | 698     | 9        | 23*     | 없음        | 없음    |
| collins  | 40971     | 970     | 19       | 26      | 없음        | 없음    |
| texture  | 40499     | 5,500   | 40       | 11      | 없음        | 없음    |

*energy: 10개 미만 샘플인 class 제거 후 실제 사용 class 수 감소

**UCI 6개 (Leakage-Free, Appendix)**

| Dataset  | UCI ID | Samples | Features | Classes | 비고 |
|----------|--------|---------|----------|---------|------|
| clinical | 890    | 2,139   | 23       | 2       | TabPFN 학습 데이터와 미중복 확인 |
| support2 | 880    | 9,105   | 42       | 2       | |
| mushroom | 73     | 8,124   | 22       | 2       | |
| auction  | 713    | 2,043   | 7        | 2       | |
| abalone  | 1      | 4,153   | 8        | 19      | |
| statlog  | 144    | 1,000   | 20       | 2       | |

#### 1.2.2 데이터 분할 프로토콜

```
1. 전체 데이터 (N개)
2. Test set 분리: N_test = min(N/2, 500), stratified
3. 나머지 pool에서 N_real개 subsample (stratified)
4. 전처리 (train에 fit, test에 transform):
   - 결측치 imputation: 수치형=mean, 범주형=mode
   - 범주형 → Leave-one-out Target Statistic (category_encoders)
   - Z-score normalization (StandardScaler)
5. 10 random splits 반복 (seed 다르게)
```

#### 1.2.3 실험 조건

| 항목 | 값 |
|------|-----|
| N_real | {20, 50, 100, 200, 500} |
| N_syn | 500 (고정, class 비율에 따라 stratified 생성) |
| Random splits | 10회 |
| Downstream classifiers | LR, KNN, MLP, RF, XGBoost, TabPFN (모두 sklearn default) |
| Baselines | Baseline(no aug), SMOTE, TVAE, CTGAN, NFLOW, TabDDPM, ARF, GOGGLE, TabPFGen |
| Metric | Balanced Accuracy (%), 6 classifier 평균 → 10 split mean±std |
| 집계 | ADTM (Average Distance to Minimum), Average Rank |

#### 1.2.4 TabEBM 하이퍼파라미터

| Parameter         | Symbol          | 논문 (v1) | 현재 코드 (v2) |
|-------------------|-----------------|-----------|---------------|
| SGLD step size    | alpha_step      | 0.1       | 0.1           |
| SGLD noise scale  | alpha_noise     | 0.01      | 0.01          |
| SGLD steps        | T               | 200       | 200           |
| 초기 perturbation | sigma_start     | 0.01      | 0.01          |
| Negative distance | alpha_neg_dist  | 5 (std)   | 5 (std)       |
| Negative 개수     | \|X_neg\|       | 4         | 4~8 (대칭쌍)  |
| **TabPFN version** | -              | **v1 (0.1.9)** | **v2 (2.1.2)** |
| **TabPFN ensemble** | n_estimators | **3**     | **1**         |
| **Preprocessing** | -              | 활성화    | **비활성화 (gradient 호환)** |

#### 1.2.5 논문 vs 우리 환경의 주요 차이점

| 항목 | 논문 | 우리 환경 | 영향 |
|------|------|----------|------|
| TabPFN version | v1 (0.1.9), n_estimators=3 | v2 (2.1.2), n_estimators=1 | EBM 내부 + downstream TabPFN 성능 모두 차이 가능 |
| GPU | NVIDIA Quadro RTX 8000 (48GB) | NVIDIA RTX 6000 Ada (49GB) x 4 | 성능 충분, 병렬 가능 |
| Baselines | 9개 (Synthcity 등) | Tier별 점진 확장 | Tier 1: 3개만 |
| Splits | 10 | Tier별 조정 (5→10) | std 영향 |

> **핵심**: TabPFN v1→v2 차이 때문에 **정확한 수치 재현은 불가능**. 방향(trend)과 상대적 순위가 일치하는지 확인하는 것이 목표.

### 1.3 재현 Tier 구조

전체 규모 재현(140,000 모델)은 비현실적이므로, 3단계로 점진 확장한다.

#### Tier 1: 핵심 빠른 검증 (완료)

```
데이터셋: biodeg, stock (2개)
N_real: 100
Methods: Baseline, SMOTE, TabEBM (3개)
Classifiers: KNN, RF, TabPFN (3개)
Splits: 5회
GPU: biodeg → GPU 0, stock → GPU 1

총 runs: 2 × 3 × 3 × 5 = 90
소요 시간: ~3분 (병렬)
```

**실행 방법**:
```bash
# 병렬 실행 (GPU 0, 1 사용)
bash experiments/launch_tier1.sh

# 단일 데이터셋 실행
conda run -n TabEBM python experiments/run_experiment.py \
    --dataset biodeg --n_real 100 --n_splits 5 \
    --methods baseline smote tabebm \
    --classifiers knn rf tabpfn \
    --gpu 0 --output_dir experiments/results/tier1

# 결과 요약 + 시각화
conda run -n TabEBM python experiments/summarize_results.py \
    --results_dir experiments/results/tier1 --plot
```

**커스텀 파라미터 예시**:
```bash
# TabEBM 하이퍼파라미터 조정
python experiments/run_experiment.py \
    --dataset biodeg --n_real 100 --gpu 0 \
    --sgld_steps 300 --sgld_step_size 0.05 --sgld_noise_std 0.02

# 다른 데이터셋 크기
python experiments/run_experiment.py \
    --dataset biodeg --n_real 50 --gpu 2

# 전체 classifier 사용
python experiments/run_experiment.py \
    --dataset biodeg --n_real 100 --gpu 0 \
    --classifiers lr knn mlp rf xgboost tabpfn
```

**Tier 1 결과** (2025-04-09, TabPFN v2, 5 splits):

biodeg (N_real=100, 41 features, 2 classes):

| Method | KNN | RF | TabPFN | MEAN |
|--------|-----|-----|--------|------|
| **Baseline** | **78.78 ± 1.80** | 77.25 ± 3.84 | **79.37 ± 2.27** | **78.47** |
| SMOTE | 75.15 ± 3.87 | 77.23 ± 5.11 | 77.40 ± 2.79 | 76.59 |
| TabEBM | 75.78 ± 2.73 | **79.28 ± 3.14** | 75.50 ± 2.34 | 76.86 |

stock (N_real=100, 9 features, 2 classes):

| Method | KNN | RF | TabPFN | MEAN |
|--------|-----|-----|--------|------|
| Baseline | 88.22 ± 3.10 | **92.99 ± 2.35** | 92.72 ± 2.55 | 91.31 |
| **SMOTE** | **92.02 ± 2.28** | 92.05 ± 2.51 | **92.91 ± 3.15** | **92.33** |
| TabEBM | 91.52 ± 2.27 | 92.18 ± 2.35 | 91.86 ± 2.87 | 91.85 |

Average Rank:

| | biodeg | stock |
|---|--------|-------|
| Baseline | **1.33** | 1.93 |
| SMOTE | 2.33 | **1.73** |
| TabEBM | 2.33 | 2.33 |

**Tier 1 분석**:
- **stock**: augmentation(SMOTE, TabEBM) 모두 KNN에서 +3~4pp 향상. 저차원(9d)에서 효과적.
- **biodeg**: TabEBM이 RF에서만 +2pp 향상, 나머지는 하락. 고차원(41d)에서 제한적.
- **논문과의 차이**: 논문은 6개 classifier 평균 + 10 splits로 TabEBM이 1등. 우리는 3개 classifier + 5 splits로 SMOTE가 stock에서 우세. Tier 2에서 classifier 확장/splits 증가로 재확인 필요.
- **TabPFN v2 영향**: downstream TabPFN 결과가 논문과 다를 수 있음 (v1은 3 ensembles, v2는 1).

#### Tier 2: 중간 규모 확장 (TODO)

```
데이터셋: 8개 전부
N_real: {50, 100, 200}
Methods: Baseline, SMOTE, TabEBM (3개)
Classifiers: LR, KNN, MLP, RF, XGBoost, TabPFN (6개 전부)
Splits: 10회
GPU: 4개에 데이터셋 분배

총 runs: 8 × 3 × 3 × 6 × 10 = 4,320
```

**실행 방법**:
```bash
# 8개 데이터셋을 4 GPU에 분배
for GPU in 0 1 2 3; do
    DATASETS=(...) # GPU별 할당
    for DS in "${DATASETS[@]}"; do
        for NR in 50 100 200; do
            conda run -n TabEBM python experiments/run_experiment.py \
                --dataset $DS --n_real $NR --n_splits 10 \
                --classifiers lr knn mlp rf xgboost tabpfn \
                --gpu $GPU --output_dir experiments/results/tier2 &
        done
    done
done
```

**GPU 할당 계획** (데이터 크기 기준 균등 분배):
- GPU 0: protein, biodeg (고차원)
- GPU 1: fourier, steel
- GPU 2: stock, energy (저차원)
- GPU 3: collins, texture

#### Tier 3: 풀 재현 (TODO)

```
Tier 2 + Synthcity baselines (TVAE, CTGAN, TabDDPM) 추가
+ N_real 전체 {20, 50, 100, 200, 500}
+ UCI 6개 데이터셋
+ Statistical fidelity metrics (Inverse KL, KS test, Chi-squared)
+ ADTM 집계

총 runs: ~21,600+ (baselines 포함)
필요 패키지: synthcity (TVAE, CTGAN, TabDDPM, ARF, GOGGLE, NFLOW)
```

**Synthcity baselines 추가 시 필요 작업**:
- `pip install synthcity` 설치
- 각 generator의 wrapper 함수 작성 (run_experiment.py에 추가)
- TabPFGen은 공식 코드 없음 → 직접 구현 또는 생략
- 실행 시간 대폭 증가: Synthcity 모델은 TabEBM 대비 3~30배 느림 (논문 Figure 6)

### 1.4 파이프라인 확인사항

```
각 class c에 대해:
1. Surrogate binary dataset 구성
   - Positive: 해당 class의 실제 데이터 X_c (label=0)
   - Negative: hypercube 꼭짓점 4~8개 (label=1), ±alpha_neg_dist in each dim
2. TabPFN으로 binary classifier 학습 (v2, n_estimators=1, no preprocessing)
3. Classifier logits → Energy 변환: E_c(x) = -logsumexp(f[0], f[1])
4. SGLD로 sampling: x_0 ~ N(X_c, sigma_start^2 * I)
   x_{t+1} = x_t - alpha_step * grad(E_c(x_t)) + alpha_noise * N(0, I)
   (주의: total_energy를 num_features로 나누는 비표준 normalization 있음, TabEBM.py:493)
5. 생성된 샘플로 augmentation → downstream classifier 학습 → balanced accuracy 측정
```

### 1.5 기록할 메트릭

- **Balanced Accuracy** (classifier별, split별 개별 기록 → mean±std 집계)
- Baseline 대비 improvement (percentage points)
- Average Rank (방법 간 순위)
- Augmentation 소요 시간 (초)

### 1.6 실험 코드 구조

```
experiments/
├── run_experiment.py       # 메인 실험 runner (argparse 기반)
│   ├── --dataset           # 데이터셋 선택 (8개 OpenML)
│   ├── --n_real            # 학습 데이터 크기
│   ├── --n_syn             # 생성 샘플 수 (default: 500)
│   ├── --n_splits          # Random split 수
│   ├── --methods           # augmentation 방법 선택
│   ├── --classifiers       # downstream classifier 선택
│   ├── --gpu               # GPU 인덱스 (-1 for CPU)
│   ├── --sgld_steps/step_size/noise_std  # TabEBM HP 조정
│   └── --output_dir/seed   # 출력 디렉토리, 시드
├── launch_tier1.sh         # Tier 1 병렬 실행 스크립트
├── summarize_results.py    # 결과 취합, 테이블, 시각화
│   ├── --results_dir       # 결과 CSV 디렉토리
│   └── --plot              # 막대그래프/개선도 그래프 생성
└── results/
    └── tier1/              # Tier 1 결과
        ├── biodeg_n100.csv          # Raw 결과 (split별, method별, clf별)
        ├── biodeg_n100_summary.csv  # 요약 테이블
        ├── biodeg_n100_config.json  # 실험 설정 기록
        ├── biodeg_n100_barplot.png  # 성능 비교 막대그래프
        ├── biodeg_n100_improvement.png  # Baseline 대비 개선도
        └── biodeg_log.txt           # 실행 로그
```

---

## Phase 2: Ensemble 구성 및 다양성 분석

### 2.1 목적

- EBM을 여러 개 만들었을 때 실제로 **다양한** energy landscape이 나오는지 확인
- 다양하지 않으면 이후 variance-preconditioned SGLD의 의미가 없음
- 앙상블의 uncertainty가 의미 있는 정보를 담고 있는지 검증
- **어떤 diversity source 조합이 가장 효과적인지** 체계적으로 탐색

### 2.2 이론적 엄밀성 점검

앙상블이 epistemic uncertainty를 유효하게 포착하려면, member들이 **같은 target distribution에 대한 서로 다른 plausible belief**를 표현해야 한다 (Lakshminarayanan et al. 2017, Wilson & Izmailov 2020).
"서로 다른 target distribution"을 생성하는 것은 uncertainty가 아니라 **modeling choice의 sensitivity analysis**이다.

아래 점검 결과에 따라 각 방법에 유효성 태그를 부여한다:
- **[VALID]**: Bayesian posterior sampling과의 formal connection이 있거나, 확립된 ensemble 이론에 근거
- **[PARTIAL]**: 이론적으로 일부 정당화 가능하나 주의사항 존재
- **[INVALID]**: Uncertainty ensemble로 부적합 — 다른 용도(sensitivity analysis, sampling 효율 등)로만 사용

| 방법 | 유효성 | 핵심 근거 |
|------|--------|----------|
| N1 (Corner subset) | **[VALID]** | Bagging 아날로그, 같은 concept의 다른 training data sample |
| N2 (Variable distance) | **[PARTIAL]** | KDE bandwidth uncertainty 아날로그이나, formal Bayesian 근거 부재 |
| N3 (Non-corner geometry) | **[PARTIAL]** | N1과 유사하나, 다른 geometry = 다른 inductive bias → modeling choice에 가까움 |
| N4 (Data-adaptive) | **[PARTIAL]** | Inter-class는 문제 정의 자체를 변경, PCA-aligned은 정당화 가능 |
| C1 (Heterogeneous clf) | **[PARTIAL]** | BMA로 해석 가능하나 **energy scale 비호환** 문제 심각 |
| C2 (TabPFN config) | **[PARTIAL]** | TabPFN v2는 n_estimators=1 + no preprocessing으로 고정 → 실질적 variation 제한적 |
| C3 (MLP architecture) | **[PARTIAL]** | BMA의 일종이나, 다른 architecture = 다른 함수 공간 |
| E1 (Temperature) | **[INVALID]** | 다른 T = 다른 target distribution (parallel tempering), gradient 방향 불변 |
| E2 (Energy definition) | **[INVALID]** | 다른 energy 정의 = 다른 density model, scale 비호환 |
| F1 (Feature subset) | **[VALID]** | Random Subspace Method, BMA over features. 단, VP-SGLD gradient 결합 문제 |
| F2 (Random projection) | **[PARTIAL]** | JL 보장이 있으나, back-projection artifact |
| F3 (Normalization) | **[INVALID]** | 다른 좌표계 = energy/gradient 직접 비교 불가 |
| D1 (Noise injection) | **[PARTIAL]** | Aleatoric uncertainty만 포착, sigma가 measurement noise 반영 시 정당 |
| D2 (Feature masking) | **[PARTIAL]** | F1의 stochastic 버전, 같은 문제 공유 |
| D3 (Mixup) | **[PARTIAL]** | Data manifold densification, 소량 데이터에서 의미 있으나 이론적 근거 약함 |
| D4 (Bayesian bootstrap) | **[VALID]** | **Rubin (1981) formal Bayesian justification**, alpha=1에서 정당 |
| M1 (Multi-scale) | 구성에 따름 | VALID 방법들만 결합 시 유효 |
| M2 (Anchored MLP) | **[VALID]** | Pearce et al. (2020), posterior sampling 근사 증명 있음 |
| M3 (Snapshot) | **[VALID]** | Huang et al. (2017), loss landscape의 다른 mode 탐색 |

**[INVALID] 방법 처리 방침**:
- E1, E2, F3는 **uncertainty ensemble에서 제외**
- E1(Temperature)은 SGLD sampling 품질 조절 용도로만 별도 실험 가능
- E2는 energy 정의 robustness 점검(sensitivity analysis)으로만 활용

---

### 2.3 앙상블 구성 방법

모든 앙상블 component는 **전체 데이터를 다 사용**한다. Randomness의 source만 다르게 한다.

**앙상블 크기**: K in {3, 5, 10}

---

#### Category 1: Negative Sample Variations

TabEBM에서 diversity의 가장 자연스러운 source. TabPFN은 pretrained + deterministic이므로,
입력(surrogate dataset)을 바꾸는 것이 classifier를 바꾸는 가장 직접적인 방법이다.

**방법 N1: Hypercube Corner Subset Randomization [VALID]**
- 원논문 기본: d차원에서 2^d개의 hypercube 꼭짓점 중 4개만 사용
- 각 앙상블 member는 다른 random subset의 꼭짓점을 선택
- 고차원(d=41, 77 등)에서 4개는 2^d의 극히 일부 → 높은 다양성 기대
- 변수: 꼭짓점 수 k in {4, 8, 16, 32}, seed별 다른 조합

**방법 N2: Variable-Distance Negative Shells [PARTIAL]**
- 원논문 기본: alpha_neg_dist = 5 (표준편차 단위)로 고정
- 각 member가 다른 거리 사용: alpha_neg_dist in {2, 3, 5, 7, 10, 15}
- 가까운 거리 → 날카로운(sharp) energy boundary, 먼 거리 → 부드러운(smooth) boundary
- KDE bandwidth uncertainty 아날로그로 해석 가능
- **주의**: formal Bayesian 근거 없음. 다른 distance는 다른 binary classification problem을 정의.
  Variance가 실험자의 거리 범위 선택에 좌우됨 → sensitivity analysis로 해석이 더 적절.
  단, distance에 대한 prior를 정의하고 marginal likelihood로 weighting하면 정당화 가능.

**방법 N3: Non-Corner Negative Geometries**
- Hypercube 꼭짓점 대신 다른 기하학적 배치:

| 배치 | 설명 | 특성 |
|------|------|------|
| Spherical Shell | 반지름 r의 hypersphere 위에 uniform 배치 | Isotropic (등방성) boundary |
| Sobol/Halton QMC | 준난수 시퀀스로 negative 영역 균일 커버 | Low-discrepancy, 균일한 커버리지 |
| Gaussian Shell | N(0, sigma^2 * I)에서 샘플, sigma >> data scale | 거리가 확률적, 부드러운 boundary |
| Axis-Aligned | 각 좌표축 방향으로만 ±d 배치 (2*dim개) | 십자형 negative, feature 독립성 가정 |
| Mixed Distance | 일부는 가까이(d=2~3), 일부는 멀리(d=8~10) | Hard negative + easy negative 혼합 |

- 각 geometry는 질적으로 다른 energy landscape 형태를 유도
- Corner → 좌표축 정렬된 energy basin, Spherical → 등방 energy basin

**방법 N4: Data-Adaptive Negatives**
- 데이터 분포 구조를 반영한 negative 배치:

| 전략 | 설명 |
|------|------|
| PCA-Aligned | 주성분 방향으로 negative 배치 (공분산 구조 존중) |
| Convex Hull Expansion | 데이터 convex hull의 꼭짓점을 확대 |
| Inter-Class | 다른 class의 데이터를 negative로 사용 |
| KNN Boundary | 각 데이터 포인트의 KNN 경계 바깥에 negative 배치 |

- 주의: Inter-Class는 TabEBM의 "각 class 독립 모델링" 가정을 변경함 → 별도 실험

---

#### Category 2: Classifier Variations

**방법 C1: Heterogeneous Classifier Ensemble**
- TabPFN 대신 다른 classifier family를 사용하여 surrogate binary task 학습:

| Classifier | Energy 변환 | Gradient | 특성 |
|-----------|-------------|----------|------|
| TabPFN | -logsumexp(logits) | Backprop | 원논문 기본, Bayesian prior |
| Logistic Regression | -logsumexp(log-odds) | Backprop | 선형 boundary → smooth convex energy |
| MLP (2-3 layers) | -logsumexp(pre-softmax) | Backprop | 비선형, architecture 다양성 |
| SVM (RBF) | -decision_function(x) | Finite diff | Band-like energy, kernel-based |
| KNN-based | -log(local density ratio) | Finite diff | 비모수, 매우 다른 inductive bias |

- **핵심 고려사항**: SGLD는 gradient 필요 → TabPFN/LR/MLP는 backprop 가능, RF/XGBoost/KNN/SVM은 finite difference 근사 필요
- 다른 model family는 근본적으로 다른 inductive bias → **가장 강한 형태의 다양성**
- Fort et al. (2019): 같은 architecture의 다른 init보다 다른 architecture가 훨씬 다양

```python
# Finite difference gradient for non-differentiable classifiers
def finite_diff_grad(E_func, x, eps=1e-4):
    grad = torch.zeros_like(x)
    for i in range(x.shape[-1]):
        x_plus = x.clone(); x_plus[..., i] += eps
        x_minus = x.clone(); x_minus[..., i] -= eps
        grad[..., i] = (E_func(x_plus) - E_func(x_minus)) / (2 * eps)
    return grad
```

**방법 C2: TabPFN Configuration Variations**
- TabPFN 내부의 설정을 다르게:
  - 다른 internal ensemble seed
  - 다른 preprocessing config (gradient 호환성 확인 필요)
  - n_estimators 변경
- TabPFN이 pretrained model이므로, 내부 variation이 제한적일 수 있음

**방법 C3: MLP Architecture Variations**
- MLP를 classifier로 사용하되, architecture를 다르게:
  - Hidden sizes: [32], [64, 32], [128, 64, 32]
  - Activation: ReLU, GELU, Tanh, SiLU
  - 각 architecture가 다른 function class를 표현 → 다른 energy landscape

---

#### Category 3: Energy Function Variations — [INVALID for uncertainty ensemble]

> **이론적 점검 결과**: 이 카테고리의 방법들은 **같은 classifier에서 다른 target distribution을 정의**하는 것이지,
> 같은 distribution에 대한 다른 belief를 표현하는 것이 아님. Uncertainty ensemble로 사용 불가.
> **Sensitivity analysis 또는 sampling 품질 조절 용도로만 활용**.

**방법 E1: Temperature Scaling [INVALID]**
```
E_T(x) = -T * log(exp(f[0]/T) + exp(f[1]/T))
```
- 다른 T는 **다른 분포** p_T(x) ∝ exp(-E_T(x))를 정의 — parallel tempering과 동일한 구조
- Gradient **방향**은 T에 무관 (크기만 변함) → VP-SGLD에서 방향적 disagreement 제공 불가
- Variance가 T 범위 선택에 의해 결정됨 (data-driven이 아님)
- **용도 제한**: SGLD의 exploration-exploitation 조절, 생성 샘플 sharpness 조절

**방법 E2: Alternative Energy Definitions [INVALID]**
- 같은 classifier logits f[0], f[1]에서 다른 energy 유도:

| Energy 정의 | 수식 | 해석 |
|-------------|------|------|
| LogSumExp (원논문) | -log(exp(f[0]) + exp(f[1])) | 총 confidence |
| Positive Logit | -f[0] | Positive class logit만 사용 (JEM-style) |
| Ratio (Log-prob) | -log(sigmoid(f[0] - f[1])) | Positive일 확률의 log |
| Margin | -(f[0] - f[1]) | Raw logit 차이 |
| Softplus | -log(1 + exp(f[0] - f[1])) | Margin의 smoothed 버전 |

- **문제점**: 각 정의의 energy scale이 완전히 다름 → Var[E_k(x)]가 scale 차이에 지배됨
- 서로 다른 density model을 정의하므로, disagreement = epistemic uncertainty가 아님
- **용도 제한**: "어떤 energy 정의가 TabEBM에 가장 적합한지" robustness check

---

#### Category 4: Feature Space Variations

**방법 F1: Random Feature Subsets (Random Subspace Method) [VALID — gradient 결합 주의]**
- 각 member가 전체 d개 feature 중 d'개만 사용하여 EBM 구성
- d'/d in {0.5, 0.6, 0.7, 0.8, 0.9}
- 고차원 데이터에서 특히 효과적: protein(77), fourier(76), biodeg(41)
- Random Forest의 핵심 원리와 동일, BMA over feature subsets로 해석 가능
- **VP-SGLD gradient 결합 문제**: 각 member의 gradient가 다른 subspace에 존재
  - Feature를 안 쓴 member의 grad=0이 평균에 포함되면 systematic bias 발생
  - **해결**: per-feature로, 해당 feature를 사용한 member들만으로 mean/var 계산
  - SGLD는 해당 subspace에서만 수행, 나머지 feature는 원본 유지

**방법 F2: Random Projections**
- 랜덤 projection 행렬 R (d x d')로 차원 축소 후 EBM 학습
- Johnson-Lindenstrauss: 거리 보존하면서 다양한 관점
- Projection type: {Gaussian, sparse, structured}
- 주의: back-projection (d' → d)시 pseudoinverse 필요, artifact 가능

**방법 F3: Different Normalization Schemes [INVALID]**
- 각 member가 다른 feature normalization 적용 후 EBM 학습

> **이론적 점검 결과**: 다른 normalization = 다른 좌표계에서 energy/gradient 계산.
> Energy 값, gradient 방향 모두 직접 비교 불가능. 비선형 normalization(quantile, Yeo-Johnson)의 경우
> Jacobian 변환 없이는 gradient 결합이 수학적으로 무의미.
> 겉보기에는 합리적이지만, **가장 미묘한 함정을 가진 방법**.
> **Uncertainty ensemble에서 제외.**

---

#### Category 5: Data Perturbation Methods

모든 데이터를 사용하되, 약간의 변형을 가한다.

**방법 D1: Gaussian Noise Injection on Positives**
- 각 member의 positive data에 소량의 noise 추가: X_perturbed = X + N(0, sigma^2 * I)
- sigma in {0.01, 0.05, 0.1, 0.2} (feature std 기준)
- TabPFN은 in-context learner → 입력의 작은 변화에도 prediction 변화 가능
- 주의: sigma가 너무 크면 데이터 분포 자체를 왜곡

**방법 D2: Feature Masking (Input Dropout)**
- 각 member의 각 데이터 포인트에서 일부 feature를 mask (0 또는 feature mean으로 대체)
- Masking probability: p in {0.05, 0.1, 0.2, 0.3}
- Random Subspace (F1)과 달리, 포인트마다 다른 feature가 mask됨 → 더 복잡한 perturbation

**방법 D3: Mixup-Based Augmented Positives**
- Positive sample 쌍 (x_i, x_j)에서 interpolation: x_mix = lambda * x_i + (1-lambda) * x_j
- lambda ~ Beta(alpha, alpha), alpha in {0.1, 0.5, 1.0, 2.0}
- 각 member가 다른 mixup realization → 다른 densified positive set → 다른 classifier
- 소량 데이터(N=20, 50)에서 특히 유효: positive가 너무 적을 때 energy landscape이 불안정

**방법 D4: Bayesian Bootstrap Reweighting [VALID — 가장 강력한 이론적 근거]**
- 데이터를 빼지 않고, 각 positive sample에 랜덤 가중치 부여
- Rubin (1981)의 **formal Bayesian justification**: Dirichlet process prior 하에서의 posterior sample
- 가중치: w ~ Dirichlet(alpha * 1_N)
  - **alpha = 1만 이론적으로 정당** (standard Bayesian bootstrap)
  - alpha < 1: uncertainty 과대추정, alpha > 1: uncertainty 과소추정
- TabPFN에서의 구현: 가중치가 높은 포인트를 중복 포함 (duplicate) → 연속 가중치의 이산 근사
- 각 member는 **같은 concept에 대한 다른 plausible data-generating distribution** → 정확히 epistemic uncertainty
- **주의**: N=20에서는 Bayesian bootstrap variance가 매우 클 수 있음 (이는 정확한 행동: 20개로는 정말 불확실)

---

#### Category 6: Composite / Advanced Methods

**방법 M1: Multi-Scale Ensemble (추천)**
- Category 1~5의 variation을 체계적으로 결합
- Sobol sequence 또는 Latin Hypercube Sampling으로 hyperparameter 조합 선택

```
Member k의 설정 = (negative_distance_k, negative_geometry_k, temperature_k, energy_type_k)
```

- 예: K=10일 때
  - Member 1: corners, dist=2, T=0.3, logsumexp
  - Member 2: spherical, dist=5, T=1.0, positive_logit
  - Member 3: sobol, dist=10, T=2.0, ratio
  - ... (LHS로 조합)
- **가장 강력한 다양성 기대**: 여러 source의 variation이 곱해짐

**방법 M2: Anchored MLP Ensemble (trainable classifier 사용 시)**
- Pearce et al. (2020) 방식: MLP classifier 각각을 다른 random init에 anchor
- Loss = CrossEntropy + lambda * ||theta - theta_anchor||^2
- 다른 anchor가 다른 weight space 영역으로 유도 → 이론적으로 Bayesian posterior sampling 근사
- TabPFN이 아닌 MLP를 classifier로 쓸 때만 적용 가능

**방법 M3: Snapshot Ensemble (trainable classifier 사용 시)**
- Cyclic learning rate로 MLP를 학습, 각 cycle의 valley에서 snapshot
- 단일 training run에서 K개의 functionally diverse한 모델 획득
- 효율적이지만 TabPFN에는 적용 불가 (TabPFN은 training 없음)

---

### 2.4 실험 우선순위

이론적 엄밀성 + 구현 난이도 + 기대 다양성을 고려한 실행 순서.
**[INVALID] 방법은 uncertainty ensemble 실험에서 제외.**

#### Tier 1: 이론적으로 정당한 방법 (반드시 수행)

| 순위 | 방법 | 유효성 | 기대 다양성 | 구현 난이도 | 근거 |
|------|------|--------|-------------|------------|------|
| **1** | D4 (Bayesian bootstrap) | **[VALID]** | 중~높 | 중 | Rubin (1981), 유일한 formal Bayesian method |
| **2** | N1 (Corner subset) | **[VALID]** | 중 | 낮음 | Bagging 아날로그, 가장 자연스러운 source |
| **3** | F1 (Feature subset) | **[VALID]** | 높 | 중 | Random Subspace Method, gradient 결합 주의 |

#### Tier 2: 이론적으로 부분 정당 (Tier 1 부족 시 추가)

| 순위 | 방법 | 유효성 | 기대 다양성 | 구현 난이도 | 비고 |
|------|------|--------|-------------|------------|------|
| **4** | N2 (Variable distance) | **[PARTIAL]** | 중~높 | 낮음 | Sensitivity analysis로 해석, 결과 해석 주의 |
| **5** | D1 (Noise injection) | **[PARTIAL]** | 중 | 낮음 | Aleatoric uncertainty만, sigma 선택 근거 필요 |
| **6** | N3 (Non-corner geometry) | **[PARTIAL]** | 높 | 중 | 다양한 geometry로 bagging 확장 |
| **7** | D3 (Mixup) | **[PARTIAL]** | 중 | 중 | 소량 데이터(N=20,50)에서만 |
| **8** | M2 (Anchored MLP) | **[VALID]** | 높 | 높 | MLP classifier 사용 시에만, TabPFN 불가 |

#### Tier 3: 별도 실험 (uncertainty ensemble이 아닌 다른 용도)

| 방법 | 유효성 | 용도 |
|------|--------|------|
| E1 (Temperature) | **[INVALID]** | SGLD sampling 품질 조절 (별도 ablation) |
| E2 (Energy definition) | **[INVALID]** | Energy 정의 robustness check |
| F3 (Normalization) | **[INVALID]** | 사용하지 않음 (좌표계 비호환) |
| C1 (Heterogeneous clf) | **[PARTIAL]** | Energy scale 정규화 해결 후에만 |

> **Phase 2 실행 전략**:
> 1. Tier 1 (D4 + N1) 먼저 수행 → 다양성 측정
> 2. 부족하면 Tier 1의 F1 추가 (gradient 결합 구현 후)
> 3. 여전히 부족하면 Tier 2 (N2, D1) 추가
> 4. VALID 방법들로만 M1(Multi-scale) 구성

---

### 2.4 다양성 측정 메트릭

각 class c에 대해 K개의 EBM {E_1, E_2, ..., E_K}를 학습한 뒤, 아래 메트릭들로 다양성을 정량화한다.

**평가 포인트 집합 구성**:
- X_real: 해당 class의 실제 데이터
- X_near: 실제 데이터 주변 (X_real + N(0, 0.5^2 * I))
- X_far: 데이터에서 먼 영역 (X_real + N(0, 3.0^2 * I))
- X_grid: 2D projection (PCA 상위 2개 주성분) 위의 regular grid

---

#### 2.4.1 Pointwise Disagreement Metrics

각 평가 포인트 x에서 K개 EBM이 얼마나 다른 energy를 할당하는지.

| 메트릭 | 수식 | 해석 |
|--------|------|------|
| **Energy Variance** | `Var(x) = (1/K) * sum_k (E_k(x) - E_bar(x))^2` | 기본, 반드시 계산 |
| **Coefficient of Variation** | `CV(x) = sqrt(Var(x)) / |E_bar(x)|` | Scale-independent, 다른 energy 정의 비교시 필요 |
| **Energy Range** | `Range(x) = max_k E_k(x) - min_k E_k(x)` | 극단적 disagreement 포착 |
| **Entropy of Softmaxed Energies** | `w_k = exp(-E_k) / sum exp(-E_j)`, `H = -sum w_k log w_k` | Boltzmann weight 관점의 disagreement |
| **Median Abs Deviation** | `MAD(x) = median_k |E_k(x) - median(E(x))|` | Outlier EBM에 강건 |

**핵심 분석**: X_real vs X_near vs X_far에서의 Variance 분포 비교
- 이상적 결과: X_real에서 Var 작고, X_far에서 Var 큼 → uncertainty가 의미 있음

---

#### 2.4.2 Pairwise Model Similarity Metrics

K개 EBM 쌍들 간의 유사도. K x K matrix로 표현.

| 메트릭 | 수식 | 해석 |
|--------|------|------|
| **Pearson Correlation** | `rho_ij = Corr(E_i(X), E_j(X))` | 에너지 값의 선형 상관 |
| **Spearman Rank Corr** | `rho_s = Corr(rank(E_i), rank(E_j))` | 순서 기반, 비선형 관계도 포착 |
| **Energy Difference Variance** | `Var_x[E_i(x) - E_j(x)]` | 0이면 두 분포 동일 (up to constant) |
| **Gradient Cosine Similarity** | `cos(nabla E_i(x), nabla E_j(x))` 의 평균 | SGLD 방향의 일치도, 직접 관련 |
| **Gradient Magnitude Ratio** | `||nabla E_i|| / ||nabla E_j||` 의 분포 | 방향 같아도 sharpness 다를 수 있음 |

**판단 기준**:
- Pearson corr > 0.99: 거의 동일 → 앙상블 의미 없음
- Pearson corr 0.80~0.95: 유의미한 다양성
- Pearson corr < 0.80: 매우 다양 (혹은 일부 member가 불안정)

---

#### 2.4.3 Ensemble-Level Diversity Metrics

전체 앙상블을 하나의 수치로 요약. Kuncheva & Whitaker (2003) 프레임워크 기반.

**연속형 (EBM에 직접 적용)**:

| 메트릭 | 수식 | 해석 |
|--------|------|------|
| **Continuous KW Variance** | `(1/N) * sum_n Var_k[E_k(x_n)]` | Kohavi-Wolpert의 연속 버전, 핵심 단일 수치 |
| **Mean Pairwise Disagreement** | `(2/K(K-1)) * sum_{i<j} Var_x[E_i - E_j]` | 쌍별 disagreement 평균 |
| **Spectral Diversity** | Pairwise correlation matrix의 eigenvalue entropy | 높으면 독립적 member 많음 |

**이진화 후 (threshold tau 적용)**:
- `y_k(x) = 1 if E_k(x) < tau` (모델이 x를 "likely"로 판단) 로 이진화

| 메트릭 | 수식 | 해석 |
|--------|------|------|
| **Q-statistic** | `Q_ij = (ad - bc) / (ad + bc)` | 1=항상 동의, 0=독립, <0=반대 |
| **Disagreement Measure** | `Dis_ij = (b+c) / (a+b+c+d)` | Disagree하는 포인트 비율 |
| **Entropy Measure** | 각 포인트에서 K개 중 l개가 "correct"일 때의 entropy | 1에 가까울수록 다양 |

---

#### 2.4.4 Gradient-Based Diversity (VP-SGLD 직결)

VP-SGLD의 preconditioner를 직접 결정하는 메트릭들. Phase 3과 직접 연결됨.

| 메트릭 | 수식 | VP-SGLD 관련성 |
|--------|------|---------------|
| **Gradient Variance (per-dim)** | `Var_k[partial E_k / partial x_d]` | 직접 preconditioner G의 분모 |
| **Gradient Agreement Ratio (GAR)** | `||mean grad||^2 / mean(||grad||^2)` | 0=완전 disagreement (상쇄), 1=완전 동의 |
| **Gradient Subspace Effective Rank** | G=[nabla E_1,...,nabla E_K]의 SVD → `exp(H(s_i/sum s))` | 1이면 gradient 다 같은 방향, K면 독립 |
| **SNR (Signal-to-Noise Ratio)** | `||mean grad||^2 / Var_grad` | 낮으면 앙상블 신호가 disagreement에 묻힘 |

**핵심 분석**:
- GAR이 X_real에서 높고 X_far에서 낮으면 → 데이터 근처에서 동의, 먼 곳에서 불일치 → 이상적
- Gradient Variance의 per-feature 분해 → 어떤 feature에서 모델 간 disagreement가 큰지 확인

---

#### 2.4.5 Generated Sample Diversity

각 EBM에서 독립적으로 SGLD → K 세트의 synthetic data 생성 후 비교.

| 메트릭 | 설명 | 구현 |
|--------|------|------|
| **MMD (Maximum Mean Discrepancy)** | 두 sample set 간 분포 차이 | Gaussian RBF kernel, median heuristic |
| **Coverage** | Real data 중 생성 샘플에 의해 cover되는 비율 | k-NN 기반 (Naeem et al. 2020) |
| **Coverage Gain** | Union(S_1,...,S_K)의 coverage - 개별 coverage | 앙상블이 실제로 더 넓게 커버하는지 |
| **Pairwise Overlap** | 두 sample set 간 가까운 샘플 비율 | epsilon-ball 기반 |
| **Column-wise KS test** | Feature별로 S_i vs S_j 2-sample KS test | 유의한 차이 나는 feature 비율 |
| **Correlation Matrix Distance** | `||Corr(S_i) - Corr(S_j)||_F` | Feature 간 관계 차이 포착 |

**핵심 분석**: Coverage Gain이 클수록 앙상블이 diverse한 영역에서 샘플 생성
- Coverage(union) >> Coverage(individual) → 효과적인 다양성
- Coverage(union) ≈ Coverage(individual) → 다양성 부족

---

#### 2.4.6 Functional Diversity (심화 분석)

| 메트릭 | 설명 | 비고 |
|--------|------|------|
| **CKA (Centered Kernel Alignment)** | 모델의 내부 표현 비교 | MLP classifier 사용 시 적용 가능 |
| **Mode Analysis** | 각 EBM의 energy 극소점(mode) 위치 비교 → Hausdorff distance | 다른 mode 발견 여부 확인 |
| **Function Space L2 Distance** | `sqrt((1/N) sum (E_i(x) - E_j(x))^2)` normalized | Energy function 자체의 거리 |

---

### 2.5 시각화 계획

#### 2.5.1 Energy Landscape Overlay

- 2D projection (PCA 상위 2 주성분)에서 regular grid 생성
- K개 EBM의 energy contour를 겹쳐서 표시 (다른 색, 투명도)
- 평균 energy contour + uncertainty (std) heatmap 별도 표시
- **stock** (9 features) 데이터셋에서 우선 수행

#### 2.5.2 Disagreement Heatmap

- 2D grid에서 `Var_E(x)` 또는 `GAR(x)` 를 heatmap으로 표시
- Real data points를 scatter로 overlay
- 기대: 데이터 주변은 cool (낮은 disagreement), 먼 영역은 hot (높은 disagreement)

#### 2.5.3 Gradient Field Visualization

- 2D projection에서 각 EBM의 gradient를 quiver plot으로 표시
- 평균 gradient field + gradient variance ellipse overlay
- 모델 간 gradient 방향이 다른 영역 강조

#### 2.5.4 K x K Pairwise Similarity Heatmap

- Pearson/Spearman correlation matrix를 heatmap으로 표시
- Hierarchical clustering으로 유사한 모델끼리 그룹화
- 각 앙상블 구성 방법(N1, N2, E1 등)별로 비교

#### 2.5.5 Diversity Profile Plots

- x축: 데이터로부터의 거리 (distance from data manifold)
- y축: 해당 거리에서의 평균 Energy Variance 또는 GAR
- 각 앙상블 구성 방법별 curve → 어떤 방법이 "멀어질수록 uncertainty 증가" 패턴을 잘 보이는지

#### 2.5.6 Generated Sample Comparison

- Real data vs 각 EBM의 생성 샘플 (scatter, 다른 색)
- Union of all members' 생성 샘플 vs 개별 생성 샘플
- Feature별 histogram (real vs generated per member)

#### 2.5.7 Parallel Coordinates

- 각 데이터 포인트의 K개 energy 값을 parallel coordinates로 표시
- 교차(crossing)가 많으면 disagreement, bundle이면 agreement
- K = 5~10에서 효과적

---

### 2.7 실험 프로토콜

#### Step 2-1: Tier 1 — 이론적으로 정당한 방법

```
데이터셋: biodeg (N_real=100), stock (N_real=100)
앙상블 크기: K=5

(a) D4: Bayesian bootstrap (alpha=1, Dirichlet(1,...,1))
    → K=5개의 다른 Dirichlet weight로 TabPFN 학습
    → 구현: 가중치에 비례하여 positive sample duplicate

(b) N1: Corner subset randomization (5개 다른 seed)
    → K=5개의 다른 hypercube corner 조합

각각에 대해 다양성 메트릭 전체 계산 + 시각화
→ 어떤 방법이 "데이터 근처 agree, 먼 곳 disagree" 패턴을 보이는지 확인
```

#### Step 2-2: Tier 1 확장 — Feature Subsets (고차원 데이터)

```
다양성이 부족하거나 추가 diversity source 필요 시:

(c) F1: Feature subsets (d' = 0.7d, 0.8d, 0.9d)
    → gradient 결합 시: per-feature, 해당 feature 사용한 member만으로 mean/var 계산
    → biodeg(41d), protein(77d)에서 테스트
```

#### Step 2-3: Tier 2 — 부분 정당 방법 (Tier 1 부족 시)

```
(d) N2: Variable distance (dist = {3, 5, 7, 10})
    → 결과 해석 시 "sensitivity analysis" 프레이밍
(e) D1: Noise injection (sigma = {0.01, 0.05, 0.1})
    → sigma 선택 근거 명시 필요 (measurement noise로 해석 가능한 범위)
```

#### Step 2-4: Composite Ensemble (최종)

```
VALID 방법들만으로 M1 구성:
(f) M1: Bayesian bootstrap(D4) member 3개 + Corner subset(N1) member 2개
    → 또는 D4 x 5 (가장 이론적으로 깔끔)
    → K=5 composite ensemble

[INVALID 방법은 uncertainty ensemble에 포함하지 않음]
[E1(Temperature)은 별도 ablation: "T가 생성 샘플 품질에 미치는 영향" 실험]
```

#### Step 2-5: 전체 데이터셋 확장

```
Step 2-1~2-4에서 확정된 best method를 전체 8개 데이터셋으로 확장
N_real in {20, 50, 100, 200, 500}
```

---

### 2.7 판단 기준 및 의사결정 트리

```
                    Continuous KW Variance 계산
                           │
                    ┌──────┴──────┐
                    │             │
              매우 작음          유의미
          (< 1% of E_mean²)     │
                    │        ┌───┴───┐
              다양성 부족      │       │
                    │     E_std:     E_std:
              Extended   near≈far   near<far
              methods     │          │
              (Step 2-2)  │       ✅ 이상적
                       모델 불안정    │
                          │      Phase 3
                       HP 조정      진행
                       필요
```

**구체적 수치 기준 (tentative, 실험 결과 보며 조정)**:

| 지표 | 다양성 부족 | 유의미 | 매우 다양 |
|------|------------|--------|----------|
| Mean Pairwise Pearson Corr | > 0.98 | 0.80~0.95 | < 0.80 |
| GAR (데이터 근처) | < 0.5 | 0.7~0.95 | > 0.95 |
| GAR (먼 영역) | > 0.8 | 0.3~0.7 | < 0.3 |
| Coverage Gain (union vs individual) | < 5% | 10~30% | > 30% |

---

## Phase 3: Variance-Preconditioned SGLD

### 3.1 목적

- Ensemble의 uncertainty 정보를 SGLD sampling에 반영
- Uncertainty가 큰 영역에서 보수적으로 sampling하여 생성 데이터 품질 향상

### 3.2 방법

#### 3.2.1 Standard SGLD (TabEBM 원본)

```python
# 원본 TabEBM
x_{t+1} = x_t - alpha_step * grad(E(x_t)) + N(0, alpha_noise^2 * I)
```

#### 3.2.2 Ensemble Mean SGLD (단순 앙상블 평균)

```python
# 단순히 K개 EBM의 gradient 평균 사용
grad_mean = (1/K) * sum_k grad(E_k(x_t))
x_{t+1} = x_t - alpha_step * grad_mean + N(0, alpha_noise^2 * I)
```

#### 3.2.3 Variance-Preconditioned SGLD (핵심 제안)

```python
def vp_sgld_step(x, energy_models, alpha_step, alpha_noise, lam=1e-4):
    K = len(energy_models)
    
    # 각 EBM의 gradient 계산
    grads = [grad(E_k, x) for E_k in energy_models]
    grad_mean = sum(grads) / K
    grad_var = sum((g - grad_mean)**2 for g in grads) / K  # per-dimension
    
    # Preconditioner: variance 크면 G 작아짐 → 보수적
    G = 1.0 / (lam + sqrt(grad_var))
    
    # Preconditioned update
    x_new = x - (alpha_step / 2) * G * grad_mean + sqrt(alpha_noise * G) * randn_like(x)
    
    return x_new
```

**핵심 아이디어**:
- `G(x) = 1 / (lambda + sigma(x))` where sigma는 gradient의 ensemble std
- Gradient variance가 큰 dimension → G가 작음 → step size 작음 → 보수적
- Gradient variance가 작은 dimension → G가 큼 → step size 큼 → 적극적

#### 3.2.4 Energy Variance Thresholding (대안)

```python
# 단순한 대안: energy variance가 threshold 이상이면 step 거부
if E_std(x_t) > tau:
    x_{t+1} = x_t  # reject, stay in place
else:
    x_{t+1} = standard_sgld_step(x_t)
```

### 3.3 실험 조건

**비교 대상 (4가지)**:

| Method | 설명 |
|--------|------|
| (a) Baseline | Augmentation 없음 |
| (b) TabEBM (single) | 원논문 그대로, 단일 EBM |
| (c) TabEBM (ensemble mean) | K개 EBM gradient 평균만 사용 |
| (d) TabEBM (VP-SGLD) | Variance-preconditioned SGLD |

**Ablation 대상**:
- 앙상블 크기: K in {3, 5, 10}
- Lambda 값: lam in {1e-2, 1e-3, 1e-4, 1e-5}
- SGLD step 수: T in {100, 200, 500}
- 앙상블 구성 방법: {D4 (Bayesian bootstrap), N1 (Corner subset), D4+N1 composite}
  (VALID 방법만 사용, INVALID 방법은 uncertainty ensemble에서 제외)

**데이터셋**: Phase 1과 동일 (우선 biodeg, stock에서 검증 → 전체 확장)

**데이터 크기**: N_real in {20, 50, 100, 200, 500}

### 3.4 하이퍼파라미터 tuning 전략

1. 우선 원논문 기본값 유지 (alpha_step=0.1, alpha_noise=0.01, T=200)
2. Lambda만 grid search: {1e-2, 1e-3, 1e-4, 1e-5}
3. 최적 lambda 확정 후, alpha_step과 T에 대해 추가 탐색

### 3.5 기록할 메트릭

- **Balanced Accuracy** (6 classifiers x 10 runs)
- 방법 (a)~(d) 간 성능 비교
- N_real 크기별 성능 변화 추이
- 통계적 유의성: paired t-test 또는 Wilcoxon signed-rank test

---

## Phase 4: 시각화 및 분석

### 4.1 Energy Landscape 시각화

- 2D feature space (또는 PCA projection)에서:
  - 단일 EBM의 energy contour
  - 앙상블 평균 energy contour
  - Uncertainty (E_std) heatmap
  - Gradient variance heatmap

### 4.2 SGLD Trajectory 시각화

- Standard SGLD vs VP-SGLD의 sampling trajectory 비교
  - 같은 초기점에서 출발하여 궤적이 어떻게 달라지는지
  - VP-SGLD가 uncertainty 높은 영역을 실제로 피하는지 확인
- Trajectory 위에 uncertainty colormap overlay

### 4.3 Generated Sample 품질 시각화

- Real data vs Generated data scatter plot
- Feature별 histogram 비교 (real vs synthetic)
- 방법 (b) vs (d)의 생성 샘플 분포 비교

### 4.4 Uncertainty vs Performance 분석

- Ensemble uncertainty가 큰 영역에서의 생성 샘플 비율 비교
  - Standard SGLD: uncertainty 높은 곳에서도 많이 생성될 수 있음
  - VP-SGLD: uncertainty 높은 곳에서 생성이 억제됨
- Uncertainty threshold별 생성 샘플 수 분포

### 4.5 Statistical Fidelity (원논문 메트릭 따라가기)

- Inverse KL divergence (real vs synthetic, per-feature)
- KS test p-value
- Chi-squared test p-value
- 방법 (b) vs (d) 비교

---

## 전체 실험 순서 요약

```
Step 1: 환경 셋업 + TabEBM 코드 확보 ✅ 완료
  └─ TabEBM repo, conda env (TabEBM), 의존성 설치
  └─ openml, imbalanced-learn, xgboost, category_encoders 추가 설치
  └─ GPU 4장 확인 (RTX 6000 Ada x4), device 선택 기능 추가

Step 2: TabEBM reproduce (Phase 1)
  ├─ Tier 1 ✅ 완료 (2025-04-09)
  │   └─ biodeg, stock / N_real=100 / Baseline,SMOTE,TabEBM / KNN,RF,TabPFN / 5 splits
  │   └─ 결과: experiments/results/tier1/
  │   └─ stock에서 augmentation 효과 확인 (KNN +3~4pp)
  │   └─ biodeg에서 TabEBM RF만 +2pp, 나머지 하락
  ├─ Tier 2 (TODO)
  │   └─ 8개 전체 데이터셋 / {50,100,200} / 6개 classifier / 10 splits
  │   └─ GPU 4개에 데이터셋 분배하여 병렬 실행
  └─ Tier 3 (TODO)
      └─ Synthcity baselines 추가 + UCI 데이터셋 + Statistical fidelity

Step 3: 앙상블 구성 + 다양성 분석 (Phase 2)
  └─ Tier 1: D4 (Bayesian bootstrap) + N1 (Corner subset) 먼저
  └─ 다양성 측정 + 시각화
  └─ 다양성 부족 시 → Tier 2 methods 추가

Step 4: VP-SGLD 구현 + 실험 (Phase 3)
  └─ (a)~(d) 4가지 방법 비교
  └─ 소규모 → 전체 데이터셋 확장
  └─ Ablation study

Step 5: 시각화 + 분석 정리 (Phase 4)
  └─ Energy landscape, trajectory, sample quality 시각화
  └─ 정량적/정성적 분석

현재 위치: Step 2 Tier 1 완료, Tier 2 대기
```

---

## 주의사항

1. **TabPFN 버전**: 현재 코드는 TabPFN v2.1.2 사용 (requirements.txt). 원논문 재현은 tabpfn==0.1.9 (requirements_paper.txt). Reproduce 시 버전 확인 필수.
2. **TabPFN은 대체로 deterministic**: n_estimators=1, preprocessing=none으로 설정됨. 같은 입력 → 같은 출력. 따라서 TabPFN seed variation(C2)은 실질적 diversity 제공이 매우 제한적.
3. **Energy normalization**: 현재 코드에서 total_energy를 num_features로 나눔 (TabEBM.py:493). 이는 비표준적이며 SGLD dynamics에 영향. 재현 시 확인 필요.
4. **Computation**: 앙상블 크기 K가 커지면 gradient 계산이 K배 → SGLD step마다 K번 forward/backward. K=10이면 10배 느려짐에 유의.
5. **VP-SGLD Correction term**: Position-dependent preconditioner G(x) 사용 시, divergence term div(G(x))가 이론적으로 필요. Drop하면 stationary distribution이 의도한 것과 달라짐. pSGLD (Li et al. 2016)에서 RMSProp 기반 preconditioner 시 drop해도 경험적으로 잘 작동하나, ensemble variance 기반 preconditioner에서의 영향은 미검증. 우선 drop하고, correction 유무 비교 실험 권장.
6. **Bayesian bootstrap 구현**: TabPFN은 sample weighting을 직접 지원하지 않음. Duplication으로 근사할 때, weight를 정수로 반올림해야 하므로 연속 가중치의 이산화 오차 발생. 이를 최소화하려면 총 sample 수를 충분히 크게 (예: 100개로 scaling) 설정.
7. **재현성**: 모든 실험에 seed 고정. 결과 기록시 seed, 하이퍼파라미터, 데이터 split 번호 전부 저장.
8. **여주경 학생과 협업**: 코드 구조와 실험 결과를 공유하기 쉽게 정리. 실험 config를 yaml/json으로 관리 권장.
9. **[INVALID] 방법 사용 시 프레이밍**: E1(Temperature)이나 E2(Energy definition)를 실험에 포함하더라도, "uncertainty estimation"이 아닌 "ablation study" 또는 "sensitivity analysis"로 명확히 프레이밍해야 함. 논문 작성 시 이를 uncertainty로 해석하면 reviewer에게 지적받을 수 있음.

---

## 참고 문헌

- TabEBM 원논문 — Margeloiu et al. (NeurIPS 2024), arXiv:2409.16118
- Welling & Teh (2011) — Bayesian Learning via Stochastic Gradient Langevin Dynamics
- Li et al. (2016) — Preconditioned Stochastic Gradient Langevin Dynamics (pSGLD), arXiv:1512.07666
- Lakshminarayanan et al. (2017) — Simple and Scalable Predictive Uncertainty Estimation using Deep Ensembles
- **Rubin (1981) — The Bayesian Bootstrap** (D4 방법의 이론적 근거)
- Wilson & Izmailov (2020) — Bayesian Deep Learning and a Probabilistic Perspective of Generalization
- Fort, Hu & Lakshminarayanan (2019) — Deep Ensembles: A Loss Landscape Perspective
- Grathwohl et al. (2020) — Your Classifier is Secretly an Energy Based Model (JEM)
- Pearce et al. (2020) — Uncertainty in Neural Networks: Approximately Bayesian Ensembling (M2 방법 근거)
- Kuncheva & Whitaker (2003) — Measures of Diversity in Classifier Ensembles
- Ho (1998) — The Random Subspace Method (F1 방법 근거)
- 교수님 공유 자료 (variance-preconditioned Langevin augmentation 관련)
