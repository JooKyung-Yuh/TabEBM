# Integrated Critical Plan for Bayesian TabEBM

## 문서 목적

이 문서는 다음 두 문서를 비판적으로 통합한 실행 계획서다.

- `experiment_plan.md`
- `ensemble_experiment_protocol.md`

이 문서의 목표는 단순한 아이디어 정리가 아니라, 다음 세 가지를 동시에 만족하는 연구 계획을 만드는 것이다.

1. 교수님이 요청한 큰 그림을 충실히 따른다.
2. 실험이 실제로 공정한 baseline 비교가 되도록 설계한다.
3. 결과 해석이 과도해지지 않도록, 어떤 방법이 uncertainty baseline으로 유효한지 명확히 제한한다.


## 한 줄 요약

가장 엄밀하고 방어 가능한 proof-of-concept는 아래 순서다.

1. `Single TabEBM`을 먼저 재현한다.
2. `Bayesian bootstrap ensemble`과 `corner-subset ensemble`만 먼저 구현한다.
3. 이 ensemble들이 실제로 uncertainty-like diversity를 보이는지 먼저 검증한다.
4. diversity가 확인되면 같은 ensemble으로 `Union`, `MeanGrad`, `VP-SGLD`를 비교한다.
5. 성능과 시각화를 함께 제시한다.

이 순서를 벗어나면, 결과가 좋아 보여도 해석이 약해질 가능성이 크다.


## 기존 `experiment_plan.md`에 대한 비판적 평가

기존 문서는 매우 강한 초안을 갖고 있다. 특히 다음 점들이 좋다.

- 연구 질문이 명확하다.
- deterministic EBM의 한계를 ensemble/Bayesian 관점으로 확장하려는 동기가 분명하다.
- invalid/partial/valid를 구분하려는 시도가 있다.
- diversity 측정과 VP-SGLD를 분리해서 보려는 의도가 보인다.
- 구현 리스크와 이론적 caveat를 문서 후반부에서 미리 인지하고 있다.

하지만 현재 상태 그대로 바로 실행에 들어가면 몇 가지 문제가 생긴다.


## 핵심 비판 1: `VALID` 판정이 일부 방법에서 너무 강하다

기존 문서에서는 다음 방법들이 `VALID`로 분류되어 있다.

- N1: Corner subset
- F1: Feature subset
- D4: Bayesian bootstrap
- M2: Anchored MLP
- M3: Snapshot ensemble

이 중에서 `D4`와 `M2`는 비교적 강한 정당화가 가능하지만, `N1`, `F1`, `M3`는 같은 수준의 `VALID`로 두기에는 과감하다.

### N1: Corner subset

- 장점: TabEBM 내부에서 가장 자연스럽고 강한 ensemble source다.
- 문제: Bayesian posterior sampling과 직접 연결된 것은 아니다.
- 따라서 `strong ensemble baseline`으로는 매우 좋지만, `formal Bayesian uncertainty`로 부르는 것은 무리다.

권장 수정:

- `VALID` 대신 `STRONG_BASELINE` 또는 `PARTIAL+` 정도로 내리는 것이 더 정직하다.

### F1: Feature subset

- 장점: random subspace라는 고전적 ensemble 근거가 있다.
- 문제: gradient가 다른 subspace에 존재하므로, variance-preconditioned SGLD의 핵심 연산이 바로 깨질 수 있다.
- 즉, classification ensemble로는 성립해도, `gradient uncertainty ensemble`로는 주의가 훨씬 크다.

권장 수정:

- `VALID`보다 `PARTIAL`이 적절하다.

### M3: Snapshot ensemble

- 장점: trainable MLP classifier를 쓸 때 실용적일 수 있다.
- 문제: TabPFN 기반 TabEBM에는 직접 적용되지 않는다.
- 현재 proof-of-concept의 main path에는 사실상 들어갈 수 없다.

권장 수정:

- `future external baseline`으로 이동한다.


## 핵심 비판 2: main proof-of-concept에 너무 많은 ensemble source가 들어가 있다

기존 계획은 Category 1~6까지 상당히 넓게 열어두고 있다. 아이디어 차원에서는 좋지만, proof-of-concept 단계에서는 위험하다.

이유:

- ensemble source가 너무 많으면 결과가 좋아도 해석이 흐려진다.
- method space가 넓을수록 hyperparameter sensitivity를 uncertainty처럼 착각하기 쉽다.
- 구현 난이도와 실험 비용이 급격히 늘어난다.

특히 다음들은 main path에서 빼는 것이 맞다.

- N2: Variable distance
- N3: Non-corner geometry
- D1: Noise injection
- D3: Mixup
- C1: Heterogeneous classifier ensemble
- E1/E2/F3

이들은 `안 하는 것`이 아니라, `main uncertainty baseline으로 두지 않는 것`이다.


## 핵심 비판 3: Phase 3 비교군이 아직 충분히 공정하지 않다

기존 문서의 Phase 3 비교 대상은 아래 4개다.

- Baseline
- Single TabEBM
- Ensemble mean
- VP-SGLD

하지만 이것만으로는 아직 부족하다.

### 빠진 핵심 비교군: `Ens-Union`

같은 ensemble을 쓰되,

- 각 member가 standard SGLD로 synthetic data를 생성하고
- 최종 샘플을 합치는 baseline

이 반드시 있어야 한다.

왜냐하면:

- VP-SGLD가 좋아 보여도, 그게 variance-aware dynamics 때문인지
- 아니면 단순히 ensemble member들을 여러 개 쓴 덕분인지

를 구분할 수 있어야 하기 때문이다.

권장 비교군은 아래 5개다.

1. `Real-only`
2. `Single-TabEBM`
3. `Ens-Union`
4. `Ens-MeanGrad`
5. `Ens-VP-SGLD`


## 핵심 비판 4: `모든 ensemble component는 모든 샘플을 다 쓴다`와 `Bayesian bootstrap` 사이의 긴장이 정리되어야 한다

기존 계획은 ensemble component가 전체 데이터를 다 쓴다고 적고 있다. 교수님 말씀도 그 방향에 가깝다.

하지만 엄밀한 Bayesian bootstrap은 다음을 허용한다.

- 어떤 샘플은 중복 선택된다.
- 어떤 샘플은 0회 선택될 수 있다.

즉, 엄밀한 bootstrap은 “모든 샘플을 반드시 1회 이상 쓴다”와 다르다.

따라서 이 부분은 문서에서 분리해야 한다.

### 권장 분리

#### BB-resample

- multinomial/duplication 기반
- 엄밀성 우선

#### BB-all-data

- 모든 샘플을 최소 1회 포함
- 추가 duplication만 weight로 구현
- 교수님 의도 반영

main uncertainty baseline은 `BB-resample`, 보조 분석은 `BB-all-data`가 적절하다.


## 핵심 비판 5: `Composite ensemble`은 너무 이르다

기존 계획은 `D4 + N1` composite도 초기부터 염두에 둔다.

하지만 composite는 다음 문제가 있다.

- 어떤 source가 diversity를 만든 것인지 분리가 안 된다.
- variance가 커져도 epistemic uncertainty 때문인지 설계 혼합 때문인지 불분명해진다.
- proof-of-concept 첫 결과를 설명하기 어려워진다.

권장 수정:

- 1차 proof-of-concept에서는 `순수 D4`, `순수 N1`만 사용한다.
- composite는 두 방법이 각각 성공한 뒤에만 appendix 또는 2차 실험으로 넣는다.


## 핵심 비판 6: Phase 1과 현재 코드 경로를 더 분명히 분리해야 한다

기존 문서는 후반에서 이 리스크를 언급하지만, 계획의 상위 구조에서 더 분명히 드러나야 한다.

### 현재 중요한 차이

- 논문 재현은 `tabpfn==0.1.9`
- 현재 로컬 코드는 `tabpfn==2.1.2`
- 현재 구현은 `n_estimators=1`
- 현재 구현은 energy normalization과 caching behavior에서 논문과 차이가 있다

결론:

- `Phase 1: paper reproduction`
- `Phase 2+: v2-based research extension`

이 둘을 문서 구조상 아예 분리해야 한다.


## 핵심 비판 7: VP-SGLD의 이론적 표현이 너무 강해질 위험이 있다

기존 문서는 correction term 문제를 이미 잘 짚고 있다.

하지만 실험 설계에서도 표현을 조심해야 한다.

- position-dependent preconditioner `G(x)`를 쓰면, 정확한 Langevin sampler 해석에는 보정항이 필요하다.
- 이를 생략하면 실제론 heuristic에 가깝다.

따라서 1차 결과에서는 아래 표현이 안전하다.

- `variance-aware SGLD`
- `variance-preconditioned heuristic`

반면 아래 표현은 피한다.

- `exact Bayesian sampler`
- `posterior-correct sampler`


## 통합된 연구 질문

이제 문서를 아래 두 개의 핵심 질문으로 재구성한다.

### RQ1. TabEBM ensemble은 uncertainty-like diversity를 보이는가?

보다 구체적으로:

- 데이터가 있는 곳에서는 agreement가 높고
- 데이터가 없는 곳에서는 disagreement가 높아지는가?

### RQ2. 그 diversity를 활용한 variance-aware sampling이 실제로 synthetic data quality를 높이는가?

보다 구체적으로:

- 같은 ensemble에 대해 `Ens-Union`, `Ens-MeanGrad`, `Ens-VP-SGLD`를 비교했을 때
- VP-SGLD가 downstream 성능과 synthetic fidelity를 개선하는가?


## 통합된 최종 실험 구조

## Phase 0. 환경 분리 및 구현 검증

### 목표

- 논문 재현 환경과 현재 연구용 환경을 분리한다.

### 트랙

#### Track A: Paper reproduction

- `requirements_paper.txt`
- `tabpfn==0.1.9`
- 목적: 원논문 재현

#### Track B: Research extension

- 현재 저장소 기준
- `tabpfn==2.1.2`
- 목적: ensemble + variance-aware SGLD extension

### 체크포인트

- [ ] 두 환경을 별도 디렉터리/conda env로 분리
- [ ] 결과 테이블도 paper vs extension 분리


## Phase 1. Single TabEBM reproduce

### 목표

- 원논문 augmentation 효과를 먼저 확인한다.

### 데이터셋

초기:

- `biodeg`
- `stock`

이후:

- OpenML 8개

### 데이터 크기

- `N_real in {20, 50, 100, 200, 500}`
- 초기 디버그는 `N_real = 100`

### splits

- debug: `3 splits`
- final: `10 splits`

### 비교

1. `Real-only`
2. `Single-TabEBM`

### downstream predictors

debug:

- Logistic Regression
- Random Forest
- TabPFN

final:

- Logistic Regression
- KNN
- MLP
- Random Forest
- XGBoost
- TabPFN

### 목적

- `augmentation 자체가 유의미한가?`
- 이게 먼저 성립하지 않으면 이후 확장은 의미가 약해진다.


## Phase 2. Ensemble diversity validation

이 단계에서는 오직 아래 두 ensemble만 main path에 포함한다.

1. `Bayesian bootstrap ensemble`
2. `Corner-subset ensemble`

### 2.1 Bayesian bootstrap ensemble

#### 목적

- positive empirical distribution에 대한 epistemic uncertainty 근사

#### main variant

- `BB-resample`

#### optional variant

- `BB-all-data`

#### 구현 원칙

- negative set은 member 간 동일하게 유지
- preprocessing은 member 간 동일
- classifier 구조는 동일

### 2.2 Corner-subset ensemble

#### 목적

- TabEBM 내부에서 가장 자연스러운 surrogate-task variation

#### 구현 원칙

- positive set은 고정
- negative distance는 고정
- number of negatives는 고정
- member마다 corner identity만 변경

### 2.3 보조 확장

다양성이 부족할 때만:

- `Feature-subspace ensemble`

이 단계에서는 아직 아래는 하지 않는다.

- variable distance
- geometry variation
- noise injection
- mixup
- heterogeneous classifier
- composite ensemble


## Phase 2의 diversity 판정 기준

다양성이 있다는 말은 단순히 “값이 다르다”가 아니라, 아래 패턴이 있어야 한다.

### 바람직한 패턴

- `X_real`에서 energy variance가 작다
- `X_far`에서 energy variance가 커진다
- `X_real`에서 gradient agreement ratio가 높다
- `X_far`에서 gradient agreement ratio가 낮다

### 실패 패턴

- 모든 영역에서 member가 거의 동일
- pairwise correlation이 대부분 `0.99+`
- toy 2D에서만 차이가 나고 real dataset에서는 spread가 거의 없음

### 주요 평가 포인트

- `X_real`
- `X_near`
- `X_far`
- 가능하면 `between-class corridor`

### 주요 메트릭

- energy variance
- Pearson/Spearman correlation
- mean pairwise disagreement
- gradient agreement ratio
- per-dim gradient variance
- coverage gain

### go / no-go

다양성 검증이 실패하면:

- VP-SGLD로 바로 가지 않는다.
- ensemble source를 재설계한다.


## Phase 3. Variance-aware sampling

Phase 2에서 diversity가 확인된 ensemble에 대해서만 진행한다.

### 공정한 비교군

1. `Real-only`
2. `Single-TabEBM`
3. `Ens-Union`
4. `Ens-MeanGrad`
5. `Ens-VP-SGLD`

### 왜 이 5개가 필요한가

- `Real-only`: augmentation 전체 효과 확인
- `Single-TabEBM`: ensemble 자체 효과 분리
- `Ens-Union`: 여러 member를 독립적으로 활용한 효과 확인
- `Ens-MeanGrad`: variance 없이 ensemble averaging만 한 효과 확인
- `Ens-VP-SGLD`: variance-aware dynamics의 추가 기여 확인

### 절대 지켜야 할 공정성 조건

- 총 synthetic sample 수 동일
- 같은 초기점 사용
- 같은 SGLD noise seed 사용
- 같은 step 수 사용
- 같은 preprocessing 사용

### 추천 SGLD 기본값

- `alpha_step = 0.1`
- `alpha_noise = 0.01`
- `T = 200`
- `sigma_start = 0.01`

### lambda grid

- `1e-2`
- `1e-3`
- `1e-4`
- `1e-5`

우선 lambda만 조정하고, 이후에 필요 시 `alpha_step`, `T`를 튜닝한다.


## Phase 4. 시각화 및 해석

숫자만으로는 부족하다. 최소한 아래를 함께 보여야 한다.

### 필수 시각화

1. `Energy variance heatmap`
2. `Gradient agreement map`
3. `Standard SGLD vs VP-SGLD trajectory`
4. `Real vs synthetic comparison`
5. `Coverage/overlap comparison`

### 시각화 메시지

- uncertainty가 큰 영역에서 VP-SGLD가 더 보수적으로 움직이는가?
- synthetic samples가 data manifold 밖으로 덜 이탈하는가?
- 그 결과 downstream 성능이 개선되는가?


## 최종 baseline 정책

## main proof-of-concept baseline으로 인정할 것

- `Real-only`
- `Single-TabEBM`
- `BB ensemble`
- `Corner-subset ensemble`
- `Ens-Union`
- `Ens-MeanGrad`
- `Ens-VP-SGLD`

## 2차/보조 baseline

- `Feature-subspace ensemble`
- `Anchored MLP ensemble`
- `MLP deep ensemble`

## appendix / stress test

- variable distance
- geometry variation
- noise injection
- mixup
- temperature
- alternative energy definition

## 제외

- normalization scheme ensemble
- TabPFN seed-only를 main uncertainty baseline으로 사용하는 것


## 구현 리스크와 보완책

### 1. 현재 TabEBM 객체 재사용 위험

현재 구현에는 cache가 있으며, ensemble member 사이에서 같은 객체를 재사용하면 오염될 위험이 있다.

조치:

- member마다 독립 `TabEBM` 인스턴스를 사용한다.

### 2. paper reproduction vs current optimized code 혼합 위험

재현 실패 원인이 method 때문인지 버전 차이 때문인지 구분이 안 될 수 있다.

조치:

- 환경 분리
- 결과 테이블 분리

### 3. BB duplication approximation

sample weighting을 duplication으로 구현하면 이산화 오차가 생긴다.

조치:

- scaling factor를 충분히 크게 두고
- `BB-resample`과 `BB-all-data`를 분리 기록한다

### 4. feature-subspace gradient aggregation 문제

feature를 안 본 member의 gradient를 0으로 넣으면 편향이 생긴다.

조치:

- 해당 feature를 실제로 사용한 member만으로 mean/var를 계산한다.

### 5. VP-SGLD의 이론적 표현 과장 위험

조치:

- 1차 문서에서는 `heuristic` 또는 `variance-aware`라고 표현한다.


## 최종 추천 실행 순서

### Step 1

논문 환경으로 `Single TabEBM reproduce`

### Step 2

현재 코드 환경에서 `Single TabEBM` baseline 정리

### Step 3

`BB ensemble` 구현

### Step 4

`Corner-subset ensemble` 구현

### Step 5

다양성 검증

### Step 6

go/no-go 판단

- 실패 시: ensemble source 재설계
- 성공 시: VP-SGLD 진행

### Step 7

같은 ensemble으로 `Union`, `MeanGrad`, `VP-SGLD` 비교

### Step 8

성능 + 시각화 + 통계 검정 정리


## 바로 실행 가능한 체크리스트

- [ ] Paper reproduction environment 분리
- [ ] Research extension environment 분리
- [ ] `Real-only` vs `Single-TabEBM` 먼저 비교
- [ ] `BB-resample` 구현
- [ ] `BB-all-data`는 optional로 구현
- [ ] `Corner-subset ensemble` 구현
- [ ] diversity metrics 계산
- [ ] diversity go/no-go 판단
- [ ] `Ens-Union` 구현
- [ ] `Ens-MeanGrad` 구현
- [ ] `Ens-VP-SGLD` 구현
- [ ] synthetic budget 통제 확인
- [ ] 같은 초기점 / 같은 noise seed 통제 확인
- [ ] trajectory 시각화
- [ ] downstream 성능 비교
- [ ] 통계 검정
- [ ] proof-of-concept 결과 정리


## 결론

기존 `experiment_plan.md`는 아이디어가 풍부하고 방향성이 매우 좋다. 다만 proof-of-concept 단계에서 너무 많은 ensemble source를 열어두고 있어서, 그대로 실행하면 결과 해석이 약해질 위험이 있다.

따라서 통합된 최종 전략은 아래처럼 압축하는 것이 맞다.

- 먼저 재현
- 그다음 가장 엄밀한 두 ensemble만
- diversity를 먼저 확인
- 그다음 variance-aware sampling
- 마지막에 성능과 시각화

이 경로가 가장 보수적이고, 가장 설득력 있고, 가장 baseline다운 연구 계획이다.
