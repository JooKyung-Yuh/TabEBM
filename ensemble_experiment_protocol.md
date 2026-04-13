# TabEBM Ensemble Experiment Protocol

## 목적

이 문서는 TabEBM 기반 연구를 진행할 때, 다음 질문들을 엄밀하게 검증하기 위한 실험 기준 문서다.

1. TabEBM 내부에서 ensemble spread가 실제 uncertainty로 해석 가능한가?
2. 그 uncertainty를 variance-preconditioned SGLD에 반영하면 synthetic data quality가 좋아지는가?
3. 어떤 ensemble 구성 방법이 baseline으로 유효하고, 어떤 방법은 appendix나 stress test로만 다뤄야 하는가?

이 문서의 핵심 원칙은 다음과 같다.

- `TabEBM reproduce -> ensemble diversity 검증 -> variance-aware sampling 검증` 순서를 반드시 지킨다.
- `논문 재현`과 `새로운 연구 baseline`을 동일한 테이블에서 바로 섞지 않는다.
- ensemble spread를 `Bayesian uncertainty`라고 부르기 전에, 그 spread가 정말 epistemic uncertainty를 반영하는지 먼저 검토한다.


## 가장 중요한 구조적 판단

실험 질문은 반드시 두 단계로 분리해야 한다.

### 질문 A: Ensemble diversity의 유효성

TabEBM 여러 개를 만들었을 때, 같은 입력 `x`에 대해 member별 energy 또는 gradient가 실제로 의미 있는 차이를 보이는가?

- 데이터가 있는 영역에서는 member들이 대체로 동의해야 한다.
- 데이터가 없는 영역에서는 disagreement가 커져야 한다.
- 이 패턴이 없다면 ensemble variance는 uncertainty proxy로 쓰기 어렵다.

### 질문 B: Variance-aware sampling의 유효성

ensemble variance가 있다고 해도, 그것을 SGLD에 넣었을 때 실제 synthetic sample quality가 좋아지는지는 별도 질문이다.

- 따라서 `diversity가 있다`와 `sampling에 도움이 된다`는 별도로 검증해야 한다.
- 두 질문을 섞으면, 개선이 variance 때문인지 단순 ensemble averaging 때문인지 구분할 수 없다.


## 현재 코드/논문 스택 분리 원칙

현재 로컬 저장소의 구현과 논문 재현 환경은 동일하지 않다.

### 논문 재현 환경

- `requirements_paper.txt` 기준
- `tabpfn==0.1.9`
- 논문은 TabPFN ensemble 설정과 원논문용 실험 구성이 있음

### 현재 로컬 구현

- `requirements.txt` 기준
- `tabpfn==2.1.2`
- `src/tabebm/TabEBM.py`는 최적화된 단일 클래스 구현
- `n_estimators=1`

### 결론

- `paper reproduction`과 `new research baseline`은 별도 트랙으로 기록한다.
- 최종 표도 가능하면 분리한다.
- 첫 번째 목표는 논문 재현이고, 두 번째 목표가 ensemble/variance-aware extension이다.


## Ensemble 방법별 엄밀성 평가

아래 평가는 다음 기준으로 한다.

- `baseline 적합성`: proof-of-concept의 main table에 둘 수 있는가?
- `uncertainty 해석`: spread를 epistemic uncertainty로 해석할 수 있는가?
- `실험 우선순위`: 지금 당장 구현해야 하는가?

| 방법 | baseline 적합성 | uncertainty 해석 | 권장 수준 | 비고 |
|---|---|---|---|---|
| Bayesian bootstrap positive reweighting | 높음 | 높음 | 최우선 | 가장 엄밀한 출발점 |
| Corner-subset negative ensemble | 높음 | 중간 | 최우선 | TabEBM 내부 baseline으로 매우 강함 |
| Feature-subspace ensemble | 중간 | 중간 | 보조 | gradient aggregation 주의 |
| TabPFN seed-only ensemble | 낮음 | 낮음 | 비추천 | 현재 구조상 다양성 부족 가능성 큼 |
| Variable negative distance | 낮음 | 낮음 | stress test only | surrogate task 자체가 달라짐 |
| Alternative negative geometry | 낮음 | 낮음 | stress test only | inductive bias 변화에 가까움 |
| Positive noise injection | 낮음 | 낮음~중간 | appendix | aleatoric perturbation 성격 |
| Mixup positives | 낮음 | 낮음 | appendix | manifold smoothing 쪽에 가까움 |
| Temperature variation | 매우 낮음 | 매우 낮음 | 제외 | uncertainty보다 target distribution 변화 |
| Alternative energy definition | 매우 낮음 | 매우 낮음 | 제외 | scale/meaning 모두 달라짐 |
| Different normalization schemes | 매우 낮음 | 매우 낮음 | 제외 | gradient 비교가 불안정해짐 |
| Trainable MLP deep ensemble | 높음 | 높음 | 2차 단계 | strong external baseline |
| Anchored MLP ensemble | 높음 | 높음 | 2차 단계 | Bayesian-like 해석이 더 깔끔 |


## 왜 어떤 방법은 main baseline으로 쓰면 안 되는가

### 1. TabPFN seed-only ensemble

- 현재 TabEBM은 pretrained TabPFN을 다시 활용하는 구조다.
- 학습 가능한 neural net를 seed 다르게 여러 번 학습하는 deep ensemble과는 다르다.
- 현재 구현은 `n_estimators=1`이라 내부 stochasticity도 작다.
- 따라서 seed만 바꾸는 방식은 diversity가 거의 안 나올 가능성이 크다.

결론:

- `main uncertainty baseline`으로는 부적절하다.
- 해도 “almost deterministic sanity check” 정도로만 둔다.

### 2. Variable negative distance

- negative distance를 바꾸면 동일한 posterior belief를 샘플링하는 것이 아니라 surrogate binary problem 정의가 달라진다.
- 따라서 member spread는 uncertainty라기보다 model specification sensitivity에 가깝다.

결론:

- main table에는 넣지 않는다.
- appendix의 robustness/stress test로만 적절하다.

### 3. Alternative negative geometry

- corner, sphere, axis-aligned, Gaussian shell은 모두 energy landscape inductive bias를 바꾼다.
- diversity는 클 수 있지만, epistemic uncertainty라고 부르기는 어렵다.

결론:

- main uncertainty baseline으로는 약하다.
- “geometry sensitivity analysis”로는 유효하다.

### 4. Temperature / alternative energy / normalization

- 이 셋은 uncertainty ensemble이 아니라, 다른 density 혹은 다른 좌표계/스케일을 정의하는 문제에 가깝다.
- 특히 normalization이 달라지면 gradient variance를 직접 합치는 것이 수학적으로 위험하다.

결론:

- proof-of-concept main 실험에서는 제외한다.


## 1차 proof-of-concept에서 반드시 필요한 baseline

최소한 아래 4개가 있어야 한다.

1. `Real-only`
2. `Single-TabEBM`
3. `Ensemble-mean control`
4. `Variance-preconditioned SGLD`

여기서 `Ensemble-mean control`이 매우 중요하다.

- 만약 이것이 없으면, 개선이 `variance 사용` 때문인지 `그냥 모델 여러 개 평균냈기 때문`인지 분리할 수 없다.

실제로는 아래 5종 비교가 가장 좋다.

1. `Real-only`
2. `Single-TabEBM`
3. `Ens-Union`
4. `Ens-MeanGrad`
5. `Ens-VP-SGLD`


## 추천하는 1차 ensemble 세트

### A. Bayesian Bootstrap Ensemble

가장 엄밀한 uncertainty baseline으로 추천한다.

#### 목표

- class-specific positive empirical distribution에 대한 epistemic uncertainty를 ensemble로 근사한다.

#### member 생성 원리

각 class `c`에 대해 positive set `X_c = {x_i}`가 있을 때:

- `w^(k,c) ~ Dirichlet(1, ..., 1)`를 샘플링한다.
- 그 weight를 정수 duplication count로 근사해 pseudo-training set을 만든다.
- negative sample 배치는 모든 member에서 동일하게 고정한다.
- classifier와 SGLD 설정도 동일하게 고정한다.

#### 구현 방식

두 버전으로 나누는 것이 가장 좋다.

##### BB-resample

- weight로부터 multinomial count를 뽑아 duplication한다.
- 통계적으로 가장 엄밀한 쪽이다.
- 일부 sample은 0회 선택될 수 있다.

##### BB-all-data

- 모든 sample을 최소 1회 포함한 뒤, 추가 duplication만 weight로 부여한다.
- 교수님 말씀의 “모든 ensemble component는 모든 sample을 다 쓴다”는 해석에 더 가깝다.
- 다만 Bayesian bootstrap의 엄밀한 구현은 아니다.

#### 장점

- uncertainty 해석이 가장 깔끔하다.
- ensemble member 간 차이가 positive empirical distribution belief 차이로 해석 가능하다.

#### 약점

- TabPFN이 sample weighting을 직접 지원하지 않으므로 duplication approximation이 필요하다.
- 최종 논문 표현은 `Bayesian-bootstrap-inspired ensemble` 정도가 더 정직하다.

#### baseline 적합성 판단

- `main baseline 가능`
- `uncertainty baseline으로 매우 강함`


### B. Corner-Subset Negative Ensemble

TabEBM 내부에서 가장 강한 engineering baseline이다.

#### 목표

- TabEBM 고유의 surrogate negative design 내부에서 diversity를 유도한다.

#### member 생성 규칙

- positive set은 모든 member에서 동일
- negative distance는 고정
- negative 개수도 고정
- 바꾸는 것은 `which corners are selected` 뿐

#### 강하게 고정해야 하는 값

- `distance_negative_class`
- `num_surrogate_negatives`
- negative geometry type은 hypercube로 고정

#### 장점

- 교수님이 직접 언급한 방향과 가장 일치한다.
- 구현이 쉽고 TabEBM 내부 논리를 거의 유지한다.
- proof-of-concept용 baseline으로 매우 강하다.

#### 약점

- Bayesian posterior approximation이라고 보기는 어렵다.
- surrogate task perturbation에 가깝다.

#### baseline 적합성 판단

- `main baseline 가능`
- 다만 표현은 `ensemble baseline`이지 `Bayesian baseline`은 아니다.


### C. Feature-Subspace Ensemble

다양성이 부족할 때 추가하는 보조 baseline이다.

#### 방식

- 각 member가 전체 feature 중 일부만 사용해 EBM을 구성
- 예: `0.7d`, `0.8d`, `0.9d`

#### 장점

- 고차원 데이터에서 diversity를 크게 만들 수 있다.
- classical random subspace baseline으로 설명 가능하다.

#### 위험

- 각 member의 gradient가 다른 subspace에 존재한다.
- VP-SGLD에서 per-dimension mean/variance를 합칠 때, 해당 feature를 실제로 사용한 member만 계산에 포함해야 한다.

#### baseline 적합성 판단

- `보조 baseline 가능`
- 첫 번째 main proof-of-concept 세트에는 넣지 않는 것이 좋다.


## 지금은 제외하는 것이 맞는 세트

아래는 “하면 안 된다”가 아니라, `main uncertainty baseline`으로는 부적절하다는 뜻이다.

- variable negative distance
- negative geometry variants
- positive noise injection
- mixup positives
- temperature variation
- alternative energy definitions
- different normalization schemes
- TabPFN seed-only ensemble

이들은 appendix, ablation, stress test로는 의미가 있지만, proof-of-concept의 핵심 테이블에 올리면 해석이 약해진다.


## 추천 baseline 조합

1차 proof-of-concept의 내부 baseline은 아래 구성이 가장 균형이 좋다.

### 내부 비교용

1. `Real-only`
2. `Single-TabEBM`
3. `TabEBM-Ens(BB)-Union`
4. `TabEBM-Ens(BB)-MeanGrad`
5. `TabEBM-Ens(BB)-VP-SGLD`
6. `TabEBM-Ens(Corner)-Union`
7. `TabEBM-Ens(Corner)-MeanGrad`
8. `TabEBM-Ens(Corner)-VP-SGLD`

### 외부 생성 baseline

proof-of-concept에서는 너무 많이 넣지 말고 아래 정도만 추천한다.

- `SMOTE`
- `TabPFGen`
- 필요 시 `CTGAN` 또는 `TabDDPM`


## 실험 순서

실험은 아래 순서를 유지한다.

### Step 1. Single TabEBM reproduce

- 논문 스택으로 먼저 재현
- augmentation이 실제로 baseline 대비 성능 향상을 주는지 확인

### Step 2. Ensemble diversity 검증

- `Bayesian bootstrap ensemble`
- `Corner-subset ensemble`

이 둘만 먼저 구현한다.

### Step 3. Variance-aware sampling 검증

같은 ensemble member 집합으로 아래 3개 비교:

- `Ens-Union`
- `Ens-MeanGrad`
- `Ens-VP-SGLD`

### Step 4. 성능 + 시각화

- downstream accuracy
- ensemble diversity
- trajectory behavior
- synthetic sample quality


## 추천 실험 세팅

### 데이터셋

디버그/초기 검증:

- `biodeg`
- `stock`

이유:

- 작고 빠르다
- class 수가 적어 해석이 쉽다
- 시각화와 실험 루프 점검에 적합하다

확장 단계:

- 논문과 동일한 OpenML 8개 데이터셋

### 데이터 크기

초기 디버그:

- `N_real = 100`

최종:

- `N_real in {20, 50, 100, 200, 500}`

### splits

초기:

- `3 random stratified splits`

최종:

- `10 random stratified splits`

### ensemble size

기본:

- `K = 5`

ablation:

- `K in {3, 5, 10}`

### SGLD

재현 단계에서는 논문 기본값 유지:

- `alpha_step = 0.1`
- `alpha_noise = 0.01`
- `T = 200`
- `sigma_start = 0.01`

### synthetic sample budget

모든 방법은 총 synthetic sample 수를 동일하게 맞춘다.

예:

- 총 `N_syn = 500`
- `Ens-Union`에서 member가 `K=5`면, member당 `100`개씩 생성하여 합친다.

이 원칙을 지키지 않으면, union baseline이 단순히 더 많은 샘플을 생성해서 유리해지는 문제가 생긴다.


## downstream predictor 세팅

### debug 세트

빠른 실험용:

- Logistic Regression
- Random Forest
- TabPFN

### final 세트

논문 정렬용:

- Logistic Regression
- KNN
- MLP
- Random Forest
- XGBoost
- TabPFN

### 기록 지표

- Balanced Accuracy
- mean/std across splits
- real-only 대비 improvement
- single TabEBM 대비 improvement
- ensemble-mean 대비 VP-SGLD improvement


## diversity 검증 지표

ensemble variance를 uncertainty proxy로 쓰기 전에 아래를 반드시 측정한다.

### 평가 포인트 집합

- `X_real`: 실제 train points
- `X_near`: 실제 점 주변 small Gaussian perturbation
- `X_far`: 실제 manifold에서 멀리 떨어진 perturbation
- 가능하면 `between-class corridor`: class 경계 부근 points

### 점별 disagreement

- energy variance
- coefficient of variation
- energy range
- median absolute deviation

### pairwise similarity

- Pearson correlation of energies
- Spearman correlation
- variance of energy differences
- cosine similarity of gradients

### ensemble-level 요약

- continuous KW variance
- mean pairwise disagreement
- spectral diversity

### gradient-based metrics

- per-dimension gradient variance
- gradient agreement ratio
- gradient subspace effective rank
- signal-to-noise ratio


## diversity가 유효하다고 볼 기준

엄밀한 이론 threshold는 아니고, go/no-go 기준이다.

### 바람직한 패턴

- `X_real`에서 energy variance가 작다
- `X_far`에서 energy variance가 커진다
- `X_real`에서 gradient agreement ratio가 높다
- `X_far`에서 gradient agreement ratio가 낮다

### 실패 패턴

- 모든 영역에서 member들이 거의 완전히 동일
- pairwise correlation이 대부분 `0.99+`
- toy 2D에서만 다르고 real data에서는 spread가 거의 없음

이 경우:

- ensemble uncertainty 사용을 바로 진행하지 않고
- ensemble source를 다시 설계해야 한다


## VP-SGLD 비교 방법

같은 ensemble member 집합으로 아래 3개를 비교해야 공정하다.

### 1. Standard per-member SGLD + Union

- 각 member가 standard SGLD로 independently sample 생성
- 최종 synthetic data를 합친다

### 2. Mean-gradient SGLD

- member gradient 평균만 사용
- variance는 사용하지 않음

### 3. Variance-preconditioned SGLD

- gradient mean과 gradient variance를 동시에 사용

수식:

- `g_bar(x) = (1/K) * sum_k g_k(x)`
- `v(x) = (1/K) * sum_k (g_k(x) - g_bar(x))^2`
- `G(x) = 1 / (lambda + sqrt(v(x)))`
- `x_{t+1} = x_t - alpha_step * G(x) * g_bar(x) + alpha_noise * sqrt(G(x)) * eps_t`

### 엄밀성 관련 주의

- `G(x)`가 위치 의존적이면, 엄밀한 Langevin 해석에서는 correction term이 필요하다.
- correction 없이 쓰면 실용적 heuristic은 되지만, 정확한 stationary distribution 보장을 주장하기 어렵다.

따라서 1차 단계 표현은 아래가 더 안전하다.

- `variance-aware SGLD`
- `variance-preconditioned heuristic sampler`

반면 아래 표현은 조심해야 한다.

- `theoretically exact Bayesian sampler`


## 시각화 항목

성능 숫자만으로는 부족하다.

### 필수 시각화

1. `Energy variance heatmap`
2. `Gradient agreement map`
3. `Standard SGLD vs VP-SGLD trajectory`
4. `Real vs synthetic scatter`
5. `Union coverage comparison`

### 핵심적으로 보여줘야 하는 메시지

- uncertainty가 큰 영역에서 VP-SGLD가 더 보수적으로 움직인다
- synthetic sample이 데이터 manifold 밖으로 덜 이탈한다
- 그 결과 downstream 성능이 좋아진다


## 구현 시 주의할 점

### 1. ensemble member마다 독립 객체 사용

현재 `src/tabebm/TabEBM.py`에는 model cache가 있다.

- cache는 fitted model object 자체를 저장하지 않고 class key만 저장한다.
- ensemble 실험에서 같은 인스턴스를 재사용하면 결과가 오염될 위험이 있다.

따라서:

- `ensemble member마다 독립 TabEBM 인스턴스`를 사용한다.

### 2. preprocessing은 모든 member에서 동일

- normalization
- categorical encoding
- train/test split

이 셋은 member 간 동일해야 한다.

바뀌는 것은 ensemble source만이어야 한다.

### 3. sampler 비교 시 초기점과 noise를 통제

`MeanGrad`와 `VP-SGLD`를 비교할 때:

- starting points
- SGLD random seed
- number of steps
- synthetic budget

이 네 개는 반드시 동일하게 맞춘다.

### 4. union baseline의 sample budget 통제

member 수가 늘어났다고 synthetic sample 총량이 늘어나면 안 된다.


## 최종 추천

지금 단계에서 가장 합리적이고 엄밀한 경로는 다음과 같다.

1. 논문 환경으로 `Single TabEBM reproduce`
2. `Bayesian bootstrap ensemble` 구현
3. `Corner-subset ensemble` 구현
4. diversity가 실제 uncertainty 패턴을 보이는지 확인
5. 같은 ensemble으로 `Union vs MeanGrad vs VP-SGLD` 비교
6. 성능 + trajectory + uncertainty 시각화

이 순서가 교수님 미팅 내용과 가장 잘 맞고, proof-of-concept로도 가장 방어가 잘 된다.


## 간단한 실행 체크리스트

- [ ] 논문 스택 재현 환경 분리
- [ ] Single TabEBM reproduce
- [ ] Real-only vs Single-TabEBM 성능 비교
- [ ] Bayesian bootstrap ensemble 구현
- [ ] Corner-subset ensemble 구현
- [ ] diversity metrics 계산
- [ ] diversity가 유효한지 go/no-go 판단
- [ ] Ens-Union 구현
- [ ] Ens-MeanGrad 구현
- [ ] Ens-VP-SGLD 구현
- [ ] 성능 비교
- [ ] trajectory/variance 시각화
- [ ] proof-of-concept 결과 정리
