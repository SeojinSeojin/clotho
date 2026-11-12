# Clotho에 복잡한 Gaussian mixture가 필요한가?

**판단부터 말하면, 새 density variant를 만드는 연구 분기는 여기서 멈추는 것이 타당하다.** 같은 PCA16·참조·ridge에서 mixture를 클래스당 Gaussian 하나로 줄이면 미지 템플릿 AUROC가 개선된다. 그러나 “그러므로 LR이 모든 Gaussian보다 좋다”도 맞지 않는다. 공유 공분산 Gaussian은 LR과 경쟁적이다. 이 실험은 새 예측기 이름을 붙이는 근거보다, 어떤 복잡도가 실제로 필요한지 검증한 실증 근거다.

아래 수치는 Llama/Gemma 순서다. AUROC는 FAIL 위험점수가 PASS보다 높을 확률(동점0.5), AP는 FAIL 양성 average precision이다. 평균은 시스템별로 계산한 후5개 seed에서 macro 평균을 낸다.95% 구간은 시스템을4,000번 bootstrap하며 그 시스템의5개 seed와 방법쌍을 함께 보존한다. 로그 줄·seed·참조 재표집10회를 독립 시스템처럼 세지 않았다.

## 1. 이번 비교가 왜 필요했는가?

이전 실험에서 상대밀도가 PASS밀도보다 좋았고, PCA/ridge/초기화에 따른 민감도도 확인했다. 하지만 두 mixture가 좋지 않다고 해서 “FAIL 분포를 모델링하는 것이 틀렸다”거나 “density보다 discriminative learning이 본질적으로 낫다”고 할 수는 없다. 복잡한 covariance와 여러 성분을 작은 표본에서 추정하는 것이 문제일 수도 있다.

이번 질문은 그 둘을 분리한다. Llama13/Gemma12개 시스템, seed0–4, 표준/GT비중복, 원래100참조(실제90+초기10), 기존 layer16/21, 참조 전용 raw PCA16 좌표를 전부 고정했다.5회 생성 samples5–9에서3회 이상 strict match이면 PASS,2회 이하면 FAIL이라는 라벨도 그대로다. 참조 획득정책의 ‘balanced’는 uncertainty/diversity 균형이지 PASS/FAIL 수가 반반이라는 뜻이 아니다. 실제 PASS 수는 Llama21–61(중앙값33), Gemma27–78(45.5)로 다르다.

기존 GMM 결과를 재사용하고, 누락된 단일 Gaussian 비교만 새로 계산했다. 기존 MDSA도 Mahalanobis 계열이지만 PCA10/transductive/pseudoinverse 설정이 달라 이 통제 비교를 대신하지 못한다. 새 PCA·LLM 생성·GPU·파서·출력 특징은 없다. 실행 중이거나 이미 끝난 동등 실험도 없었다. 프로토콜은 결과를 보기 전에 commit3a6bbad로 고정했다.

## 2. GMM, 공유 covariance, 별도 covariance는 무엇이 다른가?

Gaussian 하나는 중심 μ와 타원 모양 Σ로 한 집단을 설명한다. Mixture는 여러 타원을 가중합한다:

\[
p(z\mid c)=\sum_{j=1}^k\pi_{cj}\,\mathcal N(z;\mu_{cj},\Sigma_{cj}).
\]

여러 유형의 로그를 나타낼 수 있지만 각 타원의 중심·공분산·가중치를 추정해야 하고 EM이 다른 국소해로 갈 수 있다.16차원 covariance 하나에도16×17/2=136개의 서로 다른 원소가 있다. 평균까지 포함한 Gaussian 하나는152개, 클래스당5성분 mixture는764개다. PASS/FAIL 두 mixture는1,528개를 추정한다.6성분인 한 작업은1,834개다.

이번 새 모델은 다음과 같다.

- **PASS Gaussian:** PASS 표본만 한 타원으로 적합하고−log p_PASS를 위험으로 쓴다.152개 parameter다.
- **Shared Gaussian:** 클래스별 중심은 두 개지만 covariance 하나를 공유한다.168개를 추정하며 log p_FAIL−log p_PASS로 평가한다.
- **Separate Gaussian:** 중심과 covariance를 각각 추정한다.304개다. shared보다 covariance parameter136개가 더 필요하다.
- **LR:** 같은16차원 정보를 참조 전체 covariance로 whitening한 뒤 balanced C1 logistic loss로 학습한다.경계 계수는 절편 포함17개다.

Shared에서는 두 밀도의 제곱항이 상쇄되어 w=Σ⁻¹(μ_FAIL−μ_PASS), 위험=zᵀw+b인 **선형 경계**가 된다. Separate에서는 zᵀ(Σ_PASS⁻¹−Σ_FAIL⁻¹)z/2가 남아 **이차 경계**가 된다. 예를 들어 FAIL이 두 변수의 조합이 큰 곳에만 모이면 이차 경계가 도움될 수 있다. 복잡한 함수가 가능하다는 것과, 작은 표본으로 그 함수를 잘 추정한다는 것은 다르다. 예를 들어 PASS 중심(0,0), FAIL 중심(2,0), 공유 covariance I이면 상대위험은2z₁−2라 경계가 z₁=1이다. FAIL covariance만 diag(4,1)로 바꾸면 위험은3z₁²/8+z₁/2−1/2−log2로 제곱항이 남는다. 이는 실제 측정값이 아닌 수식 설명 예시다.

통계적 자유도와 위 nominal parameter 수는 같지 않다. 참조 covariance를 쓰는 공통 전처리/ridge도 자료에서 추정하지만 모든 방법에 같고 추가 탐색하지 않았다. Shared와 LR은 경계 함수족은 선형으로 같아도 generative likelihood와 discriminative logistic loss가 다르다. 그 차이를 단순히 ‘선형 대 비선형’이라고 설명하면 안 된다.

## 3. covariance와 class imbalance를 공정하게 맞췄는가?

각 클래스 평균은 같고 covariance는 S_c=Σ(z_i−μ_c)(z_i−μ_c)ᵀ/n_c로 계산한다. 이전에 고정한 R=αCov_reference(z)를 모든 새 density와 GMM에 똑같이 더한다. α도 이전 값 그대로다. 중요한 출처상 한계는 이 α가 이전 all510 global PCA의 참조분산으로 보정되었다는 점이다. Global PCA는 unlabeled test covariate를 보았으므로 α에도 간접적인 transductive 의존성이 있다. 이번 covariance 모양/PCA16은 참조만 사용하지만 전체 inherited Gaussian pipeline을 완전히 inductive하다고 부르면 안 된다. 이 α는 모든 density에 동일하게 고정했고 방법별로 유리하게 고르거나 테스트 라벨로 조정하지 않는다. 원래 참조 획득정책도 고정한 조건부 비교다.

Shared의 primary covariance는(S_PASS+S_FAIL)/2+R다. 두 클래스에 같은 영향력을 주어 balanced LR과 맞춘다. Separate는 S_PASS+R, S_FAIL+R다. 따라서 shared↔separate의 핵심 변경은 covariance를 같게 제약하느냐다. 표본 수로 가중한 pooled covariance도 사전에 정한 보조 대조 하나만 계산했다. pooled−balanced의 disjoint AUROC는 Llama-0.0008 [-0.0029, +0.0009], Gemma+0.0015 [-0.0020, +0.0065]로 작고 구간은0을 포함한다.

상대밀도에 경험적 class prior는 넣지 않았다. log(n_FAIL/n_PASS)를 더하면 같은 작업 안에서 상수만 바뀌므로 AUROC/AP 순위는 수학적으로 같다. 다만 covariance의 class 가중치는 모양 자체를 바꾸므로 이 상수 문제와 구별해야 한다. 위험0을 확률0.5처럼 해석하거나 calibration 성능을 주장하지 않는다.

수식이 같은 다른 구현도 검사했다. Shared의 선형식, separate의 이차식, PASS의 Mahalanobis식과 SciPy Gaussian density가 같은 순위를 낸다. 단일성분 EM도 확인했다. 총2,004검사의 최대 정규화 오차2.61e−12, 순위·동점·AUROC/AP 차이는0이다. 여기서 증명한 것은 **구현/수학적 동등성**이지 정확도 향상이 아니다. 원래 Clotho AUROC는250작업 모두 저장된 canonical 값과1e−12에서 일치하고 기존 GMM/LR 점수는 bitwise 재현된다.

## 4. mixture를 제거하면 실제로 좋아지는가?

변경한 것은 density family다. 참조·라벨·PCA16·평가 입력·R·점수 방향은 고정했다. Relative GMM과 비교하면 shared/separate 모두 FAIL과 PASS를 모델링하므로, 새 FAIL 라벨을 더 준 효과도 아니다.

| Estimator | Standard AUROC | Disjoint AUROC | Standard AP | Disjoint AP |
|---|---|---|---|---|
| Original Clotho (context) | 0.7488 ± 0.0101 | 0.6448 ± 0.0220 | 0.8520 | 0.7938 |
| PASS GMM | 0.7430 ± 0.0099 | 0.6528 ± 0.0197 | 0.8489 | 0.7978 |
| Relative GMM | 0.7764 ± 0.0077 | 0.6774 ± 0.0159 | 0.8665 | 0.8014 |
| PASS Gaussian | 0.7391 ± 0.0053 | 0.6869 ± 0.0184 | 0.8498 | 0.8203 |
| Shared Gaussian | 0.7625 ± 0.0058 | 0.7294 ± 0.0077 | 0.8563 | 0.8400 |
| Separate Gaussian | 0.7873 ± 0.0034 | 0.7141 ± 0.0173 | 0.8724 | 0.8264 |
| LR | 0.7603 ± 0.0053 | 0.7188 ± 0.0140 | 0.8548 | 0.8310 |
| Shared pooled (secondary) | 0.7625 ± 0.0038 | 0.7285 ± 0.0108 | 0.8547 | 0.8383 |

| Estimator | Standard AUROC | Disjoint AUROC | Standard AP | Disjoint AP |
|---|---|---|---|---|
| Original Clotho (context) | 0.7805 ± 0.0029 | 0.6594 ± 0.0246 | 0.8325 | 0.7073 |
| PASS GMM | 0.7647 ± 0.0094 | 0.6518 ± 0.0128 | 0.8166 | 0.7152 |
| Relative GMM | 0.7974 ± 0.0122 | 0.6691 ± 0.0192 | 0.8432 | 0.7176 |
| PASS Gaussian | 0.7484 ± 0.0069 | 0.6822 ± 0.0130 | 0.8113 | 0.7323 |
| Shared Gaussian | 0.7785 ± 0.0054 | 0.7179 ± 0.0112 | 0.8114 | 0.7431 |
| Separate Gaussian | 0.8105 ± 0.0028 | 0.7235 ± 0.0131 | 0.8545 | 0.7506 |
| LR | 0.7802 ± 0.0055 | 0.7138 ± 0.0064 | 0.8153 | 0.7406 |
| Shared pooled (secondary) | 0.7776 ± 0.0064 | 0.7194 ± 0.0129 | 0.8102 | 0.7430 |

| Disjoint AUROC contrast | Llama Δ [95% CI] | Gemma Δ [95% CI] |
|---|---|---|
| PASS Gaussian − PASS GMM | +0.0341 [+0.0152, +0.0560] | +0.0304 [-0.0015, +0.0619] |
| Shared Gaussian − Relative GMM | +0.0520 [+0.0324, +0.0735] | +0.0487 [+0.0154, +0.0791] |
| Separate Gaussian − Relative GMM | +0.0367 [+0.0240, +0.0489] | +0.0543 [+0.0393, +0.0722] |
| Separate Gaussian − Shared Gaussian | -0.0153 [-0.0418, +0.0059] | +0.0056 [-0.0172, +0.0282] |
| Shared Gaussian − LR | +0.0105 [-0.0004, +0.0284] | +0.0041 [-0.0025, +0.0119] |
| Separate Gaussian − LR | -0.0048 [-0.0338, +0.0230] | +0.0097 [-0.0147, +0.0346] |
| Shared pooled (secondary) − Shared Gaussian | -0.0008 [-0.0029, +0.0009] | +0.0015 [-0.0020, +0.0065] |

**가장 강한 새 관찰:** shared−relative GMM의 disjoint AUROC는+0.0520[0.0324,0.0735]/+0.0487[0.0154,0.0791]이다. Separate도+0.0367[0.0240,0.0489]/+0.0543[0.0393,0.0722]다. 두 모델 모두 interval이0보다 크다. 같은 참조 정보로 더 단순한 상대밀도가 잘하므로, 이 조건에서 여러 mixture 성분이 필요하다는 근거가 없다.

AP도 shared는0.8400/0.7431, separate는0.8264/0.7506, relative GMM은0.8014/0.7176이다. Shared의 Gemma AP 이득 구간은0을 포함하므로 모든 metric에서 확정적으로 이긴다고 하면 안 된다. Separate−GMM AP 구간은 두 모델 모두 양수다. Failure prevalence는 비중복0.6725/0.5425로 같지 않아서 AP 숫자의 크기를 모델 간에 그대로 비교하면 안 된다.

PASS-only에서 single−GMM은+0.0341/+0.0304지만 Gemma 구간은0을 포함한다. 따라서 ‘어떤 단순화든 항상 좋다’는 결론은 아니다. 원래 Clotho와의 더 큰 차이는 표현·차원·점수도 바뀌므로 estimator complexity 하나의 인과 효과라고 부르지 않는다.

## 5. 별도 covariance가 주는 비선형 정보는 유용한가?

Shared와 separate는 같은 두 중심과 같은100개 라벨을 쓴다. 차이는136개의 covariance parameter와 경계의 제곱항이다. Disjoint separate−shared는−0.0153[−0.0418,0.0059]/+0.0056[−0.0172,0.0282]다. 어느 모델에서도 일관된 이득을 확정할 수 없다. 이것은 ‘비선형 정보가 없다’는 동등성 검정 결과가 아니다.

**중요한 반례:** 표준 분할에서는 separate−shared가+0.0248[0.0101,0.0410]/+0.0320[0.0078,0.0559]로 두 모델 모두 양수다. 복잡한 covariance가 쓸모없는 것은 아니다. 다만 그 이득이 미지 템플릿으로 안정적으로 이전된다는 근거가 부족하다. 이것만으로 템플릿 암기를 인과적으로 입증한 것도 아니다.

시스템별로 보면 Gemma Zookeeper는 separate가 shared보다+0.1023, relative GMM도 shared보다+0.0955 좋다. 반대로 Llama Zookeeper의 separate−shared는−0.0460이다. Gemma HDFS는−0.0825, Llama OpenSSH는−0.1465다. 이런 사후 시스템 반례를 숨기면 안 되지만, 한 시스템의5개 seed를 독립 표본처럼 세어 새 선택 규칙의 유의성을 선언해서도 안 된다. 이 Zookeeper 예외조차 mixture 고유 정보의 증거는 아니다. Separate Gaussian이 GMM보다 약0.0068 더 높다. 그렇다고 테스트를 본 뒤 시스템마다 좋은 방법을 고르는 oracle를 배포 가능한 선택기로 제시하지 않는다. 이 예외로 density method 전체 연구를 계속할 일반 조건이 확립되지는 않았다.

## 6. 표본이 작아 불안정하다는 근거는 무엇인가?

클래스 전체 covariance의 raw rank는 모두16이다. 하지만 GMM은 클래스를 다시 여러 작은 집단으로 나눈다. 비중복 mixture component의 median effective sample size는 Llama9/Gemma10, raw rank는8/9다. Raw rank가16 미만인 비율은77.69%/93.69%다. 예를 들어16차원에서 성분에10개 관측이 있으면 중심을 뺀 sample covariance rank는 최대9다. R을 더하면 행렬이 full rank가 되지만 새 관측 증거가 생기지는 않는다. 조건수가 높다는 사실만으로 overfitting을 증명하지는 않는다.

이를 실제 예측 민감도로 점검하려고 각 참조 집합을10번 stratified bootstrap했다. 클래스 수를 고정하고 모든 방법에 같은 draw를 준다. PCA·whitening·R은 원래100개에서 고정한다. GMM 초기 책임도도 원래 것을 같은 row draw로 재표집하여 새 KMeans 선택을 섞지 않았다. 이 실험은 기존 표본에 대한 조건부 perturbation이며 새로운100개 표본을 열 번 얻은 것이 아니다. 줄 사이 template 의존성을 그대로 모델링한 sampling CI도 아니다.

| Estimator | Llama AUROC SD | Gemma AUROC SD | Llama rank SD | Gemma rank SD |
|---|---|---|---|---|
| PASS GMM | 0.0529 | 0.0466 | 0.1249 | 0.1351 |
| Relative GMM | 0.0623 | 0.0545 | 0.1557 | 0.1650 |
| PASS Gaussian | 0.0510 | 0.0424 | 0.1331 | 0.1242 |
| Shared Gaussian | 0.0551 | 0.0481 | 0.1400 | 0.1360 |
| Separate Gaussian | 0.0562 | 0.0459 | 0.1422 | 0.1393 |
| LR | 0.0534 | 0.0469 | 0.1381 | 0.1327 |

서로 다른 risk 단위의 표준편차를 직접 비교하지 않고, 테스트 입력 안의 percentile rank를 만든 뒤10회 SD를 평균했다. Shared−GMM rank SD는−0.0157[−0.0239,−0.0083]/−0.0290[−0.0387,−0.0181]로 감소한다. Relative mixture가 참조 변화에 더 민감하다는 H1의 근거다. 그러나 AUROC SD의 Llama 감소 구간은0을 포함한다.

또 하나의 핵심 반례는 PASS-only Llama다. Single Gaussian의 rank SD는 GMM보다+0.0082[0.0024,0.0133] **커지는데도** 평균 AUROC는 좋아진다. Parameter가 줄면 항상 안정성이 좋아진다거나, 모든 성능 손실을 variance만으로 설명할 수는 없다.

5,000개 bootstrap GMM fit은 모두 수렴했다.2,468개는30개 초기 partition 중 적어도 하나에 빈 성분이 있었고 이를 기록했다. 따라서 perturbation 차이에는 covariance 추정뿐 아니라 nonconvex optimization의 반응도 포함된다. 최악의 condition은3.35e5, 최소 covariance 고유값은9.29e−6이며 numerical failure/fallback은 없다. 계산이 무너져서 GMM이 낮았다고 설명할 근거가 없다. 분포 모델의 부적합, regularization의 효과, 작은 표본, 라벨 노이즈도 남아 있다.

## 7. LR과 비교하면 density가 필요한가? gap은 어떤가?

LR disjoint AUROC는0.7188/0.7138, shared Gaussian은0.7294/0.7179다. Shared−LR 구간은−0.0004~+0.0284/−0.0025~+0.0119로0을 포함한다. Separate도 LR보다 확정적으로 좋다는 모델 간 일관된 근거가 없다. 그렇다고 LR이 모든 Gaussian보다 좋다고 말하는 것도 틀리다. **H4는 복잡한 GMM과 비교할 때 지지되지만, 단순 Gaussian으로 통제하면 일반적인 discriminative 우월성 주장은 지지되지 않는다.**

| Model | Estimator | Outer standard−disjoint AUROC | Identical-input gap |
|---|---|---|---|
| llama | Original Clotho (context) | 0.1040 | 0.1187 |
| llama | Relative GMM | 0.0990 | 0.1086 |
| llama | PASS Gaussian | 0.0522 | 0.0536 |
| llama | Shared Gaussian | 0.0332 | 0.0478 |
| llama | Separate Gaussian | 0.0732 | 0.0810 |
| llama | LR | 0.0414 | 0.0574 |
| gemma | Original Clotho (context) | 0.1211 | 0.1199 |
| gemma | Relative GMM | 0.1282 | 0.1288 |
| gemma | PASS Gaussian | 0.0663 | 0.0612 |
| gemma | Shared Gaussian | 0.0606 | 0.0582 |
| gemma | Separate Gaussian | 0.0870 | 0.0841 |
| gemma | LR | 0.0664 | 0.0655 |

표의 ordinary gap은 서로 다른 outer 평가분포를, identical-input gap은 양 split에서 같은 입력/라벨만 맞춘 차이를 쓴다. Shared의 같은입력 gap은0.0478/0.0582로 원래 Clotho0.1186/0.1199보다 작고 절대 cold AUROC도 좋아진다. 하지만 그 전체 감소를 mixture 하나의 효과라고 할 수 없다. 원래 Clotho는 표현·차원도 다르기 때문이다.

Gemma separate는 shared보다 gap이 크면서 disjoint AUROC는 조금 높다. 작은 gap만 보고 shared가 더 잘 예측한다고 선택하면 안 되는 구체적인 사례다. 기존 S1은0.7490/0.7619, U는0.7764/0.7732로 아직 높지만 다른 층·소스·참조훈련집단(U에는 출력 정보)이라 그 잔여 차이를 estimator 하나의 몫이라고 계산하지 않는다.

## 8. H1–H4를 어떻게 판정하고 어디서 멈출 것인가?

- **H1 mixture overcomplexity:** 더 단순한 상대밀도의 cold AUROC와 rank 안정성 이득이 지지한다. 다만 PASS-only 반례, system 예외와 optimization 효과 때문에 ‘overfitting이 유일한 원인’은 아니다.
- **H2 covariance overcomplexity:** 별도 covariance의 cold 평균 이득이 일관되지 않아 기본 복잡도로 정당화하기 어렵다. 표준 분할과 Gemma Zookeeper는 반례다. 별도 covariance가 항상 더 불안정한 것도 아니다.
- **H3 nonlinear information:** 표준과 일부 미지GT 시스템에서 실제 이득이 있다. 그러나 전체 cold/LR 대비 재현 가능한 우위가 없어 새 density variant를 제안할 정도의 근거는 아니다.
- **H4 discriminative advantage:** LR은 강한 필수 대조지만 shared Gaussian과의 차이는 불확실하다. ‘density 자체가 실패했다’가 아니라 ‘현재 mixture 복잡도를 정당화하지 못했다’가 정확하다.

Reference200은 생략했다. 저장된 비중복 test 목록은100과200에서 같고200개의 실제 획득 reference ID/순서/PCA snapshot이 없으므로 임의 표본으로 대체하지 않는다. Bootstrap100은 ref200 learning curve가 아니며, 자료가 늘면 순위가 바뀌는지는 모른다. 기존5회 라벨의 노이즈, 상관된 template/log, 작은 시스템 수, 같은 데이터를 공유하는 두 LLM, 원래 획득정책에 조건부인 비교도 한계다. Confidence interval은 다중비교 보정 없이 기술하며 동등성/독창성을 주장하지 않는다.

**연구 결정:** 복잡한 Clotho density variant 개발 분기를 종료한다. Shared Gaussian과 LR은 고정된 단순 baseline으로 남긴다. Gaussian은 아래 설명한 inherited α의 transductive 출처를 보존한 archival control이지 완전히 inductive한 새 방법이라고 부르지 않는다. 소수 시스템의 사후 이득을 새 covariance/GMM/PCA 탐색의 이유로 삼지 않는다. 더 단순해서 나아졌다는 사실만으로 새 알고리즘의 novelty도 주장하지 않는다.

**다음 단계는 하나:** 검증된 실험들을 template-disjoint 평가 패키지와 집중된 empirical paper로 정리한다. Split·reference·label·고정 LR/shared Gaussian 대조와 반례를 함께 제공하여 “표현 안에 신호는 남는데 estimator/evaluation shift가 무엇을 바꾸는가”라는 통제 근거를 기여로 삼는다. 새로운 density 탐색 실험은 권하지 않는다.

가장 강한 새 사실은 같은 정보·ridge에서 mixture를 없애면 약4–5 AUROC point를 회복하고 상대점수 순위 변동도 줄어든다는 것이다. 가장 강한 반례는 표준의 비선형 covariance 이득/Gemma Zookeeper와 PASS-only 안정성의 역전이다. 두 사실을 함께 보고해야 연구자가 결과를 방어할 수 있다.

## 재현·수행비용·보존

| Estimator | Canonical fit ms | Prediction batch ms (20-repeat median) | Bootstrap fit ms (GMM only) |
|---|---|---|---|
| PASS GMM | not archived | 0.8991 | 77.8456 |
| Relative GMM | not archived | 1.7815 | 161.4634 |
| PASS Gaussian | 0.1642 | 0.1554 | — |
| Shared Gaussian | 0.2718 | 0.3070 | — |
| Separate Gaussian | 0.2420 | 0.3110 | — |
| LR | 5.9337 | 0.2300 | — |

시간은 CPU4 worker/BLAS1, prediction은 작업당20번 batch median이다. Canonical GMM fit time은 원래 저장되지 않아 **새 bootstrap population의 시간만** 표시한다. Gaussian/LR 원래100개 fit 시간과 완전히 같은 canonical speed ratio로 해석하지 않는다. 모든 방법은 기존 hidden state만 쓰므로 속도 개선을 새 LLM 성능 기여로 주장하지 않는다.

250작업,1,500 primary+250 weighting+375 context=2,125 metric rows,15,000 perturbation rows,888 system-bootstrap 추정치를 저장했다. 원래 결과는 변경하지 않았다.750 source/canonical hash 확인, 모든 점수 metric 재계산, 두 전체 job의109개 배열/비시간 metadata 재현, 모든 bootstrap/집계의 독립 검증을 통과했다. Fit/prediction 시간만 비결정적이라 replay 비교에서 제외한다. 원격 hash 보존을 확인한 다음 PROGRESS를 갱신한다.

코드/명령은 [기술보고서](clotho_gaussian_complexity.md), 결과는 `experiments/results_phase3/clotho_gaussian_complexity/`의 job JSON/NPZ·CSV·bootstrap·validation·PDF/PNG에 있다. `decision.json`에는 분기 종료와 허용되지 않는 과장 주장을 기계 판독 형태로 남겼다.
