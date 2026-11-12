# PCA가 남긴 방향과 실패 예측: 연구자가 결과를 설명하는 방법

이 실험의 결론은 “PCA를 다른 것으로 바꾸면 해결된다”가 아니다. 참조 데이터로만 정한16차원 공간은 일부 비교에서 실패 정보를 더 잘 활용하게 하지만, 밀도 추정기의 ridge와 초기화도 결과를 크게 바꾼다. 특히 **큰 분산의 방향은 템플릿 정체성과 미지 템플릿의 실패 정보를 동시에 담고 있다.** 따라서 큰 주성분을 템플릿 정보라고 보고 제거할 근거는 없다.

아래에서 AUROC는 FAIL 입력의 위험점수가 PASS 입력보다 클 확률이며 동점은0.5로 센다. AUPRC는 `average_precision_score`, 즉 FAIL을 양성으로 한 AP다. “±”는 시스템 평균을 먼저 낸 뒤 다섯 seed 평균의 표본표준편차다.95% 구간은 시스템을4,000번 재추출하고 각 시스템의5개 seed를 함께 보존한 bootstrap이다. 줄 수나 seed 수를 독립 표본 수라고 세지 않았다. 기술보고서와 기계 판독 결과에 시스템별 수치가 있다.

## 1. 무엇을 고정해야 ‘부분공간의 효과’라고 말할 수 있는가?

질문은 “어느 방향을 남겼는가?”다. Llama layer16, Gemma layer21의 기존 마지막 토큰 hidden state, 참조100개(실제90+초기10), 표준/GT비중복 outer 입력, seed0–4, samples5–9의 엄격 일치 다수결 라벨을 고정했다.5회 중3회 이상 맞으면 PASS,2회 이하면 FAIL이다. 모든 방법의 참조 순서와 평가 순서도 같다. Llama13/Gemma12개 시스템, 총250작업이다. 새로운 LLM 생성·GPU 추출·파서·출력 특징은 없다.

이전 연구는 raw→표준화 PCA를 바꾸면서 방향, 단위, ridge, KMeans 시작점도 바꿨다. 그래서 성능 차이 전체를 방향의 정보량으로 읽을 수 없었다. 이번에는 차원16을 이전 실험 근거로 미리 고정했고 테스트 AUROC를 보고 고르지 않았다. 원래 Clotho의 자동 차원은5–50이며16인 작업은 없다. 따라서 새 global16과 원래 가변 차원 Clotho는 별도 대조다. 원래 점수는 저장된 좌표/점수로 보존하고 canonical AUROC와1e−12에서 확인했다. 재적합을 완벽히 재현했다고 과장하지 않는다.

네 공간은 다음과 같다. **global**은 원래 방식대로 전체510입력에 raw PCA16을 적합한다. **reference**는 참조100개에만 raw PCA16을 적합한다. **pooled**는 참조∪평가입력에만 적합한다. 표준 분할에서는 이 합집합이510개이므로 global과 완전히 같은 대조이며, 비중복 분할에서만 사용하지 않은 참조후보가 빠져 다르다. **scaled_span**은 참조 특징별 표준화 후 PCA16이 정한 방향의 공간이다. 둘 다 테스트 라벨은 쓰지 않지만 global/pooled는 테스트의 **공변량**을 사용하므로 transductive 진단이다. reference/scaled_span은 테스트 공변량도 필요 없다.

표준화 PCA의 계수를 V, 특징 표준편차를 D라 하면 원래 hidden state에서 방향은 W=D⁻¹V다. W=QT로 분해하여 QᵀQ=I인 raw 직교기저를 쓴다. T가 가역이므로 기존 scaled PCA의 정보는 그대로다. 좌표 단위를 바꾼 효과를 방향 효과와 분리하려는 것이다. 모든 공간은 z=(h−참조평균)Q로 계산한다. randomized PCA의 `fit_transform` 근사와 실제 component projection은 조금 다르므로 그 차이도 저장했다. 기존 점수를 새 projection으로 덮어쓰거나 원래 AUROC 허용오차를 늘리지 않았다.

## 2. 같은 평면을 회전했을 뿐인데 결과가 바뀌는가?

이것은 실제 예측 향상이 아니라 **수학적 일관성 검사**다. 예를 들어2차원 좌표(1,2)를90도 회전해도 정보는 같다. Gaussian의 평균과 공분산까지 같이 회전해야 같은 모형이다:

\[
z'=zO,\quad \mu'=\mu O,\quad \Sigma'=O^T\Sigma O,\quad R'=O^TRO.
\]

O가 직교행렬이면 |det O|=1이고 log p′(z′)=log p(z)다. FAIL−PASS log 밀도비도 같아야 한다. 이미 적합한 Gaussian을 **분석적으로 이동**한 경우와, 같은 초기 책임도에서 **EM을 다시 적합**한 경우를 따로 검사했다. 실제로9,000검사에서 순위 역전·동점 변화·AUROC/AP 변화가 모두0이다. 분석적 Gaussian 최대 정규화 오차1.65e−9, 재적합5.03e−10로 고정된1e−7 기준을 통과했다.

LR 재적합은 두 표준분할 작업에서 기준을 넘었다. Gemma HealthApp seed2 scaled_span은1.14e−7, Llama HPC seed4 reference는1.65e−7이다. 기준을 바꾸지 않아 **LR 점수 오차 gate는 실패로 남긴다**. 다만 두 경우도 순위와 AUROC/AP는 동일하다. 이것을 “모든 수치 검증을 통과했다”고 쓰면 안 된다. 사후 수치 진단에서 whitening한 회전 특징의 오차는2.22e−14 이하였고, L-BFGS 종료 gradient는1.30e−9~5.17e−9로 약간 다른 근사해에 멈췄다. 같은 두 목적함수만 Newton–Cholesky로 정밀하게 다시 풀면 회전 오차가3.22e−15/7.88e−15다. 즉 solver 종료 오차라는 근거가 있지만 원래 LR 점수를 바꾸거나 실패 gate를 통과로 바꾸지 않았다. 이 계산은 새 성능 설정을 선택하는 실험이 아니다. 최소 공분산 고유값1.12e−6, 최대 조건수7.30e5이며 비수렴·공분산 보정 fallback·비유한 점수는 없다. 조건수가 크다는 사실만으로 예측 격차의 원인을 수치 불안정성이라고 할 수 없다.

이 검사로 증명되는 것은 같은 모형의 표현을 일관되게 바꾸면 같은 예측이라는 사실이다. 어떤 PCA 방향이 더 좋다거나 미지 템플릿 문제가 해결됐다는 뜻은 아니다. 이전 대각 scaling 실험은 반복하지 않았다.

## 3. 정말 다른 방향을 남겼는가? 분산이 많으면 실패 정보도 많은가?

부분공간의 주각은 Q_aᵀQ_b의 특이값으로 계산한다. 모두0도면 기저가 달라도 같은 평면이다. global↔reference의 비중복 평균 주각은 Llama26.50도/Gemma24.70도다. global↔scaled_span은42.15도/51.99도로 더 다르다. 원래 방향과 참조 방향의 차이는 단순 회전이 아니다. 표준 global↔pooled는 같은 평면으로 별도 증거가 아니다.

| Model | Subspace | Raw variance fraction, references | Raw variance fraction, all510 | Disjoint LR AUROC |
|---|---|---|---|---|
| llama | global | 0.8695 | 0.8801 | 0.6948 |
| llama | reference | 0.9080 | 0.8215 | 0.7188 |
| llama | pooled | 0.8653 | 0.8704 | 0.6837 |
| llama | scaled_span | 0.6630 | 0.6027 | 0.7200 |
| gemma | global | 0.8832 | 0.9025 | 0.6954 |
| gemma | reference | 0.9146 | 0.8667 | 0.7138 |
| gemma | pooled | 0.8765 | 0.8958 | 0.6824 |
| gemma | scaled_span | 0.4808 | 0.4568 | 0.7138 |

표의 분산은 모든 방법에 **같은 raw hidden-state 거리**를 적용해 계산한 비율이다. 예를 들어 Gemma global은 전체 분산의90.25%, scaled_span은45.68%를 남기는데, 비중복 LR은0.6954와0.7138이다. Llama도88.01% 대60.27%지만 LR은0.6948 대0.7200이다. 총 분산을 더 많이 보존하는 것이 실패 예측을 더 잘한다는 뜻은 아니다.

반대로 reference raw와 scaled_span의 각도는 크지만 LR 차이는 Llama+0.0012 [-0.0008, +0.0037], Gemma+0.0000 [-0.0080, +0.0088]로 작다. 서로 다른 공간에 비슷하게 이용 가능한 실패 정보가 남을 수 있다. 이것만으로 두 공간의 전체 정보량이 같다고 증명한 것은 아니다. probe의 용량과 학습 자료가 한정되어 있다.

## 4. 실제 AUROC 이득은 방향 때문인가, 추정기 때문인가?

밀도 모형은 동일 k,30시작점,500회 EM 한도와 참조 라벨을 쓴다. PASS 위험은−log p_PASS, 상대 위험은 log p_FAIL−log p_PASS다. FAIL 모형을 추가하므로 두 점수의 차이는 순수한 목적함수 변경만은 아니다. 이전 용량 대조의 결과를 참고하되 이번 연구가 용량 차이까지 제거했다고 말하지 않는다.

원래 ridge R=λI와 좌표일관적 R=αCov_ref(z)를 비교했다. α는 global의 평균 ridge가 원래 λ가 되도록 한 번만 정하고 모든 공간에서 고정한다. 이 규칙은 참조 covariance의 공통 raw-space prior를 투영한다. 테스트 라벨로 강도를 선택하지 않는다. 모든 공간은 같은 canonical PASS/FAIL 책임도로 시작하고, 별도로 각 공간의 native KMeans를 쓴다. 서로 다른 평면에서 같은 책임도를 쓰는 것은 시작 분할을 통제할 뿐, 최적화 문제를 같게 만들지는 않는다.

LR은 같은16차원 정보를 참조 covariance 전체로 whitening한 뒤 balanced C1을 적용한다. 특징별 표준화만 쓰면 회전 후 L2 penalty가 달라질 수 있어 이 대조를 바꿨다. 따라서 과거 coordinatewise LR 수치를 새 LR로 몰래 대체하지 않고 legacy로 따로 저장했다.

| Paired disjoint AUROC contrast | Llama Δ [95% CI] | Gemma Δ [95% CI] |
|---|---|---|
| subspace reference/lr | +0.0241 [+0.0019, +0.0499] | +0.0184 [-0.0060, +0.0469] |
| subspace pooled/lr | -0.0110 [-0.0239, +0.0003] | -0.0131 [-0.0298, +0.0011] |
| subspace scaled_span/lr | +0.0253 [+0.0050, +0.0500] | +0.0184 [-0.0073, +0.0504] |
| pairwise pooled vs reference/lr | -0.0351 [-0.0650, -0.0101] | -0.0314 [-0.0603, -0.0079] |
| pairwise scaled_span vs reference/lr | +0.0012 [-0.0008, +0.0037] | +0.0000 [-0.0080, +0.0088] |
| subspace reference/isotropic/common/relative | +0.0065 [-0.0134, +0.0293] | +0.0058 [-0.0103, +0.0212] |
| subspace reference/reference_cov/common/relative | +0.0197 [-0.0049, +0.0455] | +0.0261 [+0.0023, +0.0497] |
| relative vs pass reference/isotropic | +0.0445 [+0.0199, +0.0737] | +0.0249 [+0.0027, +0.0523] |
| relative vs pass reference/reference_cov | +0.0246 [-0.0046, +0.0548] | +0.0174 [-0.0042, +0.0444] |
| LR vs relative reference/isotropic | +0.0308 [+0.0029, +0.0573] | +0.0312 [-0.0010, +0.0583] |
| LR vs relative reference/reference_cov | +0.0415 [+0.0180, +0.0650] | +0.0447 [+0.0100, +0.0754] |

**방향 효과의 근거:** reference−global LR은+0.0241/+0.0184지만 Gemma 구간은0을 포함한다. 더 제한된 pooled와 비교하면 reference의 이득은+0.0351/+0.0314로 두 모델 모두 구간이0보다 크다. LR은 mixture 초기화를 쓰지 않고 기저변환에도 순위가 유지되므로 이 차이를 GMM 시작점이나 임의 좌표 단위만으로 설명할 수 없다. 그러나 ‘전체 정보량’이 아니라 이 고정 학습기가100개 라벨로 활용할 수 있는 선형 신호/학습가능성의 차이다. transduction 자체의 악영향과 선택된 방향의 품질까지 따로 분해한 것은 아니다.

**추정기 효과의 근거:** reference−global 상대밀도는 원래 ridge에서+0.0065/+0.0058로 양쪽 구간이0을 포함한다. reference-covariance ridge에서는+0.0197/+0.0261이며 Gemma만 구간이0보다 크다. 같은 방향 비교의 결론이 ridge에 의존한다. 좌표일관적 ridge를 썼다는 사실만으로 더 좋은 추정기가 되지 않으며, 실제 절대 AUROC가 낮아지는 경우가 많다.

**초기화의 크기:** 같은 reference 공간/R_ref에서도 세 native seed AUROC 범위의 평균이 Llama0.0653, Gemma0.0583이다. 이것은 같은 라벨과 정보를 두고도 달라지는 추정 결과다. 범위는 부호 없는 변동량이며 평균 부분공간 이득과 동일한 통계량이 아니므로 “6점 중2점이 방향이다” 같은 인과 분해는 금지다. 세 seed AUROC 평균을 낸 reference−global 효과는 Llama+0.0094[−0.0148,+0.0329], Gemma+0.0310[+0.0079,+0.0568]이다. 최종 테스트에서 가장 좋은 seed를 골라 보고하지 않았다.

## 5. 큰 분산 방향은 템플릿만 기억하는가?

이 질문은 군집 그림만으로 답할 수 없다. 같은8차원 용량으로 global PC1–8과 PC9–16에 고정 LR을 학습하고 **참조와 GT가 전혀 겹치지 않는 입력**에서 실패를 예측했다. 또 표준분할에서 참조 GT 중심점으로 본 적 있는 GT를 분류했다. GT는 이 진단의 정답/제외조건일 뿐 주 예측기의 특징이나 하이퍼파라미터 선택에 쓰지 않았다.

| Model | Global PC block | Disjoint probe AUROC | Disjoint AP | Seen-GT identity accuracy | Adjusted GT scatter | Other-GT neighbor balanced accuracy |
|---|---|---|---|---|---|---|
| llama | head8 | 0.7155 ± 0.0151 | 0.8269 | 0.9570 (N=11) | 0.9367 | 0.5967 |
| llama | tail8 | 0.5098 ± 0.0295 | 0.6921 | 0.8930 (N=11) | 0.8703 | 0.5598 |
| gemma | head8 | 0.6874 ± 0.0159 | 0.7239 | 0.9435 (N=10) | 0.8813 | 0.6200 |
| gemma | tail8 | 0.5453 ± 0.0184 | 0.6107 | 0.9202 (N=10) | 0.8392 | 0.5534 |

Head−tail 비중복 AUROC 차이는 Llama+0.2057, Gemma+0.1421이고 paired 구간이 두 모델 모두0을 벗어난다. 큰 방향에서 failure 정보가 전혀 사라지지 않는다. 다른 GT의 최근접 참조 라벨만 사용한 balanced accuracy도 head8에서 Llama0.5967[0.5610,0.6246], Gemma0.6200[0.5871,0.6496]이다. 미지 템플릿에도 전달되는 국소 라벨 구조가 있다.

동시에 head8의 본 적 있는 GT 분류 정확도는0.9570/0.9435다. template scatter도 크다. 하지만 이것은 실패 정보와 정체성 정보가 배타적이라는 뜻이 아니다. 예를 들어 특정 형식이 원래 어려우면 형식 정보가 실패 예측에 도움이 될 수도 있다. 그 형식이 어려움을 **일으킨다**는 인과 실험은 하지 않았다. 높은 정확도와 AUROC는 서로 다른 과제·기회수준·대상집단이므로 숫자의 크기를 직접 비교해 ‘어느 정보를 더 많이 담는다’고 하면 안 된다.

GT 진단에서 Mac/Thunderbird는 참조GT가 singleton이고 seen-GT 평가가 불가능하다. 그래서 identity/scatter는 Llama11/Gemma10개, 실패 probe는 전체13/12개다. 비교할 때 이 분모를 밝힌다. 비중복 identity accuracy는 모든 GT가 새 것이므로 정의하지 않으며0점이라고 기록하지 않는다. η²_adj=1−(1−η²)(n−1)/(n−g)는 그룹 수를 보정한 기술통계일 뿐 인과 검정이 아니다. 참조와의 거리는 whitening한 hidden-state 거리로 측정한 친숙도이지 템플릿 문장의 의미를 검증한 수치가 아니다.

**H3의 판정:** 템플릿 구조가 강하다는 부분은 지지된다. 그 때문에 큰 분산 방향에서 전달 가능한 실패 정보가 배제된다는 설명은 약해진다. head8이 tail8보다 미지 GT 실패를 훨씬 잘 예측하는 것이 가장 중요한 반례다. head8이 좋았다고 테스트를 본 뒤 차원을8로 바꾸어 추천하지 않는다. 주 차원은 여전히 사전 고정16이다.

## 6. LR이면 충분한가? 일반화 격차도 줄었는가?

reference LR의 비중복 AUROC는 Llama0.7188±0.0140/Gemma0.7138±0.0064, AP는0.8310/0.7406이다. FAIL 유병률은 각각0.6725/0.5425라 AP의 무작위 기준도 다르다. 원래 Clotho는 AUROC0.6448/0.6594, AP0.7938/0.7073이다. 이는 동일 outer 입력에서의 절대 개선이며 단순히 gap이 작아졌다고 좋은 방법이라고 한 것이 아니다.

같은 reference 공간/R_ref 상대밀도보다 LR은+0.0415[0.0180,0.0650]/+0.0447[0.0100,0.0754] 높다. H4를 지지한다. 그러나 원래 isotropic ridge의 Gemma 상대밀도와 비교한 LR 이득은+0.0312[−0.0010,0.0583]이므로 LR이 모든 밀도 추정기보다 유의하게 좋다고 확대하면 안 된다. 시스템별 예외도 많고 C1 외의 큰 탐색도 하지 않았다. 두 density가 선형경계보다 복잡하다는 것이 이 자료에서 필요하다는 근거가 약하다는 판단이다.

| Model | Method | Standard−disjoint AUROC gap | Same-input gap | Same-input gap reduction vs original [CI] |
|---|---|---|---|---|
| llama | anchor/original | 0.1040 | 0.1187 | reference |
| llama | global/lr | 0.0682 | 0.0787 | +0.0399 [-0.0020, +0.0810] |
| llama | reference/lr | 0.0414 | 0.0574 | +0.0612 [+0.0315, +0.0933] |
| llama | reference/reference_cov/common/relative | 0.0990 | 0.1086 | +0.0101 [-0.0154, +0.0346] |
| gemma | anchor/original | 0.1211 | 0.1199 | reference |
| gemma | global/lr | 0.0897 | 0.0908 | +0.0291 [-0.0236, +0.0746] |
| gemma | reference/lr | 0.0664 | 0.0655 | +0.0545 [+0.0146, +0.0924] |
| gemma | reference/reference_cov/common/relative | 0.1282 | 0.1288 | -0.0089 [-0.0400, +0.0250] |

같은 입력에서 원래 Clotho gap은0.1186/0.1199, reference LR은0.0574/0.0655다. 감소0.0612/0.0545의 구간은 양수다. 다만 이 비교에는 점수 방식·고정 차원·whitening·부분공간이 모두 바뀐다. **그 감소를 PCA 단독 효과로 부르면 안 된다.** global LR과 reference LR의 gap을 따로 제시한 이유다. 순수하게 좌표를 회전해서 회복한 수학적 불변성은 이 gap을 전혀 줄이지 않는다.

기존 matched S1은0.7490/0.7619, U는0.7764/0.7732다. reference LR은 S1보다0.0302/0.0480, U보다0.0576/0.0594 낮다. S1은 층·소스학습이 다르고 target 참조도 실제90개만 사용한다(주실험은초기10개를포함한100개). U에는 출력 정보도 있으므로 이 차이를 점수 하나의 몫으로 분해할 수 없다. S1은 scalar AUROC만 검증 가능하여 AP는 만들어내지 않았다.

## 7. 반례와 연구 결정

평균만 보면 놓치는 반례는 OpenSSH다. reference−global LR이 Llama에서는+0.1513인데 Gemma에서는−0.0574다. Llama Apache도−0.0523이다. 반면 Gemma OpenStack/HPC는+0.1180/+0.1021이다. 사후 leave-one-system-out 기술 점검에서 Llama OpenSSH를 빼면 평균은+0.0135로 줄어든다. 본 실험에서 유리한 시스템만 남기거나 가설을 바꾸지 않았다. 실제 표는 전체 시스템을 유지한다.

- **H1:** 일부 공간에서 접근 가능한 실패 신호/학습가능성이 달라진다는 근거가 있다. reference>pooled는 두 모델에서 재현되지만 원래 global에 대한 Gemma 이득은 불확실하다. 정보이론적 우열은 증명하지 않았다.
- **H2:** ridge 및 시작점이 상당한 영향을 준다. 그렇다고 invariant LR의 공간 차이까지 모두 estimator artifact로 지울 수는 없다.
- **H3:** 정체성은 강하지만 큰 방향을 버려야 한다는 근거는 없다. 큰 방향의 미지GT 실패 정보가 명확한 반례다.
- **H4:** reference LR이 단순하고 강한 필수 대조다. 밀도 복잡도를 계속 쓰려면 이 대조를 이기는 근거가 필요하다. 보편적 LR 우월성은 주장하지 않는다.

**지금 정당화되는 변경:** GT 없이 참조만으로 적합한 raw PCA16+전체 참조 whitening+balanced LR을 연구 프로토타입/필수 baseline으로 둔다. 더 넓은 PCA 탐색은 멈춘다. 이 근거로 새 PCA 방법의 독창성이나 배포 성능을 선언하지 않는다. 추론 때도 GT는 필요 없으며, identity 진단에만 GT가 필요했다. finite reference100, 클래스 불균형, noisy5회 라벨, 두 모델이 공유하는 같은 시스템, template-level 의존성, 원래 획득정책을 고정한 조건부 비교가 남은 한계다.

**다음 실험은 하나만:** 동일 reference-only PCA16에서 클래스당 Gaussian 하나를 쓰고 공유 공분산 대 클래스별 공분산을 비교한다. 좌표일관적 ridge는 그대로 고정하고 기존 LR/mixture를 대조로 둔다. mixture의 국소 최적화 문제를 제거하면서 클래스별 공분산의 복잡도가 필요한지, 추정 비용만 늘리는지를 검정한다. 새 hidden state·생성·PCA/강도 탐색은 필요 없다.

가장 강한 새 근거는 “다른 평면의 효과가 고정 probe에도 남지만, 큰 PC는 미지 템플릿의 실패 정보를 보존한다”는 조합이다. 가장 중요한 반례는 모델 간 OpenSSH의 부호 반전과 Gemma global 대비 구간이다. PCA가 일반화 실패의 주원인이라고 확정할 수 없고, 더 단순한 추정기를 엄격히 검정할 단계다.

## 보존과 재현 확인

원래 결과는 덮어쓰지 않았다. 새 root는 `experiments/results_phase3/clotho_pca_subspace/`다.250작업의16,125 score cells, fitted bases/indices, 선택하지 않은 모든 initializer 결과, bootstrap, AUROC/AP/유병률, score separation, covariance condition, per-system/per-seed 자료와6종 PDF/PNG가 있다. 원래1,806개 추정치 스냅샷을 보존하고 전체 부분공간 쌍을 완성한 최종1,926개를 별도로 검증했다. 두 전체 작업 재실행은 모든249배열과 wall time을 뺀 JSON이 동일하다. Gaussian 불변성은 통과했지만 LR 두 score gate 예외를 명시한다. 모든 순위와 평가 지표는 회전 전후 동일하다.

초기 입력 glob이 기존 `_error.json`24개를 포함한 실행기 오류가 있었으나 실제250개 작업은 모두 끝났다. 파일 선택을 수정한 resume은 새 fit0개로 완료되었다. HF 대량 업로드 rate limit은 작은 `hub_sync.py push` batch로 처리하며 결과를 변경하지 않는다. 검증/원격 hash 보존을 마친 뒤에만 PROGRESS를 갱신한다. 실행 명령과 세부 숫자는 [기술보고서](clotho_pca_subspace.md), `validation.json`, `aggregation_validation.json`, `persistence_validation.json`을 함께 본다.
