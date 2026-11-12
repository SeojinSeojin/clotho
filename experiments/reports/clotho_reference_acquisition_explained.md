# 어떤 참조를 고르는가: 기존 연구와 고정 LR 선택 실험을 구분하기

**결론:** 참조 선택이 전혀 중요하지 않은 것은 아니다. 그러나 이번 자료에서 새 acquisition 방법을 개발할 근거는 부족하다. 원래 Clotho도 이미 uncertainty+diversity를 사용했다. 알려진 diversity 방법은 random보다 좋아지는 조건이 있지만, 원래 참조 이력보다 두 LLM에서 일관되게 나은 새 정책은 확인되지 않았다. 작은 양의 평균 차이를 새 방법의 독창성으로 포장하지 않는다.

## 1. 이미 무엇을 알고 있었고 왜 또 확인했는가?

Random 대 원래 ‘balanced’ 선택은 이미 비교되었다. 예전 Llama layer21 표준 분할에서100참조 AUROC는 random0.7296, balanced0.7639였다. 하지만 그 실험은10회 생성 라벨, adaptive PCA/GMM, 방법마다 달라지는 test complement를 썼다. 현재 five-generation GT비중복/고정LR의 답으로 그대로 가져올 수 없다.

이후 fairness 실험은 동일 GT비중복 test에서 MDSA 참조를 바꾸었다. Llama는 원래 참조 약0.656 대 random0.645, Gemma0.655 대0.648이었다. GMMb에서는 Llama가 오히려 random0.576 대 원래0.564였다. 즉 ‘참조 정책은 예측기와 상호작용한다’는 문제는 원래부터 있었다.250개 저장 fairness 작업으로 보고서 모든 AUC 셀을 다시 계산해 확인했고, 원래 Clotho100/200 quantity도250개 canonical 결과에서 재집계했다. 기존 예측기를 재학습하지 않았다.

실패 발견 budget 연구도 있었다. Llama natural에서 B100 실제 입력을 라벨링하고 나머지 중100개를 더 확인하면 Clotho는2047.2 실패, random은1690을 찾았다. 단, 이는 future-template AUROC가 아니다. 또 그 B100은실제100+초기10개라 이번 nominal100=실제90+초기10개와 다르다. 원래 짧은 strategy 설명의455 대419는 B50 수치가 B100 문장에 섞인 것이며, 같은 Clotho 대 Drain 비교의 B100 획득 FAIL 수는 상세표에서927.8 대842.0이다. 별도로 machine decomposition에서 Clotho 대 random B100은927.8 대847.6이고 총발견수는2047.2 대1690이다. Budget을바로잡으며비교대상까지몰래바꾸지않는다. 예전 파일을 덮어쓰지 않고 이 audit에 바로잡아 적었다.

기존 PCA/밀도/LR 연구는 참조를 고정했다. Gaussian 실험의 bootstrap도 같은100개 안의 perturbation이지 새 후보를 어떤 순서로 라벨링할지 비교한 것이 아니었다. 그래서 **고정 reference-PCA16 LR에서, 실제로 숨겨진 candidate 라벨을 하나씩 얻는 정책 비교**만 남은 좁은 질문이었다. 이 비교는 단순 이름 바꾸기가 아니라 predictor–acquisition interaction의 미측정 부분이다. 기존 연구가 없었다고 주장하지 않는다.

## 2. 저장된 모든 라벨이 있는데 어떻게 ‘모르는 라벨’을 흉내 내는가?

Oracle는 정답을 보관하지만 selector는 이를 보지 못한다. 비중복 outer test ID를 먼저 고정하고, 나머지 suite 입력만 candidate로 둔다. GT ID는 그 분할이 정말 비중복인지 검사할 때만 쓴다. 정책 함수는 텍스트/은닉 특징, 후보 ID, 지금까지 관측한 라벨만 받는다.

예를 들어 초기10개와 실제10개를 관측했다면 LR은20개의 라벨로만 학습한다. 다음10개 ID를 먼저 고른 다음 Oracle가 그10개 라벨을 돌려준다. 후보 전체 라벨로 LR을 학습한 뒤 ‘uncertain’을 고르는 것은 하지 않는다. Oracle는 test ID, 중복 조회와100개 초과 조회를 거절한다. Manifest에 known_before·선택 ID·반환 라벨을 남긴다.50 budget 결과는50-prefix의 정보만 학습에 사용한다.

모든510개의 원래five-generation 라벨은 예전 표준 실험의 ref/test 메타데이터를 합치면 복원된다. 예전 표준 실험에서 test였다는 이름 자체가 현재 학습 금지는 아니다. **현재의 disjoint test**만 영구 차단하고, 현재 후보 라벨은 선택 후 비용을 지불했을 때만 공개한다. 이는 이미 완전 라벨링된 자료의 acquisition simulation이지 새 사람 실험이나 미래 데이터 수집이 아니다.

원래 Clotho는100개 순서가 저장되어 있어50-prefix까지 재사용할 수 있다. 다만 원래 알고리즘은 acquired pass rate로 내부 차원을 선택하고 unlabeled test state까지 본 global PCA를 썼다. 따라서 P5는 충실한 저장 이력 대조이지 binary-only oracle로 원래 GMM 알고리즘을 새로 실행한 것이 아니다. 더 풍부한 참조 피드백과 transductive provenance, 과거 실행시간의 부재를 숨기지 않는다. 그래도 current test/future candidate 라벨을 미리 본 정책이라는 뜻은 아니다.

## 3. 무엇을 바꾸고 무엇을 고정했는가?

LLM/Llama13·Gemma12 시스템, layer16/21, seed0–4, strict samples5–9 majority 라벨, 기존 GT비중복 test를 고정했다. Predictor는 참조전용 raw PCA16+전체 참조 covariance whitening+balanced C1 LR로 미리 정했다.16차원이나C를 정책 결과에 맞춰 바꾸지 않았다. 원래 ‘balanced acquisition’은 PASS/FAIL을 반반 맞추는 것이 아니라 **GMM 성분 entropy와max-min diversity를 반반** 쓰는 것이다. 원래 entropy는 실패확률 불확실성과 같지 않다.

새 네 정책은 실제10개 random warmup이 같다. P1은계속 random, P2는숫자/hex를 마스킹한 텍스트의char TFIDF 거리로diversity, P3는candidate-only PCA16 은닉거리로diversity, P4는관측라벨 LR의결정 margin이0에가까운10개를 고른다. P5는저장Clotho순서다. 앞의 네 정책은 GT나 candidate 출력을 사용하지 않는다. 새 LLM/GPU/은닉 추출도 없다.

Nominal50/100은공통초기10+실제40/90개의라벨이다. 후보가달라지면최종PCA도다시적합되므로 **고정된 것은 학습 recipe이지 이미 학습된 projection이 아니다**. 참조 선택의 전체 파이프라인 효과를 측정하지만, 라벨 구성의 효과와 PCA 변화의 효과를 나누지는 못한다.

표준 split에서는원래Clotho참조의complement가test라서새정책이그test를선택할수있다. 그래서이번에는억지로표준test를유지하거나다른집단을같다고부르지않고, 기존disjoint test만평가했다. 새로운standard-to-disjoint gap 감소도 주장하지 않는다.

## 4. 실제 수치는 무엇이며 무엇을 뜻하는가?

AUROC는 FAIL 위험점수가 PASS보다큰확률(동점0.5)이며 AP는FAIL 양성 average precision이다. 시스템별지표를먼저구하고5seed macro의평균±표본SD를보고한다. 구간은4,000번시스템bootstrap하며그시스템의5seed를함께보존한다. 줄이나seed를독립시스템으로세지않았다.

| Total budget | Policy | Llama AUROC | Llama AP | Gemma AUROC | Gemma AP |
|---|---|---|---|---|---|
| 50 | Random | 0.6982 ± 0.0218 | 0.8113 | 0.6728 ± 0.0249 | 0.7029 |
| 50 | Masked-text diversity | 0.7069 ± 0.0278 | 0.8194 | 0.6844 ± 0.0183 | 0.7154 |
| 50 | Hidden-state diversity | 0.7070 ± 0.0205 | 0.8218 | 0.6960 ± 0.0209 | 0.7233 |
| 50 | LR uncertainty | 0.6811 ± 0.0387 | 0.8041 | 0.6677 ± 0.0236 | 0.7023 |
| 50 | Archived Clotho acquisition | 0.6900 ± 0.0268 | 0.8149 | 0.6886 ± 0.0124 | 0.7245 |
| 100 | Random | 0.7050 ± 0.0094 | 0.8127 | 0.6977 ± 0.0175 | 0.7294 |
| 100 | Masked-text diversity | 0.7233 ± 0.0195 | 0.8292 | 0.7199 ± 0.0214 | 0.7478 |
| 100 | Hidden-state diversity | 0.7264 ± 0.0200 | 0.8341 | 0.7154 ± 0.0186 | 0.7434 |
| 100 | LR uncertainty | 0.7059 ± 0.0322 | 0.8166 | 0.7001 ± 0.0204 | 0.7255 |
| 100 | Archived Clotho acquisition | 0.7188 ± 0.0140 | 0.8310 | 0.7138 ± 0.0064 | 0.7406 |

| Budget | Paired AUROC contrast | Llama Δ [95% CI] | Gemma Δ [95% CI] |
|---|---|---|---|
| 50 | Masked-text diversity − Random | +0.0087 [-0.0178, +0.0348] | +0.0115 [-0.0095, +0.0324] |
| 50 | Hidden-state diversity − Random | +0.0089 [-0.0151, +0.0349] | +0.0232 [+0.0057, +0.0423] |
| 50 | LR uncertainty − Random | -0.0170 [-0.0485, +0.0115] | -0.0052 [-0.0301, +0.0198] |
| 50 | Archived Clotho acquisition − Random | -0.0081 [-0.0391, +0.0227] | +0.0157 [-0.0081, +0.0366] |
| 50 | Masked-text diversity − Archived Clotho acquisition | +0.0169 [-0.0066, +0.0448] | -0.0042 [-0.0254, +0.0181] |
| 50 | Hidden-state diversity − Archived Clotho acquisition | +0.0170 [+0.0024, +0.0331] | +0.0074 [-0.0088, +0.0271] |
| 50 | LR uncertainty − Archived Clotho acquisition | -0.0089 [-0.0362, +0.0200] | -0.0209 [-0.0433, +0.0005] |
| 100 | Masked-text diversity − Random | +0.0183 [-0.0034, +0.0400] | +0.0222 [+0.0003, +0.0421] |
| 100 | Hidden-state diversity − Random | +0.0214 [-0.0021, +0.0451] | +0.0177 [+0.0068, +0.0293] |
| 100 | LR uncertainty − Random | +0.0009 [-0.0276, +0.0287] | +0.0023 [-0.0150, +0.0216] |
| 100 | Archived Clotho acquisition − Random | +0.0139 [-0.0134, +0.0426] | +0.0161 [+0.0030, +0.0297] |
| 100 | Masked-text diversity − Archived Clotho acquisition | +0.0045 [-0.0166, +0.0230] | +0.0061 [-0.0082, +0.0189] |
| 100 | Hidden-state diversity − Archived Clotho acquisition | +0.0075 [-0.0010, +0.0150] | +0.0016 [-0.0054, +0.0088] |
| 100 | LR uncertainty − Archived Clotho acquisition | -0.0130 [-0.0261, +0.0007] | -0.0137 [-0.0303, +0.0022] |

100개에서hidden−random은Llama+0.0214[−0.0021,0.0451]/Gemma+0.0177[0.0068,0.0293]다. Structural−random도+0.0183[−0.0034,0.0400]/+0.0222[0.0003,0.0421]로Gemma에서더명확하다. 따라서‘다양성은아무영향없다’고할수없지만, 두모델에서확실히재현됐다고도할수없다.

더중요한original과의100개비교는hidden+0.0075[−0.0010,0.0150]/+0.0016[−0.0054,0.0088], structural+0.0045[−0.0166,0.0230]/+0.0061[−0.0082,0.0189]다. 모두0을포함한다. 원래정책도diversity를이미사용하므로random만이겨새기여가생겼다고말하면안된다.

50개에서는Llama hidden−original+0.0170[0.0024,0.0331]이실제양의근거다. 하지만Gemma는+0.0074[−0.0088,0.0271]로불확실하다. 이부조건을숨기지않되, 테스트를보고‘Llama50전용새정책’을만들거나모든seed승리만으로유의성을주장하지않는다. P4 LR uncertainty는100개에서random대비+0.0009/+0.0023이고구간은0을포함한다. 최종예측기와같은uncertainty를쓴다고자동으로좋아지지않는다.

| Policy | Llama100−50 AUROC [CI] | Gemma100−50 AUROC [CI] |
|---|---|---|
| Random | +0.0068 [-0.0154, +0.0260] | +0.0249 [+0.0118, +0.0375] |
| Masked-text diversity | +0.0164 [-0.0016, +0.0316] | +0.0355 [+0.0206, +0.0506] |
| Hidden-state diversity | +0.0194 [+0.0092, +0.0300] | +0.0194 [+0.0121, +0.0277] |
| LR uncertainty | +0.0247 [+0.0112, +0.0379] | +0.0324 [+0.0178, +0.0463] |
| Archived Clotho acquisition | +0.0288 [+0.0164, +0.0428] | +0.0252 [+0.0085, +0.0431] |

Original50→100은+0.0288/+0.0252로두모델구간이양수다. 기존정책보다새정책이더좋다는100개차이보다point효과가크다. 그러나Llama random의추가50라벨효과는불확실하므로‘더많으면항상개선’도아니다. 같은test에서학습budget이늘어난pipeline효과이며PCA와classcount도함께변한다.

## 5. FAIL을 더 많이 골라서 좋아졌는가? 다양성이 원인인가?

| Model | Policy@100 | FAIL labels | Structural coverage | Structural redundancy | Candidate-GT coverage | Mean hidden covering distance |
|---|---|---|---|---|---|---|
| llama | Random | 62.4462 | 0.4769 | 0.3275 | 0.5835 | 0.1983 |
| llama | Masked-text diversity | 61.1231 | 0.6434 | 0.1713 | 0.7757 | 0.1714 |
| llama | Hidden-state diversity | 61.3846 | 0.6565 | 0.1670 | 0.7918 | 0.1441 |
| llama | LR uncertainty | 56.8000 | 0.4540 | 0.3670 | 0.5399 | 0.2854 |
| llama | Archived Clotho acquisition | 65.8615 | 0.6114 | 0.2236 | 0.7540 | 0.1641 |
| gemma | Random | 51.7833 | 0.4776 | 0.2904 | 0.5876 | 3.4548 |
| gemma | Masked-text diversity | 50.0000 | 0.6329 | 0.1304 | 0.7570 | 3.2788 |
| gemma | Hidden-state diversity | 50.8000 | 0.6082 | 0.1569 | 0.7654 | 2.5968 |
| gemma | LR uncertainty | 48.7500 | 0.4559 | 0.3135 | 0.5617 | 4.8372 |
| gemma | Archived Clotho acquisition | 54.3833 | 0.5721 | 0.2007 | 0.7409 | 2.8461 |

Coverage는candidate의관측구조/GT종류중선택한비율이지미래test GT의coverage가아니다. Test GT coverage는모든방법0이다. Hidden distance는model별원래단위가달라Llama의0.1과Gemma의3을크기비교하지않는다. 같은model/pool안의정책차이만읽는다.

P2/P3는original보다FAIL라벨을더많이얻지않는데도AUROC point가약간높다. P4는거의반반에가까운class구성이지만성능우위가없고구조중복률은더크다. 따라서‘FAIL이많아서’, ‘반반이라서’라는단순설명은충분하지않다. 그렇다고classbalance의인과효과가없다고증명한것도아니다.

Diversity가자기가최적화한coverage를높이는것은예상되는algorithm check다. 그것이transfer가능한실패정보를더잘얻었다는독립증거는아니다. Class구성·coverage·PCA가동시에변하므로저장상관을mediation/원인으로읽지않는다. P4가참조를좁게골랐다는관찰은있지만, 그라벨이본질적으로noise가많다는추가40회검증은하지않았다.

가장중요한반례는HPC다. Structural−original이Llama에서는+0.0821인데Gemma에서는−0.0395다. Llama OpenSSH는−0.0918, Gemma는−0.0180이다. 현재후보구조를다양하게커버하는것과미지GT실패를잘예측하는것이같지않다.

## 6. 라벨 수가 같으면 비용도 같은가?

| Policy@100 | New real labels | Common initial labels | Preselection state count incl.initial | Selector CPU sec mean | Adaptive selector updates |
|---|---|---|---|---|---|
| Random | 90 | 10 | 100.0000 | 0.0009 | none |
| Masked-text diversity | 90 | 10 | 100.0000 | 0.0073 | none |
| Hidden-state diversity | 90 | 10 | 264.8800 | 0.0074 | none |
| LR uncertainty | 90 | 10 | 264.8800 | 0.2469 | 8 LR fits |
| Archived Clotho acquisition | 90 | 10 | 510.0000 | unavailable | original adaptive GMM/CV, not retimed |

모든100개방법은실제90개+공통초기10개라벨을쓴다. 그러나random/텍스트diversity는선택된입력의state만필요하고, hidden/uncertainty는전체candidate state를라벨링전에필요로한다. 원래Clotho는510개전체와GMM/CV까지썼다. 지금CPU실험은저장state를읽어새추출0회지만배포에서도공짜라는뜻이아니다. 최종test입력의state는예측시별도로모두필요하다. 실제사람annotation시간도측정하지않았으므로equal count를equal human minutes라고부르지않는다. P5과거선택시간은NA이지0이아니다.

## 7. 무엇이 검증되었고 무엇이 남는가?

125작업×5정책×2budget=1,250셀이완료되었다. 추가제외/fallback은0이다. Test ID/라벨은원래와같고모든125개GT비중복을독립확인했다. Oracle는현재test/중복/초과조회를거절한다.14개poison검사에서아직공개하지않은라벨을바꿔도LR의다음선택은같고, label-free정책은전체라벨을바꿔도같다. Test state를바꿔도다음adaptive선택은같다. 원래LR100risk는bitwise재사용되고raw재계산AUC도1e−12에서일치한다. 두전체job재실행26개배열/비시간metadata와440개cluster추정치도독립검증했다.

하지만현재13/12개시스템을여러연구에재사용했고GT random holdout이지실제미래시간shift가아니다. 고정recipe에서PCA가참조선택에따라바뀌는mediator도남는다. P5는원래더풍부한feedback/global covariate를썼고, 다른정책은binary feedback/candidate-only라access가완전히같지않다. 새policy가배포가능한입력만사용한다는것과, 그효과가새시스템에확증됐다는것은다르다.

**판단:** 기존참조연구는있었지만현대simple predictor와의좁은상호작용은새롭게검사할가치가있었다. 결과는새acquisition방법개발을정당화하지못한다. 이benchmark에서다른이름의sampling/PCA/density탐색을계속하지말고, 조건부이득과반례를포함한통제된실증결과로정리한다. Originalpolicy가최적이라고증명하거나reference선택이무의미하다고도하지않는다.

**만약확인실험하나를의뢰한다면:** original대사전에정한diversity정책하나를새로운독립chronological unseen-template population에서동일label피드백/state비용으로고정비교한다. ‘현재coverage의작은이득이미래에도전달된다’는가설을가장직접반박/검증할수있다. 현재결과만으로그확장이나새방법개발을권고하는것은아니며, 여기서추가GPU실험은하지않았다.

[기존연구audit](clotho_reference_acquisition_audit.md), [상세pilot](clotho_reference_selection_pilot.md), `results_phase3/clotho_reference_selection/`의manifest·prior_inventory·metrics·paired bootstrap·validation·PDF/PNG를함께보면수치와한계를확인할수있다. 이전결과를덮어쓰지않았고원격hash보존후PROGRESS를갱신한다.
