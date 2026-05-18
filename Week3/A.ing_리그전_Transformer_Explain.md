# Aing_리그전 Transformer Fine-tune Explain

## 리그전 목적
- 목표: 고정된 pre-train Transformer 모델을 **IWSLT 2017 (de→en)** 데이터에 fine-tuning하여 번역 성능을 높입니다.
- 리그전 방식: 참가자는 마지막 실험 설정 셀에서 **`LEAGUE_MODE`와 `FT_HP`만 수정**합니다.
- 최종 점수: `LEAGUE_MODE = "final"`로 실행했을 때 출력되는 **IWSLT test BLEU**입니다.

이 리그전의 핵심은 같은 pre-train 모델에서 시작해, fine-tuning 하이퍼파라미터를 어떻게 설정하느냐를 비교하는 것입니다.

---

## 전체 실행 흐름

1. STEP1~STEP8: 라이브러리, 데이터, tokenizer, 모델, 평가 함수 준비
2. STEP9: Multi30k로 fixed pre-train 1회 실행
3. STEP10~STEP11: IWSLT fine-tuning 데이터 준비
4. STEP12-A: 참가자가 `LEAGUE_MODE`와 `FT_HP` 수정
5. STEP12-B: fine-tuning 실행 및 validation/test BLEU 확인

반복 실험할 때는 **STEP12-A와 STEP12-B만 다시 실행**합니다.

---

## 중요한 실행 구조

노트북 마지막 구간은 두 셀로 나뉩니다.

### 1) 실험 설정 셀

참가자가 수정하는 셀입니다.

```python
LEAGUE_MODE = "public_fast"  # "public_fast" / "public_full" / "final"

FT_HP = dict(
    learning_rate=1e-4,
    lr_scheduler_type="linear",
    warmup_steps=50,
    weight_decay=0.0,
    label_smoothing_factor=0.10,
)
```

### 2) Fine-tune 실행 셀

바로 아래 실행 셀은 설정한 `LEAGUE_MODE`와 `FT_HP`로 fine-tuning을 실행합니다.

- `FT_HP`만 바꾼 경우: 실험 설정 셀 + fine-tune 실행 셀만 다시 실행
- `LEAGUE_MODE`를 바꾼 경우: 실험 설정 셀 + fine-tune 실행 셀만 다시 실행
- Stage A pre-train 셀은 다시 실행하지 않음

---

## LEAGUE_MODE 사용법과 예상 시간

```python
LEAGUE_MODE = "public_fast"  # "public_fast" / "public_full" / "final"
```

| 모드 | 사용 상황 | 평가 방식 | fine-tuning 실행 셀 재실행 기준 예상 시간 |
|---|---|---|---:|
| `public_fast` | 여러 조합을 빠르게 비교할 때 | validation 일부 | 약 3분 |
| `public_full` | 좋은 후보를 전체 validation으로 확인할 때 | validation 전체 | 약 8분 |
| `final` | 최종 test BLEU를 계산할 때 | test 전체 | 약 20분 |

권장 흐름:

```text
public_fast로 여러 조합 탐색
→ public_full로 상위 후보 확인
→ final로 최종 test BLEU 계산
```

`public_full`과 `final`은 fine-tuning 설정은 비슷하지만 평가 데이터가 다릅니다. `public_full`은 validation 기준이고, `final`은 test 기준입니다.

---

## 점수 기준

### validation 점수

`public_fast`, `public_full`에서 확인하는 참고 점수입니다.

```python
LEAGUE_VALID_SCORE_IWSLT_VALID_BLEU
```

### final 점수

`final` 모드에서 직접 계산하는 최종 점수입니다.

```python
LEAGUE_FINAL_SCORE_IWSLT_TEST_BLEU
```

리그전 최종 순위는 `LEAGUE_FINAL_SCORE_IWSLT_TEST_BLEU` 기준으로 결정합니다.

---

## 튜닝 허용 범위

이번 리그전에서 바꿀 수 있는 값은 **`FT_HP` 안의 5개 값**입니다.

```python
FT_HP = dict(
    learning_rate=1e-4,            # 1e-5 ~ 5e-4
    lr_scheduler_type="linear",    # linear / cosine / inverse_sqrt
    warmup_steps=50,               # 0 ~ 300
    weight_decay=0.0,              # 0.0 ~ 0.1
    label_smoothing_factor=0.10,   # 0.0 ~ 0.20
)
```

### 바꿀 수 있는 값
- `learning_rate`
- `lr_scheduler_type`
- `warmup_steps`
- `weight_decay`
- `label_smoothing_factor`

### 바꾸지 않는 값
- pre-train 모델 구조
- pre-train 학습 설정
- tokenizer/BPE 설정
- `VOCAB_SIZE`, `MAX_LEN`
- batch size / gradient accumulation
- 데이터셋 샘플링 규칙
- 평가 방식

---

## 튜닝 하이퍼파라미터 설명

### 1) `learning_rate`

#### 정의
모델 파라미터를 한 번 업데이트할 때 얼마나 크게 움직일지 정하는 값입니다. Fine-tuning에서 가장 먼저 확인해야 하는 핵심 하이퍼파라미터입니다.

#### 값을 올리면
- IWSLT 데이터에 더 빠르게 적응할 수 있습니다.
- 짧은 step 안에서 BLEU가 빨리 오를 수 있습니다.
- 너무 크면 loss가 불안정해지거나 BLEU가 떨어질 수 있습니다.

#### 값을 내리면
- 더 안정적으로 학습할 수 있습니다.
- 너무 작으면 fine-tuning 효과가 거의 나타나지 않을 수 있습니다.

#### 실험 감각
처음에는 `5e-5`, `1e-4`, `2e-4`, `3e-4`처럼 learning rate만 바꿔 비교하는 것을 권장합니다.

---

### 2) `lr_scheduler_type`

#### 정의
학습이 진행되면서 learning rate를 어떤 모양으로 변화시킬지 정하는 값입니다.

#### 선택지
- `linear`: learning rate를 일정하게 줄입니다. 단순한 baseline으로 좋습니다.
- `cosine`: learning rate를 부드럽게 줄입니다. 후반부 안정화에 도움이 될 수 있습니다.
- `inverse_sqrt`: Transformer 계열에서 자주 쓰이는 방식입니다. warmup 이후 완만하게 줄어듭니다.

#### 바꿨을 때 효과
- 같은 learning rate라도 scheduler에 따라 초반/후반 학습 양상이 달라집니다.
- `linear`는 단순하고 예측하기 쉽습니다.
- `cosine`은 후반부가 비교적 부드러워 안정적인 경우가 있습니다.
- `inverse_sqrt`는 warmup과 함께 쓸 때 Transformer 학습에 잘 맞는 경우가 있습니다.

#### 실험 감각
먼저 좋은 `learning_rate`를 찾은 뒤, 그 값을 고정하고 scheduler를 비교하는 것이 좋습니다.

---

### 3) `warmup_steps`

#### 정의
학습 초반에 learning rate를 바로 크게 쓰지 않고, 작은 값에서 천천히 올리는 구간입니다.

#### 값을 올리면
- 초반 학습이 안정적입니다.
- 큰 learning rate를 사용할 때 폭주를 줄일 수 있습니다.
- 너무 크면 실제로 충분히 학습하는 구간이 짧아질 수 있습니다.

#### 값을 내리면
- 초반부터 빠르게 학습할 수 있습니다.
- 짧은 `public_fast` 실험에서는 빠른 성능 상승에 유리할 수 있습니다.
- 너무 작으면 초반 loss가 불안정해질 수 있습니다.

#### 실험 감각
`0`, `50`, `100`, `200` 정도를 비교해 볼 수 있습니다. learning rate가 클수록 warmup이 어느 정도 있는 편이 안정적입니다.

---

### 4) `weight_decay`

#### 정의
모델 가중치가 지나치게 커지는 것을 막는 regularization 값입니다. 과적합을 줄이는 역할을 합니다.

#### 값을 올리면
- 일반화 성능이 좋아질 수 있습니다.
- validation/test BLEU가 안정적으로 나올 수 있습니다.
- 너무 크면 모델이 충분히 학습하지 못할 수 있습니다.

#### 값을 내리면
- 학습 데이터에 더 빠르게 맞출 수 있습니다.
- 과적합 위험이 커질 수 있습니다.

#### 실험 감각
처음에는 `0.0`과 `0.01`을 비교하는 정도면 충분합니다. learning rate나 scheduler보다 영향이 작게 보일 수 있습니다.

---

### 5) `label_smoothing_factor`

#### 정의
정답 토큰 하나에만 100% 확신을 주지 않고, 정답 분포를 조금 부드럽게 만드는 값입니다.

#### 값을 올리면
- 모델이 과하게 확신하는 것을 줄일 수 있습니다.
- 일반화에 도움이 될 수 있습니다.
- 너무 크면 정답을 강하게 학습하지 못해 BLEU가 낮아질 수 있습니다.

#### 값을 내리면
- 정답에 더 강하게 맞추도록 학습합니다.
- 짧은 step에서는 빠르게 학습되는 것처럼 보일 수 있습니다.
- 과적합 위험이 커질 수 있습니다.

#### 실험 감각
`0.0`, `0.05`, `0.10`, `0.15` 정도를 비교해 볼 수 있습니다. 너무 높은 값은 학습을 둔하게 만들 수 있습니다.

---

## 데이터셋 정리

### Multi30k (de→en)
- Stage A pre-train 데이터
- 기본 독일어→영어 번역 능력을 학습하는 데 사용
- 참가자가 직접 튜닝하는 대상은 아님

### IWSLT 2017 (de→en)
- Stage B fine-tuning 데이터
- validation 점수와 final test 점수를 계산하는 데이터셋
- 이번 리그전에서 실제로 성능을 높여야 하는 대상

---

## 초보자 실험 가이드

1. `public_fast`로 기본 `FT_HP`를 실행해 기준 점수를 기록합니다.
2. `learning_rate`만 바꿔 3~4개 후보를 비교합니다.
3. 좋은 learning rate를 고정하고 `lr_scheduler_type`을 비교합니다.
4. `warmup_steps`를 조정합니다.
5. 마지막으로 `weight_decay`, `label_smoothing_factor`를 조정합니다.
6. 가장 좋은 조합 1~2개를 `public_full`로 다시 확인합니다.
7. 최종 조합을 `final`로 실행해 test BLEU를 확인합니다.

---

## 실험 기록표 예시

| 실험 번호 | LEAGUE_MODE | learning_rate | scheduler | warmup | weight_decay | label_smoothing | valid BLEU | final BLEU | 메모 |
|---|---|---:|---|---:|---:|---:|---:|---:|---|
| 1 | public_fast | 1e-4 | linear | 50 | 0.0 | 0.10 |  |  | baseline |
| 2 | public_fast | 2e-4 | linear | 50 | 0.0 | 0.10 |  |  | learning_rate 변경 |
| 3 | public_full |  |  |  |  |  |  |  | 후보 검증 |
| 4 | final |  |  |  |  |  |  |  | 최종 점수 |

---
## 📌 제작 정보 & 출처

- 제작: 가천대학교 인공지능 학술 동아리 **Aing (A.ing)**

### 사용/참고 자료
- Vaswani et al., **Attention Is All You Need**, NeurIPS 2017.
- A.ing 내부 스터디 자료: *Attention 치트시트*, *Transformer CookBook*.
- HuggingFace Transformers/Datasets 문서: Seq2Seq 학습(Trainer), Multi30k, IWSLT 로딩.
- `tokenizers`(BPE), `sacrebleu`(BLEU 평가) 라이브러리.
