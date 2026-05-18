# Aing_리그전 Transformer Fine-tune Explain

## 리그전 목적
- 목표: 고정된 pre-train Transformer 모델을 **IWSLT 2017 (de→en)** 데이터에 fine-tuning하여 번역 성능을 높입니다.
- 리그전 방식: 참가자는 코드에서 **Fine-tune 하이퍼파라미터(`FT_HP`)만 튜닝**합니다.
- 최종 점수: 참가자가 직접 `final` 모드에서 실행한 **IWSLT test SacreBLEU(BLEU)** 입니다.

이번 리그전에서는 pre-train 모델 구조와 pre-train 학습 과정은 고정되어 있습니다.  
참가자는 같은 pre-train 모델에서 시작해, fine-tuning 설정을 어떻게 잡느냐를 비교합니다.

---

## 점수(리더보드) 기준

### 실험 중 참고 점수
- `public_fast`, `public_full` 모드에서는 **IWSLT validation BLEU**를 확인합니다.
- 이 점수는 튜닝 방향을 잡기 위한 참고 점수입니다.
- 출력 이름:

```python
LEAGUE_VALID_SCORE_IWSLT_VALID_BLEU
```

### 최종 점수
- `final` 모드에서는 **IWSLT test BLEU**까지 계산합니다.
- 최종 리그전 점수는 아래 값입니다.

```python
LEAGUE_FINAL_SCORE_IWSLT_TEST_BLEU
```

> 참가자가 직접 test까지 실행합니다.  
> 따라서 최종 제출 점수는 `LEAGUE_MODE = "final"`에서 출력되는 `LEAGUE_FINAL_SCORE_IWSLT_TEST_BLEU`입니다.

---

## `LEAGUE_MODE` 상황별 사용법

노트북에는 세 가지 실행 모드가 있습니다.

```python
LEAGUE_MODE = "public_fast"  # "public_fast" / "public_full" / "final"
```

참가자는 실험 상황에 따라 `LEAGUE_MODE`를 바꿔 사용해야 합니다.

| 상황 | 사용할 모드 | 목적 |
|---|---|---|
| 여러 `FT_HP` 조합을 빠르게 비교할 때 | `public_fast` | validation 일부와 짧은 fine-tune 설정으로 빠르게 방향성 확인 |
| 제출 후보 1~2개를 더 정확히 확인할 때 | `public_full` | 전체 validation과 final에 가까운 설정으로 후보 검증 |
| 최종 리그전 점수를 계산할 때 | `final` | IWSLT test BLEU까지 계산 |

### `public_fast`
- 빠른 실험용 모드입니다.
- IWSLT train subset과 validation subset을 사용합니다.
- beam search를 줄여 실험 속도를 높입니다.
- 여러 조합을 빠르게 비교할 때 사용합니다.
- 이 모드의 점수는 최종 점수가 아니라 참고 점수입니다.

### `public_full`
- 제출 전 확인용 모드입니다.
- `final`과 같은 fine-tune step budget과 전체 validation을 사용합니다.
- test BLEU는 계산하지 않습니다.
- `public_fast`에서 좋았던 조합 1~2개만 확인하는 용도로 사용합니다.

### `final`
- 최종 점수 계산용 모드입니다.
- IWSLT validation과 test BLEU를 모두 계산합니다.
- 최종 리그전 점수인 `LEAGUE_FINAL_SCORE_IWSLT_TEST_BLEU`를 출력합니다.

권장 흐름:

```text
public_fast로 여러 조합 탐색
→ public_full로 상위 후보 확인
→ final로 최종 점수 계산
```

---

## 튜닝 허용 범위(중요)

이번 리그전에서 바꿀 수 있는 값은 **`FT_HP` 안의 5개 값만**입니다.

```python
FT_HP = dict(
    learning_rate=1e-4,            # 🔧 TUNE: 1e-5 ~ 5e-4
    lr_scheduler_type="linear",    # 🔧 TUNE: linear / cosine / inverse_sqrt
    warmup_steps=50,               # 🔧 TUNE: 0 ~ 300
    weight_decay=0.0,              # 🔧 TUNE: 0.0 ~ 0.1
    label_smoothing_factor=0.10,   # 🔧 TUNE: 0.0 ~ 0.20
)
```

### 바꿀 수 있는 값
- `learning_rate`
- `lr_scheduler_type`
- `warmup_steps`
- `weight_decay`
- `label_smoothing_factor`

### 고정값
- pre-train 모델 구조
- pre-train 학습 설정
- tokenizer/BPE 설정
- `VOCAB_SIZE`, `MAX_LEN`
- batch size / gradient accumulation
- 데이터셋 샘플링 규칙
- generation beam 설정
- validation/test 평가 방식

---

## Flow Map

1) Setup: 설치 / 임포트 / 시드 / 디바이스 설정  
2) Datasets: Multi30k + IWSLT 2017 로드  
3) Tokenizer: Multi30k + IWSLT 기반 BPE tokenizer 준비  
4) Model: 고정 Transformer 모델 생성 또는 pre-train checkpoint 로드  
5) Stage A: Multi30k 기반 pre-train 또는 cached checkpoint 사용  
6) Stage B: IWSLT fine-tune (**🔧 TUNE 가능: `FT_HP`만**)  
7) Validation: IWSLT validation BLEU 확인  
8) Final: IWSLT test BLEU 최종 점수 출력

---

## 데이터셋 정리

### Multi30k (de→en)
- **용도:** Stage A pre-train 데이터
- **역할:** 모델이 기본적인 독일어→영어 번역 능력을 학습하는 데 사용됩니다.
- **주의:** 최종 리그전 점수는 Multi30k가 아니라 **IWSLT test BLEU**입니다.

### IWSLT 2017 (de→en)
- **용도:** Stage B fine-tune 및 validation/test 평가 데이터
- **validation 점수:** 튜닝 방향을 잡기 위한 참고 점수
- **test 점수:** 리그전 최종 점수

---

## 튜닝하는 하이퍼파라미터

### 1) `learning_rate`
- 모델 파라미터를 한 번 업데이트할 때의 이동 크기입니다.
- 값이 크면 빠르게 적응할 수 있지만 학습이 불안정해질 수 있습니다.
- 값이 작으면 안정적이지만 제한된 step 안에서 충분히 적응하지 못할 수 있습니다.

탐색 후보 예시:

```python
5e-5, 1e-4, 2e-4, 3e-4
```

---

### 2) `lr_scheduler_type`
- 학습이 진행되면서 learning rate를 어떻게 변화시킬지 정합니다.

탐색 후보:

```python
"linear"
"cosine"
"inverse_sqrt"
```

- `linear`: 단순하고 안정적인 baseline
- `cosine`: 후반으로 갈수록 부드럽게 감소
- `inverse_sqrt`: Transformer 계열 학습에서 자주 쓰이는 감소 방식

---

### 3) `warmup_steps`
- 학습 초반 learning rate를 작은 값에서 천천히 올리는 구간입니다.
- 학습이 불안정하면 warmup을 늘려볼 수 있습니다.
- 점수가 너무 천천히 오르면 warmup을 줄여볼 수 있습니다.

탐색 후보 예시:

```python
0, 50, 100, 200
```

---

### 4) `weight_decay`
- 가중치가 너무 커지는 것을 막는 regularization입니다.
- 적절한 값은 test 성능을 안정시키는 데 도움이 될 수 있습니다.

탐색 후보 예시:

```python
0.0, 0.01, 0.05
```

---

### 5) `label_smoothing_factor`
- 정답 토큰에만 100% 확신하지 않도록 만드는 regularization입니다.
- 너무 크면 학습이 둔해질 수 있습니다.

탐색 후보 예시:

```python
0.0, 0.05, 0.10, 0.15
```

---

## 초보자 실험 가이드

### 1단계: 기준 점수 만들기

1. `LEAGUE_MODE = "public_fast"`로 둡니다.
2. 기본 `FT_HP`로 한 번 실행합니다.
3. `LEAGUE_VALID_SCORE_IWSLT_VALID_BLEU`를 기록합니다.

이 점수가 이후 실험의 기준점입니다.

### 2단계: `learning_rate` 먼저 비교하기

가장 먼저 `learning_rate`만 바꿔 비교합니다.

```python
5e-5, 1e-4, 2e-4, 3e-4
```

다른 값은 그대로 두고, 어떤 learning rate에서 BLEU가 잘 오르는지 확인합니다.

### 3단계: scheduler와 warmup 조정하기

좋았던 learning rate를 고정한 뒤 아래 값을 비교합니다.

```python
lr_scheduler_type: "linear" / "cosine" / "inverse_sqrt"
warmup_steps: 0 / 50 / 100 / 200
```

### 4단계: regularization 조정하기

마지막으로 아래 두 값을 조정합니다.

```python
weight_decay: 0.0 / 0.01 / 0.05
label_smoothing_factor: 0.0 / 0.05 / 0.10 / 0.15
```

### 5단계: 최종 후보 확인하기

`public_fast`에서 가장 좋았던 조합 1~2개는 먼저 `public_full`로 다시 확인합니다.

```python
LEAGUE_MODE = "public_full"
```

`public_full`에서 가장 좋은 조합을 최종 후보로 정한 뒤, 최종 점수 확인을 위해 `final` 모드로 실행합니다.

```python
LEAGUE_MODE = "final"
```

실행 후 아래 값을 기록합니다.

```python
LEAGUE_FINAL_SCORE_IWSLT_TEST_BLEU
```

---

## 실험 기록표 예시

| 실험 번호 | mode | learning_rate | scheduler | warmup | weight_decay | label_smoothing | valid BLEU | final test BLEU | 메모 |
|---|---|---:|---|---:|---:|---:|---:|---:|---|
| 1 | public_fast | 1e-4 | linear | 50 | 0.0 | 0.10 |  |  | 기준 점수 |
| 2 | public_fast | 2e-4 | linear | 50 | 0.0 | 0.10 |  |  | learning_rate 변경 |
| 3 | public_full |  |  |  |  |  |  |  | 제출 후보 확인 |
| 4 | final |  |  |  |  |  |  |  | 최종 점수 확인 |

---

## 점수가 잘 안 오를 때 확인할 것

| 현상 | 가능한 원인 | 시도해볼 것 |
|---|---|---|
| BLEU가 거의 오르지 않음 | `learning_rate`가 너무 작을 수 있음 | `learning_rate`를 한 단계 키워 보기 |
| loss가 불안정하거나 BLEU가 떨어짐 | `learning_rate`가 너무 클 수 있음 | `learning_rate`를 줄이거나 `warmup_steps` 늘리기 |
| 초반 학습이 너무 느림 | `warmup_steps`가 너무 길 수 있음 | `warmup_steps` 줄이기 |
| validation은 높은데 test가 낮음 | validation에만 잘 맞았을 수 있음 | 너무 공격적인 LR을 줄이거나 regularization 조정 |
| 여러 조합 점수가 비슷함 | 영향이 작은 파라미터만 바꿨을 수 있음 | `learning_rate`, `scheduler`, `warmup` 중심으로 다시 비교 |

---

## 제출 시 확인할 것

최종 제출 전 아래 내용을 확인합니다.

```python
FT_HP
LEAGUE_FINAL_SCORE_IWSLT_TEST_BLEU
```

`LEAGUE_MODE = "final"`에서 출력된 `LEAGUE_FINAL_SCORE_IWSLT_TEST_BLEU`가 최종 리그전 점수입니다.

---

## 제작 정보 & 출처

- 제작: 가천대학교 인공지능 학술 동아리 **Aing (A.ing)**

### 사용/참고 자료
- Vaswani et al., **Attention Is All You Need**, NeurIPS 2017.
- HuggingFace Transformers/Datasets 문서: Seq2Seq 학습(Trainer), Multi30k, IWSLT 로딩.
- `tokenizers`(BPE), `sacrebleu`(BLEU 평가) 라이브러리.
