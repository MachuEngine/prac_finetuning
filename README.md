# 냥냥마켓 CS봇 파인튜닝 실습

Qwen2.5-1.5B-Instruct 모델을 고양이 용품 쇼핑몰 **냥냥마켓**의 고객 상담 시나리오에 맞게 파인튜닝하는 실습 프로젝트입니다.

---

## 목표

베이스 LLM이 냥냥마켓 전용 상담봇 **'냥냥봇'** 처럼 동작하도록 LoRA 기반 파인튜닝을 수행합니다.
- 배송·환불·VIP 등 쇼핑몰 고유 정책을 답변에 반영
- 봇 정체성(`너는 누구니?` → `냥냥봇`) 학습

---

## 환경

| 항목 | 내용 |
|---|---|
| OS | Windows 11 |
| GPU | NVIDIA GeForce RTX 4060 Ti (8GB VRAM) |
| CUDA | 13.1 (드라이버), PyTorch `cu126` |
| Python | 3.14.4 (가상환경 `.venv`) |
| 베이스 모델 | `Qwen/Qwen2.5-1.5B-Instruct` |

---

## 설치

```bash
# 가상 환경 생성
python -m venv .venv
.venv\Scripts\activate

# CUDA 버전 PyTorch 설치 (Python 3.14 + CUDA 12.6)
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu126

# 나머지 패키지 설치
pip install -r requirements.txt
```

> **주의:** Python 3.14는 매우 최신 버전으로, PyTorch 공식 CUDA 휠은 `cu126` 인덱스에서만 제공됩니다 (`cu121`, `cu124`는 Python 3.14 미지원).

---

## 프로젝트 구조

```
prac_finetuning/
├── data.jsonl          # 학습 데이터 (Q&A 42개)
├── train.py            # 파인튜닝 스크립트
├── inference.py        # 추론 테스트 스크립트
├── requirements.txt    # 패키지 목록
├── .gitignore
├── results/            # 체크포인트 저장 (git 제외)
└── my-custom-lora/     # 학습된 LoRA 어댑터 (git 제외)
```

---

## 학습 데이터 (`data.jsonl`)

`질문: ...\n답변: ...` 형식의 JSONL 파일. 총 **42개** Q&A 쌍.

커버하는 주제:

| 카테고리 | 예시 질문 |
|---|---|
| 배송 | 며칠 걸리나요, 오늘 주문하면 언제 받나요, 주말 배송 여부 |
| 상품 | 뽀송샌드 소재·폐기 방법, 사료 추천, 알러지 사료 |
| 환불·교환 | 단순 변심 반품, 불량품 처리, 파손 배송 |
| 주문 | 주문 취소·수량 변경, 배송 조회 |
| 결제 | 지원 수단, 무이자 할부, 결제 오류 |
| 멤버십 | VIP 혜택·조건, 포인트 사용, 정기 구독 |
| 봇 정체성 | 너는 누구니, 운영 시간, 상담원 연결 |
| 기타 | 쇼핑몰 소개, 회원가입, 재입고 알림, 선물 포장 |

---

## 모델 구조 및 학습 설정

### 양자화 (4-bit QLoRA)

```python
BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_use_double_quant=True,
    bnb_4bit_compute_dtype=torch.bfloat16,
)
```

NF4 양자화로 1.5B 모델을 약 1GB VRAM에 로드합니다.

### LoRA 설정

```python
LoraConfig(
    r=8, lora_alpha=16,
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
    lora_dropout=0.05,
    task_type="CAUSAL_LM",
)
```

전체 파라미터 1.54B 중 **2.18M (0.14%)** 만 학습합니다.

### 학습 하이퍼파라미터

| 파라미터 | 값 |
|---|---|
| optimizer | paged_adamw_8bit |
| learning_rate | 2e-4 |
| batch_size | 1 (gradient_accumulation × 4) |
| max_steps | 200 (~18 epoch) |
| precision | bf16 |

---

## 학습 결과

### 손실 곡선 (step 기준 10 간격 로그)

| Step | Loss | Token Accuracy |
|---|---|---|
| 10 | 2.379 | 53.7% |
| 30 | 1.810 | 62.4% |
| 60 | 1.108 | 75.6% |
| 100 | 0.651 | 86.4% |
| 150 | 0.403 | 93.3% |
| 200 | 0.324 | 94.8% |

### 버전별 비교

| 항목 | v1 (데이터 5개 / 15 steps) | v2 (데이터 42개 / 200 steps) |
|---|---|---|
| train_loss | 2.63 | **0.32** |
| token accuracy | 45% | **95%** |
| 학습 시간 (GPU) | 17초 | 292초 |
| `너는 누구니?` | "AI 어시스턴트" | **"냥냥봇"** ✓ |
| 도메인 정보 반영 | 거의 없음 | 방향성 인식, 일부 세부 오류 |

---

## 실행 방법

### 학습

```bash
$env:PYTHONUTF8 = "1"
.venv\Scripts\python.exe train.py
```

학습 완료 후 어댑터가 `./my-custom-lora/`에 저장됩니다.

### 추론 테스트

```bash
$env:PYTHONUTF8 = "1"
.venv\Scripts\python.exe inference.py
```

---

## TRL 1.x API 변경 사항 (트러블슈팅)

원본 코드는 TRL 0.x 기준으로 작성되어 TRL 1.4.0에서 다음 수정이 필요했습니다.

| 항목 | 구버전 (TRL 0.x) | 현재 (TRL 1.x) |
|---|---|---|
| Training args 클래스 | `TrainingArguments` | `SFTConfig` |
| 텍스트 필드 지정 | `SFTTrainer(..., dataset_text_field="text")` | `SFTConfig(dataset_text_field="text")` |
| 시퀀스 길이 | `SFTTrainer(..., max_seq_length=256)` | `SFTConfig(max_length=256)` |

---

## 현재 한계 및 원인

### 1. 소형 모델의 용량 한계 (1.5B)

1.5B 파라미터 모델은 파인튜닝 데이터를 완전히 암기하더라도 답변 생성 과정에서 베이스 모델의 사전학습 패턴이 강하게 개입합니다. 결과적으로 도메인 방향성은 인식하지만 세부 수치(배송일, 할인율 등)를 잘못 생성하거나 한자·영문이 섞이는 현상이 나타납니다.

> 개선 방향: 7B 이상 모델(예: `Qwen2.5-7B-Instruct`) 사용

### 2. 학습 데이터 부족 (42개)

파인튜닝에서 도메인 특화 정보를 안정적으로 주입하려면 일반적으로 수백~수천 개의 데이터가 필요합니다. 42개는 모델이 패턴을 학습하기에 부족하며, 훈련 데이터를 암기하더라도 비슷한 질문에 대한 일반화 능력이 낮습니다.

> 개선 방향: GPT-4o 등을 활용해 냥냥마켓 시나리오 기반 데이터 200개 이상 자동 생성

### 3. LoRA rank 부족 (r=8)

`r=8`은 매우 적은 수의 파라미터(0.14%)만 학습하므로 표현력이 제한됩니다. 새로운 도메인 지식을 충분히 저장하기 어렵습니다.

> 개선 방향: `r=16` 또는 `r=32`로 증가, `lora_alpha`도 함께 조정

### 4. 짧은 학습 길이 제한 (`max_length=256`)

냥냥마켓 답변 중 일부는 256 토큰을 초과하여 잘립니다. 잘린 답변으로 학습하면 모델이 문장을 완전히 마무리하지 않는 패턴을 학습할 수 있습니다.

> 개선 방향: `max_length=512` 이상으로 설정

### 5. Greedy Decoding

현재 추론에서 `do_sample=False`(탐욕적 디코딩)를 사용하므로 모델이 확률적으로 가장 높은 토큰만 선택합니다. 작은 모델에서는 이 방식이 반복 문구나 엉뚱한 방향으로 흘러가는 경향이 있습니다.

> 개선 방향: `do_sample=True`, `temperature=0.7`, `top_p=0.9` 등 샘플링 전략 적용

---

## 향후 개선 계획

- [ ] 학습 데이터 200개 이상으로 증강 (GPT-4o 자동 생성)
- [ ] LoRA rank를 r=16으로 상향
- [ ] max_length=512로 확장
- [ ] 추론 시 샘플링 파라미터 튜닝
- [ ] 더 큰 베이스 모델(7B)로 실험
