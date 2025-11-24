# JA-BERT 기반 약기법 위반 분류 모델 (Classification Model)

## Overview

이 문서는 **LINE DistilBERT 일본어 모델(line-corporation/line-distilbert-base-japanese)** 을 기반으로 한  
**약기법(PMD Act) 위반 여부 자동 판별 분류 모델**의 학습 방법을 정리한 README입니다.

해당 모델은 다음 목적을 위해 설계되었습니다:

- 일본어 광고 문구에 대해 **약기법 위반 가능성 여부를 자동으로 분류**
- 사내 데이터셋을 기반으로 분류 모델을 학습
- Hugging Face `Trainer` + PyTorch 기반의 재현 가능한 학습 파이프라인 제공

본 README는 PDF 문서 내용만을 기반으로 구성되었습니다.

---

# Training Pipeline

아래 파이프라인은 PDF에 포함된 학습 코드를 기반으로 재구성한 **5단계 학습 절차**입니다.

## 1. Dataset Loading

- Hugging Face Hub에서 학습 데이터 로드  
  (`A1PerformaceFactory/distilbert_med_line`)
- 특정 태그(`line_ad_original`) 기준으로 학습/테스트 데이터 분할

```python
origin_dataset = load_dataset(DATASETS)["train"]

train_dataset = Dataset.from_list(
    [i for i in train_dataset if i["tag"] != "line_ad_original"]
)
test_dataset = Dataset.from_list(
    [i for i in train_dataset if i["tag"] == "line_ad_original"]
)
```

---

## 2. Tokenization

- 기본 모델의 토크나이저 사용 (`AutoTokenizer`)  
- 최대 길이: **512 tokens**  
- padding: `"max_length"`
- truncation: `True`

```python
def tokenize_function(examples):
    return tokenizer(
        examples["text"], truncation=True, padding="max_length", max_length=512
    )
```

---

## 3. Model Initialization

- Base Model: `line-corporation/line-distilbert-base-japanese`
- Task: **Binary Classification (num_labels=2)**

```python
tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL)
model = DistilBertForSequenceClassification.from_pretrained(
    BASE_MODEL, num_labels=2
)
```

---

## 4. Training Setup

학습 환경 설정은 아래와 같습니다:

| 설정 항목 | 값 |
|----------|-----|
| Epochs | 10 |
| Learning Rate | 5e-6 |
| Batch Size | 16 |
| Eval Batch Size | 1 |
| Weight Decay | 0.01 |
| Mixed Precision | bf16 |
| Optimizer | adamw_torch |
| Logging Steps | 10 |
| Eval Steps | 100 |
| Save Steps | 500 |
| report_to | wandb / None (선택) |

```python
training_args = TrainingArguments(
    output_dir=OUTPUT_DIR,
    num_train_epochs=NUM_TRAIN_EPOCHS,
    learning_rate=LEARNING_RATE,
    per_device_train_batch_size=BATCH_SIZE,
    per_device_eval_batch_size=1,
    weight_decay=0.01,
    optim="adamw_torch",
    logging_steps=LOGGING_STEPS,
    evaluation_strategy="steps",
    eval_steps=EVAL_STEPS,
    save_steps=SAVE_STEPS,
    bf16=True,
    report_to="wandb",
)
```

---

## 5. Evaluation Metrics

모델 평가는 다음 지표를 사용합니다:

- **Accuracy**
- **F1 Score (weighted)**
- **Precision (weighted)**
- **Recall (weighted)**

```python
def compute_metrics(pred):
    labels = pred.label_ids
    preds = pred.predictions[0].argmax(-1)

    return {
        "accuracy": accuracy_score(labels, preds),
        "f1": f1_score(labels, preds, average="weighted"),
        "precision": precision_score(labels, preds, average="weighted"),
        "recall": recall_score(labels, preds, average="weighted")
    }
```

---

# Requirements

## Core Libraries

```
torch
transformers
datasets
scikit-learn
```

## Optional (Logging)

```
wandb
```

## For Japanese DistilBERT Model

```
fugashi
sentencepiece
unidic-lite
```

---

# W&B (Weights & Biases) Setup

- 사용 시:
  ```
  wandb login
  report_to="wandb"
  ```
- 미사용 시:
  ```
  report_to=None
  ```

---

# Full Training Code

PDF에 제공된 전체 학습 코드를 README 구조에 맞게 정리하여 제공합니다.

```python
from transformers import AutoTokenizer, DistilBertForSequenceClassification
import torch
from transformers import Trainer, TrainingArguments
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from datasets import Dataset, load_dataset

BASE_MODEL = "line-corporation/line-distilbert-base-japanese"
DATASETS = "A1PerformaceFactory/distilbert_med_line"
OUTPUT_DIR = "./output/distil_med_line"
NUM_TRAIN_EPOCHS = 10
LEARNING_RATE = 5e-6
BATCH_SIZE = 16
LOGGING_STEPS = 10
EVAL_STEPS = 100
SAVE_STEPS = 500

def compute_metrics(pred):
    labels = pred.label_ids
    preds = pred.predictions[0].argmax(-1)

    return {
        "accuracy": accuracy_score(labels, preds),
        "f1": f1_score(labels, preds, average="weighted"),
        "precision": precision_score(labels, preds, average="weighted"),
        "recall": recall_score(labels, preds, average="weighted")
    }

origin_dataset = load_dataset(DATASETS)["train"]

train_dataset = Dataset.from_list(
    [i for i in train_dataset if i["tag"] != "line_ad_original"]
)

test_dataset = Dataset.from_list(
    [i for i in train_dataset if i["tag"] == "line_ad_original"]
)

tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL)
model = DistilBertForSequenceClassification.from_pretrained(BASE_MODEL, num_labels=2)

def tokenize_function(examples):
    return tokenizer(
        examples["text"], truncation=True, padding="max_length", max_length=512
    )

train_dataset = train_dataset.map(tokenize_function, batched=True)
test_dataset = test_dataset.map(tokenize_function, batched=True)

train_dataset.set_format(type="torch", columns=["input_ids", "attention_mask", "label"])
test_dataset.set_format(type="torch", columns=["input_ids", "attention_mask", "label"])

device = torch.device("cuda")
model.to(device)

training_args = TrainingArguments(
    output_dir=OUTPUT_DIR,
    num_train_epochs=NUM_TRAIN_EPOCHS,
    learning_rate=LEARNING_RATE,
    per_device_train_batch_size=BATCH_SIZE,
    per_device_eval_batch_size=1,
    weight_decay=0.01,
    optim="adamw_torch",
    logging_steps=LOGGING_STEPS,
    evaluation_strategy="steps",
    eval_steps=EVAL_STEPS,
    save_steps=SAVE_STEPS,
    bf16=True,
    report_to="wandb",
)

trainer = Trainer(
    model=model,
    args=training_args,
    compute_metrics=compute_metrics,
    train_dataset=train_dataset,
    eval_dataset=test_dataset,
    tokenizer=tokenizer,
)

trainer.train()
```

---

# Model Components Summary

1. **모델 및 토크나이저 초기화**  
2. **데이터셋 로드 및 태그 기반 분할**  
3. **512 토큰 기준 토큰화**  
4. **Trainer 기반 학습 루프 구성**  
5. **Accuracy/F1/Precision/Recall 평가 지표 사용**

---

# Notes

- Base Model 및 tokenizer는 다른 일본어 모델로 교체 가능  
- 데이터셋 변경 시 전처리 코드 수정 필요  
- PDF는 CUDA 환경 사용을 전제로 작성됨

---