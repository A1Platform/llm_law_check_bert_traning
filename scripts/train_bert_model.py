from transformers import AutoTokenizer, DistilBertForSequenceClassification
import torch
from transformers import Trainer, TrainingArguments
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from datasets import Dataset, load_dataset

# 설정
BASE_MODEL = "line-corporation/line-distilbert-base-japanese"
DATASETS = "A1PerformaceFactory/distilbert_med_line" # 조직 권한 및 개인 HF 로그인(or 토큰) 필요
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

    f1 = f1_score(labels, preds, average="weighted")
    acc = accuracy_score(labels, preds)
    precision = precision_score(labels, preds, average="weighted")
    recall = recall_score(labels, preds, average="weighted")
    return {"accuracy": acc, "f1": f1, "precision": precision, "recall": recall}


# 데이터셋 로드
origin_dataset = load_dataset(DATASETS)["train"]
train_dataset = []
train_dataset = Dataset.from_list(
    [i for i in train_dataset if i["tag"] != "line_ad_original"]
)
test_dataset = Dataset.from_list(
    [i for i in train_dataset if i["tag"] == "line_ad_original"]
)


# BERT 토크나이저 및 모델 초기화
tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL)
model = DistilBertForSequenceClassification.from_pretrained(BASE_MODEL, num_labels=2)


# 토큰화 함수 정의
def tokenize_function(examples):
    return tokenizer(
        examples["text"], truncation=True, padding="max_length", max_length=512
    )


# 데이터셋에 토큰화 적용
train_dataset = train_dataset.map(tokenize_function, batched=True)
test_dataset = test_dataset.map(tokenize_function, batched=True)

# 데이터셋 포맷 설정
train_dataset.set_format(type="torch", columns=["input_ids", "attention_mask", "label"])
test_dataset.set_format(type="torch", columns=["input_ids", "attention_mask", "label"])


# 모델 학습
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
    disable_tqdm=False,
    logging_steps=LOGGING_STEPS,
    evaluation_strategy="steps",
    eval_steps=EVAL_STEPS,
    save_steps=SAVE_STEPS,
    bf16=True,
    log_level="error",
    report_to="wandb",  # w&b 로그인 필요, 없을시 None
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