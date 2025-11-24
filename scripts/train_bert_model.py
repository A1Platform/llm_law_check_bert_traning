from transformers import AutoTokenizer, DistilBertForSequenceClassification
import torch
from transformers import Trainer, TrainingArguments
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from datasets import Dataset, load_dataset
from transformers import DataCollatorWithPadding
import wandb

def is_wandb_logged_in():
    try:
        return wandb.api.api_key is not None
    except:
        return False

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

# setup device
device = torch.device("cpu")
if torch.backends.mps.is_available():
    torch.mps.empty_cache()
    device = torch.device("mps")
    print("⚙️ MPS available")
elif torch.cuda.is_available():
    device = torch.device("cuda")
    print("⚙️ CUDA available")

# 데이터셋 로드
origin_dataset = load_dataset(DATASETS)["train"]

train_dataset = Dataset.from_list(
    [i for i in origin_dataset if i["tag"] != "line_ad_original"]
)

test_dataset = Dataset.from_list(
    [i for i in origin_dataset if i["tag"] == "line_ad_original"]
)


# BERT 토크나이저 및 모델 초기화
tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL)
data_collator = DataCollatorWithPadding(tokenizer=tokenizer)
model = DistilBertForSequenceClassification.from_pretrained(BASE_MODEL, num_labels=2)


# 토큰화 함수 정의
def tokenize_function(examples):
    return tokenizer(examples["text"], truncation=True)


# 데이터셋에 토큰화 적용
train_dataset = train_dataset.map(tokenize_function, batched=True)
test_dataset = test_dataset.map(tokenize_function, batched=True)

# 데이터셋 포맷 설정
train_dataset.set_format(type="torch", columns=["input_ids", "attention_mask", "label"])
test_dataset.set_format(type="torch", columns=["input_ids", "attention_mask", "label"])


# 모델 학습
model.to(device)

bf16 = True if torch.cuda.is_available() else False
# fp16 = True if torch.cuda.is_available() else False

use_wandb = is_wandb_logged_in()

report_to = "wandb" if use_wandb else "None"
print(f"📡 W&B logging: {report_to}")

training_args = TrainingArguments(
    output_dir=OUTPUT_DIR,
    num_train_epochs=NUM_TRAIN_EPOCHS,
    learning_rate=LEARNING_RATE,
    per_device_train_batch_size=BATCH_SIZE,
    per_device_eval_batch_size=BATCH_SIZE, # better to use equal to or greater than per_device_train_batch_size
    weight_decay=0.01,
    optim="adamw_torch",
    disable_tqdm=False,
    logging_steps=LOGGING_STEPS,
    evaluation_strategy="steps",
    eval_steps=EVAL_STEPS,
    save_steps=SAVE_STEPS,
    save_total_limit=3,              # 체크포인트 개수 제한 (선택)
    load_best_model_at_end=True,     # 마지막에 best model 로드 (선택)
    metric_for_best_model="f1",      # 기준 메트릭 (선택)
    greater_is_better=True,
    bf16=bf16,
    bf16=bf16,
    log_level="error",
    report_to=report_to,
)

trainer = Trainer(
    model=model,
    args=training_args,
    compute_metrics=compute_metrics,
    train_dataset=train_dataset,
    eval_dataset=test_dataset,
    tokenizer=tokenizer,
    data_collator=data_collator # this is to better memory managment
)

trainer.train()