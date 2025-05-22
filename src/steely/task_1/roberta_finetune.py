import polars as pl
from datasets import Dataset
from transformers import (
    RobertaTokenizer,
    RobertaForSequenceClassification,
    Trainer,
    TrainingArguments,
)
from sklearn.model_selection import train_test_split
import torch

from steely import DATA_TASK_1_DIR, ROOT_DIR

# === 1. Load your dataset ===
train_df = pl.read_ndjson(DATA_TASK_1_DIR / "train.jsonl").select(["text", "label"])
# Shorten the dataset to 500 samples for faster training
# train_df = df.sample(n=500, seed=42)
# Cut each text to 256 characters
# train_df = df.with_columns(pl.col("text").str.slice(0, 512))
val_df = pl.read_ndjson(DATA_TASK_1_DIR / "val.jsonl").select(["text", "label"])

# === 2. Convert to HuggingFace Dataset ===
train_dataset = Dataset.from_polars(train_df)
val_dataset = Dataset.from_polars(val_df)

# === 3. Tokenizer ===
tokenizer = RobertaTokenizer.from_pretrained("roberta-base")


def tokenize(batch):
    return tokenizer(
        batch["text"], padding="max_length", truncation=True, max_length=256
    )


train_dataset = train_dataset.map(tokenize, batched=True)
val_dataset = val_dataset.map(tokenize, batched=True)

# Remove original columns to keep only input_ids, attention_mask, labels
train_dataset.set_format(type="torch", columns=["input_ids", "attention_mask", "label"])
val_dataset.set_format(type="torch", columns=["input_ids", "attention_mask", "label"])

# === 4. Model ===
model = RobertaForSequenceClassification.from_pretrained("roberta-base", num_labels=2)

# === 5. Training Arguments ===
training_args = TrainingArguments(
    output_dir="./results",
    eval_strategy="epoch",
    save_strategy="epoch",
    learning_rate=2e-5,
    per_device_train_batch_size=4,
    per_device_eval_batch_size=8,
    num_train_epochs=3,
    weight_decay=0.01,
    load_best_model_at_end=True,
    metric_for_best_model="accuracy",
    fp16=True,
    report_to="none",
)

# === 6. Evaluation Metrics ===
from sklearn.metrics import accuracy_score, f1_score


def compute_metrics(eval_pred):
    logits, labels = eval_pred
    preds = torch.argmax(torch.tensor(logits), dim=1)
    acc = accuracy_score(labels, preds)
    f1 = f1_score(labels, preds)
    return {"accuracy": acc, "f1": f1}


# === 7. Trainer ===
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=val_dataset,
    compute_metrics=compute_metrics,
)

# === 8. Train ===
trainer.train()

# === 9. Evaluate ===
results = trainer.evaluate()
print(results)

# === 10. Save model ===
model_dir = ROOT_DIR / "roberta-text-detector"
trainer.save_model(model_dir)
tokenizer.save_pretrained(model_dir)
