import polars as pl
import torch
from transformers import RobertaTokenizer, RobertaForSequenceClassification
from torch.utils.data import DataLoader
from datasets import Dataset
from tqdm import tqdm
import json
import os

from steely import DATA_TASK_2_DIR, ROOT_DIR
from sklearn.metrics import accuracy_score, f1_score, recall_score, confusion_matrix

# === Config ===
MODEL_DIR = ROOT_DIR / "roberta-text-detector"
INPUT_FILE = DATA_TASK_2_DIR / "dev.jsonl"
OUTPUT_DIR = ROOT_DIR / "results" / "inference"
os.makedirs(OUTPUT_DIR, exist_ok=True)
OUTPUT_FILE = os.path.join(OUTPUT_DIR, "predictions.jsonl")
BATCH_SIZE = 16

# === Load model & tokenizer ===
tokenizer = RobertaTokenizer.from_pretrained("roberta-base")
model = RobertaForSequenceClassification.from_pretrained(MODEL_DIR)
model.eval()

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")
model = model.to(device)

# === Load input data ===
df = pl.read_ndjson(INPUT_FILE).select(["text", "label"])
# Add an id column that enumerates everything
df = df.with_columns(pl.Series("id", range(len(df))))
# Shorten the dataset to 500 samples for faster inference
df = df.sample(n=500, seed=42)
# Cut each text to 256 characters
df = df.with_columns(pl.col("text").str.slice(0, 256))
dataset = Dataset.from_polars(df)

# === Tokenize ===
def tokenize(batch):
    return tokenizer(batch["text"], truncation=True, padding=True)

dataset = dataset.map(tokenize, batched=True)
dataset.set_format(type="torch", columns=["input_ids", "attention_mask", "id", "label"])
dataloader = DataLoader(dataset, batch_size=BATCH_SIZE)

# === Inference ===
true_labels = []
predicted_labels = []
results = []

with torch.no_grad():
    for batch in tqdm(dataloader):
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        ids = batch["id"]
        labels = batch["label"]

        outputs = model(input_ids=input_ids, attention_mask=attention_mask)
        probs = torch.softmax(outputs.logits, dim=1)[:, 1].tolist()
        preds = (torch.tensor(probs) > 0.5).int().tolist()

        true_labels.extend(labels.tolist())
        predicted_labels.extend(preds)

        for id_, prob in zip(ids, probs):
            results.append({"id": id_, "label": round(prob, 4)})

# === Metrics ===
accuracy = accuracy_score(true_labels, predicted_labels)
macro_f1 = f1_score(true_labels, predicted_labels, average="macro")
macro_recall = recall_score(true_labels, predicted_labels, average="macro")
conf_matrix = confusion_matrix(true_labels, predicted_labels)

print(f"Accuracy: {accuracy:.4f}")
print(f"Macro F1: {macro_f1:.4f}")
print(f"Macro Recall: {macro_recall:.4f}")
print("Confusion Matrix:")
print(conf_matrix)