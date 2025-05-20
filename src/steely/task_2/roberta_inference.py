import polars as pl
import torch
from transformers import RobertaTokenizer, RobertaForSequenceClassification
from torch.utils.data import DataLoader
from datasets import Dataset
from tqdm import tqdm
import os
import json
from datetime import datetime

from steely import DATA_TASK_2_DIR, ROOT_DIR
from sklearn.metrics import accuracy_score, f1_score, recall_score, confusion_matrix

# === Config ===
MODEL_DIR = ROOT_DIR / "roberta-text-detector-task2"
INPUT_FILE = DATA_TASK_2_DIR / "test.jsonl"
OUTPUT_DIR = ROOT_DIR / "results" / "inference"
os.makedirs(OUTPUT_DIR, exist_ok=True)
current_date = datetime.now().strftime("%Y-%m-%d")
OUTPUT_FILE = os.path.join(OUTPUT_DIR, f"steely_{current_date}.jsonl")
BATCH_SIZE = 16

# === Load model & tokenizer ===
tokenizer = RobertaTokenizer.from_pretrained(MODEL_DIR)
model = RobertaForSequenceClassification.from_pretrained(MODEL_DIR)
model.eval()

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")
model = model.to(device)

# === Load input data ===
df = pl.read_ndjson(INPUT_FILE).select(["text", "id"])
# Shorten the dataset to 500 samples for faster inference
df = df.sample(n=500, seed=42)
# Cut each text to 256 characters
df = df.with_columns(pl.col("text").str.slice(0, 256))
dataset = Dataset.from_polars(df)

# === Tokenize ===
def tokenize(batch):
    return tokenizer(batch["text"], truncation=True, padding=True)

dataset = dataset.map(tokenize, batched=True, num_proc=4, batch_size=5000)
dataset.set_format(type="torch", columns=["input_ids", "attention_mask", "id"])
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

        outputs = model(input_ids=input_ids, attention_mask=attention_mask)
        probs = torch.softmax(outputs.logits, dim=1).tolist()
        preds = torch.argmax(torch.tensor(probs), dim=1).tolist()

        predicted_labels.extend(preds)

        for id_, pred in zip(ids, preds):
            results.append({"id": int(id_), "label": int(pred)})

with open(OUTPUT_FILE, "w") as f:
    for result in results:
        f.write(json.dumps(result) + "\n")

print(f"Results saved to {OUTPUT_FILE}")
