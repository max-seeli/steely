import polars as pl
import torch
from transformers import RobertaTokenizer, RobertaForSequenceClassification
from torch.utils.data import DataLoader
from datasets import Dataset
from tqdm import tqdm
import json
import os

from steely import DATA_TASK_2_DIR, ROOT_DIR

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

# === Load input data ===
df = pl.read_ndjson(INPUT_FILE).select(["text"])
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
dataset.set_format(type="torch", columns=["input_ids", "attention_mask", "id"])

dataloader = DataLoader(dataset, batch_size=BATCH_SIZE)

# === Inference ===
results = []
with torch.no_grad():
    for batch in tqdm(dataloader):
        input_ids = batch["input_ids"]
        attention_mask = batch["attention_mask"]
        ids = batch["id"]

        outputs = model(input_ids=input_ids, attention_mask=attention_mask)
        probs = torch.softmax(outputs.logits, dim=1)[:, 1].tolist()  # probability for label 1

        for id_, prob in zip(ids, probs):
            results.append({"id": id_, "label": round(prob, 4)})

# === Save to JSONL ===
with open(OUTPUT_FILE, "w") as f:
    for entry in results:
        entry = {
            "id": entry["id"].item(),
            "label": entry["label"],
        }
        f.write(json.dumps(entry) + "\n")

print(f"Predictions saved to: {OUTPUT_FILE}")
