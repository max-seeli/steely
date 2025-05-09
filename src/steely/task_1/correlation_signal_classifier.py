import json
from sklearn.feature_extraction.text import CountVectorizer
from nltk import word_tokenize
import os
import numpy as np
import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer

import polars as pl

from steely import DATA_TASK_1_DIR, ROOT_DIR

nltk.download("punkt")
nltk.download('punkt_tab')
nltk.download("stopwords")
nltk.download("wordnet")
nltk.download("omw-1.4")


# ------------------------------------------------------------------
# 1. Pre-processing helper
# ------------------------------------------------------------------
stemmer = PorterStemmer()
stop_words = set(stopwords.words("english"))          # O(1) look-ups


def stem_tokenise(text: str) -> list[str]:
    """Lower-case, tokenise, remove punctuation & stop-words, then stem."""
    tokens = [t for t in word_tokenize(text.lower())
              if t.isalpha()]  # and t not in stop_words]
    return [stemmer.stem(t) for t in tokens]


def calculate_correlations():
    # ------------------------------------------------------------------
    # 2. Load data
    # ------------------------------------------------------------------
    df = pl.read_ndjson(DATA_TASK_1_DIR / "train.jsonl")
    texts = df["text"].to_list()
    labels = df["label"].to_numpy()
    n_docs = labels.size
    print(f"Loaded {n_docs} documents")

    # ------------------------------------------------------------------
    # 3. Document-term matrix (binary of shape: n_docs × n_words)
    # ------------------------------------------------------------------
    print("Vectorising...")
    vectoriser = CountVectorizer(
        tokenizer=stem_tokenise,   # our custom pipeline
        binary=True,            # presence/absence, not counts
        lowercase=False            # already lower-cased in tokenizer
    )
    X = vectoriser.fit_transform(texts)         # shape = (n_docs, n_words)
    print(f"Vectorised to {X.shape[0]} documents and {X.shape[1]} words")

    # ------------------------------------------------------------------
    # 4. Vectorised Pearson correlation for every word
    # ------------------------------------------------------------------
    print("Calculating correlations...")
    # Means
    p = X.mean(axis=0).A1          # P(word present)
    m = labels.mean()              # P(label==1)

    # Covariance:  𝑬[XY] − 𝑬[X]𝑬[Y]
    xy_mean = (X.T @ labels) / n_docs       # shape (n_words,)
    cov = xy_mean - p * m

    # Standard deviations
    # clip p to avoid divide by zero
    p = np.clip(p, 1e-10, 1-1e-10)
    m = np.clip(m, 1e-10, 1-1e-10)
    std = np.sqrt(p * (1-p) * m * (1-m))

    correlations = cov / std                # shape (n_words,)
    print("Number of nans:", np.isnan(correlations).sum())
    # ------------------------------------------------------------------
    # 5. Create lookup table for words
    # ------------------------------------------------------------------
    word_correlations = dict(
        zip(vectoriser.get_feature_names_out(), correlations))
    return word_correlations


if __name__ == "__main__":

    if os.path.exists(ROOT_DIR / "tmp" / "word_correlations.json"):
        with open(ROOT_DIR / "tmp" / "word_correlations.json", "r") as f:
            word_correlations = json.load(f)
    else:
        word_correlations = calculate_correlations()

        # ------------------------------------------------------------------
        # 6. Save the word correlations to a file
        # ------------------------------------------------------------------
        os.makedirs(ROOT_DIR / "tmp", exist_ok=True)
        with open(ROOT_DIR / "tmp" / "word_correlations.json", "w") as f:
            f.write(json.dumps(word_correlations, indent=4))

    from tqdm import tqdm
    from sklearn.metrics import accuracy_score, f1_score


    def get_signal(text: str) -> float:
        """Get the signal of a text."""
        tokens = stem_tokenise(text)
        return np.sum([word_correlations[token] for token in tokens if token in word_correlations]) / len(tokens)

    


    # ------------------------------------------------------------------
    # 7. Find the best threshold
    # ------------------------------------------------------------------
    df = pl.read_ndjson(DATA_TASK_1_DIR / "train.jsonl")
    ids = df["id"].to_list()
    texts = df["text"].to_list()
    labels = df["label"].to_numpy()  # assumed 0/1 labels

    signals = []
    for i, text, label in tqdm(zip(ids, texts, labels), total=len(ids)):
        signals.append(get_signal(text))

    sig_min, sig_max = np.min(signals), np.max(signals)
    print(f"Signal min: {sig_min:.4f}, max: {sig_max:.4f}")
    thresholds = np.linspace(sig_min, sig_max, 1000)
    best_accuracy = 0
    best_f1 = 0
    best_threshold = 0

    # Evaluate the predictions
    for threshold in thresholds:
        predictions = [1 if signal > threshold else 0 for signal in signals]
        accuracy = accuracy_score(labels, predictions)
        f1 = f1_score(labels, predictions)

        if accuracy > best_accuracy or f1 > best_f1:
            print(f"Threshold: {threshold:.4f}, Accuracy: {accuracy:.4f}, F1: {f1:.4f}")
            best_f1 = max(f1, best_f1)
            best_accuracy = max(accuracy, best_accuracy)
            best_threshold = threshold

    print(f"Best threshold: {best_threshold:.4f}, Accuracy: {best_accuracy:.4f}, F1: {best_f1:.4f}")

    # ------------------------------------------------------------------
    # 8. Evaluate the predictions
    # ------------------------------------------------------------------
    # Load the validation data
    df = pl.read_ndjson(DATA_TASK_1_DIR / "val.jsonl")
    ids = df["id"].to_list()
    texts = df["text"].to_list()
    labels = df["label"].to_numpy()  # assumed 0/1 labels

    signals = []
    for i, text, label in tqdm(zip(ids, texts, labels), total=len(ids)):
        signals.append(get_signal(text))

    predictions = [1 if signal > best_threshold else 0 for signal in signals]
    accuracy = accuracy_score(labels, predictions)
    f1 = f1_score(labels, predictions)
    print(f"Validation accuracy: {accuracy:.4f}, F1: {f1:.4f}")
