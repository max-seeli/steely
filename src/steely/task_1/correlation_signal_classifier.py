import json
import os
from typing import List, Tuple, Union

import numpy as np
import polars as pl
from sklearn.metrics import accuracy_score, f1_score
from tqdm import tqdm

from steely import DATA_TASK_1_DIR, ROOT_DIR
from steely.task_1.word_correlations import (
    CorrelationMethod,
    stem_tokenise,
    texts_to_word_correlations,
)


class CorrelationSignalClassifier:
    def __init__(self, word_correlations: dict, n_gram: int = 1):
        self.word_correlations = word_correlations
        self.n_gram = n_gram

        self.thresholds = None

    def train(self, texts: List[str], labels: np.ndarray):
        """Train the model on the given texts and labels."""

        signals = []
        for text in tqdm(texts):
            signals.append(self.get_ngram_signal(text, self.n_gram))

        best_threshold, best_accuracy, best_f1 = self.find_thresholds(signals, labels)
        print(
            f"{self.n_gram}-gram with threshold: {best_threshold:.4f}, Accuracy: {best_accuracy:.4f}, F1: {best_f1:.4f}"
        )

        self.thresholds = best_threshold

    def predict(self, text: Union[str, List[str]]) -> Union[int, List[int]]:
        """Predict the label for the given text."""
        if isinstance(text, str):
            signal = self.get_ngram_signal(text, self.n_gram)
            return 1 if signal > self.thresholds else 0
        elif isinstance(text, list):
            signals = [self.get_ngram_signal(t, self.n_gram) for t in text]
            return [1 if signal > self.thresholds else 0 for signal in signals]
        else:
            raise ValueError("Input must be a string or a list of strings.")

    def get_ngram_signal(self, text: str, n: int) -> float:
        """Get the signal of a text."""
        tokens = stem_tokenise(text)
        ngrams = [" ".join(tokens[i : i + n]) for i in range(len(tokens) - n + 1)]
        return np.sum(
            [
                self.word_correlations[token]
                for token in ngrams
                if token in self.word_correlations
            ]
        ) / len(ngrams)

    def find_thresholds(
        self, signals: List[float], labels: np.ndarray
    ) -> Tuple[float, float]:
        """Find the thresholds for the signals."""
        thresholds = np.linspace(np.min(signals), np.max(signals), 1000)
        best_accuracy = 0
        best_f1 = 0
        best_threshold = 0

        for threshold in thresholds:
            predictions = [1 if signal > threshold else 0 for signal in signals]
            accuracy = accuracy_score(labels, predictions)
            f1 = f1_score(labels, predictions)

            best_f1 = max(f1, best_f1)

            if accuracy > best_accuracy:
                print(
                    f"Threshold: {threshold:.4f}, Accuracy: {accuracy:.4f}, F1: {f1:.4f}"
                )
                best_threshold = threshold
                best_accuracy = max(accuracy, best_accuracy)

        return best_threshold, best_accuracy, best_f1


if __name__ == "__main__":
    from argparse import ArgumentParser

    parser = ArgumentParser(
        description="Run inference on the CorrelationSignalClassifier model."
    )
    parser.add_argument("input_file", type=str, help="Path to the input JSONL file.")
    parser.add_argument(
        "output_dir", type=str, help="Directory to save the output predictions."
    )
    parser.add_argument(
        "--n_gram", type=int, default=1, help="N-gram size for the classifier."
    )
    args = parser.parse_args()

    train_df = pl.read_ndjson(DATA_TASK_1_DIR / "train.jsonl")
    print(f"Using word correlations from {ROOT_DIR / "tmp"}")
    word_correlations = texts_to_word_correlations(
        train_df,
        CorrelationMethod.PEARSON,
        n_gram=3,
        word_correlations_path=ROOT_DIR / "tmp",
        vectorized_texts_path=ROOT_DIR / "tmp",
    )
    clf = CorrelationSignalClassifier(word_correlations, n_gram=args.n_gram)
    clf.train(train_df["text"].to_list(), train_df["label"].to_numpy())

    # ------------------------------------------------------------------
    # Predict and evaluate
    # ------------------------------------------------------------------
    df = pl.read_ndjson(args.input_file)
    ids = df["id"].to_list()
    texts = df["text"].to_list()

    predictions = clf.predict(texts)
    print(f"Predictions: {predictions}")

    # ------------------------------------------------------------------
    # Save the predictions on JSONL input
    # ------------------------------------------------------------------
    out_file = os.path.join(args.output_dir, "predictions.jsonl")

    predictions = [{"id": id, "label": label} for id, label in zip(ids, predictions)]
    with open(out_file, "w") as f:
        for entry in predictions:
            f.write(json.dumps(entry) + "\n")
    print(f"Saved predictions to {out_file}")
