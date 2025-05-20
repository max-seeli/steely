import json
import os
from argparse import ArgumentParser
from enum import Enum
from typing import List, Tuple
from warnings import warn

import numpy as np
import polars as pl
import scipy.sparse as sp
from nltk import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from nltk.tokenize import word_tokenize
from scipy.stats import rankdata
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics import accuracy_score, f1_score
from tqdm import tqdm, trange

from steely import DATA_TASK_1_DIR, ROOT_DIR
from steely.nltk_loader import load_nltk_data
load_nltk_data()

# ------------------------------------------------------------------
# Pre-processing helper
# ------------------------------------------------------------------
stemmer = PorterStemmer()
stop_words = set(stopwords.words("english"))          # O(1) look-ups


def stem_tokenise(text: str) -> list[str]:
    """Lower-case, tokenise, remove punctuation & stop-words, then stem."""
    tokens = [t for t in word_tokenize(text.lower())
              if t.isalpha() and t not in stop_words]
    return [stemmer.stem(t) for t in tokens]


def pearson(X: sp.csr_matrix, y: np.ndarray) -> np.ndarray:
    """Calculate the Pearson correlation between each column of X and y."""
    # Covariance:  E[XY] − E[X]E[Y]
    xy_mean = (X.T @ y) / X.shape[0]       # shape (n_words,)
    p = X.mean(axis=0).A1               # P(word==1)
    m = y.mean()                           # P(label==1)
    cov = xy_mean - p * m

    # Standard deviations
    # clip p to avoid divide by zero
    p = np.clip(p, 1e-10, 1-1e-10)
    m = np.clip(m, 1e-10, 1-1e-10)
    std = np.sqrt(p * (1-p) * m * (1-m))

    return cov / std


def spearman(X: sp.csr_matrix, y: np.ndarray) -> np.ndarray:
    """Calculate the Spearman correlation between each column of X and y."""
    # Rank the data
    X = X.tocsc(copy=False)
    n_samples, n_features = X.shape

    # Rank once for y (average-rank method handles ties correctly)
    y_ranks = rankdata(y, method="average")

    corr = np.empty(n_features, dtype=np.float64)

    for j in trange(n_features):
        # Extract ONE column -> dense 1-D view
        x = X.getcol(j).toarray().ravel()

        x_ranks = rankdata(x, method="average")
        d2 = np.square(x_ranks - y_ranks).sum()
        corr[j] = 1.0 - (6.0 * d2) / (n_samples * (n_samples**2 - 1))

    return corr


def jaccard(X: sp.csr_matrix, y: np.ndarray) -> np.ndarray:
    """Calculate the jaccard correlation between each column of X and y."""
    X = X.tocsc(copy=False)
    n_samples, _ = X.shape

    equalities = X == y[:, np.newaxis]  # shape (n_samples, n_words)

    n_equals = equalities.sum(axis=0).A1  # shape (n_words,)
    return (n_equals / n_samples) * 2 - 1


class CorrelationMethod(Enum):
    PEARSON = "pearson"
    SPEARMAN = "spearman"
    JACCARD = "jaccard"

    def __str__(self):
        return self.value

    def get_correlation_func(self):
        if self == CorrelationMethod.PEARSON:
            return pearson
        elif self == CorrelationMethod.SPEARMAN:
            return spearman
        elif self == CorrelationMethod.JACCARD:
            return jaccard
        else:
            raise ValueError(f"Unknown correlation method: {self}")


def vectorize_texts(texts: List[str], n_gram: int, vectorized_texts_dir: str | None = None) -> Tuple[sp.csr_matrix, np.ndarray]:
    """
    Vectorise the texts using CountVectorizer and save to a file.

    Args:
        texts (List[str]): List of texts to vectorise.
        n_gram (int): N-gram size for the vectorisation.
        vectorized_texts_dir (str | None): Directory to save the vectorised texts. If None, do not save.

    Returns:
        Tuple[sp.csr_matrix, np.ndarray]: Tuple containing the vectorised texts and the tokens.
    """
    vectoriser = CountVectorizer(
        tokenizer=stem_tokenise,   # our custom pipeline
        binary=True,            # presence/absence, not counts
        lowercase=False,            # already lower-cased in tokenizer
        token_pattern=None,         # use our custom tokenizer
        ngram_range=(1, n_gram),  # n-grams
    )
    X = vectoriser.fit_transform(texts)         # shape = (n_docs, n_words)
    tokens = vectoriser.get_feature_names_out()  # shape = (n_words,)

    if vectorized_texts_dir is not None:
        os.makedirs(vectorized_texts_dir, exist_ok=True)
        sp.save_npz(os.path.join(vectorized_texts_dir,
                    "vectorized_texts.npz"), X)
        np.save(os.path.join(vectorized_texts_dir, "tokens.npy"), tokens)
        print(f"Saved vectorized texts to {vectorized_texts_dir}")
    return X, tokens


def calculate_word_correlations(vectorized_texts_tokens: Tuple[sp.csr_matrix, np.ndarray], labels: np.ndarray, correlation_method: CorrelationMethod = CorrelationMethod.PEARSON, word_correlations_dir: str | None = None) -> dict:
    """
    Calculate the word correlations using the specified method.

    Args:
        vectorized_texts_tokens (Tuple[sp.csr_matrix, np.ndarray]): Tuple containing the vectorised texts (n_docs, n_words) and the tokens (n_words,).
        labels (np.ndarray): Labels for the texts of shape (n_docs,).
        CorrelationMethod (CorrelationMethod): Method to use for calculating correlations.
        word_correlations_dir (str | None): Directory to save the word correlations. If None, do not save.

    Returns:
        np.ndarray: Array of word correlations.
    """
    vectorized_texts, tokens = vectorized_texts_tokens
    correlation_func = correlation_method.get_correlation_func()
    correlations = correlation_func(vectorized_texts, labels)

    word_correlations = dict(
        zip(tokens, correlations))

    if word_correlations_dir is not None:
        os.makedirs(word_correlations_dir, exist_ok=True)
        with open(os.path.join(word_correlations_dir, "word_correlations.json"), "w") as f:
            f.write(json.dumps(word_correlations, indent=4))
        print(f"Saved word correlations to {word_correlations_dir}")

    return word_correlations


def texts_to_word_correlations(df: pl.DataFrame, correlation_method: CorrelationMethod, n_gram: int, word_correlations_path: str | None = None, vectorized_texts_path: str | None = None) -> dict:
    """
    Calculate word correlations for the given texts.

    Args:
        df (pl.DataFrame): DataFrame containing the `text` and `label`.
        correlation_method (CorrelationMethod): Method to use for calculating correlations.
        n_gram (int): N-gram size for the vectorisation (only used, when vectorized texts are recomputed).
        word_correlations_path (str | None): Directory to load/save the word correlations. If None, do not save.
        vectorized_texts_path (str | None): Directory to load/save the vectorised texts. If None, do not save.

    Returns:
        dict: Dictionary of word correlations.
    """
    word_correlations_file = os.path.join(
        word_correlations_path, "word_correlations.json") if word_correlations_path is not None else None
    vectorized_texts_file = os.path.join(
        vectorized_texts_path, "vectorized_texts.npz") if vectorized_texts_path is not None else None
    tokens_file = os.path.join(
        vectorized_texts_path, "tokens.npy") if vectorized_texts_path is not None else None

    if word_correlations_file is not None and os.path.exists(word_correlations_file) and vectorized_texts_file is not None and os.path.exists(vectorized_texts_file) and tokens_file is not None and os.path.exists(tokens_file):
        warn("Both word_correlations and vectorized_texts paths are provided. Skipping correlation calculation -> vectorized_texts will not be used.")

    texts = df["text"].to_list()
    labels = df["label"].to_numpy()  # assumed 0/1 labels

    if word_correlations_file is not None and os.path.exists(word_correlations_file):
        with open(word_correlations_file, "r") as f:
            word_correlations = json.load(f)
    else:
        if vectorized_texts_path is not None and os.path.exists(vectorized_texts_file) and os.path.exists(tokens_file):
            vectorized_texts = sp.load_npz(vectorized_texts_file)
            tokens = np.load(tokens_file, allow_pickle=True)
        else:
            print("Vectorizing texts...")
            vectorized_texts, tokens = vectorize_texts(
                texts, n_gram=n_gram, vectorized_texts_dir=vectorized_texts_path)

        print(
            f"Calculating word correlations using {correlation_method.value}...")
        word_correlations = calculate_word_correlations(
            (vectorized_texts, tokens), labels=labels, correlation_method=correlation_method, word_correlations_dir=word_correlations_path)
    return word_correlations


if __name__ == "__main__":
    parser = ArgumentParser(
        description="Calculate word correlations and predict.")

    parser.add_argument(
        "--word-correlations-dir", type=str, default=None,
        help="Path to a directory for precomputed word_correlations.json file (loads if exists/else computes and stores)"
    )
    parser.add_argument(
        "--vectorized-texts-dir", type=str, default=None,
        help="Path to a directory for precomputed vectorized_texts.npz file and tokens.npy file (loads if exists/else computes and stores)"
    )
    parser.add_argument(
        "--correlation-method", type=str, default="pearson",
        choices=[m.value for m in CorrelationMethod],
        help="Correlation method to use (pearson, spearman, jaccard)"
    )
    parser.add_argument(
        "--n-gram", type=int, default=1,
        help="N-gram size for the vectorisation"
    )
    args = parser.parse_args()

    word_correlations_path = args.word_correlations_dir
    vectorized_texts_path = args.vectorized_texts_dir
    correlation_method = CorrelationMethod(args.correlation_method)

    df = pl.read_ndjson(DATA_TASK_1_DIR / "train.jsonl")

    word_correlations = texts_to_word_correlations(
        df,
        correlation_method=correlation_method,
        n_gram=args.n_gram,
        word_correlations_path=word_correlations_path,
        vectorized_texts_path=vectorized_texts_path
    )
    print("Word correlations:")
    print(json.dumps(word_correlations, indent=4))
