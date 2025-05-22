import json

import numpy as np
import polars as pl
from nltk.corpus import stopwords
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from tqdm import tqdm

from steely import DATA_TASK_1_DIR, ROOT_DIR
from steely.nltk_loader import load_nltk_data
from steely.task_1.correlation_signal_classifier import CorrelationSignalClassifier
from steely.task_1.word_correlations import (
    CorrelationMethod,
    texts_to_word_correlations,
)


def extract_features(df, word_correlations):
    stop_words = set(stopwords.words("english"))

    vectorizer = TfidfVectorizer(use_idf=True, smooth_idf=True)
    vectorizer.fit([text for text in df["text"]])
    idf_values = vectorizer.idf_

    features = []
    for text in tqdm(df["text"], total=len(df)):
        # document length
        doc_length = len(text)

        # avg sentence length
        sentences = text.split(".")
        avg_sentence_length = sum(
            len(sentence.split()) for sentence in sentences if sentence.strip()
        ) / len(sentences)

        # avg word length
        words = text.split()
        avg_word_length = sum(len(word) for word in words) / len(words) if words else 0

        # type-token ratio (TTR)
        unique_words = set(words)
        ttr = len(unique_words) / len(words) if words else 0

        # stopword ratio
        stopword_count = sum(1 for word in words if word.lower() in stop_words)
        stopword_ratio = stopword_count / len(words) if words else 0

        # punctuation density
        punctuation_count = sum(1 for char in text if char in ".,;:!?")
        punctuation_density = punctuation_count / len(text) if len(text) > 0 else 0

        # inverse document frequency (IDF)
        avg_idf = (
            np.mean(
                [
                    idf_values[vectorizer.vocabulary_.get(word.lower(), 0)]
                    for word in words
                    if word.lower() in vectorizer.vocabulary_
                ]
            )
            if words
            else 0
        )

        # signal score
        signal_score = CorrelationSignalClassifier.get_ngram_signal(
            text, word_correlations, 1
        )

        features.append(
            {
                "signal_score": signal_score,
                "doc_length": doc_length,
                "avg_sentence_length": avg_sentence_length,
                "avg_word_length": avg_word_length,
                "ttr": ttr,
                "stopword_ratio": stopword_ratio,
                "punctuation_density": punctuation_density,
                "avg_idf": avg_idf,
            }
        )

    return pl.DataFrame(features)


if __name__ == "__main__":
    from argparse import ArgumentParser

    parser = ArgumentParser(
        description="Run inference on the CorrelationSignalClassifier model."
    )
    parser.add_argument("input_file", type=str, help="Path to the input JSONL file.")
    parser.add_argument(
        "output_dir", type=str, help="Directory to save the output predictions."
    )

    args = parser.parse_args()

    load_nltk_data()

    train_df = pl.read_ndjson(DATA_TASK_1_DIR / f"train.jsonl")
    inference_df = pl.read_ndjson(args.input_file)

    word_correlations = texts_to_word_correlations(
        train_df,
        CorrelationMethod.PEARSON,
        n_gram=1,
        word_correlations_path=ROOT_DIR / "tmp",
        vectorized_texts_path=ROOT_DIR / "tmp",
    )

    features_train = extract_features(train_df, word_correlations)
    features_inference = extract_features(inference_df, word_correlations)

    X_train = features_train.to_numpy()
    y_train = train_df["label"].to_numpy()
    X_inference = features_inference.to_numpy()

    rf_model = RandomForestClassifier(
        random_state=777,
        n_estimators=200,
        max_depth=None,
        min_samples_split=5,
        min_samples_leaf=2,
        max_features="log2",
    )
    rf_model.fit(X_train, y_train)

    y_pred_proba = rf_model.predict_proba(X_inference)[:, 1]
    y_pred = y_pred_proba > 0.5

    output_file = f"{args.output_dir}/predictions.jsonl"

    ids = inference_df["id"].to_list()
    with open(output_file, "w") as f:
        for id, prob in zip(ids, y_pred_proba):
            result = {"id": id, "label": round(prob, 4)}
            f.write(json.dumps(result) + "\n")

    print(f"Predictions saved to {output_file}")
