from steely import DATA_TASK_1_DIR, ROOT_DIR

import polars as pl
import numpy as np
from tqdm import tqdm
import json

from nltk import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from nltk.tokenize import word_tokenize
import nltk

from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score
from sklearn.feature_extraction.text import TfidfVectorizer
from datetime import datetime

nltk.download("punkt")
nltk.download('punkt_tab')
nltk.download("stopwords")
nltk.download("wordnet")
nltk.download("omw-1.4")

def stem_tokenise(text: str) -> list[str]:
    """Lower-case, tokenise, remove punctuation & stop-words, then stem."""
    tokens = [t for t in word_tokenize(text.lower())
              if t.isalpha()]  # and t not in stop_words]
    return [stemmer.stem(t) for t in tokens]

def get_signal(text: str, word_correlations) -> float:
    """Get the signal of a text."""
    tokens = stem_tokenise(text)
    return np.sum([word_correlations[token] for token in tokens if token in word_correlations]) / len(tokens)

def load_word_correlations(word_correlations_file):
    with open(word_correlations_file, "r") as f:
        word_correlations = json.load(f)
    return word_correlations

def extract_features_with_correlation_scores(dataframe, scores):
    stop_words = set(stopwords.words('english'))
    
    vectorizer = TfidfVectorizer(use_idf=True, smooth_idf=True)
    vectorizer.fit([(doc[1]) for doc in dataframe.iter_rows()])
    idf_values = vectorizer.idf_

    features = []
    for doc in tqdm(dataframe.iter_rows(), total=len(dataframe)):
        text = doc[1]
        label = doc[3]

        # document length
        doc_length = len(text)

        # avg sentence length
        sentences = text.split('.')
        avg_sentence_length = sum(len(sentence.split()) for sentence in sentences if sentence.strip()) / len(sentences)

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
        punctuation_count = sum(1 for char in text if char in '.,;:!?')
        punctuation_density = punctuation_count / len(text) if len(text) > 0 else 0
        
        # inverse document frequency (IDF)
        avg_idf = np.mean([idf_values[vectorizer.vocabulary_.get(word.lower(), 0)] for word in words if word.lower() in vectorizer.vocabulary_]) if words else 0
        
        # signal score
        signal_score = get_signal(text, scores)

        features.append({
            'signal_score': signal_score,
            'doc_length': doc_length,
            'avg_sentence_length': avg_sentence_length,
            'avg_word_length': avg_word_length,
            'ttr': ttr,
            'stopword_ratio': stopword_ratio,
            'punctuation_density': punctuation_density,
            'avg_idf': avg_idf,
            'label': label
        })

    return pl.DataFrame(features)

if __name__ == "__main__":
    
    train_df = pl.read_ndjson(DATA_TASK_1_DIR / f"train.jsonl")
    val_df = pl.read_ndjson(DATA_TASK_1_DIR / f"val.jsonl")
    labels = {
        0: "Human",
        1: "AI"
    }

    stemmer = PorterStemmer()
    stop_words = set(stopwords.words("english"))

    word_correlations = load_word_correlations(ROOT_DIR / "tmp" / f"word_correlations.json")

    features_train = extract_features_with_correlation_scores(train_df, word_correlations)
    features_val = extract_features_with_correlation_scores(val_df, word_correlations)

    X_train = features_train.drop('label').to_numpy()
    y_train = features_train['label'].to_numpy()
    X_val = features_val.drop('label').to_numpy()
    y_val = features_val['label'].to_numpy()

    rf_model = RandomForestClassifier(random_state=777,
                                    n_estimators=100,
                                    max_depth=None,
                                    max_features='sqrt',
                                    min_samples_split=10,
                                    min_samples_leaf=1)

    rf_model.fit(X_train, y_train)

    y_pred_proba = rf_model.predict_proba(X_val)[:, 1]
    y_pred = y_pred_proba > 0.5

    print("Validation Accuracy:", accuracy_score(y_val, y_pred))
    print("\nClassification Report:\n", classification_report(y_val, y_pred))
    
    current_date = datetime.now().strftime("%Y-%m-%d")
    output_file = f"results/steely_{current_date}.jsonl"
    
    with open(output_file, "w") as f:
        for id, prob in enumerate(y_pred_proba):
            result = {"id": int(id), "label": round(prob, 4)}
            f.write(json.dumps(result) + "\n")

    print(f"Predictions saved to {output_file}")