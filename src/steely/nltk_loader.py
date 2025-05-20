import os
import nltk
from steely import DATA_TASK_1_DIR, ROOT_DIR

def load_nltk_data():
    # Define a local directory to store the NLTK data
    NLTK_DATA_DIR = os.path.join(ROOT_DIR, "nltk_data")
    os.makedirs(NLTK_DATA_DIR, exist_ok=True)
    nltk.data.path.append(NLTK_DATA_DIR)

    # Download the required NLTK data if not already present
    required_packages = ["stopwords", "punkt_tab"]
    for package in required_packages:
        if os.path.exists(os.path.join(NLTK_DATA_DIR, "corpora", package)) or os.path.exists(os.path.join(NLTK_DATA_DIR, "tokenizers", package)):
            print(f"NLTK data package '{package}' already exists.")
        else:
            print(f"Downloading NLTK data package '{package}'...")
            nltk.download(package, download_dir=NLTK_DATA_DIR)

if __name__ == "__main__":
    load_nltk_data()