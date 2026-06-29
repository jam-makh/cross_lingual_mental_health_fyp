"""
cache_embeddings.py

Usage
-----
    python cache_embeddings.py
"""

import numpy as np
import joblib
from pathlib import Path
from scipy.sparse import save_npz
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from vectorizers import build_tfidf_features, generate_distilbert_embeddings


def cache_embeddings(
    train_texts: list,
    test_texts: list,
    y_train,
    y_test,
    train_texts_raw: list,   # raw for DistilBERT
    test_texts_raw: list,
    cache_dir: str = "cache",
    use_distilbert: bool = True,
) -> None:
    """
    Vectorize train/test texts and save results to disk.

    :param train_texts: Cleaned training texts.
    :type train_texts: list[str]
    :param test_texts: Cleaned test texts.
    :type test_texts: list[str]
    :param y_train: Training labels.
    :type y_train: array-like
    :param y_test: Test labels.
    :type y_test: array-like
    :param train_texts_raw: Raw training texts used for MentalBERT embeddings.
    :type train_texts_raw: list[str]
    :param test_texts_raw: Raw test texts used for MentalBERT embeddings.
    :type test_texts_raw: list[str]
    :param cache_dir: Directory to save all cached files.
    :type cache_dir: str
    :param use_mentalbert: Whether to compute MentalBERT embeddings.
    :type use_mentalbert: bool
    :returns: None
    :rtype: None
    """
    # Create the target cache directory if it does not exist.
    cache_path = Path(cache_dir)
    cache_path.mkdir(parents=True, exist_ok=True)

    # Save labels (shared by all mains)
    joblib.dump(y_train, cache_path / "y_train.pkl")
    joblib.dump(y_test,  cache_path / "y_test.pkl")

    # # TF-IDF (commented out - only using MentalBERT)
    # print("Vectorizing with TF-IDF...")
    # # Build TF-IDF features from cleaned text for both train and test splits.
    # X_train_tfidf, X_test_tfidf, tfidf_vectorizer = build_tfidf_features(
    #     train_texts, test_texts
    # )
    # # sparse matrices use save_npz
    # save_npz(str(cache_path / "X_train_tfidf.npz"), X_train_tfidf)
    # save_npz(str(cache_path / "X_test_tfidf.npz"),  X_test_tfidf)
    # # save the fitted vectorizer so inference can reuse the same vocabulary
    # joblib.dump(tfidf_vectorizer, cache_path / "tfidf_vectorizer.pkl")
    # print(f"  TF-IDF saved → {cache_path}/X_{{train,test}}_tfidf.npz")

    # DistilBERT embeddings are computed from the raw text column only.
    if use_distilbert:
        print("Vectorizing with DistilBERT ...")
        # Generate raw-text sentence embeddings using the DistilBERT transformer.
        # FIX: Combine lists to load the model into memory/GPU exactly ONCE.
        # This prevents loading, deleting, and reloading the model.
        all_raw_texts = list(train_texts_raw) + list(test_texts_raw)
        split_idx = len(train_texts_raw)
        
        print(f"Generating embeddings for {len(all_raw_texts)} total samples...")
        all_embeddings = generate_distilbert_embeddings(all_raw_texts)
        
        # Split the resulting matrix back into train and test subsets
        X_train_db = all_embeddings[:split_idx]
        X_test_db  = all_embeddings[split_idx:]

        np.save(str(cache_path / "X_train_distilbert.npy"), X_train_db)
        np.save(str(cache_path / "X_test_distilbert.npy"),  X_test_db)
        print(f"  DistilBERT saved to   {cache_path}/X_{{train,test}}_distilbert.npy")

    print(f"\nAll embeddings cached in '{cache_dir}/'.")


if __name__ == "__main__":
    # Load the already-cleaned dataset
    df = pd.read_csv(r"C:\Users\MY-PC\OneDrive - Sagesse University\Desktop\LU DS\FYP\mental_health\processed_data_with_features.csv",
                    on_bad_lines='skip')
    df["cleaned_text"] = df["cleaned_text"].fillna("").astype(str)
    df["text"] = df["text"].fillna("").astype(str)

    # Drop rows where cleaned_text is empty after fillna
    df = df[df["cleaned_text"].str.strip() != ""].reset_index(drop=True)


    # Split the dataframe
    train_df, test_df = train_test_split(
        df, test_size=0.2, stratify=df["status"], random_state=42
    )

    train_texts = train_df["cleaned_text"].tolist()
    test_texts = test_df["cleaned_text"].tolist()
    train_texts_raw = train_df["text"].tolist()
    test_texts_raw = test_df["text"].tolist()
    y_train = train_df["status"].tolist()
    y_test = test_df["status"].tolist()

    cache_embeddings(
        train_texts=train_texts,
        test_texts=test_texts,
        y_train=y_train,
        y_test=y_test,
        train_texts_raw=train_texts_raw,
        test_texts_raw=test_texts_raw,
        cache_dir="cache",
        use_distilbert=True,
    )
