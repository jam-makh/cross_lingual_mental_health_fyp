import os
import json
import yaml
import torch
import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, f1_score, classification_report

from models import get_model
from trainer import Trainer
from tfidf import TfidfVectorizerWrapper


CONFIG_PATH = "config.yaml"
DATA_PATH = "french_cleaned.csv"

TEXT_COL = "text_nostop"
LABEL_COL = "mental_state"

CHECKPOINT_DIR = "checkpoints"
VECTORIZER_DIR = "vectorizers"
VECTORIZER_PATH = os.path.join(VECTORIZER_DIR, "tfidf_vectorizer.pkl")
EXPORTED_CONFIG_PATH = "config_exported.yaml"


def main():
    os.makedirs(CHECKPOINT_DIR, exist_ok=True)
    os.makedirs(VECTORIZER_DIR, exist_ok=True)
    os.makedirs("export_test_results", exist_ok=True)

    print("=" * 70)
    print("FRENCH LSTM + TF-IDF TRAINING AND EXPORT")
    print("=" * 70)

    with open(CONFIG_PATH, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    # Force the exact setup needed for deployment
    cfg["vectorizer"] = "tfidf"
    cfg["model"]["architecture"] = "LSTM"
    cfg["model"]["input_size"] = 5000
    cfg["model"]["num_classes"] = 2

    cfg["paths"]["data"] = DATA_PATH
    cfg["paths"]["checkpoints"] = CHECKPOINT_DIR
    cfg["paths"]["tfidf_vectorizer"] = VECTORIZER_PATH

    cfg["dataset"]["text_col"] = TEXT_COL
    cfg["dataset"]["label_col"] = LABEL_COL

    print(f"Dataset: {DATA_PATH}")
    print(f"Text column: {TEXT_COL}")
    print(f"Label column: {LABEL_COL}")
    print(f"Architecture: {cfg['model']['architecture']}")
    print(f"Vectorizer: {cfg['vectorizer']}")
    print(f"Input size: {cfg['model']['input_size']}")
    print("=" * 70)

    if not os.path.exists(DATA_PATH):
        raise FileNotFoundError(
            f"Dataset not found: {DATA_PATH}. "
            "Put french_cleaned.csv inside the same DL folder."
        )

    df = pd.read_csv(DATA_PATH)
    print(f"[DATA] Loaded shape: {df.shape}")

    for col in [TEXT_COL, LABEL_COL]:
        if col not in df.columns:
            raise ValueError(
                f"Missing column '{col}'. Available columns: {list(df.columns)}"
            )

    df = df.dropna(subset=[TEXT_COL, LABEL_COL]).reset_index(drop=True)
    df[TEXT_COL] = df[TEXT_COL].astype(str)
    df = df[df[TEXT_COL].str.strip() != ""].reset_index(drop=True)

    print("[DATA] Class distribution:")
    print(df[LABEL_COL].value_counts())

    label_encoder = LabelEncoder()
    y = label_encoder.fit_transform(df[LABEL_COL].values)
    X = df[TEXT_COL].values

    label_mapping = {
        str(label): int(label_encoder.transform([label])[0])
        for label in label_encoder.classes_
    }

    print("[DATA] Label mapping:")
    print(label_mapping)

    X_train, X_val, y_train, y_val = train_test_split(
        X,
        y,
        test_size=cfg["dataset"].get("test_size", 0.2),
        random_state=cfg["dataset"].get("random_state", 42),
        stratify=y,
    )

    print(f"[SPLIT] Train: {len(X_train)}")
    print(f"[SPLIT] Validation/Test: {len(X_val)}")

    print("[TF-IDF] Fitting vectorizer...")
    vectorizer = TfidfVectorizerWrapper(max_features=cfg["model"]["input_size"])

    X_train_vec = vectorizer.fit_transform(X_train)
    X_val_vec = vectorizer.transform(X_val)

    X_train_vec = np.asarray(X_train_vec, dtype=np.float32)
    X_val_vec = np.asarray(X_val_vec, dtype=np.float32)

    print(f"[TF-IDF] Train shape: {X_train_vec.shape}")
    print(f"[TF-IDF] Validation/Test shape: {X_val_vec.shape}")

    print(f"[TF-IDF] Saving fitted vectorizer to: {VECTORIZER_PATH}")
    vectorizer.save(VECTORIZER_PATH)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[MODEL] Device: {device}")

    model = get_model(cfg).to(device)
    trainer = Trainer(model, cfg, device)

    trainer.fit(
        X_train_vec,
        y_train,
        X_val_vec,
        y_val,
        label_encoder=label_encoder,
    )

    y_train_pred = trainer.predict(X_train_vec)
    y_val_pred = trainer.predict(X_val_vec)

    results = {
        "train_accuracy": float(accuracy_score(y_train, y_train_pred)),
        "test_accuracy": float(accuracy_score(y_val, y_val_pred)),
        "test_f1_macro": float(f1_score(y_val, y_val_pred, average="macro")),
        "label_mapping": label_mapping,
        "classes": [str(c) for c in label_encoder.classes_],
        "architecture": cfg["model"]["architecture"],
        "vectorizer": cfg["vectorizer"],
        "input_size": int(cfg["model"]["input_size"]),
    }

    print("[RESULTS]")
    print(json.dumps(results, indent=2, ensure_ascii=False))

    print("[CLASSIFICATION REPORT]")
    print(
        classification_report(
            y_val,
            y_val_pred,
            target_names=[str(c) for c in label_encoder.classes_],
            zero_division=0,
        )
    )

    checkpoint_path = os.path.join(
        CHECKPOINT_DIR,
        f"{cfg['model']['architecture']}_{cfg['vectorizer']}.pt"
    )

    print(f"[MODEL] Saving trained checkpoint to: {checkpoint_path}")
    torch.save(model.state_dict(), checkpoint_path)

    with open(EXPORTED_CONFIG_PATH, "w", encoding="utf-8") as f:
        yaml.safe_dump(cfg, f, allow_unicode=True, sort_keys=False)

    with open("export_test_results/french_export_metrics.json", "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)

    print("=" * 70)
    print("EXPORT COMPLETE")
    print("=" * 70)
    print("Send us these files/folders:")
    print(EXPORTED_CONFIG_PATH)
    print("predictor.py")
    print("models.py")
    print("tfidf.py")
    print(checkpoint_path)
    print(VECTORIZER_PATH)
    print("=" * 70)


if __name__ == "__main__":
    main()
