"""
combined_models_tfidf_arabert_3models.py

Runs the 3 PyTorch models from the provided zip:
    1) lstm
    2) bilstm
    3) cnn_rnn

Runs them with ONLY these 2 vectorizers:
    1) TF-IDF
    2) AraBERT embeddings

Output structure:
    combined_models/
        final_combined_results.csv
        TF-IDF/
            lstm/
            bilstm/
            cnn_rnn/
        AraBERT/
            lstm/
            bilstm/
            cnn_rnn/

CSV output style:
    Architecture,Vectoriser,Best Epoch,Best Val F1,Test Accuracy,Test Macro F1,Test Weighted,Precision,Recall

Edit ONLY the USER SETTINGS block before running.
"""

from __future__ import annotations

import json
import random
import re
import warnings
from collections import Counter
from pathlib import Path
from typing import Dict, List, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from scipy import sparse
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_selection import SelectKBest, chi2
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, Dataset

warnings.filterwarnings("ignore")

# =============================================================================
# USER SETTINGS — CHANGE THESE ONLY
# =============================================================================

DATA_PATH = "final_modeling_dataset.csv"
TEXT_COLUMN = "clean_text"          # change if your text column is different
LABEL_COLUMN = "umbrella"           # change if your label column is different

OUTPUT_ROOT = "combined_models"

VECTORIZERS_TO_RUN = ["TF-IDF", "AraBERT"]
ARCHITECTURES_TO_RUN = ["lstm", "bilstm", "cnn_rnn"]

# Keep only these umbrella labels if they exist in your dataset.
# Leave as [] to use all labels found in LABEL_COLUMN.
TARGET_LABELS = ["anxiety_fear", "depression", "ocd_obsessive"]
DISPLAY_LABELS = {
    "anxiety_fear": "Anxiety/Fear",
    "depression": "Depression",
    "ocd_obsessive": "OCD/Obsessive",
}

# Downsampling is required for your plan.
USE_DOWNSAMPLING = True
DOWNSAMPLE_TO_MIN_CLASS = True
DOWNSAMPLE_N_PER_CLASS = None       # ignored when DOWNSAMPLE_TO_MIN_CLASS=True

SEED = 42
TEST_SIZE = 0.15
VAL_SIZE = 0.15

EPOCHS = 100
PATIENCE = 8
BATCH_SIZE = 32
LEARNING_RATE = 1e-4
WEIGHT_DECAY = 1e-4
CLIP_GRAD_NORM = 1.0

# TF-IDF settings
TFIDF_MAX_FEATURES = 70000
TFIDF_SELECTED_FEATURES = 30000
TFIDF_NGRAM_RANGE = (1, 2)
TFIDF_MIN_DF = 2
TFIDF_MAX_DF = 0.95

# AraBERT vectorizer settings. This is used as a frozen feature extractor only.
ARABERT_MODEL_NAME = "aubmindlab/bert-base-arabertv02"
ARABERT_MAX_LENGTH = 128
ARABERT_BATCH_SIZE = 16
ARABERT_POOLING = "mean"            # "mean" or "cls"
CACHE_ARABERT_FEATURES = True

# Shared model settings from the zip-model family
EMBED_DIM = 128
HIDDEN_SIZE = 128
NUM_LAYERS = 1
DROPOUT = 0.35
CNN_FILTERS = 128
CNN_KERNEL_SIZE = 3

# =============================================================================
# REPRODUCIBILITY
# =============================================================================


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = False
    torch.backends.cudnn.benchmark = True


# =============================================================================
# MODELS FROM THE ZIP: LSTM, BiLSTM, CNN-RNN
# =============================================================================


def _init_embedding(embedding: nn.Embedding) -> None:
    nn.init.xavier_uniform_(embedding.weight.data)
    if embedding.padding_idx is not None:
        embedding.weight.data[embedding.padding_idx].zero_()


def _get_last_real_timestep(out: torch.Tensor, lengths: torch.Tensor) -> torch.Tensor:
    batch_size = out.size(0)
    idx = (lengths - 1).clamp(min=0)
    idx = idx.unsqueeze(1).unsqueeze(2).expand(batch_size, 1, out.size(2))
    return out.gather(1, idx).squeeze(1)


class LSTMClassifier(nn.Module):
    def __init__(
        self,
        vocab_size: int = 1,
        embed_dim: int = 128,
        hidden_size: int = 256,
        num_layers: int = 2,
        num_classes: int = 3,
        dropout: float = 0.3,
        use_pretrained_features: bool = True,
        input_dim: int = 768,
    ):
        super().__init__()
        self.use_pretrained_features = use_pretrained_features

        if use_pretrained_features:
            self.proj = nn.Linear(input_dim, hidden_size)
            self.act = nn.ReLU()
            self.drop_in = nn.Dropout(dropout)
            lstm_input = hidden_size
        else:
            self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=0)
            _init_embedding(self.embedding)
            lstm_input = embed_dim

        self.lstm = nn.LSTM(
            input_size=lstm_input,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(hidden_size, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.use_pretrained_features:
            x = self.drop_in(self.act(self.proj(x)))
            x = x.unsqueeze(1)
            out, _ = self.lstm(x)
            out = out.squeeze(1)
        else:
            lengths = (x != 0).sum(dim=1).clamp(min=1)
            emb = self.embedding(x)
            out, _ = self.lstm(emb)
            out = _get_last_real_timestep(out, lengths)
        out = self.dropout(out)
        return self.fc(out)


class BiLSTMClassifier(nn.Module):
    def __init__(
        self,
        vocab_size: int = 1,
        embed_dim: int = 128,
        hidden_size: int = 256,
        num_layers: int = 2,
        num_classes: int = 3,
        dropout: float = 0.3,
        use_pretrained_features: bool = True,
        input_dim: int = 768,
    ):
        super().__init__()
        self.use_pretrained_features = use_pretrained_features

        if use_pretrained_features:
            self.proj = nn.Linear(input_dim, hidden_size)
            self.act = nn.ReLU()
            self.drop_in = nn.Dropout(dropout)
            lstm_input = hidden_size
        else:
            self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=0)
            _init_embedding(self.embedding)
            lstm_input = embed_dim

        self.lstm = nn.LSTM(
            input_size=lstm_input,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(hidden_size * 2, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.use_pretrained_features:
            x = self.drop_in(self.act(self.proj(x)))
            x = x.unsqueeze(1)
            out, _ = self.lstm(x)
            out = out.squeeze(1)
        else:
            lengths = (x != 0).sum(dim=1).clamp(min=1)
            emb = self.embedding(x)
            out, _ = self.lstm(emb)
            out = _get_last_real_timestep(out, lengths)
        out = self.dropout(out)
        return self.fc(out)


class CNNRNNClassifier(nn.Module):
    def __init__(
        self,
        vocab_size: int = 1,
        embed_dim: int = 128,
        hidden_size: int = 256,
        num_layers: int = 1,
        num_classes: int = 3,
        dropout: float = 0.3,
        num_filters: int = 128,
        kernel_size: int = 3,
        use_pretrained_features: bool = True,
        input_dim: int = 768,
    ):
        super().__init__()
        self.use_pretrained_features = use_pretrained_features
        self.kernel_size = kernel_size

        if use_pretrained_features:
            self.proj = nn.Linear(input_dim, hidden_size)
            self.act = nn.ReLU()
            self.drop_in = nn.Dropout(dropout)
            conv_in_channels = hidden_size
        else:
            self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=0)
            _init_embedding(self.embedding)
            conv_in_channels = embed_dim

        self.conv = nn.Conv1d(
            in_channels=conv_in_channels,
            out_channels=num_filters,
            kernel_size=kernel_size,
            padding=kernel_size // 2,
        )
        self.relu = nn.ReLU()
        self.rnn = nn.GRU(
            input_size=num_filters,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(hidden_size * 2, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.use_pretrained_features:
            x = self.drop_in(self.act(self.proj(x)))
            x = x.unsqueeze(1).expand(-1, self.kernel_size, -1)
            x = x.transpose(1, 2)
            x = self.relu(self.conv(x))
            x = x.transpose(1, 2)
            out, _ = self.rnn(x)
            out = out[:, -1, :]
        else:
            lengths = (x != 0).sum(dim=1).clamp(min=1)
            emb = self.embedding(x)
            conv_in = emb.transpose(1, 2)
            conv_out = self.relu(self.conv(conv_in))
            rnn_in = conv_out.transpose(1, 2)
            out, _ = self.rnn(rnn_in)
            out = _get_last_real_timestep(out, lengths)
        out = self.dropout(out)
        return self.fc(out)


def get_model(architecture: str, input_dim: int, num_classes: int) -> nn.Module:
    architecture = architecture.lower().strip()
    common = dict(
        input_dim=input_dim,
        num_classes=num_classes,
        embed_dim=EMBED_DIM,
        hidden_size=HIDDEN_SIZE,
        num_layers=NUM_LAYERS,
        dropout=DROPOUT,
        use_pretrained_features=True,
    )
    if architecture == "lstm":
        return LSTMClassifier(**common)
    if architecture == "bilstm":
        return BiLSTMClassifier(**common)
    if architecture in {"cnn_rnn", "cnnrnn", "cnn-rnn"}:
        return CNNRNNClassifier(num_filters=CNN_FILTERS, kernel_size=CNN_KERNEL_SIZE, **common)
    raise ValueError("Architecture must be one of: lstm | bilstm | cnn_rnn")


# =============================================================================
# DATA + VECTORIZERS
# =============================================================================


def normalize_text_basic(text: str) -> str:
    text = str(text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def find_column(df: pd.DataFrame, preferred: str, alternatives: List[str]) -> str:
    if preferred in df.columns:
        return preferred
    for col in alternatives:
        if col in df.columns:
            print(f"Column '{preferred}' not found. Using detected column '{col}' instead.")
            return col
    raise ValueError(
        f"Column '{preferred}' was not found. Available columns are: {list(df.columns)}"
    )


def load_and_prepare_dataframe() -> pd.DataFrame:
    df = pd.read_csv(DATA_PATH, on_bad_lines="skip")

    text_col = find_column(
        df,
        TEXT_COLUMN,
        ["input_text", "text", "question", "clean_text", "text_nostop", "sentence", "content"],
    )
    label_col = find_column(
        df,
        LABEL_COLUMN,
        ["target", "target_class", "class", "label", "umbrella", "mental_state"],
    )

    df = df[[text_col, label_col]].dropna().copy()
    df.columns = ["text", "label"]
    df["text"] = df["text"].map(normalize_text_basic)
    df["label"] = df["label"].astype(str).str.strip()
    df = df[df["text"] != ""].copy()

    if TARGET_LABELS:
        existing = set(df["label"].unique())
        chosen = [x for x in TARGET_LABELS if x in existing]
        if chosen:
            df = df[df["label"].isin(chosen)].copy()
        else:
            print("TARGET_LABELS were not found exactly. Using all labels in the dataset.")

    df = df.reset_index(drop=True)
    print(f"Loaded rows after class filtering: {len(df):,}")
    print("\nOriginal class distribution:")
    print(df["label"].value_counts())
    return df


def downsample_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    if not USE_DOWNSAMPLING:
        return df.sample(frac=1.0, random_state=SEED).reset_index(drop=True)

    counts = df["label"].value_counts()
    if DOWNSAMPLE_TO_MIN_CLASS:
        n = int(counts.min())
    else:
        if DOWNSAMPLE_N_PER_CLASS is None:
            raise ValueError("Set DOWNSAMPLE_N_PER_CLASS or use DOWNSAMPLE_TO_MIN_CLASS=True")
        n = int(DOWNSAMPLE_N_PER_CLASS)

    parts = []
    for label, group in df.groupby("label"):
        take_n = min(n, len(group))
        parts.append(group.sample(n=take_n, random_state=SEED))
    balanced = pd.concat(parts, axis=0).sample(frac=1.0, random_state=SEED).reset_index(drop=True)

    print("\nBalanced class distribution:")
    print(balanced["label"].value_counts())
    print(f"\nBalanced rows used: {len(balanced):,}")
    return balanced


def make_label_mapping(labels: List[str]) -> Tuple[Dict[str, int], Dict[int, str], List[str]]:
    if TARGET_LABELS and all(x in set(labels) for x in TARGET_LABELS):
        ordered = [x for x in TARGET_LABELS if x in set(labels)]
    else:
        ordered = sorted(set(labels))

    label_to_id = {label: i for i, label in enumerate(ordered)}
    id_to_label = {i: label for label, i in label_to_id.items()}
    display = [DISPLAY_LABELS.get(label, label) for label in ordered]
    return label_to_id, id_to_label, display


def split_data(df: pd.DataFrame):
    label_to_id, id_to_label, display_names = make_label_mapping(df["label"].tolist())
    df["y"] = df["label"].map(label_to_id).astype(int)

    train_val_df, test_df = train_test_split(
        df,
        test_size=TEST_SIZE,
        random_state=SEED,
        stratify=df["y"],
    )
    relative_val = VAL_SIZE / (1.0 - TEST_SIZE)
    train_df, val_df = train_test_split(
        train_val_df,
        test_size=relative_val,
        random_state=SEED,
        stratify=train_val_df["y"],
    )

    print("\nClasses:")
    for i, name in id_to_label.items():
        print(f"{i}: {name} ({display_names[i]})")
    print(f"\nTrain/Validation/Test rows: {len(train_df):,}/{len(val_df):,}/{len(test_df):,}")
    return train_df.reset_index(drop=True), val_df.reset_index(drop=True), test_df.reset_index(drop=True), id_to_label, display_names


class DenseFeatureDataset(Dataset):
    def __init__(self, X, y):
        if sparse.issparse(X):
            X = X.toarray()
        self.X = torch.tensor(np.asarray(X), dtype=torch.float32)
        self.y = torch.tensor(np.asarray(y), dtype=torch.long)

    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]


def build_tfidf_features(train_df, val_df, test_df):
    print("\nBuilding TF-IDF features...")
    vectorizer = TfidfVectorizer(
        max_features=TFIDF_MAX_FEATURES,
        ngram_range=TFIDF_NGRAM_RANGE,
        min_df=TFIDF_MIN_DF,
        max_df=TFIDF_MAX_DF,
        sublinear_tf=True,
        norm="l2",
        dtype=np.float32,
    )

    X_train_raw = vectorizer.fit_transform(train_df["text"])
    X_val_raw = vectorizer.transform(val_df["text"])
    X_test_raw = vectorizer.transform(test_df["text"])

    print(f"Combined TF-IDF features before selection: {X_train_raw.shape[1]:,}")

    k = min(TFIDF_SELECTED_FEATURES, X_train_raw.shape[1])
    selector = SelectKBest(score_func=chi2, k=k)
    X_train = selector.fit_transform(X_train_raw, train_df["y"].values)
    X_val = selector.transform(X_val_raw)
    X_test = selector.transform(X_test_raw)

    print(f"Selected TF-IDF features after chi2: {X_train.shape[1]:,}")
    return X_train, X_val, X_test, X_train.shape[1]


def build_arabert_features(train_df, val_df, test_df, output_root: Path):
    try:
        from transformers import AutoModel, AutoTokenizer
    except Exception as exc:
        raise ImportError(
            "AraBERT vectorizer needs transformers installed. Run: pip install transformers sentencepiece"
        ) from exc

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    cache_dir = output_root / "_cache"
    cache_dir.mkdir(parents=True, exist_ok=True)
    cache_key = f"arabert_{ARABERT_MODEL_NAME.replace('/', '_')}_len{ARABERT_MAX_LENGTH}_{ARABERT_POOLING}_seed{SEED}"
    paths = {
        "train": cache_dir / f"{cache_key}_train.npy",
        "val": cache_dir / f"{cache_key}_val.npy",
        "test": cache_dir / f"{cache_key}_test.npy",
    }

    if CACHE_ARABERT_FEATURES and all(p.exists() for p in paths.values()):
        print("\nLoading cached AraBERT features...")
        X_train = np.load(paths["train"])
        X_val = np.load(paths["val"])
        X_test = np.load(paths["test"])
        return X_train, X_val, X_test, X_train.shape[1]

    print("\nBuilding AraBERT features as frozen vectorizer...")
    tokenizer = AutoTokenizer.from_pretrained(ARABERT_MODEL_NAME)
    model = AutoModel.from_pretrained(ARABERT_MODEL_NAME).to(device)
    model.eval()

    def encode_texts(texts: List[str]) -> np.ndarray:
        all_vecs = []
        for start in range(0, len(texts), ARABERT_BATCH_SIZE):
            batch = texts[start:start + ARABERT_BATCH_SIZE]
            encoded = tokenizer(
                batch,
                padding=True,
                truncation=True,
                max_length=ARABERT_MAX_LENGTH,
                return_tensors="pt",
            )
            encoded = {k: v.to(device) for k, v in encoded.items()}
            with torch.no_grad():
                out = model(**encoded)
                hidden = out.last_hidden_state
                if ARABERT_POOLING == "cls":
                    pooled = hidden[:, 0, :]
                else:
                    mask = encoded["attention_mask"].unsqueeze(-1).float()
                    pooled = (hidden * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1e-9)
            all_vecs.append(pooled.cpu().numpy().astype(np.float32))
        return np.vstack(all_vecs)

    X_train = encode_texts(train_df["text"].tolist())
    X_val = encode_texts(val_df["text"].tolist())
    X_test = encode_texts(test_df["text"].tolist())

    if CACHE_ARABERT_FEATURES:
        np.save(paths["train"], X_train)
        np.save(paths["val"], X_val)
        np.save(paths["test"], X_test)

    print(f"AraBERT feature dimension: {X_train.shape[1]:,}")
    return X_train, X_val, X_test, X_train.shape[1]


# =============================================================================
# TRAINING + EVALUATION
# =============================================================================


def make_loaders(X_train, y_train, X_val, y_val, X_test, y_test):
    train_loader = DataLoader(DenseFeatureDataset(X_train, y_train), batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(DenseFeatureDataset(X_val, y_val), batch_size=BATCH_SIZE, shuffle=False)
    test_loader = DataLoader(DenseFeatureDataset(X_test, y_test), batch_size=BATCH_SIZE, shuffle=False)
    return train_loader, val_loader, test_loader


def run_epoch(model, loader, criterion, device, optimizer=None):
    is_train = optimizer is not None
    model.train() if is_train else model.eval()

    total_loss = 0.0
    y_true, y_pred = [], []

    context = torch.enable_grad() if is_train else torch.no_grad()
    with context:
        for X, y in loader:
            X = X.to(device)
            y = y.to(device)
            if is_train:
                optimizer.zero_grad()
            logits = model(X)
            loss = criterion(logits, y)
            if is_train:
                loss.backward()
                nn.utils.clip_grad_norm_(model.parameters(), CLIP_GRAD_NORM)
                optimizer.step()
            total_loss += loss.item() * X.size(0)
            preds = torch.argmax(logits, dim=1)
            y_true.extend(y.detach().cpu().numpy().tolist())
            y_pred.extend(preds.detach().cpu().numpy().tolist())

    avg_loss = total_loss / len(loader.dataset)
    acc = accuracy_score(y_true, y_pred)
    macro_f1 = f1_score(y_true, y_pred, average="macro", zero_division=0)
    return avg_loss, acc, macro_f1, np.array(y_true), np.array(y_pred)


def save_confusion(y_true, y_pred, display_names, output_dir: Path, title: str, percent: bool = False):
    cm = confusion_matrix(y_true, y_pred, labels=np.arange(len(display_names)))
    if percent:
        row_sums = cm.sum(axis=1, keepdims=True)
        values = np.divide(cm, row_sums, out=np.zeros_like(cm, dtype=float), where=row_sums != 0) * 100
    else:
        values = cm

    plt.figure(figsize=(8, 6))
    plt.imshow(values)
    plt.title(title)
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.xticks(range(len(display_names)), display_names, rotation=30, ha="right")
    plt.yticks(range(len(display_names)), display_names)
    for i in range(len(display_names)):
        for j in range(len(display_names)):
            txt = f"{values[i, j]:.1f}%" if percent else str(int(values[i, j]))
            plt.text(j, i, txt, ha="center", va="center")
    plt.colorbar()
    plt.tight_layout()
    filename = "confusion_matrix_percent.png" if percent else "confusion_matrix.png"
    plt.savefig(output_dir / filename, dpi=300, bbox_inches="tight")
    plt.close()


def save_training_curves(history: Dict[str, List[float]], output_dir: Path, title_prefix: str):
    epochs = np.arange(1, len(history["train_loss"]) + 1)

    plt.figure(figsize=(8, 5))
    plt.plot(epochs, history["train_loss"], label="Training Loss")
    plt.plot(epochs, history["val_loss"], label="Validation Loss")
    plt.title(f"{title_prefix} Training Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_dir / "training_loss.png", dpi=300, bbox_inches="tight")
    plt.close()

    plt.figure(figsize=(8, 5))
    plt.plot(epochs, history["train_acc"], label="Training Accuracy")
    plt.plot(epochs, history["val_acc"], label="Validation Accuracy")
    plt.title(f"{title_prefix} Training Accuracy")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_dir / "training_accuracy.png", dpi=300, bbox_inches="tight")
    plt.close()

    plt.figure(figsize=(8, 5))
    plt.plot(epochs, history["val_macro_f1"], label="Validation Macro F1")
    plt.title(f"{title_prefix} Validation Macro F1")
    plt.xlabel("Epoch")
    plt.ylabel("Macro F1")
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_dir / "validation_macro_f1.png", dpi=300, bbox_inches="tight")
    plt.close()


def train_one_model(
    architecture: str,
    vectorizer_name: str,
    input_dim: int,
    train_loader,
    val_loader,
    test_loader,
    display_names: List[str],
    output_dir: Path,
):
    set_seed(SEED)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    output_dir.mkdir(parents=True, exist_ok=True)

    model = get_model(architecture, input_dim=input_dim, num_classes=len(display_names)).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="max", factor=0.5, patience=3)

    history = {"train_loss": [], "val_loss": [], "train_acc": [], "val_acc": [], "val_macro_f1": []}
    best_val_f1 = -1.0
    best_epoch = 0
    bad_epochs = 0
    best_path = output_dir / "best_model.pt"

    print("\n" + "=" * 70)
    print(f"Training {architecture} with {vectorizer_name}")
    print(f"Input dim: {input_dim:,} | Device: {device} | Epochs: {EPOCHS} | Patience: {PATIENCE}")
    print("=" * 70)

    for epoch in range(1, EPOCHS + 1):
        train_loss, train_acc, _, _, _ = run_epoch(model, train_loader, criterion, device, optimizer)
        val_loss, val_acc, val_f1, _, _ = run_epoch(model, val_loader, criterion, device, optimizer=None)

        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)
        history["train_acc"].append(train_acc)
        history["val_acc"].append(val_acc)
        history["val_macro_f1"].append(val_f1)

        scheduler.step(val_f1)

        if val_f1 > best_val_f1:
            best_val_f1 = val_f1
            best_epoch = epoch
            bad_epochs = 0
            torch.save({"model_state": model.state_dict(), "best_epoch": best_epoch, "best_val_f1": best_val_f1}, best_path)
            status = "best"
        else:
            bad_epochs += 1
            status = f"wait {bad_epochs}/{PATIENCE}"

        print(
            f"Epoch {epoch:03d} | train_loss={train_loss:.4f} | val_loss={val_loss:.4f} | "
            f"train_acc={train_acc:.4f} | val_acc={val_acc:.4f} | val_macro_f1={val_f1:.4f} | {status}"
        )

        if bad_epochs >= PATIENCE:
            print(f"Early stopping at epoch {epoch}. Best epoch: {best_epoch}")
            break

    checkpoint = torch.load(best_path, map_location=device)
    model.load_state_dict(checkpoint["model_state"])
    test_loss, test_acc, test_macro_f1, y_true, y_pred = run_epoch(model, test_loader, criterion, device, optimizer=None)

    report_dict = classification_report(
        y_true,
        y_pred,
        target_names=display_names,
        output_dict=True,
        zero_division=0,
    )
    report_text = classification_report(
        y_true,
        y_pred,
        target_names=display_names,
        zero_division=0,
    )

    test_weighted_f1 = f1_score(y_true, y_pred, average="weighted", zero_division=0)
    test_precision = precision_score(y_true, y_pred, average="macro", zero_division=0)
    test_recall = recall_score(y_true, y_pred, average="macro", zero_division=0)

    pd.DataFrame(report_dict).transpose().to_csv(output_dir / "classification_report.csv")
    with open(output_dir / "classification_report.txt", "w", encoding="utf-8") as f:
        f.write(report_text)

    pd.DataFrame({"y_true": y_true, "y_pred": y_pred}).to_csv(output_dir / "test_predictions.csv", index=False)

    title_prefix = f"{vectorizer_name} {architecture.upper()}"
    save_confusion(y_true, y_pred, display_names, output_dir, f"{title_prefix} Confusion Matrix", percent=False)
    save_confusion(y_true, y_pred, display_names, output_dir, f"{title_prefix} Confusion Matrix (% Within Actual Class)", percent=True)
    save_training_curves(history, output_dir, title_prefix)

    summary = {
        "Architecture": architecture,
        "Vectoriser": vectorizer_name,
        "Best Epoch": best_epoch,
        "Best Val F1": best_val_f1,
        "Test Accuracy": test_acc,
        "Test Macro F1": test_macro_f1,
        "Test Weighted": test_weighted_f1,
        "Precision": test_precision,
        "Recall": test_recall,
        "Test Loss": test_loss,
        "Input Dimension": input_dim,
    }
    with open(output_dir / "run_summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

    print(
        f"DONE {architecture}/{vectorizer_name}: acc={test_acc:.4f}, "
        f"macro_f1={test_macro_f1:.4f}, weighted_f1={test_weighted_f1:.4f}"
    )
    return summary


# =============================================================================
# MAIN
# =============================================================================


def main():
    set_seed(SEED)
    output_root = Path(OUTPUT_ROOT)
    output_root.mkdir(parents=True, exist_ok=True)

    df = load_and_prepare_dataframe()
    df = downsample_dataframe(df)
    train_df, val_df, test_df, id_to_label, display_names = split_data(df)

    y_train = train_df["y"].values
    y_val = val_df["y"].values
    y_test = test_df["y"].values

    all_results = []

    for vectorizer_name in VECTORIZERS_TO_RUN:
        if vectorizer_name.lower() in {"tf-idf", "tfidf", "tf idf"}:
            clean_vectorizer_name = "TF-IDF"
            X_train, X_val, X_test, input_dim = build_tfidf_features(train_df, val_df, test_df)
        elif vectorizer_name.lower() in {"arabert", "ara-bert"}:
            clean_vectorizer_name = "AraBERT"
            X_train, X_val, X_test, input_dim = build_arabert_features(train_df, val_df, test_df, output_root)
        else:
            raise ValueError("Only TF-IDF and AraBERT vectorizers are allowed.")

        train_loader, val_loader, test_loader = make_loaders(X_train, y_train, X_val, y_val, X_test, y_test)

        for arch in ARCHITECTURES_TO_RUN:
            model_dir = output_root / clean_vectorizer_name / arch
            result = train_one_model(
                architecture=arch,
                vectorizer_name=clean_vectorizer_name,
                input_dim=input_dim,
                train_loader=train_loader,
                val_loader=val_loader,
                test_loader=test_loader,
                display_names=display_names,
                output_dir=model_dir,
            )
            all_results.append(result)

    summary_df = pd.DataFrame(all_results)[[
        "Architecture",
        "Vectoriser",
        "Best Epoch",
        "Best Val F1",
        "Test Accuracy",
        "Test Macro F1",
        "Test Weighted",
        "Precision",
        "Recall",
    ]]

    summary_path = output_root / "final_combined_results.csv"
    summary_df.to_csv(summary_path, index=False)

    print("\n" + "=" * 90)
    print("FINAL COMBINED RESULTS")
    print("=" * 90)
    print(summary_df.to_csv(index=False).strip())
    print(f"\nSaved summary to: {summary_path}")


if __name__ == "__main__":
    main()
