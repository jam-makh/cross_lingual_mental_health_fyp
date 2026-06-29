import os
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from torch.optim import Adam
from torch.optim.lr_scheduler import ReduceLROnPlateau

from metrics import evaluate, plot_confusion_matrix


class Trainer:

    def __init__(self, model, cfg, device):
        self.model = model
        self.cfg = cfg
        self.device = device

        t = cfg["training"]

        self.epochs = t["epochs"]
        self.batch_size = t["batch_size"]
        self.patience = t["patience"]
        self.clip_grad = t.get("clip_grad_norm", 1.0)

        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = Adam(
            model.parameters(),
            lr=t["learning_rate"],
            weight_decay=t.get("weight_decay", 1e-5)
        )

        self.scheduler = ReduceLROnPlateau(
            self.optimizer,
            mode="min",
            factor=t.get("scheduler_factor", 0.5),
            patience=t.get("scheduler_patience", 3)
        )

        self.best_loss = float("inf")
        self.best_weights = None
        self.counter = 0

        self.history = {
            "train_loss": [],
            "val_loss": [],
            "val_acc": []
        }

    def _loader(self, X, y, shuffle=False):
        X = torch.tensor(np.array(X), dtype=torch.float32)
        y = torch.tensor(np.array(y), dtype=torch.long)

        return DataLoader(
            TensorDataset(X, y),
            batch_size=self.batch_size,
            shuffle=shuffle
        )

    def _epoch(self, loader, train=True):
        self.model.train(train)

        total_loss, correct, total = 0.0, 0, 0

        for X_batch, y_batch in loader:
            X_batch = X_batch.to(self.device)
            y_batch = y_batch.to(self.device)

            if train:
                self.optimizer.zero_grad()

            logits = self.model(X_batch)
            loss = self.criterion(logits, y_batch)

            if train:
                loss.backward()
                nn.utils.clip_grad_norm_(self.model.parameters(), self.clip_grad)
                self.optimizer.step()

            preds = logits.argmax(dim=1)

            total_loss += loss.item() * len(y_batch)
            correct += (preds == y_batch).sum().item()
            total += len(y_batch)

        return total_loss / total, correct / total

    def fit(self, X_train, y_train, X_val, y_val, label_encoder=None):

        train_loader = self._loader(X_train, y_train, shuffle=True)
        val_loader = self._loader(X_val, y_val, shuffle=False)

        print("\n================ TRAINING ================\n")

        for epoch in range(1, self.epochs + 1):

            train_loss, train_acc = self._epoch(train_loader, train=True)
            val_loss, val_acc = self._epoch(val_loader, train=False)

            self.history["train_loss"].append(train_loss)
            self.history["val_loss"].append(val_loss)
            self.history["val_acc"].append(val_acc)

            self.scheduler.step(val_loss)

            print(f"Epoch {epoch}/{self.epochs} | "
                  f"Train Loss: {train_loss:.4f} | "
                  f"Val Loss: {val_loss:.4f} | "
                  f"Val Acc: {val_acc:.4f}")

            if val_loss < self.best_loss:
                self.best_loss = val_loss
                self.counter = 0
                self.best_weights = self.model.state_dict()
            else:
                self.counter += 1
                if self.counter >= self.patience:
                    print("[EARLY STOP]")
                    break

        if self.best_weights:
            self.model.load_state_dict(self.best_weights)

    def predict(self, X):
        self.model.eval()

        X = torch.tensor(np.array(X), dtype=torch.float32)
        loader = DataLoader(TensorDataset(X, torch.zeros(len(X))), batch_size=self.batch_size)

        preds = []

        with torch.no_grad():
            for X_batch, _ in loader:
                X_batch = X_batch.to(self.device)
                out = self.model(X_batch).argmax(dim=1)
                preds.extend(out.cpu().numpy())

        return np.array(preds)