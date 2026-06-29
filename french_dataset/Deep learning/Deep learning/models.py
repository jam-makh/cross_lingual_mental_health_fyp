import torch
import torch.nn as nn


# =========================
# LSTM
# =========================
class LSTMClassifier(nn.Module):

    def __init__(self, input_size, hidden_size=256, num_layers=2,
                 num_classes=2, dropout=0.3):
        super().__init__()
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )
        self.dropout = nn.Dropout(dropout)
        self.fc      = nn.Linear(hidden_size, num_classes)

    def forward(self, x):
        x = x.unsqueeze(1)
        out, _ = self.lstm(x)
        out = out[:, -1, :]
        out = self.dropout(out)
        return self.fc(out)


# =========================
# BiLSTM
# =========================
class BiLSTMClassifier(nn.Module):

    def __init__(self, input_size, hidden_size=256,
                 num_layers=2, num_classes=2, dropout=0.3):
        super().__init__()
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )
        self.dropout = nn.Dropout(dropout)
        self.fc      = nn.Linear(hidden_size * 2, num_classes)

    def forward(self, x):
        x = x.unsqueeze(1)
        out, _ = self.lstm(x)
        out = out[:, -1, :]
        out = self.dropout(out)
        return self.fc(out)


# =========================
# CNN + RNN
# =========================
class CNNRNNClassifier(nn.Module):
    """
    For vectorizer input (TF-IDF / CamemBERT):

    Flow:
      (B, input_size)
        → reshape to chunks    (B, num_chunks, chunk_size)
        → Conv1d over chunks   (B, num_filters, num_chunks)
        → global max pool      (B, num_filters)
        → GRU on 3 scales      (B, hidden*2)
        → Linear               (B, num_classes)

    We split the input_size into num_chunks chunks of size chunk_size.
    This gives the CNN a real local structure to detect patterns over,
    then the GRU reads the resulting feature map.
    """

    def __init__(self, input_size, hidden_size=256,
                 num_layers=1, num_classes=2,
                 dropout=0.3, num_filters=128,
                 kernel_size=3, num_chunks=50):
        super().__init__()

        self.num_chunks = num_chunks
        # chunk_size = input_size / num_chunks (e.g. 5000/50 = 100)
        self.chunk_size = input_size // num_chunks

        # 3 parallel CNNs with different kernel sizes (multi-scale)
        self.conv3 = nn.Conv1d(self.chunk_size, num_filters, kernel_size=3, padding=1)
        self.conv5 = nn.Conv1d(self.chunk_size, num_filters, kernel_size=5, padding=2)
        self.conv7 = nn.Conv1d(self.chunk_size, num_filters, kernel_size=7, padding=3)
        self.relu  = nn.ReLU()

        # GRU reads the 3 CNN outputs as a sequence of 3 timesteps
        self.gru = nn.GRU(
            input_size  = num_filters,
            hidden_size = hidden_size,
            num_layers  = num_layers,
            batch_first = True,
            bidirectional = True,
        )

        self.dropout = nn.Dropout(dropout)
        self.fc      = nn.Linear(hidden_size * 2, num_classes)

    def forward(self, x):
        B = x.size(0)

        # reshape: (B, input_size) → (B, num_chunks, chunk_size)
        x = x[:, :self.num_chunks * self.chunk_size]  # trim if not divisible
        x = x.view(B, self.num_chunks, self.chunk_size)

        # permute for Conv1d: (B, chunk_size, num_chunks)
        x = x.permute(0, 2, 1)

        # 3 parallel CNNs → global max pool → (B, num_filters) each
        c3 = self.relu(self.conv3(x)).max(dim=2)[0]   # (B, num_filters)
        c5 = self.relu(self.conv5(x)).max(dim=2)[0]
        c7 = self.relu(self.conv7(x)).max(dim=2)[0]

        # stack as sequence of 3 timesteps: (B, 3, num_filters)
        seq = torch.stack([c3, c5, c7], dim=1)

        # GRU over 3 timesteps
        out, _ = self.gru(seq)           # (B, 3, hidden*2)
        out = out[:, -1, :]             # (B, hidden*2)

        out = self.dropout(out)
        return self.fc(out)


# =========================
# MODEL FACTORY
# =========================
def get_model(cfg):
    m    = cfg["model"]
    arch = m["architecture"].lower()

    common = dict(
        input_size  = m["input_size"],
        hidden_size = m["hidden_size"],
        num_layers  = m["num_layers"],
        num_classes = m["num_classes"],
        dropout     = m["dropout"],
    )

    if arch == "lstm":
        return LSTMClassifier(**common)

    if arch == "bilstm":
        return BiLSTMClassifier(**common)

    if arch in ["cnn_rnn", "cnnrnn", "cnn-rnn"]:
        return CNNRNNClassifier(
            **common,
            num_filters = m.get("num_filters", 128),
            kernel_size = m.get("kernel_size", 3),
            num_chunks  = m.get("num_chunks", 50),
        )

    raise ValueError("Unknown architecture. Use: lstm | bilstm | cnn_rnn")