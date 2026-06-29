import numpy as np
import torch
from transformers import CamembertTokenizer, CamembertModel
from tqdm import tqdm


class CamembertVectorizer:

    def __init__(self, model_name="camembert-base", batch_size=16, max_length=128):
        self.batch_size = batch_size
        self.max_length = max_length
        self.device     = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"[CamemBERT] Loading model on {self.device}...")
        self.tokenizer = CamembertTokenizer.from_pretrained(model_name)
        self.model     = CamembertModel.from_pretrained(model_name)
        self.model.to(self.device)
        self.model.eval()

    def encode(self, text):
        return self.encode_dataset([text], verbose=False)[0]

    def encode_dataset(self, texts, verbose=True):
        if hasattr(texts, "tolist"):
            texts = texts.tolist()
        all_emb  = []
        iterator = range(0, len(texts), self.batch_size)
        if verbose:
            iterator = tqdm(iterator, desc="CamemBERT encoding")
        for i in iterator:
            batch  = texts[i : i + self.batch_size]
            inputs = self.tokenizer(batch, return_tensors="pt", truncation=True,
                                    padding=True, max_length=self.max_length)
            inputs = {k: v.to(self.device) for k, v in inputs.items()}
            with torch.no_grad():
                out = self.model(**inputs)
            all_emb.append(out.last_hidden_state[:, 0, :].cpu().numpy())
        return np.vstack(all_emb)

    @staticmethod
    def save_embeddings(embeddings, path):
        np.save(path, embeddings)
        print(f"[CamemBERT] Saved → {path}")

    @staticmethod
    def load_embeddings(path):
        emb = np.load(path)
        print(f"[CamemBERT] Loaded ← {path}  shape: {emb.shape}")
        return emb
