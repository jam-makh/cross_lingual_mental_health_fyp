import argparse
import os
import yaml
import torch
import numpy as np

from models import get_model

class Predictor:

    def __init__(self, config_path="config.yaml"):

        with open(config_path, "r", encoding="utf-8") as f:
            self.cfg = yaml.safe_load(f)

        self.device     = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.vec_choice = self.cfg["vectorizer"].lower()

        # ── Load vectorizer ───────────────────────────────────────
        if self.vec_choice == "tfidf":
            from tfidf import TfidfVectorizerWrapper
            path = self.cfg["paths"]["tfidf_vectorizer"]
            if not os.path.exists(path):
                raise FileNotFoundError(
                    f"TF-IDF vectorizer not found at '{path}'. Run main.py first.")
            self.vectorizer = TfidfVectorizerWrapper.load(path)

        elif self.vec_choice == "camembert":
            from camembert import CamembertVectorizer
            self.vectorizer = CamembertVectorizer()

        else:
            raise ValueError(f"Unknown vectorizer '{self.vec_choice}'.")

        # ── Load model ────────────────────────────────────────────
        self.model = get_model(self.cfg).to(self.device)

        arch      = self.cfg["model"]["architecture"]
        vec       = self.cfg["vectorizer"]
        ckpt_path = os.path.join(self.cfg["paths"]["checkpoints"], f"{arch}_{vec}.pt")

        if not os.path.exists(ckpt_path):
            raise FileNotFoundError(
                f"Checkpoint not found at '{ckpt_path}'. Run main.py first.")

        self.model.load_state_dict(torch.load(ckpt_path, map_location=self.device))
        self.model.eval()
        print(f"[Predictor] Loaded from {ckpt_path}")

        # Healthy=0, Unhealthy=1 (LabelEncoder sorts alphabetically)
        self.classes = ["Healthy", "Unhealthy"]

    def predict(self, text):
        if self.vec_choice == "tfidf":
            vec = self.vectorizer.transform_one(text)[0]
        else:
            vec = self.vectorizer.encode(text)

        X = torch.tensor(vec, dtype=torch.float32).unsqueeze(0).to(self.device)

        with torch.no_grad():
            probs = torch.softmax(self.model(X), dim=1).cpu().numpy()[0]

        idx  = int(np.argmax(probs))
        return self.classes[idx], float(probs[idx])


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--text",   type=str, required=True)
    parser.add_argument("--config", type=str,
        default=r"C:\Users\Admin\Documents\FYP\french dataset\Code\MODELS\deep learning lstm\config.yaml")
    args = parser.parse_args()

    p = Predictor(args.config)
    label, conf = p.predict(args.text)

    print(f"\n  Text       : {args.text}")
    print(f"  Prediction : {label}")
    print(f"  Confidence : {conf:.2%}")
