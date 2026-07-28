# cross_lingual_mental_health_fyp
# Cross-Lingual Mental Health Detection (FYP)

A deep learning framework for **Cross-Lingual Mental Health Detection & Text Classification** across multilingual datasets (English, Arabic, French). Developed as a Final Year Project (FYP), this project investigates transfer learning, multilingual representations, and fine-tuning transformer architectures to identify early signs of mental health distress in text across diverse linguistic and cultural contexts.

---

## Supervision & Acknowledgments

This Final Year Project was developed by Joseph Am-Makhlouf, Juliette Elias Daher and Jean Khalil under the **Technical Supervision of Mr. Elie Dina**. Special thanks and recognition are extended to Mr. Elie Dina for technical guidance, methodology reviews, and architectural insights throughout the design and evaluation phases.

---

## Project Overview

Mental health text classification models often perform well in resource-rich languages like English but suffer severe performance drops when deployed across low-resource or dialect-heavy languages. This repository addresses the cross-lingual transfer gap by leveraging specialized 4 different text embeddings and nine different models.

### Key Features

* **Multilingual Dataset Pipeline:** Automated extraction, cleaning, and preprocessing for English, Arabic, and French textual datasets.
* **Text Embeddings:** As part of this study, TF-IDF was used as an embedding for all languages. A pretained model was used as well for each language: DistilBERT for English, CamemBERT for French and AraBERT for Arabic. 
* **Transformer Architectures:** Fine-tuning and evaluating ML, DL and LLM models for multi-class and binary mental health classification tasks.
* **Cross-Lingual Zero-Shot:** Evaluating how models trained on high-resource language data perform when tested on unseen target languages.
* **Comprehensive Metrics:** Detailed reporting on Macro F1-score as a main metric, in addition to Accuracy, Precision, Recall, and Confusion Matrices across embeddings, models and languages.

---

## 📁 Repository Structure

```text
cross_lingual_mental_health_fyp/
├── english
│   ├── raw/                  # Raw text data across target languages
│   └── processed/            # Cleaned, tokenized, and aligned datasets
├── notebooks/                # Exploratory Data Analysis (EDA) & experimentation
├── src/
│   ├── preprocessing/        # Language normalization, tokenization, & cleaning
│   ├── models/               # Model initialization, fine-tuning scripts, & evaluation
│   ├── utils/                # Helper functions, metrics calculation, and logger
│   └── pipelines/            # End-to-end training and inference execution
├── configs/                  # Hyperparameter configurations (YAML/JSON)
├── requirements.txt          # Python dependencies
├── setup.py                  # Project package installation
└── README.md                 # Project documentation

```

---

## 🚀 Getting Started

### Prerequisites

* Python **3.9+**
* CUDA-compatible GPU (recommended for training transformer models)

### Installation

1. **Clone the repository:**
```bash
git clone https://github.com/jam-makh/cross_lingual_mental_health_fyp.git
cd cross_lingual_mental_health_fyp

```


2. **Set up a virtual environment:**
```bash
python -m venv venv
# On Windows:
venv\Scripts\activate
# On macOS/Linux:
source venv/bin/activate

```


3. **Install dependencies for each language:**
```bash
pip install --upgrade pip
pip install -r requirements.txt

```



---

## Workflow & Usage

### 1. Data Preprocessing

Preprocess and normalize text for English, Arabic (including dialectal normalization), and French datasets:

```bash
python -m src.preprocessing.clean_data --input_dir data/raw --output_dir data/processed

```

### 2. Model Fine-Tuning

Train or fine-tune a multilingual transformer (e.g., `xlm-roberta-base`) on your dataset:

```bash
python -m src.models.train --config configs/xlm_roberta_config.json

```

### 3. Cross-Lingual Evaluation

Evaluate the model's cross-lingual generalization across test sets:

```bash
python -m src.models.evaluate --model_path outputs/best_model --test_dir data/processed/test

```

---

## 📊 Methodology & Models Evaluated

| Model | Language Scope | Primary Use Case |
| --- | --- | --- |
| **BERT / RoBERTa** | Monolingual (EN) | Baseline performance comparison |
| **mBERT** | Multilingual | Multilingual fine-tuning & feature alignment |
| **XLM-RoBERTa** | Multilingual | Cross-lingual transfer & zero-shot evaluation |
| **Language-Specific Transformers** | AR / FR | Benchmarking against native language models |

---

## Citation & Attribution

If you reference or use this repository for research or academic purposes, please attribute the project appropriately:

```bibtex
@misc{ammakhlouf2026crosslingual,
  author = {Joseph Am-Makhlouf},
  title = {Cross-Lingual Mental Health Detection (FYP)},
  year = {2026},
  publisher = {GitHub},
  journal = {GitHub Repository},
  howpublished = {\url{https://github.com/jam-makh/cross_lingual_mental_health_fyp}},
  note = {Under the Technical Supervision of Mr. Elie Dina}
}

```

---

## 📄 License

This project is licensed under the [MIT License](https://www.google.com/search?q=LICENSE) - see the LICENSE file for details.
