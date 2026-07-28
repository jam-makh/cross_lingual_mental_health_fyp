# Cross-Lingual Mental Health Detection

This repository contains the code and experiments for a cross-lingual mental health text classification project covering English, Arabic, and French datasets. The work focuses on comparing classical machine learning, deep learning, and large language model-based approaches for detecting mental health related content in multilingual text.

---

## Project Team and Work Division

This project was developed by Joseph Am-Makhlouf, Juliette Elias Daher, and Jean Khalil, under the supervision of Mr. Elie Dina.

| Area | Responsible Person |
| --- | --- |
| English dataset preprocessing, experimentation, and modeling | Joseph Am-Makhlouf |
| French dataset preprocessing, experimentation, and modeling | Juliette Elias Daher |
| Arabic dataset preprocessing, experimentation, and modeling | Jean Khalil |
| Overall supervision and guidance | Mr. Elie Dina |

---

## Project Overview

The main objective of this project is to evaluate how well mental health classification models generalize across languages and to compare different representation methods and classifiers. The repository includes experiments based on:

- cross-lingual shared text preprocessing and cleaning pipeline
- classical machine learning models
- deep learning models
- transformer-based and LLM-based approaches
- evaluation using standard classification metrics

---

## Repository Structure

```text
cross_lingual_mental_health_fyp/
├── README.md
├── arabic_dataset/
│   ├── EDA_Arabic.py
│   ├── Deep learning/
│   └── Machine learning/
├── english_dataset/
│   ├── main_cl.py
│   ├── cleaning_eda/
│   ├── models/
│   └── statistical_significance/
├── french_dataset/
│   ├── Deep learning/
│   ├── EDA/
│   ├── LLM/
│   └── ML/
└── web_interface/
    └── README_web_interface.md
```

### Folder Description

- english_dataset/: English-language pipeline, preprocessing, ML, DL, LLMs, and evaluation scripts.
- arabic_dataset/: Arabic-language pipeline with EDA, ML, DL amd LLMs.
- french_dataset/: French-language pipeline with EDA, ML, DL and LLMs.
- web_interface/: Files related to the web interface for the project.

---

## Getting Started

### Prerequisites

- Python 3.9 or newer
- A working virtual environment is recommended
- GPU support is recommended for deep learning experiments

### Setup

1. Clone the repository:

```bash
git clone <repository-url>
cd chuf
```

2. Create and activate a virtual environment:

```bash
python -m venv venv
source venv/bin/activate
```

3. Install the required dependencies for the experiment you want to run. Depending on the script, you may need libraries such as Pandas, Numpy, scikit-learn, PyTorch, and Transformers.

---

## Usage Notes

This project is organized by language rather than by a single unified pipeline. To work with a specific language, navigate to the corresponding folder and run the relevant script.

- English experiments: use files inside english_dataset/
- Arabic experiments: use files inside arabic_dataset/
- French experiments: use files inside french_dataset/

The repository also includes a web interface folder for deployment or demonstration purposes.

---

## Research Focus

The project compares several modeling strategies for mental health text classification, including:

- TF-IDF-based approaches
- Transformer-based models
- Deep Learning architectures
- Language-specific and Cross-Lingual evaluation

The overall goal is to understand which methods perform best across different languages and datasets.

---

## License

This project is licensed under the MIT License. Please refer to the LICENSE file for more details.
