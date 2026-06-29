# Cross-Lingual Mental Health Classifier

This project is a web-based interface for a multilingual mental health text classification system. It supports text input in English, French, and Arabic, automatically detects the input language, routes the text to the correct trained model, and displays the predicted mental-health category with confidence information.

The system was developed as part of a Final Year Project focused on cross-lingual mental health analysis using classical machine learning, deep learning, and language-specific preprocessing pipelines.

---

## Project Overview

The application provides a single interface where users can enter a mental-health related sentence or short paragraph. Based on the detected language, the system selects the appropriate model and returns the prediction result.

Supported languages:

* English
* French
* Arabic

The interface displays:

* Detected language
* Selected model
* Predicted category
* Confidence score
* Decision strength
* Class probability breakdown

---

## Important Usage Note

The system is designed to classify complete sentences or short paragraphs, not isolated keywords.

Very short inputs such as:

```text
depressed
triste
anxiety
```

may be unreliable because the models were trained on contextual text entries, not single-word keyword matching.

For best results, enter a complete sentence such as:

```text
I feel empty and exhausted every day, and I no longer enjoy anything I used to like.
```

---

## Language-Specific Models

### English Model

The English classifier uses:

* Text cleaning and preprocessing
* DistilBERT embeddings
* SVM classifier

Displayed model name:

```text
SVM + DistilBERT
```

The English model classifies text into:

```text
Anxiety
Depression
Normal
Suicidal
```

---

### French Model

The French classifier uses:

* French text preprocessing
* TF-IDF vectorization
* LSTM model

Displayed model name:

```text
LSTM + TF-IDF
```

The French model classifies text into:

```text
Healthy
Unhealthy
```

---

### Arabic Model

The Arabic classifier uses:

* Arabic preprocessing
* TF-IDF vectorization
* Logistic Regression classifier

Displayed model name:

```text
Logistic Regression + TF-IDF
```

The Arabic model classifies text into:

```text
anxiety_fear
depression
ocd_obsessive
```

---

## Project Structure

```text
mental_health_web_interface/
|
|-- app.py
|-- requirements.txt
|
|-- adapters/
|   |-- english_adapter.py
|   |-- french_adapter.py
|   |-- arabic_adapter.py
|
|-- inference/
|   |-- model_router.py
|   |-- output_format.py
|
|-- models/
|   |-- english/
|   |   |-- cleaning.py
|   |   |-- english_config.py
|   |   |-- vectorizers.py
|   |   |-- english_svm_distilbert_probability.pkl
|   |   |-- english_distilbert_scaler.pkl
|   |
|   |-- french/
|   |   |-- predictor.py
|   |   |-- tfidf.py
|   |   |-- models.py
|   |   |-- config_exported.yaml
|   |   |
|   |   |-- vectorizers/
|   |   |   |-- tfidf_vectorizer.pkl
|   |   |
|   |   |-- checkpoints/
|   |       |-- LSTM_tfidf.pt
|   |
|   |-- arabic/
|       |-- arabic_preprocessing.py
|       |-- arabic_config.json
|       |-- arabic_tfidf_vectorizer.pkl
|       |-- arabic_logistic_regression.pkl
|       |-- arabic_label_encoder.pkl
|
|-- README.md
```

---

## Repository

The source code for this web interface is available in the following GitHub repository:

```text
https://github.com/Jay7-analyst/mental_health_web_interface
```

---

## Installation

Create and activate a Python environment, then install the required packages:

```bash
pip install -r requirements.txt
```

The project uses Python 3.12.

---

## Running the Application Locally

From the project folder, run:

```bash
streamlit run app.py
```

Then open the local URL shown in the terminal.

---

## Example Inputs

### English

```text
I keep panicking for no clear reason and I feel scared all the time.
```

### French

```text
Depuis plusieurs semaines, je me sens tres triste, epuise, isole, et je n'arrive plus a vivre normalement.
```

### Arabic

```text
أشعر بحزن عميق وفقدت الرغبة في كل شيء حتى الأشياء التي كنت أحبها.
```

---

## Decision Strength

The decision strength is derived from the confidence score of the predicted class:

```text
Weak: confidence below 55%
Medium: confidence from 55% to below 75%
Strong: confidence of 75% or higher
```

This indicator gives a simple interpretation of how confident the model is in its final prediction.

---

## Limitations

This system is a research prototype and should not be used as a medical diagnosis tool.

Main limitations:

* The models may fail on very short inputs.
* The models are not fully optimized and may not always produce the best possible prediction.
* The models depend on the quality and distribution of the training datasets.
* Predictions are sensitive to wording and language detection.
* The system is designed for academic demonstration and analysis, not clinical decision-making.

---

## Disclaimer

This application is developed for academic and research purposes only. It does not provide medical advice, diagnosis, or treatment. Any mental health concerns should be discussed with a qualified mental health professional.
