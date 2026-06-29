"""
TF-IDF vectorization utilities.
"""

from typing import Tuple, List
from tqdm import tqdm

import numpy as np
import torch

from sklearn.feature_extraction.text import TfidfVectorizer
from transformers import AutoTokenizer, AutoModel


def build_tfidf_features(
    train_texts,
    test_texts,
    max_features: int = 10000,
    ngram_range: Tuple[int, int] = (1, 2)
):
    """
    Convert text into TF-IDF vectors.

    :param train_texts: Training text samples.
    :type train_texts: list
    :param test_texts: Test text samples.
    :type test_texts: list
    :param max_features: Maximum vocabulary size.
    :type max_features: int
    :param ngram_range: N-gram range.
    :type ngram_range: Tuple[int, int]
    :returns: Train/test TF-IDF matrices and fitted vectorizer.
    :rtype: Tuple
    """

    # Initialize TF-IDF vectorizer with the requested vocabulary size and n-grams.
    vectorizer = TfidfVectorizer(
        max_features=max_features,
        ngram_range=ngram_range
    )

    # Learn vocabulary from training data.
    X_train = vectorizer.fit_transform(train_texts)

    # Transform test data using the same fitted vocabulary.
    X_test = vectorizer.transform(test_texts)

    return X_train, X_test, vectorizer

def mean_pooling(
    model_output,
    attention_mask
):
    """
    Apply mean pooling to transformer outputs.

    :param model_output: Output from a transformer model.
    :type model_output: tuple
    :param attention_mask: Attention mask for valid tokens.
    :type attention_mask: torch.Tensor
    :returns: Tensor of sentence embeddings.
    :rtype: torch.Tensor
    """

    # Use the first output tensor as the token embeddings.
    token_embeddings = model_output[0]

    # Expand the attention mask to cover the token embedding dimensions.
    input_mask_expanded = (
        attention_mask.unsqueeze(-1)
        .expand(token_embeddings.size())
        .float()
    )

    # Compute the average over valid token positions only.
    return torch.sum(
        token_embeddings * input_mask_expanded,
        1
    ) / torch.clamp(
        input_mask_expanded.sum(1),
        min=1e-9
    )


def generate_distilbert_embeddings(
    texts: List[str],
    batch_size: int = 16,
    max_length: int = 256,
    model_name: str = "distilbert-base-uncased"
) -> np.ndarray:
    """
    Generate DistilBERT sentence embeddings.

    :param texts: Input text samples.
    :type texts: List[str]
    :param batch_size: Batch size.
    :type batch_size: int
    :param max_length: Maximum token length.
    :type max_length: int
    :param model_name: Hugging Face model name.
    :type model_name: str
    :returns: Dense embedding matrix.
    :rtype: np.ndarray
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Running embedding generation on: {device}")
    # Load the tokenizer and model from Hugging Face.
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModel.from_pretrained(model_name).to(device)

    model.eval()

    # Accumulate embeddings across batches to avoid excessive GPU memory use.
    embeddings = []

    with torch.no_grad():

        # Process texts in batches for transformer encoding.
        for i in tqdm(range(0, len(texts), batch_size), desc="  Generating DistilBERT embeddings", unit="batch"):

            # Slice the current batch of raw text samples.
            batch = texts[i:i + batch_size]

            # Tokenize each batch with padding and truncation.
            encoded_input = tokenizer(
                batch,
                padding=True,
                truncation=True,
                max_length=max_length,
                return_tensors="pt"
            )

            # 2. CRITICAL: Move tokenized tensors to the GPU
            encoded_input = {k: v.to(device) for k, v in encoded_input.items()}

            # Run the transformer model and capture hidden states.
            model_output = model(**encoded_input)

            # Pool token embeddings into a single sentence vector.
            sentence_embeddings = mean_pooling(
                model_output,
                encoded_input["attention_mask"]
            )

            # Pull back to CPU safely before converting to numpy
            embeddings.append(
                sentence_embeddings.cpu().numpy()
            )

    # Stack all batches into a single matrix for downstream training.
    return np.vstack(embeddings)
