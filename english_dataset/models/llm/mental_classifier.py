"""
mental_classifier.py
====================

OOP-based mental health text classification pipeline using a local Ollama LLM.

Each dataset (potentially in a different language) is processed independently.
The model receives a user-defined prompt, a message, and a JSON schema of
possible labels with their definitions.  For every row the model returns:

- **model_result**      – one of the predefined label keys
- **model_explanation** – free-text justification

Results are persisted to CSV and final metrics (accuracy, F1, recall,
precision, in_out) are printed to the console.

.. note::
    Requires a running Ollama instance (``ollama serve``).
    Install Python dependencies::

        pip install requests pandas scikit-learn tqdm

Example
-------
See :mod:`__main__` at the bottom of this file for a full usage example.
"""

from __future__ import annotations

import csv
import json
import logging
import re
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import pandas as pd
import requests
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from tqdm import tqdm

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s – %(message)s",
)
logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------


@dataclass
class Label:
    """Represents a single classification label and its definition.

    :param key: Short identifier used as the label value (e.g. ``"depression"``).
    :type key: str
    :param definition: Human-readable description of what the label means.
    :type definition: str
    """

    key: str
    definition: str


@dataclass
class DatasetConfig:
    """Configuration for one dataset to classify.

    :param name: Human-readable name / identifier for this dataset.
    :type name: str
    :param path: Path to the CSV file that contains the dataset.
    :type path: str | Path
    :param text_column: Name of the column that holds the raw message text.
    :type text_column: str
    :param label_column: Name of the column that holds the ground-truth label.
    :type label_column: str
    :param language: Language of the dataset (informational, passed to the
        model prompt if desired).
    :type language: str
    :param output_path: Where the result CSV will be written.  Defaults to
        ``<name>_results.csv`` in the current directory.
    :type output_path: str | Path | None
    :param encoding: CSV file encoding. Defaults to ``"utf-8"``.
    :type encoding: str
    :param separator: CSV field separator. Defaults to ``","``.
    :type separator: str
    """

    name: str
    path: str | Path
    text_column: str
    label_column: str
    language: str = "English"
    output_path: Optional[str | Path] = None
    encoding: str = "utf-8"
    separator: str = ","

    def __post_init__(self) -> None:
        self.path = Path(self.path)
        if self.output_path is None:
            self.output_path = Path(f"{self.name}_results.csv")
        else:
            self.output_path = Path(self.output_path)


@dataclass
class ClassificationResult:
    """Holds the classification output for a single message.

    :param message: The original text that was classified.
    :type message: str
    :param model_result: The label key returned by the model.
    :type model_result: str
    :param model_explanation: The model's justification for the chosen label.
    :type model_explanation: str
    :param actual_result: The ground-truth label from the dataset.
    :type actual_result: str
    """

    message: str
    model_result: str
    model_explanation: str
    actual_result: str


@dataclass
class Metrics:
    """Aggregated evaluation metrics for one dataset run.

    :param accuracy: Fraction of correctly classified samples.
    :type accuracy: float
    :param f1: Macro-averaged F1 score.
    :type f1: float
    :param recall: Macro-averaged recall.
    :type recall: float
    :param precision: Macro-averaged precision.
    :type precision: float
    :param in_out: Fraction of samples where the model returned a label
        **outside** the predefined label set (lower is better).
    :type in_out: float
    :param total_samples: Total number of samples evaluated.
    :type total_samples: int
    :param out_of_vocab: Number of samples where the model's answer was not
        in the predefined label set.
    :type out_of_vocab: int
    """

    accuracy: float
    f1: float
    recall: float
    precision: float
    in_out: float
    total_samples: int
    out_of_vocab: int

    def __str__(self) -> str:  # noqa: D401
        return (
            f"Accuracy  : {self.accuracy:.4f}\n"
            f"F1 (macro): {self.f1:.4f}\n"
            f"Recall    : {self.recall:.4f}\n"
            f"Precision : {self.precision:.4f}\n"
            f"In/Out    : {self.in_out:.4f}  "
            f"({self.out_of_vocab}/{self.total_samples} out-of-vocab predictions)"
        )


# ---------------------------------------------------------------------------
# Ollama client
# ---------------------------------------------------------------------------


class OllamaClient:
    """Thin wrapper around the Ollama HTTP API.

    :param base_url: Base URL of the Ollama server.
    :type base_url: str
    :param model: Name of the model to use (e.g. ``"llama3"``).
    :type model: str
    :param timeout: Request timeout in seconds.
    :type timeout: int
    :param max_retries: Number of times to retry on transient errors.
    :type max_retries: int
    :param retry_delay: Seconds to wait between retries.
    :type retry_delay: float
    """

    _GENERATE_ENDPOINT = "/api/generate"

    def __init__(
        self,
        base_url: str = "http://localhost:11434",
        model: str = "llama3",
        timeout: int = 120,
        max_retries: int = 3,
        retry_delay: float = 2.0,
    ) -> None:
        self.base_url = base_url.rstrip("/")
        self.model = model
        self.timeout = timeout
        self.max_retries = max_retries
        self.retry_delay = retry_delay

    # ------------------------------------------------------------------
    # Public helpers
    # ------------------------------------------------------------------

    def list_models(self) -> List[str]:
        """Return the names of all models available on the Ollama server.

        :raises requests.RequestException: If the server cannot be reached.
        :return: Sorted list of model name strings.
        :rtype: list[str]
        """
        resp = requests.get(f"{self.base_url}/api/tags", timeout=self.timeout)
        resp.raise_for_status()
        data = resp.json()
        return sorted(m["name"] for m in data.get("models", []))

    def generate(self, prompt: str) -> str:
        """Send *prompt* to the model and return the raw text response.

        Automatically retries on ``requests.RequestException``.

        :param prompt: The full prompt string (system + user content combined).
        :type prompt: str
        :raises requests.RequestException: If all retries are exhausted.
        :return: The model's text response.
        :rtype: str
        """
        payload = {
            "model": self.model,
            "prompt": prompt,
            "stream": False,
        }
        last_exc: Optional[Exception] = None
        for attempt in range(1, self.max_retries + 1):
            try:
                resp = requests.post(
                    f"{self.base_url}{self._GENERATE_ENDPOINT}",
                    json=payload,
                    timeout=self.timeout,
                )
                resp.raise_for_status()
                return resp.json().get("response", "")
            except requests.RequestException as exc:
                last_exc = exc
                logger.warning(
                    "Ollama request failed (attempt %d/%d): %s",
                    attempt,
                    self.max_retries,
                    exc,
                )
                if attempt < self.max_retries:
                    time.sleep(self.retry_delay)
        raise last_exc  # type: ignore[misc]


# ---------------------------------------------------------------------------
# Prompt builder
# ---------------------------------------------------------------------------


class PromptBuilder:
    """Builds the full prompt that is sent to the LLM for each message.

    :param user_prompt_template: A template string written by the user.
        Use ``{message}`` as a placeholder for the text to classify,
        ``{language}`` for the dataset language, and ``{labels_json}`` for
        the JSON schema of labels.  Example::

            "Classify the following {language} mental-health forum post.
             Available categories: {labels_json}

             Post: {message}"

    :type user_prompt_template: str
    :param labels: List of :class:`Label` objects defining the valid outputs.
    :type labels: list[Label]
    """

    _SYSTEM_SUFFIX = (
        "\n\nIMPORTANT: You MUST respond with a single valid JSON object "
        "and nothing else.  The object must have exactly two keys:\n"
        '  "result"      – one of the allowed label keys (string)\n'
        '  "explanation" – your reasoning (string)\n'
        "Do not include markdown fences, extra keys, or any text outside "
        "the JSON object."
    )

    def __init__(self, user_prompt_template: str, labels: List[Label]) -> None:
        self.user_prompt_template = user_prompt_template
        self.labels = labels
        self._labels_json: str = json.dumps(
            {lbl.key: lbl.definition for lbl in labels}, ensure_ascii=False, indent=2
        )

    # ------------------------------------------------------------------
    # Public helpers
    # ------------------------------------------------------------------

    def build(self, message: str, language: str = "English") -> str:
        """Render the full prompt for *message*.

        :param message: The raw text to classify.
        :type message: str
        :param language: Language of the dataset (injected into the template).
        :type language: str
        :return: The complete prompt string ready to send to the model.
        :rtype: str
        """
        user_section = self.user_prompt_template.format(
            message=message,
            language=language,
            labels_json=self._labels_json,
        )
        return user_section + self._SYSTEM_SUFFIX


# ---------------------------------------------------------------------------
# Response parser
# ---------------------------------------------------------------------------


class ResponseParser:
    """Extracts ``result`` and ``explanation`` from the model's raw output.

    The model is instructed to return pure JSON, but LLMs sometimes wrap the
    output in markdown fences or add extra text.  This parser tries multiple
    strategies to recover the JSON.

    :param valid_keys: Set of label keys that are considered valid.
    :type valid_keys: set[str]
    :param fallback_result: Value to use when parsing fails completely.
    :type fallback_result: str
    """

    def __init__(
        self,
        valid_keys: set[str],
        fallback_result: str = "__PARSE_ERROR__",
    ) -> None:
        self.valid_keys = valid_keys
        self.fallback_result = fallback_result

    # ------------------------------------------------------------------
    # Public helpers
    # ------------------------------------------------------------------

    def parse(self, raw: str) -> Tuple[str, str]:
        """Parse the model's raw string into *(result, explanation)*.

        :param raw: The raw text returned by the model.
        :type raw: str
        :return: A ``(result, explanation)`` tuple.  If parsing fails the
            result is ``self.fallback_result``.
        :rtype: tuple[str, str]
        """
        data = self._try_parse_json(raw)
        if data is None:
            logger.debug("Could not parse model output: %r", raw[:200])
            return self.fallback_result, raw.strip()

        result = str(data.get("result", "")).strip()
        explanation = str(data.get("explanation", "")).strip()

        if result not in self.valid_keys:
            logger.debug("Model returned out-of-vocab label: %r", result)

        return result, explanation

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _try_parse_json(self, text: str) -> Optional[Dict[str, Any]]:
        """Attempt several strategies to extract a JSON object from *text*.

        :param text: Raw text that may or may not contain a JSON object.
        :type text: str
        :return: Parsed dictionary, or ``None`` if all strategies fail.
        :rtype: dict | None
        """
        # Strategy 1: direct parse
        try:
            return json.loads(text)
        except json.JSONDecodeError:
            pass

        # Strategy 2: strip markdown fences
        stripped = re.sub(r"```(?:json)?", "", text).strip().strip("`").strip()
        try:
            return json.loads(stripped)
        except json.JSONDecodeError:
            pass

        # Strategy 3: extract first {...} block
        match = re.search(r"\{.*?\}", text, re.DOTALL)
        if match:
            try:
                return json.loads(match.group())
            except json.JSONDecodeError:
                pass

        return None


# ---------------------------------------------------------------------------
# Classifier
# ---------------------------------------------------------------------------


class MentalHealthClassifier:
    """Orchestrates the end-to-end classification pipeline.

    :param client: An :class:`OllamaClient` instance.
    :type client: OllamaClient
    :param prompt_builder: A :class:`PromptBuilder` instance.
    :type prompt_builder: PromptBuilder
    :param labels: List of :class:`Label` objects (used to build the parser).
    :type labels: list[Label]
    :param sleep_between_calls: Seconds to sleep between API calls to avoid
        overwhelming the local Ollama server.  Set to ``0`` to disable.
    :type sleep_between_calls: float
    """

    def __init__(
        self,
        client: OllamaClient,
        prompt_builder: PromptBuilder,
        labels: List[Label],
        sleep_between_calls: float = 0.0,
    ) -> None:
        self.client = client
        self.prompt_builder = prompt_builder
        self.labels = labels
        self.sleep_between_calls = sleep_between_calls
        self._valid_keys: set[str] = {lbl.key for lbl in labels}
        self._parser = ResponseParser(valid_keys=self._valid_keys)

    # Public helpers
    def classify_dataset(self, config: DatasetConfig) -> List[ClassificationResult]:
        """Load *config.path*, classify every row, and save results to CSV.

        :param config: Dataset configuration object.
        :type config: DatasetConfig
        :raises FileNotFoundError: If the dataset CSV does not exist.
        :raises KeyError: If ``text_column`` or ``label_column`` are missing.
        :return: List of :class:`ClassificationResult` objects.
        :rtype: list[ClassificationResult]
        """
        logger.info("Loading dataset '%s' from %s", config.name, config.path)
        df = pd.read_csv(
            config.path, encoding=config.encoding, sep=config.separator
        )

        if config.text_column not in df.columns:
            raise KeyError(
                f"Column '{config.text_column}' not found in {config.path}. "
                f"Available columns: {list(df.columns)}"
            )
        if config.label_column not in df.columns:
            raise KeyError(
                f"Column '{config.label_column}' not found in {config.path}. "
                f"Available columns: {list(df.columns)}"
            )

        results: List[ClassificationResult] = []

        for idx, row in tqdm(
            df.iterrows(), total=len(df), desc=f"Classifying [{config.name}]"
        ):
            message = str(row[config.text_column])
            actual = str(row[config.label_column])

            model_result, model_explanation = self._classify_single(
                message=message, language=config.language
            )

            results.append(
                ClassificationResult(
                    message=message,
                    model_result=model_result,
                    model_explanation=model_explanation,
                    actual_result=actual,
                )
            )

            if self.sleep_between_calls > 0:
                time.sleep(self.sleep_between_calls)

        self._save_results(results, config.output_path)
        return results

    def compute_metrics(self, results: List[ClassificationResult]) -> Metrics:
        """Compute evaluation metrics from a list of classification results.

        Labels that the model returned outside the predefined set are counted
        towards ``in_out`` but are **excluded** from sklearn metrics to avoid
        unexpected label errors.

        :param results: List of :class:`ClassificationResult` objects.
        :type results: list[ClassificationResult]
        :return: A populated :class:`Metrics` instance.
        :rtype: Metrics
        """
        y_true_all = [r.actual_result for r in results]
        y_pred_all = [r.model_result for r in results]

        total = len(results)
        out_of_vocab = sum(1 for p in y_pred_all if p not in self._valid_keys)

        # Filter to rows where prediction is in-vocab for sklearn metrics
        in_vocab_pairs = [
            (true, pred)
            for true, pred in zip(y_true_all, y_pred_all)
            if pred in self._valid_keys
        ]
        y_true_iv = [p[0] for p in in_vocab_pairs]
        y_pred_iv = [p[1] for p in in_vocab_pairs]

        if not y_true_iv:
            logger.warning("All predictions were out-of-vocab; metrics set to 0.")
            return Metrics(
                accuracy=0.0,
                f1=0.0,
                recall=0.0,
                precision=0.0,
                in_out=1.0,
                total_samples=total,
                out_of_vocab=out_of_vocab,
            )

        labels_list = sorted(self._valid_keys)

        return Metrics(
            accuracy=accuracy_score(y_true_iv, y_pred_iv),
            f1=f1_score(
                y_true_iv, y_pred_iv, labels=labels_list, average="macro", zero_division=0
            ),
            recall=recall_score(
                y_true_iv, y_pred_iv, labels=labels_list, average="macro", zero_division=0
            ),
            precision=precision_score(
                y_true_iv, y_pred_iv, labels=labels_list, average="macro", zero_division=0
            ),
            in_out=out_of_vocab / total,
            total_samples=total,
            out_of_vocab=out_of_vocab,
        )

    # Private helpers
    def _classify_single(self, message: str, language: str) -> Tuple[str, str]:
        """Classify one *message* and return *(result, explanation)*.

        :param message: Text to classify.
        :type message: str
        :param language: Dataset language (forwarded to the prompt).
        :type language: str
        :return: ``(model_result, model_explanation)`` tuple.
        :rtype: tuple[str, str]
        """
        prompt = self.prompt_builder.build(message=message, language=language)
        raw_response = self.client.generate(prompt)
        return self._parser.parse(raw_response)

    @staticmethod
    def _save_results(results: List[ClassificationResult], output_path: Path) -> None:
        """Write *results* to a CSV file.

        Columns: ``message``, ``model_result``, ``model_explanation``,
        ``actual_result``.

        :param results: Classification results to persist.
        :type results: list[ClassificationResult]
        :param output_path: Destination CSV file path.
        :type output_path: Path
        """
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, "w", newline="", encoding="utf-8") as fh:
            writer = csv.writer(fh)
            writer.writerow(
                ["message", "model_result", "model_explanation", "actual_result"]
            )
            for r in results:
                writer.writerow(
                    [r.message, r.model_result, r.model_explanation, r.actual_result]
                )
        logger.info("Results saved to %s", output_path)


# Pipeline runner
class ClassificationPipeline:
    """High-level entry point that ties everything together.

    Instantiate this class, call :meth:`run_all`, and get per-dataset metrics.

    :param model: Ollama model name to use (e.g. ``"llama3"`` or ``"mistral"``).
    :type model: str
    :param labels: List of :class:`Label` instances that define the taxonomy.
    :type labels: list[Label]
    :param prompt_template: Prompt template with ``{message}``,
        ``{language}``, and ``{labels_json}`` placeholders.
    :type prompt_template: str
    :param datasets: One :class:`DatasetConfig` per dataset to process.
    :type datasets: list[DatasetConfig]
    :param ollama_base_url: Ollama server URL.
    :type ollama_base_url: str
    :param sleep_between_calls: Seconds to sleep between Ollama API calls.
    :type sleep_between_calls: float
    """

    def __init__(
        self,
        model: str,
        labels: List[Label],
        prompt_template: str,
        datasets: List[DatasetConfig],
        ollama_base_url: str = "http://localhost:11434",
        sleep_between_calls: float = 0.0,
    ) -> None:
        self.model = model
        self.labels = labels
        self.prompt_template = prompt_template
        self.datasets = datasets

        self._client = OllamaClient(base_url=ollama_base_url, model=model)
        self._prompt_builder = PromptBuilder(
            user_prompt_template=prompt_template, labels=labels
        )
        self._classifier = MentalHealthClassifier(
            client=self._client,
            prompt_builder=self._prompt_builder,
            labels=labels,
            sleep_between_calls=sleep_between_calls,
        )

    def list_available_models(self) -> List[str]:
        """Return model names available on the configured Ollama server.

        :return: Sorted list of model name strings.
        :rtype: list[str]
        """
        return self._client.list_models()

    def run_all(self) -> Dict[str, Metrics]:
        """Run the classification pipeline on every configured dataset.

        :return: A dictionary mapping ``dataset.name`` to its :class:`Metrics`.
        :rtype: dict[str, Metrics]
        """
        all_metrics: Dict[str, Metrics] = {}

        for ds_config in self.datasets:
            logger.info("=" * 60)
            logger.info("Dataset : %s  |  Language: %s", ds_config.name, ds_config.language)
            logger.info("Model   : %s", self.model)
            logger.info("=" * 60)

            results = self._classifier.classify_dataset(ds_config)
            metrics = self._classifier.compute_metrics(results)

            logger.info("\nMetrics for '%s':\n%s", ds_config.name, metrics)
            all_metrics[ds_config.name] = metrics

        return all_metrics


if __name__ == "__main__":
    # 1. Define your labels and their definitions
    labels = [
        Label(
            key="anxiety",
            definition="The text expresses excessive worry, nervousness, fear, or feelings of anxiety."
        ),
        Label(
            key="depression",
            definition="The text expresses sadness, hopelessness, loss of interest, low mood, or depressive thoughts."
        ),
        Label(
            key="suicidal",
            definition="The text expresses suicidal thoughts, self-harm intentions, a desire to die, or feelings that life is not worth living."
        ),
        Label(
            key="normal",
            definition="The text does not show significant signs of anxiety, depression, suicidal ideation, or other notable mental health concerns."
        )
    ]

    # 2. Write your prompt template
    #    Placeholders: {message}, {language}, {labels_json}
    PROMPT_TEMPLATE = (
        "You are a mental health text classifier. "
        "The following message is written in {language}.\n\n"
        "Possible categories and their definitions:\n{labels_json}\n\n"
        "Message to classify:\n\"{message}\"\n"
    )

    # 3. Configure your datasets (one per language)
    datasets = [
        DatasetConfig(
            name="sampled_25_per_label.csv",
            path=r"C:\Users\MY-PC\OneDrive - Sagesse University\Desktop\LU DS\FYP\mental_health\sampled_25_per_label.csv",
            text_column="text",
            label_column="status",
            language="English",
            output_path="results_qwen/english_results.csv",
        ),
    ]

    # 4. Create and run the pipeline
    pipeline = ClassificationPipeline(
        model="qwen3.5:4b",
        labels=labels,
        prompt_template=PROMPT_TEMPLATE,
        datasets=datasets,
        ollama_base_url="http://localhost:11434",
        sleep_between_calls=0.1,  # be kind to your GPU
    )

    # Optional: inspect available models before running
    try:
        available = pipeline.list_available_models()
        logger.info("Available Ollama models: %s", available)
    except Exception as exc:
        logger.warning("Could not list models: %s", exc)

    # Run classification on all datasets and collect metrics
    all_metrics = pipeline.run_all()

    # Print summary
    print("\n" + "=" * 60)
    print("FINAL METRICS SUMMARY")
    print("=" * 60)
    for ds_name, m in all_metrics.items():
        print(f"\n[{ds_name}]")
        print(m)
