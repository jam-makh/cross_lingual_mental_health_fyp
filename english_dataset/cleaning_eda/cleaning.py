#!/usr/bin/env python3

"""
Pydantic models and text cleaning for mental health text classification.
"""

import re
import warnings
from typing import Dict, List, Set, Tuple

import emoji
import nltk
import pandas as pd
import spacy
from bs4 import BeautifulSoup
from nltk import download
from pydantic import BaseModel, Field
from tqdm import tqdm
from nltk.corpus import stopwords

# import patterns and regex from the config file
from .config import (
    DISCOURSE_IN_SPACY,
    DISCOURSE_UNIQUE,
    STOPWORDS,
    CONTRACTIONS,
    EMOJIS, EMOTICONS, VISUAL_STOPWORDS
)

warnings.filterwarnings("ignore")

# NLTK Setup
try:
    download("punkt", quiet=True)
    download("punkt_tab", quiet=True)
    download("stopwords", quiet=True)
    download("wordnet", quiet=True)
except Exception as e:
    print(f"Warning: NLTK download failed: {e}")


class MentalHealthPost(BaseModel):
    """
    Pydantic model representing a mental health post.
    It defines the structure for individual social media posts
    used in the mental health classification dataset.

    :param unique_id: Unique identifier for the post.
    :type unique_id: int
    :param text: Social media post content.
    :type text: str
    :param status: Labels (Anxiety, Depression, Suicidal, Normal)
    :type status: str
    """

    unique_id: int = Field(..., description="Unique identifier for the post")
    text: str = Field(..., description="Social media post content")
    status: str = Field(..., description="Mental health category: Anxiety, "
                                         "Depression, Suicidal, Normal")

    class Config:
        """
        Pydantic configuration for the MentalHealthPost model.

        :param validate_assignment: Validation of fields during assignment.
        :param use_enum_values: If enums're used, they're serialized as values.
        """
        validate_assignment = True
        use_enum_values = True


class PostAnalysis(BaseModel):
    """
    Pydantic model for storing analysis features of a post.

    This model holds various textual features extracted from a post,
    such as word count, punctuation usage, emoji counts, etc.

    :param text_length: Number of words in the post.
    :type text_length: int
    :param char_count: Number of characters in the post.
    :type char_count: int
    :param punct_density: Punctuation density (total punctuation / word count).
    :type punct_density: float
    :param question_count: Number of question marks.
    :type question_count: int
    :param exclamation_count: Number of exclamation marks.
    :type exclamation_count: int
    :param ellipsis_count: Number of ellipses.
    :type ellipsis_count: int
    :param emoji_count: Number of emojis.
    :type emoji_count: int
    :param emojis: List of emoji characters found.
    :type emojis: List[str]
    :param emoticon_count: Number of emoticons.
    :type emoticon_count: int
    :param emoticons: List of emoticon strings found.
    :type emoticons: List[str]
    :param hashtags: List of hashtags extracted from the post.
    :type hashtags: List[str]
    """

    text_length: int = Field(default=0, description="Number of words")
    char_count: int = Field(default=0, description="Number of characters")
    punct_density: float = Field(default=0.0,
                                 description="total punctuation / word count")
    question_count: int = Field(default=0,
                                description="Number of question marks")
    exclamation_count: int = Field(default=0,
                                   description="Number of exclamation marks")
    ellipsis_count: int = Field(default=0,
                                description="Number of ellipsis")
    emoji_count: int = Field(default=0,
                             description="Number of emojis")
    emojis: List[str] = Field(default_factory=list,
                              description="List of emojis")
    emoticon_count: int = Field(default=0,
                                description="Number of emoticons")
    emoticons: List[str] = Field(default_factory=list,
                                 description="List of emoticons")
    hashtags: List[str] = Field(default_factory=list,
                                description="List of hashtags")


def _load_spacy_model(
    model_name: str = "en_core_web_sm",
) -> spacy.language.Language:
    """
    Load (or auto-download) a spaCy language model with NER and the
    dependency parser disabled for speed.

    :param model_name: spaCy model identifier, e.g. ``en_core_web_sm``.
    :type model_name: str
    :returns: Ready-to-use spaCy language pipeline.
    :rtype: spacy.language.Language
    """
    try:
        return spacy.load(model_name, disable=["ner", "parser"])
    except OSError:
        print(f"Model '{model_name}' not found. Downloading…")
        spacy.cli.download(model_name)  # type: ignore
        return spacy.load(model_name, disable=["ner", "parser"])


class Cleaning:
    """
    Cleaning logic for mental-health social-media posts.

    :param model_name: spaCy model to load (NER and parser disabled).
    :type model_name: str
    """

    # Strings treated as missing / non-informative (case-insensitive).
    _PLACEHOLDERS: Tuple[str, ...] = (
        "na", "n/a", "nan", "none", "null", "undefined", "",
    )

    def __init__(self, model_name: str = "en_core_web_sm") -> None:
        """
        Initialise the class instance.
        Loads the spaCy model once and builds the combined stopword set
        from spaCy defaults plus the project-level custom list.

        :param model_name: spaCy model identifier.
        :type model_name: str
        """
        self.nlp: spacy.language.Language = _load_spacy_model(
            model_name
        )
        self.stopwords_set: Set[str] = set()
        self.stopwords_set.update(self.nlp.Defaults.stop_words)
        self.stopwords_set.update(nltk.corpus.stopwords.words("english"))
        self.stopwords_set.update(STOPWORDS)
        self.stopwords_set = self.stopwords_set - {
            "never", "no", "nobody", "none", "not", "nothing",
        }
        self.stopwords_set = self.stopwords_set - DISCOURSE_IN_SPACY

        self.wordcloud_stopwords_set: Set[str] = set(self.stopwords_set)
        self.wordcloud_stopwords_set.update(DISCOURSE_UNIQUE)
        self.wordcloud_stopwords_set.update(DISCOURSE_IN_SPACY)
        self.wordcloud_stopwords_set.update(VISUAL_STOPWORDS)

        self.contraction_map: Dict[str, str] = CONTRACTIONS
        self.emojis_map: str = EMOJIS
        self.emoticons_map: List[str] = EMOTICONS

    def drop_missing(
        self, df: pd.DataFrame, text_column: str
    ) -> pd.DataFrame:
        """
        Remove rows where text_column is NaN or None.

        :param df: Input DataFrame.
        :type df: pandas.DataFrame
        :param text_column: Name of the text column to check.
        :type text_column: str
        :returns: DataFrame with NaN rows removed; index reset.
        :rtype: pandas.DataFrame
        """
        before: int = len(df)
        df = df.dropna(subset=[text_column]).reset_index(drop=True)
        dropped: int = before - len(df)
        if dropped:
            print(f"  [drop_missing]      removed {dropped:,} rows.")
        return df

    def drop_duplicates(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Remove fully duplicate rows from df.

        :param df: Input DataFrame.
        :type df: pandas.DataFrame
        :returns: Deduplicated DataFrame; index reset.
        :rtype: pandas.DataFrame
        """
        before: int = len(df)
        df = df.drop_duplicates().reset_index(drop=True)
        dropped: int = before - len(df)
        if dropped:
            print(f"  [drop_duplicates]   removed {dropped:,} rows.")
        return df

    def drop_placeholders(
        self, df: pd.DataFrame, text_column: str
    ) -> pd.DataFrame:
        """
        Remove rows whose text is empty, whitespace-only, or matches a
        known placeholder ('na', 'n/a', 'none', 'nan', etc.).

        :param df: Input DataFrame.
        :type df: pandas.DataFrame
        :param text_column: Name of the text column to check.
        :type text_column: str
        :returns: Filtered DataFrame; index reset.
        :rtype: pandas.DataFrame
        """
        ph: Set[str] = set(self._PLACEHOLDERS)
        before: int = len(df)
        mask = df[text_column].apply(
            lambda t: (
                isinstance(t, str)
                and t.strip() != ""
                and t.strip().lower() not in ph
            )
        )
        df = df[mask].reset_index(drop=True)
        dropped: int = before - len(df)
        if dropped:
            print(f"  [drop_placeholders] removed {dropped:,} rows.")
        return df

    def drop_invalid_cleaned(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Remove rows where the cleaned_text column produced by
        :meth:full_preprocess is still empty or punctuation-only.

        Expects cleaned_text to already exist in df.

        :param df: DataFrame containing a cleaned_text column.
        :type df: pandas.DataFrame
        :returns: Filtered DataFrame; index reset.
        :rtype: pandas.DataFrame
        """
        before: int = len(df)
        mask = df["cleaned_text"].apply(self.is_valid_text)
        df = df[mask].reset_index(drop=True)
        dropped: int = before - len(df)
        if dropped:
            print(
                f"  [drop_invalid]      removed {dropped:,} rows "
                f"(empty after cleaning)."
            )
        return df

    def remove_hashtags(self, text: str) -> str:
        """
        Removes hashtags from text

        :param text: original text, possibly containing hashtags.
        :type text: str
        :return: text with hashtags removed
        :rtype: str
        """
        text = re.sub(r"#\w+", " ", text)
        return re.sub(r"\s+", " ", text).strip()

    def remove_html(self, text: str) -> str:
        """
        Strip HTML/XML tags while preserving emoticons and internal
        placeholders (``<ELLIPSIS>``, ``<3``, ``>_<``, etc.).

        Known false-positives are swapped with sentinel strings before
        BeautifulSoup runs, then restored afterwards.

        :param text: original text possibly containing HTML markup.
        :type text: str
        :returns: Text with HTML tags removed; emoticons intact.
        :rtype: str
        """
        guards: List[Tuple[str, str]] = [
            ("<ELLIPSIS>", "__ELLIPSIS_S__"),
            ("</3", "__BROKENHEART_S__"),
            ("<3", "__HEART_S__"),
            (">_<", "__UPSET_S__"),
            (">.<", "__GRUMPY_S__"),
            (">=(",  "__ANGRY_S__"),
            (">:(", "__ANGRY2_S__"),
            (">:)", "__EVIL_S__"),
        ]
        for original, sentinel in guards:
            text = text.replace(original, sentinel)

        text = BeautifulSoup(text, "lxml").get_text(separator=" ")

        for original, sentinel in guards:
            text = text.replace(sentinel, original)

        return re.sub(r"\s+", " ", text).strip()

    def remove_emojis(self, text: str) -> str:
        """
        Remove Unicode emoji characters using the :mod:emoji library
        and a supplementary regex, then strip stray control characters.

        :param text: Input string.
        :type text: str
        :returns: String with emoji and control characters removed.
        :rtype: str
        """
        text = emoji.replace_emoji(text, replace="")
        text = re.sub(self.emojis_map, "", text)
        text = re.sub(r"[\x00-\x1F\x7F-\x9F\uFFFD\uFEFF]", "", text)
        return text

    def remove_emoticons(self, text: str) -> str:
        """
        Remove ASCII emoticons using the project's regex pattern list.

        :param text: Input string.
        :type text: str
        :returns: String with emoticons removed; extra whitespace
            collapsed.
        :rtype: str
        """
        for pattern in self.emoticons_map:
            text = re.sub(pattern, " ", text, flags=re.IGNORECASE)
        return re.sub(r" +", " ", text).strip()

    def expand_contractions(self, text: str) -> str:
        """
        Replace contractions with their expanded forms.

        Longer contractions are matched first to prevent partial
        substitution (e.g. can't've before can't).

        :param text: Input string.
        :type text: str
        :returns: Text with contractions expanded.
        :rtype: str
        """
        # can't and can't've : sort is used here to take can't've before
        sorted_map: List[Tuple[str, str]] = sorted(
            self.contraction_map.items(),
            key=lambda kv: len(kv[0]),
            reverse=True,
        )
        for contraction, expansion in sorted_map:
            # \b ensures we only match the exact contraction, not substrings
            # re.escape handles special characters
            # in contractions (like apostrophes)
            pattern = r"\b" + re.escape(contraction) + r"\b"
            text = re.sub(
                pattern, f" {expansion} ", text, flags=re.IGNORECASE
            )
        # collapse any resulting multiple spaces
        return re.sub(r"\s+", " ", text).strip()

    def replace_urls(self, text: str) -> str:
        """
        Replace HTTP(S) URLs with the placeholder token URL.

        :param text: Input string.
        :type text: str
        :returns: String with URLs replaced.
        :rtype: str
        """
         # Match http(s):// URLs
        text = re.sub(
            r"http[s]?://"
            r"(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\\(\\),]"
            r"|(?:%[0-9a-fA-F][0-9a-fA-F]))+",
            " url ",
            text
        )

        # Match bare URLs: www.x.com or x.com/path (no scheme)
        text = re.sub(
            r"\b(?:www\.)\S+\.\S+|\b\w+\.\w{2,4}/\S*",
            " url ",
            text
        )
            
        return text

    def replace_mentions(self, text: str) -> str:
        """
        Replace @username mentions with the placeholder PEOPLE.

        :param text: Input string.
        :type text: str
        :returns: String with mentions replaced.
        :rtype: str
        """
        # \w : matches any alphanumeric character or underscore (john123)
        return re.sub(r"@\w+", " people ", text)

    def standardize_text(self, text: str) -> str:
        """
        Lower-case text and replace newline characters with spaces.

        :param text: Input string.
        :type text: str
        :returns: Lower-cased string with newlines replaced.
        :rtype: str
        """
        return text.lower().replace("\n", " ").replace("\r", " ")

    def add_spaces_after_punctuation(self, text: str) -> str:
        """
        Ensure a space follows every period or comma to aid tokenisation.
        Ellipses (...) are left untouched.

        :param text: Input string.
        :type text: str
        :returns: Text with spaces inserted after punctuation.
        :rtype: str
        """
        text = re.sub(r"\.(?!\.)", ". ", text)
        text = re.sub(r",(?!\s)", ", ", text)
        return re.sub(r"\s+", " ", text).strip()

    def is_valid_text(self, text: str) -> bool:
        """
        Return True if text contains at least one non-punctuation
        character and is not a known placeholder string.

        :param text: String to validate.
        :type text: str
        :returns: True if the text is usable; False otherwise.
        :rtype: bool
        """
        if not isinstance(text, str):
            return False

        # Remove punctuation and check if anything remains
        stripped = text.strip()
        if stripped.lower() in set(self._PLACEHOLDERS):
            return False

        # Check if it's only punctuation marks
        no_punct = re.sub(r"[?!.\s]", "", stripped)
        return len(no_punct) > 0

    def full_preprocess(self, text: str) -> str:
        """
        Text cleaning method that combines different previously defined
        methods.

        :param text: text from a row
        :type text: str
        :return: cleaned_text after the pipeline
        :rtype: str
        """
        text = self.standardize_text(text)  # lowercase + newline handling
        text = self.remove_html(text)  # remove html tags
        text = self.remove_emojis(text)  # remove emojis
        text = self.remove_emoticons(text)  # remove emoticons
        text = self.expand_contractions(text)  # expand contractions
        text = self.replace_urls(text)  # replaces urls by URL
        text = self.replace_mentions(text)  # replaces @user by People
        text = self.remove_hashtags(text)  # remove hashtags
        text = self.add_spaces_after_punctuation(text)  # add spaces

        text = text.replace("...", "<ELLIPSIS>")
        text = re.sub(r"[^\w\s!?\-<>']", "", text)  # keeps ?, ! and ...
        text = text.replace("<ELLIPSIS>", " ... ")
        text = re.sub(r"\s+", " ", text).strip()

        # remove remaining unicode
        cleaned_text = text.encode("ascii", "ignore").decode()

        return cleaned_text

    def clean_dataframe(
        self, df: pd.DataFrame, text_column: str = "text"
            ) -> pd.DataFrame:
        """
        Clean the DataFrame by removing invalid rows and preprocessing text.

        :param df: DataFrame containing the raw data.
        :type df: pandas.DataFrame
        :param text_column: Name of the column holding raw post text.
        :type text_column: str
        :returns: Cleaned DataFrame with a new ``cleaned_text`` column
        and a reset index.
        :rtype: pandas.DataFrame
        :raises ValueError: If text_column is absent from df.
        """
        if text_column not in df.columns:
            raise ValueError(
                f"Column '{text_column}' not found. "
                f"Available: {df.columns.tolist()}"
            )

        # Drop rows where the text column is NaN/None
        df = self.drop_missing(df, text_column)

        # Drop fully duplicate rows
        df = self.drop_duplicates(df)

        # Remove placeholder / empty-text rows
        df = self.drop_placeholders(df, text_column)
        
        cleaned = [
            self.full_preprocess(str(row))
            for row in tqdm(df[text_column], desc="  Cleaning text")
        ]
        
        # Create a shallow copy of the DataFrame
        df = df.copy()
        # Assign the list of preprocessed text to a new column
        df["cleaned_text"] = cleaned

        # Drop rows that are empty or punctuation-only after cleaning
        df = self.drop_invalid_cleaned(df)

        docs = list(tqdm(
            self.nlp.pipe(df["cleaned_text"], batch_size=500),
            total=len(df),
            desc="  spaCy pass",
            dynamic_ncols=True
        ))

        token_lists = [
            [
                token.lemma_.lower()
                for token in doc
                if token.is_alpha
                and token.lemma_.lower() not in self.stopwords_set
            ]
            for doc in docs
        ]
        df["tokens"] = token_lists

        df["raw_cleaned_text"] = df["cleaned_text"].copy()

        df["cleaned_text"] = [" ".join(toks) for toks in token_lists]

        self._cached_docs = docs
        self._cached_index = df.index.tolist()
        self._cached_tokens = token_lists
        return df.reset_index(drop=True)
