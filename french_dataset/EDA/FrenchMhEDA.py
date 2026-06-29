"""
mental_health_EDA.py
====================
Exploratory Data Analysis (EDA) pipeline for a French mental-health social-media dataset.

PIPELINE OVERVIEW
-----------------
  0. Config         – single place for all settings (paths, column names, visual style)
  1. PlotHelper     – handles saving figures and sanitising filenames
  2. TextCleaner    – loads raw CSV, filters to French rows, removes duplicates,
                      cleans raw text, tokenises, lemmatises, extracts features
  3. EDAAnalysis    – all ten graph groups in one class; run individually or via run_all()

COLUMNS PRODUCED BY TextCleaner.fit_transform()
------------------------------------------------
  cleaned_text      – full sentence after noise removal (keeps !, ?, -, ')
  hashtags          – list of #tags extracted before cleaning
  tokens            – lemmatised, stop-word-free list of meaningful words
  char_count        – number of characters in cleaned_text
  text_length       – number of words (space-split) in cleaned_text
  punct_count       – total count of ?, !, and ... in cleaned_text
  question_count    – count of ? marks
  exclamation_count – count of ! marks
  ellipsis_count    – count of ... sequences
  text_nostop       – tokens joined back into a single string (for NLP graphs)
  emoji_count       – emoji count from the ORIGINAL raw text
  emoticon_count    – ASCII emoticon count from the ORIGINAL raw text

WHERE EACH COLUMN IS USED
--------------------------
  cleaned_text  → char_count, text_length, all punct counts, sent_tokenize (Graph 3)
  tokens        → text_nostop; also filtered with WORDCLOUD_STOPWORDS for Graph 4
  text_nostop   → Graph 5 (co-occurrence), Graph 6 (n-grams)
  char_count    → Graph 2c (char count boxplot)
  text_length   → Graph 2a (histogram), Graph 2b (boxplot)
  punct counts  → Graph 3a (bar chart), Graph 3b (table)
  emoji_count   → Graph 10
  emoticon_count→ Graph 10

STOPWORD SETS — TWO SEPARATE LISTS
------------------------------------
  STOPWORDS           – used during text cleaning (tokenise → lemmatise → remove_stopwords)
                        keeps negations and intensity modifiers (they matter for meaning)
  WORDCLOUD_STOPWORDS – used ONLY in Graph 4 (word clouds)
                        extends STOPWORDS to also remove negations and intensity modifiers
                        because these words dominate word clouds and hide content words
"""

# ══════════════════════════════════════════════════════════════════════════════
# IMPORTS
# ══════════════════════════════════════════════════════════════════════════════

import os
import re
import string
import warnings
from collections import Counter
from itertools import combinations
from matplotlib_venn import venn2, venn3
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from wordcloud import WordCloud
import spacy
import emoji
from sklearn.feature_extraction.text import CountVectorizer

from typing import List, Tuple, Set, Dict, Optional
from nltk.tokenize import sent_tokenize

import nltk
try:
    nltk.download('punkt',     quiet=True)
    nltk.download('punkt_tab', quiet=True)
    nltk.download('wordnet',   quiet=True)
except Exception as e:
    print(f"Warning: NLTK download failed: {e}")

warnings.filterwarnings("ignore")
print("All imports OK")


# ══════════════════════════════════════════════════════════════════════════════
# 0. CONFIGURATION
# ══════════════════════════════════════════════════════════════════════════════
class Config:
    """
    Central store for every tuneable constant in the pipeline.
    One change here propagates everywhere automatically.
    """
    CSV_PATH       = r"C:\Users\Admin\Documents\FYP\french dataset\Dataset\french_data.csv"
    OUTPUT_DIR     = r"MyResults"

    LANGUAGE_COL   = "language"
    LANGUAGE_VALUE = "French"
    TEXT_COL       = "text"
    LABEL_COL      = "mental_state"

    BG      = "#FFFFFF"
    DPI     = 150
    PALETTE = "Set2"

    TOP_N_WORDS = 20
    TOP_N_COOC  = 20


cfg = Config()
os.makedirs(cfg.OUTPUT_DIR, exist_ok=True)
print(f"Config ready — output folder: '{cfg.OUTPUT_DIR}'")


# ══════════════════════════════════════════════════════════════════════════════
# 1. PLOT HELPER
# ══════════════════════════════════════════════════════════════════════════════
class PlotHelper:
    """
    Centralises all figure-saving logic.

    __init__  → applies shared visual style globally to all matplotlib figures.
    save()    → chains tight_layout → savefig → close → prints confirmation.
    safe_name → converts a label string into an OS-safe filename.
    """

    def __init__(self, cfg: Config):
        self.cfg = cfg
        plt.rcParams.update({
            "figure.facecolor" : cfg.BG,
            "axes.facecolor"   : cfg.BG,
            "axes.spines.top"  : False,
            "axes.spines.right": False,
            "font.size"        : 11,
        })

    def save(self, filename: str) -> str:
        path = os.path.join(self.cfg.OUTPUT_DIR, filename)
        plt.tight_layout()
        plt.savefig(path, dpi=self.cfg.DPI, bbox_inches="tight")
        plt.close()
        print(f"  [SAVED] {filename}")
        return path

    @staticmethod
    def safe_name(text: str) -> str:
        return re.sub(r'[\\/*?"<>|]+', "_", str(text)).strip()


helper = PlotHelper(cfg)
print("PlotHelper ready")


# ══════════════════════════════════════════════════════════════════════════════
# STOPWORD DEFINITIONS  (module-level — shared by TextCleaner and EDAAnalysis)
# ══════════════════════════════════════════════════════════════════════════════

PRONOUNS = {
    "je", "j", "tu", "il", "elle", "nous", "vous", "ils", "elles", "on",
    "me", "moi", "te", "toi", "se",
    "mon", "ma", "mes", "ton", "ta", "tes", "son", "sa", "ses",
    "notre", "nos", "votre", "vos", "leur", "leurs"
}

EXTRA_REMOVE = {
    "le", "la", "les", "un", "une", "des", "du", "au", "aux",
    "de", "à", "en", "dans", "sur", "avec", "pour", "par", "sans", "chez",
    "et", "ou", "mais", "donc", "or", "ni", "car",
    "que", "qui", "quand", "lorsque", "comme", "puisque",
    "quoique", "quoi", "si", "afin", "bien", "pendant",
    "avant", "après", "depuis", "jusqu", "malgré",
    "chaque", "tous", "toutes", "tout", "toute",
    "ce", "cet", "cette", "ces",
    "toujours", "parfois","autre","autres", "même", "mêmes",
    # Demonstrative pronouns
    "cela", "ceci",
    "celui", "celle", "ceux", "celles",
    "celui-ci", "celui-là",
    "celle-ci", "celle-là",
    "ceux-ci", "ceux-là",
    "celles-ci", "celles-là",

}

NOISE = {"j", "m", "n", "s", "t", "quelqu", "aujourd", "hui", "pa"}

REMOVE_VERBS = {
    "être", "avoir",
    "suis", "es", "est", "sommes", "êtes", "sont",
    "étais", "était", "étions", "étiez", "étaient",
    "serai", "seras", "sera", "serons", "serez", "seront",
    "serais", "serait", "serions", "seriez", "seraient",
    "sois", "soit", "soyons", "soyez", "soient",
    "ai", "as", "a", "avons", "avez", "ont",
    "avais", "avait", "avions", "aviez", "avaient",
    "aurai", "auras", "aura", "aurons", "aurez", "auront",
    "aurais", "aurait", "aurions", "auriez", "auraient",
    "aie", "aies", "ait", "ayons", "ayez", "aient",
    "été", "eu",
}

KEEP_WORDS = {
    "ne", "pas", "rien", "personne", "jamais",
    "plus", "toujours", "parfois", "tellement", "trop",
    "dépression", "pensées", "vide", "douleur", "désespoir",
    "espoir", "suicidaires", "lumière", "obscurité", "âme",
    "résilience", "guérison",
}

# Primary stopwords — used during cleaning; preserves negations and intensifiers
STOPWORDS = (PRONOUNS | EXTRA_REMOVE | NOISE | REMOVE_VERBS) - KEEP_WORDS

# Word-cloud-only stopwords — also removes negations/intensifiers for visual clarity
WORDCLOUD_STOPWORDS = STOPWORDS | {
    "ne", "pas", "rien", "personne", "jamais",
    "très", "trop", "toujours", "parfois", "tellement",
    "plus", "bien", "vraiment", "encore", "déjà",
    "assez", "peu", "beaucoup", "moins", "autant",
}

# Load French spaCy model once at module level (expensive)
nlp = spacy.load("fr_core_news_sm", disable=["ner", "parser"])


# ══════════════════════════════════════════════════════════════════════════════
# 2. TEXT CLEANER
# ══════════════════════════════════════════════════════════════════════════════
class TextCleaner:
    """
    Responsible for everything that happens to the raw data before analysis:
      • Loading the CSV
      • Filtering to French rows and deduplicating
      • Cleaning text (emoji removal, URL/mention replacement, etc.)
      • Tokenisation, lemmatisation, stopword removal
      • Feature extraction (char_count, text_length, punct counts,
        emoji_count, emoticon_count)

    PUBLIC INTERFACE
    ────────────────
    cleaner = TextCleaner(cfg)
    df      = cleaner.load_and_clean()   # returns the fully enriched DataFrame

    COLUMNS ADDED BY load_and_clean()
    ──────────────────────────────────
      cleaned_text, hashtags, tokens, char_count, text_length,
      punct_count, question_count, exclamation_count, ellipsis_count,
      text_nostop, emoji_count, emoticon_count

    BRANCHES (both stem from cleaned_text)
    ───────────────────────────────────────
    Branch 1 — Statistical: cleaned_text keeps !, ? so counts are correct.
    Branch 2 — Linguistic : tokenise → lemmatise → remove_stopwords produces
                            pure content lemmas for NLP graphs.
    """

    def __init__(self, cfg: Config) -> None:
        self.cfg           = cfg
        self.nlp           = nlp
        self.stopwords_set = STOPWORDS

        self.emoji_regex = (
            r'[\U0001F600-\U0001F64F]|[\U0001F300-\U0001F5FF]|'
            r'[\U0001F680-\U0001F6FF]|[\U00002600-\U000026FF]|'
            r'[\U00002700-\U000027BF]|[\U0001F900-\U0001F9FF]|'
            r'[\U0001FA00-\U0001FA6F]|[\U0001FA70-\U0001FAFF]'
        )

        self.emoticon_patterns: List[str] = [
            r':\)|:-\)|:\]|=\]|=\)',
            r':\(|:-\(|:\[|=\[|=\(',
            r':D|:-D|=D',
            r';\)|;-\)',
            r':P|:-P|=P',
            r':o|:-o|:O|:-O',
            r':/|:-/',
            r":'\(",
            r'<3',
        ]

    # ──────────────────────────────────────────────────────────────────────────
    # STEP 1: DATA LOADING & DEDUPLICATION
    # ──────────────────────────────────────────────────────────────────────────

    def _load_raw(self) -> pd.DataFrame:
        """
        Load CSV → filter French rows → remove duplicate posts.
        Returns a clean pd.DataFrame (French-only, unique posts).
        """
        df = pd.read_csv(self.cfg.CSV_PATH, encoding="utf-8-sig")
        print(f"[TextCleaner] Total rows loaded  : {len(df)}")

        mask = (
            df[self.cfg.LANGUAGE_COL].str.strip().str.lower()
            == self.cfg.LANGUAGE_VALUE.lower()
        )
        df = df[mask].copy()

        before = len(df)
        df = df.drop_duplicates(subset=self.cfg.TEXT_COL).reset_index(drop=True)
        print(f"[TextCleaner] French rows kept   : {before}")
        print(f"[TextCleaner] After dedup        : {len(df)} ({before - len(df)} removed)")
        return df

    # ──────────────────────────────────────────────────────────────────────────
    # STEP 2: LOW-LEVEL CLEANING HELPERS
    # ──────────────────────────────────────────────────────────────────────────

    def _remove_emojis(self, text: str) -> str:
        """Two-pass emoji removal: emoji library then regex safety net."""
        text = emoji.replace_emoji(text, replace="")
        text = re.sub(self.emoji_regex, "", text)
        return text

    def _replace_urls(self, text: str) -> str:
        return re.sub(r'https?://\S+|www\.\S+', ' URL ', text)

    def _replace_mentions(self, text: str) -> str:
        return re.sub(r'@\w+', ' PEOPLE ', text)

    def _extract_hashtags(self, text: str) -> Tuple[List[str], str]:
        hashtags = re.findall(r'#\w+', text)
        text_without = re.sub(r'#\w+', '', text)
        return hashtags, text_without

    def _standardize(self, text: str) -> str:
        return text.lower().replace('\n', ' ').replace('\r', ' ')

    def _clean_text(self, text: str) -> Tuple[str, List[str]]:
        """
        Full cleaning pipeline for one post.
        Order: standardize → remove_emojis (×2) → replace URLs/mentions
               → extract hashtags → regex cleanup → collapse spaces.
        Returns (cleaned_text, hashtag_list).
        """
        text = self._standardize(text)
        text = self._remove_emojis(text)
        text = self._replace_urls(text)
        text = self._replace_mentions(text)
        hashtags, text = self._extract_hashtags(text)
        text = re.sub(r"[^\w\s!?\-']", "", text)
        text = self._remove_emojis(text)     # pass 2 safety net
        text = re.sub(r"\s+", " ", text).strip()
        return text, hashtags

    # ──────────────────────────────────────────────────────────────────────────
    # STEP 3: SPACY NLP HELPERS
    # ──────────────────────────────────────────────────────────────────────────

    def _tokenize(self, cleaned_text: str) -> List[str]:
        """Tokenise with spaCy; keep only alphabetic tokens (drops all punctuation)."""
        doc = self.nlp(cleaned_text)
        return [token.text.lower() for token in doc if token.is_alpha]

    def _lemmatize(self, tokens: List[str]) -> List[str]:
        """Lemmatise by re-joining tokens into a sentence for context."""
        doc = self.nlp(" ".join(tokens))
        return [
            token.lemma_.lower()
            for token in doc
            if token.is_alpha and token.lemma_ != ""
        ]

    def _remove_stopwords(self, tokens: List[str]) -> List[str]:
        """Filter using STOPWORDS (KEEP_WORDS are protected and will never be removed)."""
        return [t for t in tokens if t not in self.stopwords_set]

    def _preprocess(self, cleaned_text: str) -> List[str]:
        """Convenience: tokenize → lemmatize → remove_stopwords in one call."""
        return self._remove_stopwords(self._lemmatize(self._tokenize(cleaned_text)))

    # ──────────────────────────────────────────────────────────────────────────
    # PUBLIC METHOD: load raw CSV and return enriched DataFrame
    # ──────────────────────────────────────────────────────────────────────────

    def load_and_clean(self) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        Main entry point. Loads the CSV, deduplicates, cleans every row,
        and extracts all features.

        Returns
        -------
        df_raw : pd.DataFrame   — raw French rows (for emoji log at the end)
        df     : pd.DataFrame   — fully enriched, ready for EDA
        """
        df_raw = self._load_raw()
        df     = df_raw.copy()

        # ── Clean text + extract hashtags ────────────────────────────────────
        cleaned_results    = df[self.cfg.TEXT_COL].apply(self._clean_text)
        df["cleaned_text"] = cleaned_results.apply(lambda x: x[0])
        df["hashtags"]     = cleaned_results.apply(lambda x: x[1])

        # ── Diagnostics: noise elements ──────────────────────────────────────
        url_count   = df[self.cfg.TEXT_COL].apply(
            lambda x: len(re.findall(r'https?://\S+|www\.\S+', x))
        ).sum()
        posts_w_url = df[self.cfg.TEXT_COL].str.contains(r'https?://|www\.', regex=True).sum()

        mention_count   = df[self.cfg.TEXT_COL].apply(
            lambda x: len(re.findall(r'@\w+', x))
        ).sum()
        posts_w_mention = df[self.cfg.TEXT_COL].str.contains(r'@\w+', regex=True).sum()

        hashtag_count   = df["hashtags"].apply(len).sum()
        posts_w_hashtag = (df["hashtags"].apply(len) > 0).sum()

        print(f"\n── Noise Element Counts (from original text) ──")
        print(f"   URLs found      : {url_count}  (in {posts_w_url} posts)")
        print(f"   @mentions found : {mention_count}  (in {posts_w_mention} posts)"
              + ("  ← none in dataset" if mention_count == 0 else ""))
        print(f"   #hashtags found : {hashtag_count}  (in {posts_w_hashtag} posts)")

        # ── Branch 2: linguistic pipeline ────────────────────────────────────
        df["tokens"] = df["cleaned_text"].apply(self._preprocess)

        # ── Branch 1: surface-level statistics ───────────────────────────────
        df["char_count"]        = df["cleaned_text"].apply(len)
        df["text_length"]       = df["cleaned_text"].apply(lambda x: len(x.split()))
        df["punct_count"]       = df["cleaned_text"].apply(
            lambda x: x.count('?') + x.count('!') + x.count('...')
        )
        df["question_count"]    = df["cleaned_text"].apply(lambda x: x.count('?'))
        df["exclamation_count"] = df["cleaned_text"].apply(lambda x: x.count('!'))
        df["ellipsis_count"]    = df["cleaned_text"].apply(lambda x: x.count('...'))

        # ── Join tokens back to string for NLP graphs ─────────────────────────
        df["text_nostop"] = df["tokens"].apply(lambda t: " ".join(t))

        # ── Emoji / emoticon counts from ORIGINAL text ────────────────────────
        df["emoji_count"]    = df[self.cfg.TEXT_COL].apply(lambda x: len(emoji.emoji_list(x)))
        df["emoticon_count"] = df[self.cfg.TEXT_COL].apply(
            lambda x: sum(len(re.findall(p, x, re.IGNORECASE)) for p in self.emoticon_patterns)
        )

        # ── Emoji removal verification ────────────────────────────────────────
        df["emoji_count_after"] = df["cleaned_text"].apply(lambda x: len(emoji.emoji_list(x)))
        before  = df["emoji_count"].sum()
        after   = df["emoji_count_after"].sum()
        removed = before - after
        print(f"\n── Emoji Removal Summary ──")
        print(f"   Before: {before}  |  After: {after}  |  Removed: {removed}")
        print("   ✓ All emojis removed" if after == 0 else f"   ⚠ {after} emojis remain")
        df.drop(columns=["emoji_count_after"], inplace=True)

        # ── Drop rows with missing labels ─────────────────────────────────────
        missing = df[self.cfg.LABEL_COL].isna().sum()
        print(f"\n── Missing Labels: {missing} / {len(df)}")
        if missing > 0:
            df = df.dropna(subset=[self.cfg.LABEL_COL]).reset_index(drop=True)

        print(f"\n[TextCleaner] Pipeline complete — {len(df)} rows ready for EDA")
        return df_raw, df

    # ──────────────────────────────────────────────────────────────────────────
    # PUBLIC HELPERS exposed for EDAAnalysis diagnostics / final log
    # ──────────────────────────────────────────────────────────────────────────

    def tokenize_public(self, text: str) -> List[str]:
        return self._tokenize(text)

    def lemmatize_public(self, tokens: List[str]) -> List[str]:
        return self._lemmatize(tokens)


# ══════════════════════════════════════════════════════════════════════════════
# 3. EDA ANALYSIS
# ══════════════════════════════════════════════════════════════════════════════
class EDAAnalysis:
    """
    All ten graph groups in one class.

    Each graph group is a public method (graph_01, graph_02, … graph_10).
    Call them individually during development, or use run_all() to produce
    every plot in sequence.

    USAGE
    ─────
    eda = EDAAnalysis(cfg, helper, df)
    eda.run_all()                # run everything
    eda.graph_01()               # run one graph only

    GRAPH INDEX
    ───────────
    graph_01  → Label distribution (bar + pie)
    graph_02  → Text length analysis (histogram, word-count boxplot, char-count boxplot)
    graph_03  → Punctuation analysis (normalised bar chart + table)
    graph_04  → Word clouds per label
    graph_05  → Co-occurrence analysis per label
    graph_06  → Common words / bigrams / trigrams per label
    graph_07  → Category distribution (bar + pie)
    graph_08  → Category × Label heatmap
    graph_08b → Per-category healthy vs unhealthy word comparison (console print)
    graph_10  → Emoji & emoticon analysis
    """

    def __init__(self, cfg: Config, helper: PlotHelper, df: pd.DataFrame) -> None:
        self.cfg    = cfg
        self.helper = helper
        self.df     = df

    # ──────────────────────────────────────────────────────────────────────────
    # GRAPH 1 — Label Distribution
    # ──────────────────────────────────────────────────────────────────────────
    def graph_01(self) -> str:
        """
        Bar chart (absolute counts) + pie chart (proportions with n=) per label.
        Output: 01_label_distribution.png
        """
        counts = self.df[self.cfg.LABEL_COL].value_counts()

        fig, axes = plt.subplots(1, 2, figsize=(13, 5))
        fig.suptitle("Label Distribution — mental_state", fontsize=14, fontweight="bold")

        sns.barplot(x=counts.values, y=counts.index.astype(str),
                    palette=self.cfg.PALETTE, ax=axes[0])
        axes[0].set_xlabel("Count")
        axes[0].set_title("Count per Label")
        for bar, val in zip(axes[0].patches, counts.values):
            axes[0].text(bar.get_width() + 0.3,
                         bar.get_y() + bar.get_height() / 2,
                         str(val), va="center", fontsize=9)

        def make_autopct(values):
            def autopct(pct):
                count = int(round(pct * sum(values) / 100.0))
                return f"{pct:.1f}%\n(n={count})"
            return autopct

        axes[1].pie(
            counts.values, labels=counts.index,
            autopct=make_autopct(counts.values),
            colors=sns.color_palette(self.cfg.PALETTE, len(counts)),
            startangle=140,
        )
        axes[1].set_title("Proportion per Label")

        return self.helper.save("01_label_distribution.png")

    # ──────────────────────────────────────────────────────────────────────────
    # GRAPH 2 — Text Length Analysis
    # ──────────────────────────────────────────────────────────────────────────
    def graph_02(self) -> None:
        """
        2a — histogram of word count per label (with median line)
        2b — boxplot of word count per label (annotated statistics)
        2c — boxplot of character count per label (annotated statistics)
        """
        df     = self.df
        labels = df[self.cfg.LABEL_COL].unique()
        colors = sns.color_palette(self.cfg.PALETTE, len(labels))

        # ── 2a: Histogram ────────────────────────────────────────────────────
        fig, axes = plt.subplots(1, len(labels), figsize=(6 * len(labels), 5))
        fig.suptitle("Text Length Distribution by Label", fontsize=14, fontweight="bold")
        if len(labels) == 1:
            axes = [axes]

        for ax, label, color in zip(axes, labels, colors):
            subset = df[df[self.cfg.LABEL_COL] == label]["text_length"]
            counts, bins, patches = ax.hist(subset, bins=30, color=color,
                                            edgecolor="white", alpha=0.85)
            median_val = subset.median()
            ax.axvline(median_val, color="red", linestyle="--", linewidth=1.8,
                       label=f"Median = {median_val:.0f}")
            ax.legend(fontsize=9)
            for count, patch in zip(counts, patches):
                if count > 0:
                    ax.text(patch.get_x() + patch.get_width() / 2,
                            patch.get_height() + max(counts) * 0.01,
                            f"{int(count)}", ha="center", va="bottom", fontsize=8)
            ax.set_title(f"{label}", fontsize=11, fontweight="bold")
            ax.set_xlabel("Text Length")
            ax.set_ylabel("Frequency")

        plt.tight_layout()
        self.helper.save("02a_textlength_histogram_by_label.png")

        # ── Helper: annotated boxplot ─────────────────────────────────────────
        def _boxplot(col: str, title: str, ylabel: str, fname: str) -> None:
            fig, axes = plt.subplots(1, len(labels), figsize=(5 * len(labels), 5))
            fig.suptitle(title, fontsize=14, fontweight="bold")
            for ax, label, color in zip(axes, labels, colors):
                subset = df[df[self.cfg.LABEL_COL] == label][col]
                ax.boxplot(subset, patch_artist=True,
                           boxprops=dict(facecolor=color, color="gray"),
                           medianprops=dict(color="black", linewidth=2),
                           flierprops=dict(marker='o', markersize=3,
                                           markerfacecolor=color, alpha=0.4))
                mn     = subset.min();   q1 = subset.quantile(0.25)
                median = subset.median(); mean = subset.mean()
                q3     = subset.quantile(0.75); mx = subset.max()
                for val, lbl, offset in [
                    (mn,     f"Min: {mn:.0f}",       -0.32),
                    (q1,     f"Q1: {q1:.0f}",          0.32),
                    (median, f"Median: {median:.0f}",   0.32),
                    (mean,   f"Mean: {mean:.0f}",      -0.32),
                    (q3,     f"Q3: {q3:.0f}",           0.32),
                    (mx,     f"Max: {mx:.0f}",         -0.32),
                ]:
                    ax.text(1 + offset, val, lbl, ha="center", va="center", fontsize=8)
                ax.set_title(f"{ylabel} — {label}")
                ax.set_ylabel(ylabel)
                ax.set_xticks([])
            plt.tight_layout()
            self.helper.save(fname)

        _boxplot("text_length", "Text Length Boxplot by Label",
                 "Text Length", "02b_textlength_boxplot_by_label.png")
        _boxplot("char_count",  "Char Count Distribution by Label",
                 "Char Count",  "02c_charcount_boxplot_by_label.png")

    # ──────────────────────────────────────────────────────────────────────────
    # GRAPH 3 — Punctuation Analysis
    # ──────────────────────────────────────────────────────────────────────────
    def graph_03(self) -> None:
        """
        3a — normalised bar chart (avg count per sentence per label)
        3b — same data as a formatted table
        """
        df          = self.df
        punct_cols  = ['question_count', 'exclamation_count', 'ellipsis_count']
        sent_counts = df['cleaned_text'].apply(lambda t: max(len(sent_tokenize(t)), 1))

        def _build_norm_df() -> pd.DataFrame:
            summary = []
            for label in df[self.cfg.LABEL_COL].unique():
                mask      = df[self.cfg.LABEL_COL] == label
                avg_sents = sent_counts[mask].mean()
                row = {"Label": label}
                for col in punct_cols:
                    avg = df[mask][col].mean()
                    row[col] = avg / avg_sents if avg_sents > 0 else 0
                summary.append(row)
            norm = pd.DataFrame(summary).set_index("Label")
            norm.columns = ["Question", "Exclamation", "Ellipsis"]
            return norm

        norm_df = _build_norm_df()

        # ── 3a: Bar chart ─────────────────────────────────────────────────────
        fig, ax = plt.subplots(figsize=(10, 5))
        fig.suptitle("Punctuation Usage Normalized by Sentence Length",
                     fontsize=14, fontweight="bold")
        x      = np.arange(len(norm_df.columns))
        labels = norm_df.index.tolist()
        n      = len(labels)
        width  = 0.35
        colors = sns.color_palette(self.cfg.PALETTE, n)

        for i, (label, color) in enumerate(zip(labels, colors)):
            offset = (i - n / 2 + 0.5) * width
            bars   = ax.bar(x + offset, norm_df.loc[label],
                            width=width, label=label, color=color, edgecolor="white")
            for bar in bars:
                ax.text(bar.get_x() + bar.get_width() / 2,
                        bar.get_height() + 0.001,
                        f"{bar.get_height():.4f}", ha="center", va="bottom", fontsize=9)

        ax.set_xticks(x)
        ax.set_xticklabels(norm_df.columns, fontsize=11)
        ax.set_ylabel("Avg count per sentence")
        ax.set_xlabel("Punctuation type")
        ax.legend(title="Label")
        ax.set_ylim(0, norm_df.values.max() * 1.25)
        self.helper.save("03_punctuation_normalized.png")

        # ── 3b: Table ─────────────────────────────────────────────────────────
        norm_df_rounded = _build_norm_df().round(4)
        fig, ax = plt.subplots(figsize=(8, 2 + len(norm_df_rounded) * 0.6))
        ax.axis("off")
        fig.suptitle("Punctuation Usage Normalized by Sentence Length",
                     fontsize=13, fontweight="bold", y=1.02)

        colors_rows = sns.color_palette(self.cfg.PALETTE, len(norm_df_rounded))
        row_colors  = [[c] + ["#f9f9f9"] * len(norm_df_rounded.columns) for c in colors_rows]

        table = ax.table(
            cellText    = norm_df_rounded.reset_index().values,
            colLabels   = ["Label"] + list(norm_df_rounded.columns),
            cellLoc     = "center",
            loc         = "center",
            cellColours = row_colors,
        )
        table.auto_set_font_size(False)
        table.set_fontsize(11)
        table.scale(1.3, 2)
        for j in range(len(norm_df_rounded.columns) + 1):
            table[0, j].set_text_props(fontweight="bold", color="white")
            table[0, j].set_facecolor("#4C72B0")
        self.helper.save("03b_punctuation_table.png")

    # ──────────────────────────────────────────────────────────────────────────
    # GRAPH 4 — Word Clouds per Label
    # ──────────────────────────────────────────────────────────────────────────
    def graph_04(self) -> list:
        """
        One word cloud per label. Tokens filtered with WORDCLOUD_STOPWORDS to
        remove high-frequency function words that obscure content words.
        Output: 04_wordclouds_per_label.png
        """
        df     = self.df
        labels = df[self.cfg.LABEL_COL].unique()
        n      = len(labels)
        cols   = min(3, n)
        rows   = (n + cols - 1) // cols

        fig, axes = plt.subplots(rows, cols, figsize=(cols * 6, rows * 4))
        axes = np.array(axes).flatten()
        fig.suptitle("Word Clouds per Label", fontsize=15, fontweight="bold")
        cmaps = ["Blues", "Reds", "Greens", "Purples", "Oranges", "YlOrBr"]

        for i, label in enumerate(labels):
            tokens_filtered = (
                df[df[self.cfg.LABEL_COL] == label]["tokens"]
                .apply(lambda t: [w for w in t if w not in WORDCLOUD_STOPWORDS])
            )
            text = " ".join([" ".join(t) for t in tokens_filtered])
            if not text.strip():
                axes[i].axis("off")
                continue
            wc = WordCloud(width=600, height=350, background_color="white",
                           colormap=cmaps[i % len(cmaps)], max_words=100,
                           collocations=False).generate(text)
            axes[i].imshow(wc, interpolation="bilinear")
            axes[i].axis("off")
            axes[i].set_title(str(label), fontsize=12, fontweight="bold")

        for j in range(i + 1, len(axes)):
            axes[j].axis("off")

        path = self.helper.save("04_wordclouds_per_label.png")
        return [path]

    # ──────────────────────────────────────────────────────────────────────────
    # GRAPH 5 — Co-occurrence Analysis
    # ──────────────────────────────────────────────────────────────────────────
    def graph_05(self) -> list:
        """
        Top co-occurring word pairs per label (unique words per post).
        Output: 05_cooccurrence_<label>.png
        """
        df     = self.df
        paths  = []
        labels = df[self.cfg.LABEL_COL].unique()

        def _cooccurrence(texts, top_n):
            co = Counter()
            for sentence in texts:
                words = list(set(sentence.split()))
                for pair in combinations(sorted(words), 2):
                    co[pair] += 1
            return co.most_common(top_n)

        for label in labels:
            texts = df[df[self.cfg.LABEL_COL] == label]["text_nostop"]
            pairs = _cooccurrence(texts, self.cfg.TOP_N_COOC)
            if not pairs:
                continue
            pair_labels = [f"{a} & {b}" for (a, b), _ in pairs]
            counts      = [c for _, c in pairs]

            fig, ax = plt.subplots(figsize=(10, 6))
            sns.barplot(x=counts, y=pair_labels, palette="mako", ax=ax)
            ax.set_title(f"Top Word Co-occurrences — {label}",
                         fontsize=13, fontweight="bold")
            ax.set_xlabel("Co-occurrence count")
            for bar, val in zip(ax.patches, counts):
                ax.text(bar.get_width() + max(counts) * 0.01,
                        bar.get_y() + bar.get_height() / 2,
                        str(val), va="center", fontsize=9)
            ax.set_xlim(0, max(counts) * 1.12)

            fname = f"05_cooccurrence_{self.helper.safe_name(label)}.png"
            paths.append(self.helper.save(fname))

        return paths

    # ──────────────────────────────────────────────────────────────────────────
    # GRAPH 6 — Common Words / Bigrams / Trigrams
    # ──────────────────────────────────────────────────────────────────────────
    def graph_06(self) -> list:
        """
        Top N unigrams, bigrams, and trigrams per label using CountVectorizer.
        Output: 06_common_words_per_label.png, 06_bigrams_per_label.png,
                06_trigrams_per_label.png
        """
        df = self.df

        def _get_ngram_freq(texts, n):
            vectorizer = CountVectorizer(ngram_range=(n, n), min_df=2, max_df=0.95)
            X     = vectorizer.fit_transform(texts)
            freqs = X.toarray().sum(axis=0)
            result = pd.DataFrame({"ngram": vectorizer.get_feature_names_out(), "count": freqs})
            return result.sort_values("count", ascending=False).head(self.cfg.TOP_N_WORDS)

        def _plot_ngrams(n, title_prefix, filename):
            labels = df[self.cfg.LABEL_COL].unique()
            cols   = min(2, len(labels))
            rows   = (len(labels) + cols - 1) // cols
            fig, axes = plt.subplots(rows, cols, figsize=(cols * 8, rows * 5))
            axes = np.array(axes).flatten()
            fig.suptitle(f"Top {self.cfg.TOP_N_WORDS} {title_prefix} per Label",
                         fontsize=14, fontweight="bold")

            for i, label in enumerate(labels):
                texts      = df[df[self.cfg.LABEL_COL] == label]["text_nostop"].dropna().astype(str)
                top_ngrams = _get_ngram_freq(texts, n)
                if top_ngrams.empty:
                    axes[i].axis("off")
                    continue
                sns.barplot(data=top_ngrams, x="count", y="ngram",
                            palette="rocket", ax=axes[i])
                axes[i].set_title(str(label), fontsize=11, fontweight="bold")
                axes[i].set_xlabel("Frequency")
                axes[i].set_ylabel("")
                for bar, val in zip(axes[i].patches, top_ngrams["count"]):
                    axes[i].text(bar.get_width() + max(top_ngrams["count"]) * 0.01,
                                 bar.get_y() + bar.get_height() / 2,
                                 str(val), va="center", fontsize=8)
                axes[i].set_xlim(0, max(top_ngrams["count"]) * 1.12)

            for j in range(i + 1, len(axes)):
                axes[j].axis("off")
            plt.tight_layout()
            return self.helper.save(filename)

        return [
            _plot_ngrams(1, "Common Words", "06_common_words_per_label.png"),
            _plot_ngrams(2, "Bigrams",      "06_bigrams_per_label.png"),
            _plot_ngrams(3, "Trigrams",     "06_trigrams_per_label.png"),
        ]

    # ──────────────────────────────────────────────────────────────────────────
    # GRAPH 7 — Category Distribution
    # ──────────────────────────────────────────────────────────────────────────
    def graph_07(self) -> str:
        """
        Bar chart + pie chart of post counts per mental-health category.
        Output: 07_category_distribution.png
        """
        counts = self.df["category"].value_counts()

        fig, axes = plt.subplots(1, 2, figsize=(14, 5))
        fig.suptitle("Distribution by Mental Health Category", fontsize=14, fontweight="bold")

        sns.barplot(x=counts.values, y=counts.index.astype(str), palette="Set2", ax=axes[0])
        axes[0].set_xlabel("Count")
        axes[0].set_title("Count per Category")
        for bar, val in zip(axes[0].patches, counts.values):
            axes[0].text(bar.get_width() + 10, bar.get_y() + bar.get_height() / 2,
                         str(val), va="center", fontsize=9)

        def make_autopct(values):
            def autopct(pct):
                count = int(round(pct * sum(values) / 100.0))
                return f"{pct:.1f}%\n(n={count})"
            return autopct

        axes[1].pie(counts.values, labels=counts.index,
                    autopct=make_autopct(counts.values),
                    colors=sns.color_palette("Set2", len(counts)),
                    startangle=140)
        axes[1].set_title("Proportion per Category")

        return self.helper.save("07_category_distribution.png")

    # ──────────────────────────────────────────────────────────────────────────
    # GRAPH 8 — Category × Label Heatmap
    # ──────────────────────────────────────────────────────────────────────────
    def graph_08(self) -> str:
        """
        Heatmap cross-tabulation of category vs. mental_state with totals.
        Output: 08_category_label_heatmap.png
        """
        cross = pd.crosstab(self.df["category"], self.df[self.cfg.LABEL_COL])
        cross.loc["Total"] = cross.sum()
        cross["Total"]     = cross.sum(axis=1)

        fig, ax = plt.subplots(figsize=(9, 7))
        sns.heatmap(cross, annot=True, fmt="d", cmap="YlOrRd",
                    linewidths=0.5, ax=ax, annot_kws={"size": 11, "weight": "bold"})
        ax.set_title("Category × Mental State Heatmap", fontsize=13, fontweight="bold")
        ax.set_xlabel("Mental State")
        ax.set_ylabel("Category")

        return self.helper.save("08_category_label_heatmap.png")

    # ──────────────────────────────────────────────────────────────────────────
    # GRAPH 8b — Per-Category Word Comparison (console print)
    # ──────────────────────────────────────────────────────────────────────────
    def graph_08b(self) -> None:
        """
        Prints top 15 words for Healthy vs Unhealthy within each category.
        No plot is produced — output is to the console for qualitative inspection.
        """
        df = self.df
        for category in df["category"].unique():
            subset = df[df["category"] == category]
            healthy_words   = " ".join(subset[subset["mental_state"] == "Healthy"]["text_nostop"])
            unhealthy_words = " ".join(subset[subset["mental_state"] == "Unhealthy"]["text_nostop"])
            h_freq = Counter(healthy_words.split()).most_common(15)
            u_freq = Counter(unhealthy_words.split()).most_common(15)
            print(f"\n=== {category} ===")
            print("Healthy top words:",   h_freq)
            print("Unhealthy top words:", u_freq)

    # ──────────────────────────────────────────────────────────────────────────
# GRAPH 9 — Venn Diagram: Shared vs. Unique Vocabulary per Label
# ──────────────────────────────────────────────────────────────────────────
def graph_09(self) -> str:
    """
    Venn diagram showing vocabulary overlap between mental_state labels.
    Uses text_nostop (lemmatised, stop-word-free tokens) so only meaningful
    content words are compared.

    Supports exactly 2 labels (venn2) or exactly 3 labels (venn3).
    If more than 3 labels exist, the first 3 are used and a warning is printed.

    Output: 09_venn_vocabulary_overlap.png
    """
    df     = self.df
    labels = df[self.cfg.LABEL_COL].unique().tolist()

    if len(labels) > 3:
        print(f"  [graph_09] WARNING: {len(labels)} labels found — using first 3 only.")
        labels = labels[:3]

    # Build a set of unique vocabulary words per label from text_nostop
    label_word_sets = {}
    for label in labels:
        all_words = (
            df[df[self.cfg.LABEL_COL] == label]["text_nostop"]
            .dropna()
            .astype(str)
            .apply(lambda x: set(x.split()))
        )
        # Union of all per-post word sets → unique vocab for this label
        vocab = set().union(*all_words) if len(all_words) > 0 else set()
        label_word_sets[label] = vocab

    # ── Compute overlap statistics for the subtitle ───────────────────────
    sets       = list(label_word_sets.values())
    shared     = set.intersection(*sets)
    total_all  = set.union(*sets)
    pct_shared = len(shared) / len(total_all) * 100 if total_all else 0

    fig, ax = plt.subplots(figsize=(9, 7))
    fig.suptitle(
        "Vocabulary Overlap Between Mental State Labels\n"
        f"(unique lemmas from text_nostop  |  {len(shared)} shared words = {pct_shared:.1f}% of total vocabulary)",
        fontsize=13, fontweight="bold",
    )

    palette = sns.color_palette(self.cfg.PALETTE, len(labels))
    colors  = [(*c, 0.55) for c in palette]   # add alpha for overlap readability

    if len(labels) == 2:
        v = venn2(
            subsets=(label_word_sets[labels[0]], label_word_sets[labels[1]]),
            set_labels=labels,
            set_colors=colors[:2],
            ax=ax,
        )
    elif len(labels) == 3:
        v = venn3(
            subsets=(
                label_word_sets[labels[0]],
                label_word_sets[labels[1]],
                label_word_sets[labels[2]],
            ),
            set_labels=labels,
            set_colors=colors[:3],
            ax=ax,
        )

    # ── Annotate each region with its count ──────────────────────────────
    if v:
        for patch_id in v.id2label:
            lbl = v.get_label_by_id(patch_id)
            if lbl and lbl.get_text():
                try:
                    count = int(lbl.get_text())
                    lbl.set_text(f"{count:,}")
                    lbl.set_fontsize(12)
                    lbl.set_fontweight("bold")
                except ValueError:
                    pass  # skip non-numeric labels (set name labels)

    # ── Console summary ───────────────────────────────────────────────────
    print(f"\n── Vocabulary Overlap Summary ──")
    for label, words in label_word_sets.items():
        unique = words - set.union(*[v for k, v in label_word_sets.items() if k != label])
        print(f"   {label:<15}: {len(words):>6,} total words  |  {len(unique):>6,} exclusive")
    print(f"   {'Shared':<15}: {len(shared):>6,} words  ({pct_shared:.1f}% of full vocabulary)")

    return self.helper.save("09_venn_vocabulary_overlap.png")

    # ──────────────────────────────────────────────────────────────────────────
    # GRAPH 10 — Emoji & Emoticon Analysis
    # ──────────────────────────────────────────────────────────────────────────
    def graph_10(self) -> str:
        """
        Distribution histograms, pie chart (emoji presence), and per-label
        averages for emojis and emoticons.
        ⚠ <1% of posts contain emojis — results are noted as not statistically
        meaningful in the plot title.
        Output: 10_emoji_emoticon.png
        """
        df  = self.df
        fig, axes = plt.subplots(2, 3, figsize=(18, 10))
        fig.suptitle(
            "Emoji & Emoticon Analysis\n"
            "⚠️ <1% of posts contain emojis — results not statistically meaningful",
            fontsize=13, fontweight="bold",
        )

        for col, ax, color, title, xlabel in [
            ("emoji_count",    axes[0, 0], "#F4A460", "Emoji Count Distribution",    "Emojis per text"),
            ("emoticon_count", axes[0, 1], "#87CEEB", "Emoticon Count Distribution", "Emoticons per text"),
        ]:
            ax.hist(df[col], bins=20, color=color, edgecolor="white")
            median = df[col].median()
            ax.axvline(median, color="navy", linestyle=":",  label=f"Median={median:.4f}")
            ax.set_title(title)
            ax.set_xlabel(xlabel)
            ax.legend()
            for patch in ax.patches:
                h = patch.get_height()
                if h > 0:
                    ax.text(patch.get_x() + patch.get_width() / 2,
                            h + 0.5, str(int(h)), ha="center", va="bottom", fontsize=8)

        ax_pie         = axes[0, 2]
        emoji_presence = (df["emoji_count"] > 0).value_counts()
        no_emoji       = emoji_presence.get(False, 0)
        has_emoji      = emoji_presence.get(True,  0)
        ax_pie.pie([no_emoji, has_emoji],
                   labels=[f"No Emoji\n(n={no_emoji})", f"Contains Emoji\n(n={has_emoji})"],
                   autopct="%1.2f%%", colors=["#d3d3d3", "#F4A460"], startangle=90)
        ax_pie.set_title("Emoji Presence Distribution")

        for col, ax, title in [
            ("emoji_count",    axes[1, 0], "Avg Emoji Count by Label"),
            ("emoticon_count", axes[1, 1], "Avg Emoticon Count by Label"),
        ]:
            avg = df.groupby(self.cfg.LABEL_COL)[col].mean().sort_values(ascending=False)
            sns.barplot(x=avg.values, y=avg.index.astype(str), palette=self.cfg.PALETTE, ax=ax)
            ax.set_title(title)
            ax.set_xlabel("Avg count")
            for bar, val in zip(ax.patches, avg.values):
                ax.text(bar.get_width() + avg.values.max() * 0.01,
                        bar.get_y() + bar.get_height() / 2,
                        f"{val:.4f}", va="center", fontsize=9)
            ax.set_xlim(0, avg.values.max() * 1.15)

        axes[1, 2].axis("off")
        plt.tight_layout()
        return self.helper.save("10_emoji_emoticon.png")

    # ──────────────────────────────────────────────────────────────────────────
    # run_all — convenience method
    # ──────────────────────────────────────────────────────────────────────────
    def run_all(self) -> None:
        """Run every graph method in sequence."""
        print("\n── Running all EDA graphs ──")
        self.graph_01()
        self.graph_02()
        self.graph_03()
        self.graph_04()
        self.graph_05()
        self.graph_06()
        self.graph_07()
        self.graph_08()
        self.graph_08b()
        self.graph_10()
        print("\n── All EDA graphs complete ──")


# ══════════════════════════════════════════════════════════════════════════════
# 4. MAIN EXECUTION
# ══════════════════════════════════════════════════════════════════════════════

# ── Step 1: clean data ────────────────────────────────────────────────────────
cleaner        = TextCleaner(cfg)
df_raw, df     = cleaner.load_and_clean()

# ── Step 2: emoji / emoticon diagnostics ──────────────────────────────────────
print("\n── Emoji & Emoticon Diagnostics ──")
print(df["emoji_count"].value_counts())
print(df["emoticon_count"].value_counts())
print(f"Posts with any emoji:    {(df['emoji_count'] > 0).sum()}")
print(f"Posts with any emoticon: {(df['emoticon_count'] > 0).sum()}")

# ── Step 3: run all EDA graphs ────────────────────────────────────────────────
eda = EDAAnalysis(cfg, helper, df)
eda.run_all()

# ── Step 4: save cleaned CSV ──────────────────────────────────────────────────
df.to_csv(f"{cfg.OUTPUT_DIR}/french_cleaned.csv", index=False, encoding="utf-8-sig")

# ── Step 5: emoji + hashtag log ───────────────────────────────────────────────
all_emojis = []
for text in df_raw[cfg.TEXT_COL]:
    all_emojis.extend([item['emoji'] for item in emoji.emoji_list(str(text))])

emoji_counts   = Counter(all_emojis)
emoji_txt_path = os.path.join(cfg.OUTPUT_DIR, "emojis_hashtags_found.txt")

with open(emoji_txt_path, "w", encoding="utf-8") as f:
    f.write(f"Emojis found in ORIGINAL dataset ({len(emoji_counts)} unique)\n")
    f.write("=" * 45 + "\n")
    for em, count in emoji_counts.most_common():
        f.write(f"  {em}  →  {count} times\n")

all_hashtags = Counter([tag for tags in df["hashtags"] for tag in tags])

with open(emoji_txt_path, "a", encoding="utf-8") as f:
    f.write(f"\n\nHashtags found in dataset ({len(all_hashtags)} unique)\n")
    f.write("=" * 45 + "\n")
    for tag, count in all_hashtags.most_common():
        f.write(f"  {tag}  →  {count} times\n")

print(f"\n── Hashtags found ({len(all_hashtags)} unique) ──")
for tag, count in all_hashtags.most_common():
    print(f"   {tag}  →  {count} times")

print(f"\n── Emojis found in ORIGINAL dataset ({len(emoji_counts)} unique) ──")
for em, count in emoji_counts.most_common():
    print(f"   {em}  →  {count} times")

# ── Step 6: lemmatisation examples ───────────────────────────────────────────
print("\n── Lemmatisation Examples from Dataset ──\n")
for i in range(3):
    raw        = df_raw[cfg.TEXT_COL].iloc[i]
    cleaned    = df["cleaned_text"].iloc[i]
    tokens_raw = cleaner.tokenize_public(cleaned)
    lemmas     = cleaner.lemmatize_public(tokens_raw)
    final      = df["tokens"].iloc[i]
    print(f"POST {i+1}")
    print(f"  Original : {raw[:120]}")
    print(f"  Cleaned  : {cleaned[:120]}")
    print(f"  Tokens   : {tokens_raw[:10]}")
    print(f"  Lemmas   : {lemmas[:10]}")
    print(f"  Final    : {final[:10]}")
    print()

# ── Step 7: final summary ─────────────────────────────────────────────────────
print("=" * 55)
print("  FRENCH EDA PIPELINE — COMPLETE")
print("=" * 55)

saved = [f for f in os.listdir(cfg.OUTPUT_DIR) if f.endswith(".png")]
print(f"\n📊 {len(saved)} plots saved to '{cfg.OUTPUT_DIR}/':")
for f in sorted(saved):
    print(f"   • {f}")

print(f"\n📄 Cleaned CSV : {cfg.OUTPUT_DIR}/french_cleaned.csv")
print(f"📄 Emoji log   : {emoji_txt_path}")
print("\n✅ All done!")

