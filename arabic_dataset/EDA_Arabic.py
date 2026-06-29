from __future__ import annotations

# Standard-library imports used for math, file paths, regex cleanup, and label wrapping.
import math
import os
import re
import textwrap
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Set, Tuple

# Third-party libraries used for EDA tables and chart generation.
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import pandas as pd

# Optional Arabic display helpers. The script still runs if these libraries are missing.
try:
    import arabic_reshaper  # type: ignore
    from bidi.algorithm import get_display  # type: ignore
    ARABIC_OK = True
except Exception:
    ARABIC_OK = False


# Central configuration: changing these values controls the whole pipeline without editing logic.
@dataclass
class EDAConfig:
    """Configuration container for the Arabic mental health EDA pipeline.

    Attributes:
        csv_path: Path to the source CSV file.
        output_dir: Root directory where charts, tables, and reports are saved.
        text_column: Name of the column containing the main Arabic question text.
        label_column: Name of the column containing the hierarchical diagnosis label.
        title_column: Optional question title column used for future extensions.
        answer_column: Optional answer text column.
        doctor_column: Optional doctor name column.
        consultation_column: Optional consultation identifier column.
        date_column: Optional answer date column.
        top_n_words: Maximum number of frequent words to export per umbrella.
        top_n_sublabels: Maximum number of sublabels to visualize per umbrella.
        top_n_common_sublabels: Legacy setting kept for compatibility; no longer used by the all-sublabel chart.
        bottom_n_rare_sublabels: Legacy setting kept for compatibility; no longer used by the all-sublabel chart.
        top_n_ngrams: Maximum number of n-grams to export per umbrella.
        min_token_length: Minimum token length accepted after normalization.
        min_class_size: Minimum class size required to keep an umbrella category.
        keep_other_class: Whether to retain the other or unclear umbrella.
        normalize_definite_article_for_analysis: Whether to normalize the Arabic definite article.
        enable_clitic_cleanup: Whether to apply conservative clitic cleanup.
        normalize_repeated_chars: Whether to reduce excessive Arabic character elongation.
        max_repeated_chars: Maximum number of repeated Arabic characters to preserve.
        preserve_negation_terms: Whether to keep Arabic negation terms out of the stopword list.
        enable_question_pov_analysis: Whether to classify questions by patient/doctor/mixed/unclear POV.
        enable_short_text_analysis: Whether to export short-text token-count inspection tables.
        enable_experimental_stemming_audit: Whether to export a safe stemming audit without changing the main tokens.
        figsize_wide: Default figure size for wider plots.
        figsize_tall: Default figure size for taller plots.
        font_family: Matplotlib font family used for rendering text.
    }""".replace("}", "")
    csv_path: str
    output_dir: str = "umbrella_eda_output_report_outputs_v6"

    text_column: str = "Question"
    label_column: str = "Hierarchical Diagnosis"
    title_column: str = "Question Title"
    answer_column: str = "Answer"
    doctor_column: str = "Doctor Name"
    consultation_column: str = "Consultation Number"
    date_column: str = "Date of Answer"

    top_n_words: int = 12
    top_n_sublabels: int = 8
    top_n_common_sublabels: int = 3
    bottom_n_rare_sublabels: int = 2
    top_n_ngrams: int = 12

    min_token_length: int = 2
    min_class_size: int = 1
    keep_other_class: bool = True

    normalize_definite_article_for_analysis: bool = True
    enable_clitic_cleanup: bool = True

    # Added after comparison with Elie's cleaning strategy and last year's report.
    normalize_repeated_chars: bool = True
    max_repeated_chars: int = 2
    preserve_negation_terms: bool = True
    enable_question_pov_analysis: bool = True
    enable_short_text_analysis: bool = True
    enable_experimental_stemming_audit: bool = True

    figsize_wide: Tuple[int, int] = (14, 8)
    figsize_tall: Tuple[int, int] = (12, 8)

    font_family: str = "DejaVu Sans"


class ArabicMentalHealthUmbrellaEDA:
    # The class keeps all EDA logic together: loading, cleaning, mapping, charts, and exports.
    def __init__(self, config: EDAConfig) -> None:
        """Initialize the EDA pipeline and prepare runtime resources.

        Args:
            config: Fully populated configuration object controlling input paths,
                preprocessing options, and output settings.
        """
        self.config = config

        self.df: Optional[pd.DataFrame] = None
        self.raw_df: Optional[pd.DataFrame] = None

        self._ensure_output_dirs()
        self._setup_patterns()
        self._setup_display_names()
        self._setup_normalization_resources()
        self._setup_stopwords()

        plt.rcParams["figure.dpi"] = 120
        plt.rcParams["savefig.dpi"] = 200
        plt.rcParams["font.family"] = self.config.font_family
        plt.rcParams["axes.unicode_minus"] = False

    def _ensure_output_dirs(self) -> None:
        """Create the output directory structure used by the pipeline.

        The method prepares dedicated folders for charts, tables, and text reports
        so every downstream export step can write files safely.
        """
        self.charts_dir = os.path.join(self.config.output_dir, "charts")
        self.tables_dir = os.path.join(self.config.output_dir, "tables")
        self.reports_dir = os.path.join(self.config.output_dir, "reports")
        self.modeling_dir = os.path.join(self.config.output_dir, "modeling")

        os.makedirs(self.config.output_dir, exist_ok=True)
        os.makedirs(self.charts_dir, exist_ok=True)
        os.makedirs(self.tables_dir, exist_ok=True)
        os.makedirs(self.reports_dir, exist_ok=True)
        os.makedirs(self.modeling_dir, exist_ok=True)

    def _setup_patterns(self) -> None:
        """Compile reusable regular expression patterns for preprocessing.

        The patterns cover structural cleanup, Arabic token extraction, punctuation,
        emoji and emoticon detection, time/date handling, and utility normalization rules.
        """
        self.html_pattern = re.compile(r"<[^>]+>")
        self.url_pattern = re.compile(r"http\S+|www\S+|https\S+", re.IGNORECASE)
        self.email_pattern = re.compile(r"\b[\w\.-]+@[\w\.-]+\.\w+\b")
        self.phone_pattern = re.compile(r"(\+?\d[\d\-\s]{7,}\d)")
        self.money_pattern = re.compile(
            r"(\d+(?:[\.,]\d+)?)\s*(دولار|ريال|درهم|دينار|جنيه|€|\$|£|ليرة)"
        )
        self.date_like_pattern = re.compile(r"\b\d{1,2}[\/\-\._]\d{1,2}[\/\-\._]\d{2,4}\b")
        self.time_like_pattern = re.compile(r"\b\d{1,2}\s*(?::|h|H|س|ساعة)\s*\d{0,2}\b")

        self.latin_digits_pattern = re.compile(r"[A-Za-z0-9]")
        self.arabic_diacritics_pattern = re.compile(r"[\u0617-\u061A\u064B-\u0652\u0670]")
        self.tatweel_pattern = re.compile(r"ـ+")
        self.non_arabic_pattern = re.compile(r"[^\u0600-\u06FF\s]")
        self.multi_space_pattern = re.compile(r"\s+")
        self.repeated_punct_pattern = re.compile(r"([!؟?\.،,؛;:])\1+")

        self.arabic_token_pattern = re.compile(r"[ء-ي]+")
        self.punct_pattern = re.compile(r"[،؟.!,:؛\"'“”‘’()\[\]{}<>/\\\-_=+*…]+")
        self.repeated_arabic_char_pattern = re.compile(r"([\u0600-\u06FF])\1{2,}")

        self.emoji_pattern = re.compile(
            "["
            "\U0001F600-\U0001F64F"
            "\U0001F300-\U0001F5FF"
            "\U0001F680-\U0001F6FF"
            "\U0001F1E0-\U0001F1FF"
            "\U00002700-\U000027BF"
            "\U000024C2-\U0001F251"
            "]+",
            flags=re.UNICODE,
        )

        self.emoticon_pattern = re.compile(
            r"(:-\)|:\)|:-\(|:\(|;-\)|;\)|:D|XD|<3|:\'\(|:\'\)|:-D)"
        )

    def _setup_display_names(self) -> None:
        """Define human readable labels and fixed chart colors for umbrella labels."""
        self.umbrella_display: Dict[str, str] = {
            "depression": "الاكتئاب (Depression)",
            "anxiety_fear": "القلق والمخاوف (Anxiety/Fear)",
            "ocd_obsessive": "الوساوس (OCD/Obsessive)",
            "bipolar_mania": "ثنائي القطب والهوس (Bipolar/Mania)",
            "suicidality_selfharm": "الانتحارية وإيذاء النفس (Suicidality/Self-Harm)",
            "attention_hyperactivity": "فرط الحركة وتشتت الانتباه (ADHD/Attention)",
            "normal_reassurance": "سوي / طمأنة (Normal/Reassurance)",
            "other_unclear": "غير واضح (Other/Unclear)",
        }

        # English-only names used in clean legends and report-facing chart keys.
        self.umbrella_display_en: Dict[str, str] = {
            "depression": "Depression",
            "anxiety_fear": "Anxiety/Fear",
            "ocd_obsessive": "OCD/Obsessive",
            "bipolar_mania": "Bipolar/Mania",
            "suicidality_selfharm": "Suicidality/Self-Harm",
            "attention_hyperactivity": "ADHD/Attention",
            "normal_reassurance": "Normal/Reassurance",
            "other_unclear": "Other/Unclear",
        }

        # Juliet-style soft palette. Matching labels reuse the same visual role:
        # Anxiety/Fear is coral/red, Depression is purple/blue.
        self.umbrella_colors: Dict[str, str] = {
            "anxiety_fear": "#F06C64",
            "depression": "#6F6CC3",
            "suicidality_selfharm": "#45B39D",
            "normal_reassurance": "#F4C542",
            "ocd_obsessive": "#F39C5A",
            "bipolar_mania": "#8E6BBE",
            "attention_hyperactivity": "#5DADE2",
            "other_unclear": "#95A5A6",
        }

    def _setup_normalization_resources(self) -> None:
        """Prepare helper resources used during token normalization.

        This includes words whose definite article should be preserved, discourse
        bases used by conservative clitic cleanup, and negation terms preserved
        after comparison with previous cleaning strategies.
        """
        self.article_keep_words: Set[str] = {
            "الله", "الهي", "الهيه", "الرحمن", "الرحيم",
            "الان", "الانسان", "الناس", "الذي", "التي", "الذين",
            "اللاتي", "اللاتى", "اللهم", "القران", "الحديث",
        }

        # Added after cleaning review: negation is meaningful in mental-health text.
        raw_negation_terms = {
            "لا", "لم", "لن", "ليس", "ليست", "لست", "لسنا",
            "بدون", "بلا", "مش", "مو", "مافي", "ما في"
        }
        self.negation_terms = {
            self._simple_normalize_token(x) for x in raw_negation_terms if x
        }

        raw_discourse_bases = {
            "شكرا", "شكرًا", "الحمد", "الله", "السلام", "عليكم",
            "رحمه", "رحمة", "بركاته", "جزاكم", "جزاك", "ارجو",
            "أرجو", "افيدوني", "أفيدوني", "دكتور", "الدكتور",
            "طبيب", "الطبيب", "سؤالي", "السؤال", "تشخيصكم",
            "حالتي", "لو", "سمحتم", "خير", "الشكر", "جزيلا",
        }
        self.discourse_bases = {
            self._simple_normalize_token(x) for x in raw_discourse_bases if x
        }

    def _setup_stopwords(self) -> None:
        """Build stopword and stopphrase resources used in text filtering.

        The stopword inventory combines general Arabic stopwords, consultation style
        filler terms, hierarchy specific junk, and dataset specific noise. Negation
        terms are optionally preserved because they can reverse clinical meaning.
        """
        general_stopwords = {
            "في", "من", "على", "إلى", "الى", "عن", "ما", "ماذا", "هذا", "هذه",
            "ذلك", "تلك", "هو", "هي", "هم", "هن", "أنا", "انا", "نحن", "كان",
            "كانت", "يكون", "يمكن", "لقد", "قد", "ثم", "أو", "او", "و", "ف", "ب",
            "ل", "ك", "مع", "عند", "بعد", "قبل", "كل", "أي", "اي", "أحد", "احد",
            "هناك", "هنا", "لكن", "ولكن", "لأن", "لان", "إن", "ان", "إذا", "اذا",
            "حتى", "حتي", "بين", "أكثر", "اكثر", "أقل", "اقل", "أيضًا", "ايضا",
            "أحيانًا", "احيانا", "ولا", "لا", "لم", "لن", "لي", "لدي", "عندي",
            "عنده", "عندها", "الذي", "التي", "الذين", "اللاتي", "اللاتى",
            "شيء", "بعض", "أمر", "أمور", "كنت", "وأنا", "وانا",
            "أنني", "انني", "أنها", "انها", "أنه", "انه", "اني", "هل",
            "عندما", "منذ", "الآن", "الان", "الا", "إلا", "حول", "حيث", "بسبب",
            "جدا", "جدًا", "كما", "تم", "علي", "عليه", "عليها",
            "اشعر", "أشعر", "واشعر", "اعاني", "أعاني", "احس", "أحس",
            "اريد", "أريد", "استطيع", "أستطيع", "اعرف", "أعرف",
            "امي", "أمي", "ابي", "أبي",
            "الي", "وفي", "وهل", "فهل", "وما", "فانا", "بان", "ام", "مره",
            "غير", "الحاله", "الحالة", "حالتي", "حالاتي", "وعدم", "كثيرا",
            "سنه", "سنة", "لمده", "مدة", "لله", "ورحمه", "ورحمة",
            "وقد", "فقد", "وكان", "وكانت", "ايضا", "ايضاً",
        }

        consultation_filler_stopwords = {
            "السلام", "عليكم", "ورحمة", "وبركاته", "جزاكم", "خيرا", "خيرًا",
            "شكرا", "شكرًا", "أرجو", "ارجو", "افيدوني", "أفيدوني", "سؤالي",
            "السؤال", "الإجابة", "الاخ", "الأخ", "الاخت", "الأخت", "حفظه",
            "حفظها", "بارك", "بكم", "نسأل", "تعالى", "وفقكم", "الله", "بسم",
            "الرحمن", "الرحيم", "وبعد", "الفاضل", "الفاضلة", "كريم", "كريمة",
            "استشارتي", "استشارة", "جواب", "سؤالك", "رسالتك", "الموقع", "ويب",
            "اسلام", "إسلام", "المكرم", "المكرمة", "الدكتور", "الدكتورة",
            "استفسار", "استفساري", "ارجوكم", "لو", "سمحتم", "جزاك", "الرجاء",
            "أرجوكم", "حضرتك", "ممكن", "ممکن", "طبيب", "الطبيب", "دكتور",
            "علما", "علماً", "اعلم", "جزيل", "الشكر", "خير", "الجزاء",
            "صاحب", "الاستشاره", "الاستشارة", "رقم", "الرسول", "صلي", "وسلم",
            "الحمد",
        }

        hierarchy_junk = {
            "الحالات", "النفسيه", "العصبيه", "العصبية",
            "عموما", "عموماً"
        }

        dataset_specific_noise = {
            "محمد", "عبد", "العليم", "عمري", "العمر", "ابلغ",
            "اصبحت", "بدات", "بدأت", "ذهبت", "افكر",
        }

        raw_stopwords = (
            general_stopwords
            | consultation_filler_stopwords
            | hierarchy_junk
            | dataset_specific_noise
        )

        self.stopwords = {
            self.canonicalize_token(word)
            for word in raw_stopwords
            if word and self.canonicalize_token(word)
        }

        if self.config.preserve_negation_terms:
            self.stopwords = self.stopwords - self.negation_terms

        raw_ngram_stopphrases = {
            # Greetings / openings
            "السلام عليكم",
            "وعليكم السلام",
            "السلام عليكم ورحمه",
            "السلام عليكم ورحمة",
            "ورحمة الله وبركاته",
            "بسم الله الرحمن",
            "بسم الله الرحمن الرحيم",
            "في البدايه اود",
            "في البداية أود",
            "البدايه اود اشكركم",
            "البداية أود أشكركم",

            # Gratitude / blessings
            "شكرا لكم",
            "وشكرا لكم",
            "وشكرا جزيلا لكم",
            "جزاكم الله خيرا",
            "جزاكم الله خير",
            "جزاك الله خيرا",
            "جزاك الله خير",
            "جزيل الشكر",
            "ولكم جزيل الشكر",
            "جعله ميزان حسناتكم",
            "وجعله ميزان حسناتكم",
            "ميزان حسناتكم",
            "بارك الله فيكم",
            "عز وجل",

            # Repeated religious/discourse phrases
            "الحمد لله",
            "والحمد لله",
            "والله الحمد",
            "سليمه والحمد لله",
            "سليمه والله الحمد",
            "وكانت النتيجه سليمه",
            "الصلاه وقراءه القران",
            "الصلاة وقراءة القران",
            "قراءه القران",
            "قراءة القرآن",

            # Forum / request style phrases
            "ما تشخيصكم لحالتي",
            "ماذا تشخصكم لحالتي",
            "فما تشخيصكم لحالتي",
            "فما تشخصيكم لحالتي",
            "ارجو الرد",
            "أرجو الرد",
            "ارجو الافاده",
            "أرجو الإفادة",
            "افيدوني جزاكم الله",
            "أفيدوني جزاكم الله",
            "لو سمحتم",
            "يا دكتور",
            "الى طبيب",
            "الي طبيب",

            # Age / template phrasing
            "عمري سنه",
            "عمري سنة",
            "العمر سنه",
            "العمر سنة",
            "شاب بعمر سنه",
            "شاب بعمر سنة",
            "فتاه بعمر سنه",
            "فتاه بعمر سنة",
            "فتاة بعمر سنة",
            "شاب سنه",
            "شاب سنة",
            "فتاه سنه",
            "فتاه سنة",
            "بعمر سنه",
            "بعمر سنة",

            # Generic dosage / template junk
            "نصف حبه",
            "نصف حبة",
            "حبه يوميا",
            "حبة يوميا",
            "مرتين يوميا",
            "مرة يوميا",
        }

        self.ngram_stopphrases = {
            self.normalize_phrase_for_matching(phrase)
            for phrase in raw_ngram_stopphrases
            if phrase
        }

    @staticmethod
    def safe_filename(name: str) -> str:
        """Convert arbitrary text into a filesystem safe filename.

        Args:
            name: Raw filename candidate.

        Returns:
            Sanitized filename string with reserved characters removed or replaced.
        """
        name = str(name).strip()
        name = re.sub(r'[\\/*?:"<>|]', "_", name)
        name = re.sub(r"\s+", "_", name)
        return name[:180]

    @staticmethod
    def _contains_arabic(text: str) -> bool:
        """Return whether the input text contains Arabic script characters."""
        return bool(re.search(r"[\u0600-\u06FF]", str(text)))

    def shape_arabic_text(self, text: str) -> str:
        """Apply Arabic reshaping and bidi display fixes when libraries exist.

        Args:
            text: Raw text to display.

        Returns:
            Display ready Arabic text when shaping support is available, otherwise
            the original text.
        """
        if text is None:
            return ""
        text = str(text)

        if ARABIC_OK and self._contains_arabic(text):
            try:
                reshaped = arabic_reshaper.reshape(text)
                return get_display(reshaped)
            except Exception:
                return text

        return text

    def wrap_arabic_label(self, text: str, width: int = 28) -> str:
        """Wrap and shape Arabic labels for cleaner chart rendering.

        Args:
            text: Label text to wrap.
            width: Maximum line width used before wrapping.

        Returns:
            Wrapped and display shaped label string.
        """
        wrapped = "\n".join(
            textwrap.wrap(str(text), width=width, break_long_words=False)
        )
        return self.shape_arabic_text(wrapped)

    def get_umbrella_english_name(self, umbrella: str) -> str:
        """Return the English-only report name for an umbrella id."""
        return self.umbrella_display_en.get(umbrella, str(umbrella).replace("_", " ").title())

    def get_umbrella_color(self, umbrella: str) -> str:
        """Return the fixed Juliet-style color for an umbrella id."""
        return self.umbrella_colors.get(umbrella, "#7F8C8D")

    def get_report_chart_title(self, title: str = "", umbrella: Optional[str] = None) -> str:
        """Return the restricted report title required for every chart.

        Chart titles are intentionally limited to either:
        - Arabic Dataset
        - Arabic Dataset - <Umbrella Name>

        The method detects umbrella-specific charts from the provided umbrella id
        when available, otherwise from the title text passed by the caller.
        """
        if umbrella:
            return f"Arabic Dataset - {self.get_umbrella_english_name(umbrella)}"

        title_text = str(title or "")
        for key, english_name in self.umbrella_display_en.items():
            arabic_name = self.umbrella_display.get(key, "")
            if key in title_text or english_name in title_text or arabic_name in title_text:
                return f"Arabic Dataset - {english_name}"

        return "Arabic Dataset"

    def dataset_chart_colors(self, n: int) -> List[str]:
        """Return non-blue Juliet-style colors for generic dataset-level charts."""
        if n <= 0:
            return []

        base = [
            self.umbrella_colors["anxiety_fear"],
            self.umbrella_colors["depression"],
            self.umbrella_colors["ocd_obsessive"],
            self.umbrella_colors["suicidality_selfharm"],
            self.umbrella_colors["bipolar_mania"],
            self.umbrella_colors["normal_reassurance"],
            self.umbrella_colors["attention_hyperactivity"],
            self.umbrella_colors["other_unclear"],
        ]
        return [base[i % len(base)] for i in range(n)]


    def juliet_gradient_colors(self, n: int) -> List:
        """Return a soft sequential color list for generic ranked bar charts."""
        if n <= 0:
            return []
        cmap = plt.colormaps.get_cmap("YlOrRd")
        if n == 1:
            return [cmap(0.75)]
        return [cmap(0.25 + 0.65 * (i / (n - 1))) for i in range(n)]

    def umbrella_gradient_colors(self, umbrella: str, values: Sequence[float]) -> List[str]:
        """Return a clearer Juliet-style palette tied to the umbrella color family.

        This version uses a stronger hand-picked light-to-dark palette for each
        umbrella, so the n-gram bars visibly look like a true gradient while
        still matching the pie-chart color family.
        """
        values_list = [float(v) for v in values]
        if not values_list:
            return []

        palette_map: Dict[str, List[str]] = {
            "anxiety_fear": ["#FDE5E1", "#FCC9BE", "#F9A78F", "#F57F6C", "#F06C64", "#D94B45", "#B93A36"],
            "ocd_obsessive": ["#FDEAD8", "#FCD5B5", "#F8BE8D", "#F4A865", "#F39C5A", "#E07E3C", "#C96A2E"],
            "depression": ["#E7E5F8", "#CDC8F2", "#ACA4E6", "#8F84D9", "#6F6CC3", "#554FAD", "#3E388F"],
            "bipolar_mania": ["#F0E8F8", "#DDCDEF", "#C6ACE3", "#AE8DD6", "#8E6BBE", "#744FA6", "#5C388A"],
            "other_unclear": ["#F2F4F4", "#E3E7E8", "#D2D7D8", "#BCC5C7", "#95A5A6", "#7C8C8D", "#657172"],
            "suicidality_selfharm": ["#DFF5EF", "#BEEADE", "#92DDCB", "#66D0B8", "#45B39D", "#2C9680", "#1E7765"],
            "normal_reassurance": ["#FFF4CF", "#FEE7A0", "#FCD972", "#F8CC52", "#F4C542", "#D8A82C", "#B58916"],
            "attention_hyperactivity": ["#E2F1FC", "#C5E4FA", "#9FD0F6", "#78BCEE", "#5DADE2", "#3F91C8", "#2C76AE"],
        }

        palette = palette_map.get(umbrella)
        if palette is None:
            base_rgb = mcolors.to_rgb(self.get_umbrella_color(umbrella))
            palette = []
            for strength in [0.15, 0.28, 0.42, 0.58, 0.74, 0.88, 1.0]:
                mixed_rgb = tuple((1 - strength) * 1.0 + strength * channel for channel in base_rgb)
                palette.append(mcolors.to_hex(mixed_rgb))

        n = len(values_list)
        if n == 1:
            return [palette[-2]]

        # Values are passed in descending order, so assign darkest shades to the
        # largest bars at the top and lighter shades to smaller bars below.
        if n <= len(palette):
            indices = [round((len(palette) - 1) - i * (len(palette) - 1) / (n - 1)) for i in range(n)]
            return [palette[idx] for idx in indices]

        # If there are more bars than palette steps, interpolate across the same color family.
        dark_to_light = list(reversed(palette))
        colors: List[str] = []
        for i in range(n):
            pos = i / (n - 1)
            scaled = pos * (len(dark_to_light) - 1)
            left = int(math.floor(scaled))
            right = min(left + 1, len(dark_to_light) - 1)
            frac = scaled - left
            c1 = mcolors.to_rgb(dark_to_light[left])
            c2 = mcolors.to_rgb(dark_to_light[right])
            mix = tuple((1 - frac) * a + frac * b for a, b in zip(c1, c2))
            colors.append(mcolors.to_hex(mix))
        return colors

    def load_data(self) -> None:
        """Load the source CSV into memory and normalize column names."""
        self.raw_df = pd.read_csv(self.config.csv_path)
        self.raw_df.columns = [str(c).strip() for c in self.raw_df.columns]
        self.df = self.raw_df.copy()

        print(f"Loaded {len(self.df):,} rows and {self.df.shape[1]} columns.")

    def validate_columns(self) -> None:
        """Validate that the required text and label columns are present.

        Raises:
            ValueError: If one or more required columns are missing.
        """
        assert self.df is not None

        required = [self.config.text_column, self.config.label_column]
        missing = [c for c in required if c not in self.df.columns]

        if missing:
            raise ValueError(f"Missing required columns: {missing}")

    def normalize_repeated_characters(self, text: str) -> str:
        """Reduce excessive repeated Arabic characters without fully deleting emphasis.

        Added after the cleaning comparison: Arabic elongation may indicate emotion,
        so the method compresses long repetitions to a controlled maximum instead
        of reducing them to one character.
        """
        text = str(text)

        if not self.config.normalize_repeated_chars:
            return text

        max_rep = max(1, int(self.config.max_repeated_chars))

        def repl(match: re.Match) -> str:
            return match.group(1) * max_rep

        return self.repeated_arabic_char_pattern.sub(repl, text)

    def light_structural_preprocess(self, text: str) -> str:
        """Apply lightweight structural cleanup before Arabic normalization.

        The method removes or replaces artifacts such as HTML, URLs, emails,
        phone numbers, money values, date/time values, and repeated punctuation
        while preserving enough structure for later normalization.
        """
        text = str(text)
        # Remove markup and neutralize structured artifacts before Arabic normalization.
        text = self.html_pattern.sub(" ", text)
        text = self.url_pattern.sub(" URLTOKEN ", text)
        text = self.email_pattern.sub(" EMAILTOKEN ", text)
        text = self.phone_pattern.sub(" PHONETOKEN ", text)
        text = self.money_pattern.sub(" MONEYTOKEN ", text)
        text = self.date_like_pattern.sub(" DATETOKEN ", text)
        text = self.time_like_pattern.sub(" TIMETOKEN ", text)
        text = self.repeated_punct_pattern.sub(r"\1", text)
        text = self.multi_space_pattern.sub(" ", text).strip()
        return text

    def normalize_arabic(self, text: str) -> str:
        """Normalize Arabic text into a simplified analytical form.

        Args:
            text: Input text after structural preprocessing.

        Returns:
            Normalized text with Arabic character variants unified and non Arabic
            noise reduced.
        """
        text = str(text)
        text = self.normalize_repeated_characters(text)

        text = self.url_pattern.sub(" ", text)
        text = self.email_pattern.sub(" ", text)
        text = self.phone_pattern.sub(" ", text)
        text = self.money_pattern.sub(" ", text)
        text = self.date_like_pattern.sub(" ", text)
        text = self.time_like_pattern.sub(" ", text)

        # Arabic-specific letter normalization.
        text = re.sub(r"[إأآا]", "ا", text)
        text = re.sub(r"ى", "ي", text)
        text = re.sub(r"ة", "ه", text)

        # Remove diacritics/tatweel and keep only Arabic analytical content.
        text = self.arabic_diacritics_pattern.sub("", text)
        text = self.tatweel_pattern.sub("", text)
        text = self.punct_pattern.sub(" ", text)
        text = self.latin_digits_pattern.sub(" ", text)
        text = self.non_arabic_pattern.sub(" ", text)
        text = self.multi_space_pattern.sub(" ", text).strip()
        return text

    def clean_arabic_text(self, text: str) -> str:
        """Run the full text cleaning pipeline on a single string.

        Args:
            text: Raw text.

        Returns:
            Cleaned Arabic text ready for tokenization.
        """
        text = self.light_structural_preprocess(text)
        text = self.normalize_arabic(text)
        return text

    def _simple_normalize_token(self, token: str) -> str:
        """Apply basic normalization to a single token without clitic logic."""
        token = self.normalize_arabic(token)
        token = self.multi_space_pattern.sub(" ", token).strip()
        return token

    def _strip_conservative_clitics(self, token: str) -> str:
        """Remove selected attached clitics using conservative heuristics.

        Args:
            token: Candidate token.

        Returns:
            Token after conservative clitic stripping when the pattern is judged
            safe enough for analysis.
        """
        tok = token

        tok = re.sub(r"^لل(?=[\u0600-\u06FF]{2,}$)", "ال", tok)

        changed = True
        while changed:
            changed = False

            if re.match(r"^[وفبكل]ال[\u0600-\u06FF]{2,}$", tok):
                candidate = tok[1:]
                if len(candidate) >= self.config.min_token_length:
                    tok = candidate
                    changed = True
                    continue

            if re.match(r"^[وف][\u0600-\u06FF]{3,}$", tok):
                candidate = tok[1:]
                if candidate in self.discourse_bases:
                    tok = candidate
                    changed = True
                    continue

        return tok

    def _normalize_definite_article(self, token: str) -> str:
        """Normalize the Arabic definite article for analytical comparison.

        Args:
            token: Input token.

        Returns:
            Token after optional definite article normalization, except for
            protected words that should keep the article.
        """
        tok = token

        if not self.config.normalize_definite_article_for_analysis:
            return tok

        if tok in self.article_keep_words:
            return tok

        if tok.startswith("ال") and len(tok) >= 5:
            candidate = tok[2:]
            if len(candidate) >= self.config.min_token_length:
                return candidate

        return tok

    def canonicalize_token(self, token: str) -> str:
        """Produce the final canonical form used for token level analysis.

        Args:
            token: Raw token extracted from text.

        Returns:
            Canonical token after normalization, optional clitic cleanup, and
            optional definite article handling.
        """
        tok = self._simple_normalize_token(token)

        if not tok:
            return ""

        if self.config.enable_clitic_cleanup:
            tok = self._strip_conservative_clitics(tok)

        tok = self._normalize_definite_article(tok)

        tok = self.punct_pattern.sub("", tok)
        tok = tok.strip()

        return tok

    def tokenize_arabic(self, text: str) -> List[str]:
        """Tokenize Arabic text into cleaned analytical tokens.

        Args:
            text: Raw question text.

        Returns:
            List of canonical tokens that pass minimum length and stopword filters.
        """
        text = self.clean_arabic_text(text)
        raw_tokens = self.arabic_token_pattern.findall(text)

        clean_tokens: List[str] = []

        for tok in raw_tokens:
            canon = self.canonicalize_token(tok)

            if not canon:
                continue
            if len(canon) < self.config.min_token_length:
                continue
            if canon in self.stopwords:
                continue

            clean_tokens.append(canon)

        return clean_tokens

    def normalize_phrase_for_matching(self, phrase: str) -> str:
        """Normalize a multiword phrase for stopphrase matching.

        Args:
            phrase: Raw phrase text.

        Returns:
            Canonical phrase string used during n-gram filtering.
        """
        phrase = self.clean_arabic_text(phrase)
        raw_tokens = self.arabic_token_pattern.findall(phrase)

        normalized_tokens: List[str] = []
        for tok in raw_tokens:
            canon = self.canonicalize_token(tok)
            if canon:
                normalized_tokens.append(canon)

        return " ".join(normalized_tokens).strip()

    def should_filter_ngram(self, gram: str) -> bool:
        """Decide whether an n-gram should be excluded from analysis.

        Args:
            gram: Candidate n-gram string.

        Returns:
            True if the n-gram should be filtered out, otherwise False.
        """
        normalized_gram = self.normalize_phrase_for_matching(gram)

        if not normalized_gram:
            return True

        if normalized_gram in self.ngram_stopphrases:
            return True

        gram_tokens = normalized_gram.split()
        if not gram_tokens:
            return True

        discourse_hits = sum(1 for t in gram_tokens if t in self.stopwords)
        if len(gram_tokens) >= 2 and discourse_hits >= len(gram_tokens) - 1:
            return True

        return False

    def generate_ngrams(self, tokens: List[str], n: int) -> List[str]:
        """Generate filtered n-grams from a token sequence.

        Args:
            tokens: Ordered list of cleaned tokens.
            n: Size of each n-gram.

        Returns:
            List of n-grams that survive phrase level filtering.
        """
        if n <= 0 or len(tokens) < n:
            return []

        out: List[str] = []
        for i in range(len(tokens) - n + 1):
            gram = " ".join(tokens[i:i + n])
            if self.should_filter_ngram(gram):
                continue
            out.append(gram)
        return out

    def clean_label_text(self, label: str) -> str:
        """Clean hierarchical diagnosis labels into a more consistent form."""
        label = str(label).strip()
        label = re.sub(r"\s*-\s*", " - ", label)
        label = re.sub(r"\s+", " ", label)

        label = re.sub(r"^الحالات\s+النفسيه\s+العصبيه\s*-\s*", "", label)
        label = re.sub(r"^الحالات\s+النفسية\s+العصبية\s*-\s*", "", label)

        return label.strip(" -")

    def map_to_umbrella(self, label: str) -> str:
        """Map a cleaned diagnosis label to a broader umbrella category.

        Args:
            label: Original or cleaned hierarchical label.

        Returns:
            Canonical umbrella identifier.
        """
        normalized = self.normalize_arabic(self.clean_label_text(label)).lower()

        if re.search(r"انتحار|انتحاري|افكار انتحاريه|قتل النفس|ايذاء النفس|ايذاء الذات", normalized):
            return "suicidality_selfharm"

        if re.search(r"ثنائي القطب|اضطراب وجداني|هوس واكتئاب|هوس", normalized):
            return "bipolar_mania"

        if re.search(r"فرط الحركه|تشتت|نقص الانتباه|ضعف الانتباه|قصور الانتباه", normalized):
            return "attention_hyperactivity"

        if re.search(r"وسواس|وساوس|قهري|افكار وسواسيه|افعال وسواسيه", normalized):
            return "ocd_obsessive"

        if re.search(
            r"قلق|توتر|هلع|هرع|رهاب|مخاوف|خوف|فوبيا|انطواء|عزله|القلق الاكتئابي|اكتئابي قلقي",
            normalized
        ):
            return "anxiety_fear"

        if re.search(r"اكتئاب|اكتياب|كابه|كآبه|حزن|مزاج منخفض", normalized):
            return "depression"

        if re.search(r"طبيعي|سليم|لا اعاني|لا يوجد مرض|عادي|طمانه|اطمئنان", normalized):
            return "normal_reassurance"

        return "other_unclear"

    def detect_question_pov(self, text: str) -> str:
        """Classify question style as patient POV, doctor POV, mixed, or unclear.

        Added after the cleaning/report review to separate direct patient symptom
        narration from consultation-style or diagnosis-seeking language.
        """
        normalized = self.normalize_arabic(text)

        doctor_patterns = (
            r"تشخيص|شخص|ما رايكم|ما رأيكم|رايكم|دكتور|طبيب|"
            r"العلاج|دواء|جرعه|جرعة|استشاره|استشارة|افيدوني|تنصحوني"
        )
        patient_patterns = (
            r"اشعر|اعاني|احس|اخاف|افكر|لا استطيع|لااستطيع|"
            r"انا|عندي|لدي|ينتابني|اصبت|اعيش|اكره|احب|انام|اخشى"
        )

        doctor_hit = bool(re.search(doctor_patterns, normalized))
        patient_hit = bool(re.search(patient_patterns, normalized))

        if patient_hit and not doctor_hit:
            return "patient_pov"

        if doctor_hit and not patient_hit:
            return "doctor_pov"

        if patient_hit and doctor_hit:
            return "mixed_pov"

        return "unclear_pov"

    def light_stem_arabic_token(self, token: str) -> str:
        """Apply a conservative experimental Arabic light stem for audit only.

        This does not replace the main tokens. It only creates an optional audit to
        estimate vocabulary reduction before deciding whether stemming is useful.
        """
        tok = self.canonicalize_token(token)

        if not tok:
            return ""

        prefixes = ("وال", "بال", "كال", "فال", "لل", "ال", "و", "ف", "ب", "ك", "ل")
        suffixes = ("كما", "هما", "كم", "كن", "نا", "ها", "هم", "هن", "ات", "ون", "ين", "ان", "ه", "ي")

        for pref in prefixes:
            if tok.startswith(pref) and len(tok) - len(pref) >= 3:
                tok = tok[len(pref):]
                break

        for suff in suffixes:
            if tok.endswith(suff) and len(tok) - len(suff) >= 3:
                tok = tok[:-len(suff)]
                break

        return tok

    def export_short_text_analysis(self) -> None:
        """Export token-count summaries for very short cleaned questions.

        Added after prior-work comparison: short texts are inspected but not deleted,
        because short mental-health questions can still be meaningful.
        """
        assert self.df is not None

        short_text_summary = (
            self.df["token_count_clean"]
            .value_counts()
            .sort_index()
            .reset_index()
        )
        short_text_summary.columns = ["token_count_clean", "row_count"]
        short_text_summary["percentage"] = (
            short_text_summary["row_count"] / len(self.df) * 100
        ).round(2)

        short_text_summary.to_csv(
            os.path.join(self.tables_dir, "short_text_token_count_summary.csv"),
            index=False,
            encoding="utf-8-sig",
        )

        # Compact token-count distribution: vertical orientation is clearer here.
        self.plot_barv(
            labels=short_text_summary["token_count_clean"].astype(str).tolist(),
            values=short_text_summary["row_count"].tolist(),
            title="Clean Token Count Distribution",
            ylabel="Number of Questions",
            filename="short_text_token_count_distribution_vertical.png",
            annotation_labels=[
                f"{c:,} ({p:.1f}%)"
                for c, p in zip(short_text_summary["row_count"], short_text_summary["percentage"])
            ],
            wrap_width=10,
            figsize=(11, 6),
            rotation=0,
        )

        short_text_samples = self.df[
            self.df["token_count_clean"] <= 2
        ][
            ["raw_text", "clean_text", "analysis_text", "token_count_clean", "umbrella", "question_pov"]
        ].head(200)

        short_text_samples.to_csv(
            os.path.join(self.tables_dir, "short_text_samples.csv"),
            index=False,
            encoding="utf-8-sig",
        )

    def export_question_pov_analysis(self) -> None:
        """Export patient/doctor/mixed/unclear question point-of-view summaries."""
        assert self.df is not None

        pov_summary = (
            self.df.groupby(["question_pov", "umbrella"])
            .size()
            .reset_index(name="count")
        )

        pov_summary["percentage_within_pov"] = (
            pov_summary["count"] /
            pov_summary.groupby("question_pov")["count"].transform("sum") * 100
        ).round(2)

        pov_summary["umbrella_display"] = pov_summary["umbrella"].map(self.umbrella_display)

        pov_summary.to_csv(
            os.path.join(self.tables_dir, "question_pov_by_umbrella.csv"),
            index=False,
            encoding="utf-8-sig",
        )

        pov_counts = self.df["question_pov"].value_counts().reset_index()
        pov_counts.columns = ["question_pov", "count"]
        pov_counts["percentage"] = (pov_counts["count"] / len(self.df) * 100).round(2)

        pov_counts.to_csv(
            os.path.join(self.tables_dir, "question_pov_distribution.csv"),
            index=False,
            encoding="utf-8-sig",
        )

        # Compact non-ranked distribution: vertical chart follows the orientation rule.
        self.plot_barv(
            labels=pov_counts["question_pov"].tolist(),
            values=pov_counts["count"].tolist(),
            title="Question Point of View Distribution",
            ylabel="Number of Questions",
            filename="question_pov_distribution_vertical.png",
            annotation_labels=[
                f"{c:,} ({p:.1f}%)"
                for c, p in zip(pov_counts["count"], pov_counts["percentage"])
            ],
            wrap_width=18,
            figsize=(10, 6),
            rotation=15,
        )

    def export_experimental_stemming_audit(self) -> None:
        """Export a safe stemming audit without changing the main cleaned tokens.

        Lemmatization is not forced here because reliable Arabic lemmatization
        depends on external tools. This audit only checks whether simple stemming
        might reduce vocabulary size before using it in modeling.
        """
        assert self.df is not None

        baseline_tokens = [t for tokens in self.df["tokens"] for t in tokens]
        stemmed_tokens = [
            self.light_stem_arabic_token(t)
            for t in baseline_tokens
        ]
        stemmed_tokens = [t for t in stemmed_tokens if t]

        audit = pd.DataFrame([{
            "baseline_total_tokens": len(baseline_tokens),
            "baseline_unique_tokens": len(set(baseline_tokens)),
            "experimental_stemmed_total_tokens": len(stemmed_tokens),
            "experimental_stemmed_unique_tokens": len(set(stemmed_tokens)),
            "unique_vocab_reduction_pct": round(
                (
                    (len(set(baseline_tokens)) - len(set(stemmed_tokens))) /
                    len(set(baseline_tokens)) * 100
                ),
                2
            ) if baseline_tokens else 0,
            "lemmatization_status": "not_applied_requires_reliable_arabic_lemmatizer",
            "main_pipeline_changed": False,
        }])

        audit.to_csv(
            os.path.join(self.tables_dir, "experimental_stemming_lemmatization_audit.csv"),
            index=False,
            encoding="utf-8-sig",
        )

    def export_cleaning_comparison_table(self) -> None:
        """Export a comparison table against Elie's and last year's cleaning steps."""
        rows = [
            {
                "cleaning_point": "HTML, URLs, emails, phones, money, dates",
                "status": "handled",
                "decision": "Detected and neutralized during structural preprocessing.",
            },
            {
                "cleaning_point": "Time expressions",
                "status": "added_now",
                "decision": "Time-like expressions are detected and neutralized.",
            },
            {
                "cleaning_point": "Arabic normalization",
                "status": "handled",
                "decision": "Alef, Ya, Ta Marbuta, diacritics, tatweel, and non-Arabic noise are normalized.",
            },
            {
                "cleaning_point": "Repeated character normalization",
                "status": "added_now",
                "decision": "Excessive Arabic elongation is reduced while preserving limited emphasis.",
            },
            {
                "cleaning_point": "Negation handling",
                "status": "added_now",
                "decision": "Core negation terms are preserved because they affect clinical meaning.",
            },
            {
                "cleaning_point": "Stopwords and consultation phrases",
                "status": "handled",
                "decision": "Arabic stopwords and consultation-style phrases are filtered.",
            },
            {
                "cleaning_point": "Short-text filtering",
                "status": "analyzed_not_deleted",
                "decision": "Very short questions are exported for review but not removed automatically.",
            },
            {
                "cleaning_point": "Patient vs doctor point of view",
                "status": "added_now",
                "decision": "Question POV is classified as patient, doctor, mixed, or unclear for analysis.",
            },
            {
                "cleaning_point": "Stemming",
                "status": "audited_not_forced",
                "decision": "Experimental stemming audit is exported without changing the main text pipeline.",
            },
            {
                "cleaning_point": "Lemmatization",
                "status": "postponed",
                "decision": "Requires reliable Arabic lemmatization tool and model comparison.",
            },
            {
                "cleaning_point": "Rare-word removal",
                "status": "postponed_to_feature_engineering",
                "decision": "Better handled later with vectorizer parameters such as min_df.",
            },
            {
                "cleaning_point": "Spelling correction",
                "status": "not_applied",
                "decision": "Risky for Arabic mental-health text without reliable correction tool.",
            },
            {
                "cleaning_point": "Written-number conversion",
                "status": "not_applied",
                "decision": "Can be tested later if written number expressions are frequent.",
            },
            {
                "cleaning_point": "Full names/location anonymization",
                "status": "not_applied",
                "decision": "Only general structured artifacts are handled now; full NER anonymization needs separate validation.",
            },
            {
                "cleaning_point": "Email-chain truncation, banking anonymization, hashtags, contractions, uppercase tagging",
                "status": "not_suitable",
                "decision": "These are not suitable for standalone Arabic mental-health consultation questions.",
            },
        ]

        pd.DataFrame(rows).to_csv(
            os.path.join(self.tables_dir, "cleaning_comparison_with_prior_work.csv"),
            index=False,
            encoding="utf-8-sig",
        )

    def export_chart_orientation_guide(self) -> None:
        """Export the chart-orientation rules used for the report and annexes.

        This table makes the visualization decision explicit: ranked outputs use
        horizontal charts, while compact non-ranked summaries use vertical charts.
        """
        rows = [
            {
                "chart_family": "ranked_or_ordered_results",
                "orientation": "horizontal",
                "examples": "top words, top ngrams, top sublabels, combined all-sublabel chart, distinctive words, ranked umbrella counts",
                "reason": "Long Arabic labels and ordered comparisons are easier to read horizontally.",
            },
            {
                "chart_family": "compact_distributions",
                "orientation": "vertical",
                "examples": "patient/doctor POV distribution, duplicate/non-duplicate status, clean token-count distribution",
                "reason": "Small non-ranked categories are clearer as vertical summary charts.",
            },
            {
                "chart_family": "percentage_summaries",
                "orientation": "pie_or_table_only_when_useful",
                "examples": "umbrella proportion overview, duplicate percentage overview",
                "reason": "Pie charts are kept as optional annex/supporting visuals, not the main ranked comparison.",
            },
        ]

        pd.DataFrame(rows).to_csv(
            os.path.join(self.tables_dir, "chart_orientation_guide.csv"),
            index=False,
            encoding="utf-8-sig",
        )

    def export_cleaning_row_count_summary(
        self,
        rows_before_cleaning: int,
        rows_after_empty_clean_text_filter: int,
        rows_after_final_filtering: int,
    ) -> None:
        """Export before/after row counts for report-ready cleaning documentation.

        The table separates the original loaded rows from the rows kept after
        empty cleaned questions are removed and after any optional class filtering
        is applied. This does not change the cleaning logic; it only documents
        the effect of the existing preprocessing filters.
        """
        removed_empty_clean_text = rows_before_cleaning - rows_after_empty_clean_text_filter
        removed_optional_class_filtering = rows_after_empty_clean_text_filter - rows_after_final_filtering
        total_removed = rows_before_cleaning - rows_after_final_filtering

        def pct(value: int, denominator: int) -> float:
            return round((value / denominator * 100), 2) if denominator else 0.0

        rows = [
            {
                "stage": "Before cleaning",
                "row_count": rows_before_cleaning,
                "rows_removed_at_stage": 0,
                "removed_pct_of_original": 0.0,
                "note": "Rows loaded directly from the source CSV before preprocessing.",
            },
            {
                "stage": "After empty cleaned-text filtering",
                "row_count": rows_after_empty_clean_text_filter,
                "rows_removed_at_stage": removed_empty_clean_text,
                "removed_pct_of_original": pct(removed_empty_clean_text, rows_before_cleaning),
                "note": "Rows kept after removing questions whose cleaned Arabic text became empty.",
            },
            {
                "stage": "After final filtering",
                "row_count": rows_after_final_filtering,
                "rows_removed_at_stage": removed_optional_class_filtering,
                "removed_pct_of_original": pct(removed_optional_class_filtering, rows_before_cleaning),
                "note": "Rows kept after optional other-class and minimum-class-size filters.",
            },
            {
                "stage": "Total removed",
                "row_count": total_removed,
                "rows_removed_at_stage": total_removed,
                "removed_pct_of_original": pct(total_removed, rows_before_cleaning),
                "note": "Total number of rows removed between the original CSV and final EDA/modeling dataset.",
            },
        ]

        pd.DataFrame(rows).to_csv(
            os.path.join(self.tables_dir, "cleaning_before_after_row_counts.csv"),
            index=False,
            encoding="utf-8-sig",
        )

    def preprocess_dataframe(self) -> None:
        """Create all derived columns required for downstream analysis.

        This step fills missing values, cleans text, tokenizes questions, generates
        n-grams, computes structural indicators, applies label mapping, exports
        audit tables, and adds the extra review outputs selected after comparison
        with Elie's cleaning strategy and the previous year's report.
        """
        assert self.df is not None

        rows_before_cleaning = len(self.df)

        # Step 1: protect the two required fields from missing-value errors.
        self.df[self.config.text_column] = self.df[self.config.text_column].fillna("").astype(str)
        self.df[self.config.label_column] = self.df[self.config.label_column].fillna("").astype(str)

        # Step 2: keep the raw text, clean the labels, and map labels to umbrella classes.
        self.df["raw_text"] = self.df[self.config.text_column]
        self.df["label_clean"] = self.df[self.config.label_column].map(self.clean_label_text)
        self.df["label_normalized"] = self.df["label_clean"].map(self.normalize_arabic)
        self.df["umbrella"] = self.df["label_clean"].map(self.map_to_umbrella)

        # Step 3: create cleaned text and token sequences used by lexical EDA.
        self.df["structured_text"] = self.df["raw_text"].map(self.light_structural_preprocess)
        self.df["clean_text"] = self.df["raw_text"].map(self.clean_arabic_text)
        self.df["tokens"] = self.df["raw_text"].map(self.tokenize_arabic)

        # Step 4: build analysis text and short-text indicators without deleting rows.
        self.df["analysis_text"] = self.df["tokens"].map(lambda x: " ".join(x))
        self.df["token_count_clean"] = self.df["tokens"].map(len)
        self.df["is_very_short_clean"] = self.df["token_count_clean"] <= 2

        if self.config.enable_question_pov_analysis:
            self.df["question_pov"] = self.df["raw_text"].map(self.detect_question_pov)
        else:
            self.df["question_pov"] = "not_analyzed"

        # Step 5: generate n-grams after stopword and stopphrase filtering.
        self.df["unigrams"] = self.df["tokens"]
        self.df["bigrams"] = self.df["tokens"].map(lambda x: self.generate_ngrams(x, 2))
        self.df["trigrams"] = self.df["tokens"].map(lambda x: self.generate_ngrams(x, 3))

        # Step 6: structural features are measured on raw text to preserve writing style.
        self.df["word_count"] = self.df["raw_text"].str.split().map(len)
        self.df["char_count"] = self.df["raw_text"].str.len()

        self.df["punct_count"] = self.df["raw_text"].str.count(r"[،؟.!,:؛]")
        self.df["emoji_count"] = self.df["raw_text"].str.count(self.emoji_pattern)
        self.df["emoticon_count"] = self.df["raw_text"].str.count(self.emoticon_pattern)

        # Step 7: duplication is flagged for quality analysis rather than removed blindly.
        self.df["is_duplicate_clean"] = self.df["clean_text"].duplicated(keep=False)
        self.df["is_duplicate_analysis_text"] = self.df["analysis_text"].duplicated(keep=False)

        self.df = self.df[self.df["clean_text"].str.len() > 0].copy()
        rows_after_empty_clean_text_filter = len(self.df)

        counts_before = self.df["umbrella"].value_counts().reset_index()
        counts_before.columns = ["umbrella", "count_before_filter"]
        counts_before["umbrella_display"] = counts_before["umbrella"].map(self.umbrella_display)
        counts_before.to_csv(
            os.path.join(self.tables_dir, "umbrella_counts_before_filter.csv"),
            index=False,
            encoding="utf-8-sig",
        )

        if not self.config.keep_other_class:
            self.df = self.df[self.df["umbrella"] != "other_unclear"].copy()

        if self.config.min_class_size > 1:
            keep = self.df["umbrella"].value_counts()
            keep = keep[keep >= self.config.min_class_size].index
            self.df = self.df[self.df["umbrella"].isin(keep)].copy()

        rows_after_final_filtering = len(self.df)
        self.export_cleaning_row_count_summary(
            rows_before_cleaning=rows_before_cleaning,
            rows_after_empty_clean_text_filter=rows_after_empty_clean_text_filter,
            rows_after_final_filtering=rows_after_final_filtering,
        )

        mapping_audit = (
            self.df[[self.config.label_column, "label_clean", "label_normalized", "umbrella"]]
            .drop_duplicates()
            .sort_values(["umbrella", "label_clean"])
        )
        mapping_audit["umbrella_display"] = mapping_audit["umbrella"].map(self.umbrella_display)
        mapping_audit.to_csv(
            os.path.join(self.tables_dir, "label_mapping_audit.csv"),
            index=False,
            encoding="utf-8-sig",
        )

        token_audit = self.df[
            ["raw_text", "clean_text", "analysis_text", "tokens", "token_count_clean", "question_pov"]
        ].copy()
        token_audit.head(500).to_csv(
            os.path.join(self.tables_dir, "tokenization_audit_sample.csv"),
            index=False,
            encoding="utf-8-sig",
        )

        self.export_cleaning_comparison_table()
        self.export_chart_orientation_guide()

        if self.config.enable_short_text_analysis:
            self.export_short_text_analysis()

        if self.config.enable_question_pov_analysis:
            self.export_question_pov_analysis()

        if self.config.enable_experimental_stemming_audit:
            self.export_experimental_stemming_audit()

        # Final ML-ready output generated from the validated cleaning pipeline.
        self.export_final_modeling_dataset()

        print(f"After preprocessing/filtering: {len(self.df):,} rows kept.")
        print("Umbrellas kept:", sorted(self.df["umbrella"].unique().tolist()))

    def _save_plot(self, filename: str) -> None:
        """Save the current matplotlib figure to the charts directory."""
        plt.tight_layout()
        plt.savefig(os.path.join(self.charts_dir, filename), bbox_inches="tight")
        plt.close()

    def _annotate_barh(self, ax, values: Sequence[float], labels: Optional[Sequence[str]] = None) -> None:
        """Annotate horizontal bar plots with value labels.

        Args:
            ax: Matplotlib axes object containing horizontal bars.
            values: Numeric values represented by the bars.
            labels: Optional custom annotation labels. When omitted, raw values are
                used.
        """
        max_val = max(values) if len(values) else 0
        offset = max_val * 0.01 if max_val else 0.1
        labels = labels or [str(v) for v in values]

        for patch, txt in zip(ax.patches, labels):
            ax.text(
                patch.get_width() + offset,
                patch.get_y() + patch.get_height() / 2,
                str(txt),
                va="center",
                ha="left",
                fontsize=10,
            )

    def _annotate_barv(self, ax, values: Sequence[float], labels: Optional[Sequence[str]] = None) -> None:
        """Annotate vertical bar plots with value labels above each bar.

        This helper is used for compact, non-ranked distributions where vertical
        charts are easier to read, such as POV distribution or duplicate status.
        """
        max_val = max(values) if len(values) else 0
        offset = max_val * 0.01 if max_val else 0.1
        labels = labels or [str(v) for v in values]

        for patch, txt in zip(ax.patches, labels):
            ax.text(
                patch.get_x() + patch.get_width() / 2,
                patch.get_height() + offset,
                str(txt),
                va="bottom",
                ha="center",
                fontsize=10,
            )

    def plot_barh(
        self,
        labels: Sequence[str],
        values: Sequence[float],
        title: str,
        xlabel: str,
        filename: str,
        annotation_labels: Optional[Sequence[str]] = None,
        wrap_width: int = 26,
        figsize: Optional[Tuple[int, int]] = None,
        colors: Optional[Sequence] = None,
        invert_yaxis: bool = True,
    ) -> None:
        """Create and save a horizontal bar chart."""
        plt.figure(figsize=figsize or self.config.figsize_tall)
        ax = plt.gca()

        y_labels = [self.wrap_arabic_label(x, width=wrap_width) for x in labels]
        final_colors = list(colors) if colors is not None else self.dataset_chart_colors(len(values))
        ax.barh(y_labels, values, color=final_colors)
        if invert_yaxis:
            ax.invert_yaxis()

        ax.set_title(self.get_report_chart_title(title), fontsize=15, fontweight="bold", pad=14)
        ax.set_xlabel(self.shape_arabic_text(xlabel), fontsize=12)
        ax.grid(axis="x", alpha=0.20)

        for spine in ["top", "right"]:
            ax.spines[spine].set_visible(False)

        self._annotate_barh(ax, values, labels=annotation_labels)
        self._save_plot(filename)

    def plot_barv(
        self,
        labels: Sequence[str],
        values: Sequence[float],
        title: str,
        ylabel: str,
        filename: str,
        annotation_labels: Optional[Sequence[str]] = None,
        wrap_width: int = 18,
        figsize: Optional[Tuple[int, int]] = None,
        rotation: int = 20,
        colors: Optional[Sequence] = None,
    ) -> None:
        """Create and save a vertical bar chart for compact non-ranked distributions."""
        plt.figure(figsize=figsize or (10, 6))
        ax = plt.gca()

        x_labels = [self.wrap_arabic_label(x, width=wrap_width) for x in labels]
        final_colors = list(colors) if colors is not None else self.dataset_chart_colors(len(values))
        ax.bar(x_labels, values, color=final_colors)

        ax.set_title(self.get_report_chart_title(title), fontsize=15, fontweight="bold", pad=14)
        ax.set_ylabel(self.shape_arabic_text(ylabel), fontsize=12)
        ax.tick_params(axis="x", labelrotation=rotation)

        self._annotate_barv(ax, values, labels=annotation_labels)
        self._save_plot(filename)

    def plot_pie(
        self,
        labels: Sequence[str],
        values: Sequence[float],
        title: str,
        filename: str,
        wrap_width: int = 24,
        english_labels: Optional[Sequence[str]] = None,
        colors: Optional[Sequence[str]] = None,
    ) -> None:
        """Create and save a readable pie chart with a boxed English side legend.

        The pie itself stays clean. Percentages are shown inside slices, while
        the side legend contains English names, percentages, and n row counts.
        """
        total = sum(values)
        if total == 0:
            return

        fig, ax = plt.subplots(figsize=(12, 8))

        if colors is None:
            colors = self.juliet_gradient_colors(len(values))

        wedges, _, _ = ax.pie(
            values,
            labels=None,
            colors=colors,
            autopct=lambda pct: f"{pct:.1f}%" if pct >= 1 else "",
            startangle=90,
            counterclock=False,
            pctdistance=0.72,
            textprops={"fontsize": 10, "color": "white", "fontweight": "bold"},
            wedgeprops={"linewidth": 1.2, "edgecolor": "white"},
        )

        legend_names = list(english_labels) if english_labels is not None else [str(x) for x in labels]
        legend_text = [
            f"{name}: {value / total * 100:.1f}% (n={int(value):,})"
            for name, value in zip(legend_names, values)
        ]

        legend = ax.legend(
            wedges,
            legend_text,
            title="Labels",
            loc="center left",
            bbox_to_anchor=(1.02, 0.5),
            frameon=True,
            fontsize=10,
            title_fontsize=11,
            borderpad=1.0,
            labelspacing=1.0,
        )
        legend.get_frame().set_facecolor("white")
        legend.get_frame().set_edgecolor("#CCCCCC")
        legend.get_frame().set_linewidth(1.0)

        ax.set_title(self.get_report_chart_title(title), fontsize=15, fontweight="bold", pad=14)
        ax.axis("equal")
        self._save_plot(filename)

    def plot_ranked_frequency_barh(
        self,
        labels: Sequence[str],
        values: Sequence[float],
        title: str,
        filename: str,
        xlabel: str = "Frequency",
        wrap_width: int = 24,
        figsize: Optional[Tuple[int, int]] = None,
        umbrella: Optional[str] = None,
    ) -> None:
        """Create a Juliet-style ranked horizontal frequency chart.

        The first input item is shown at the top, so pass values in descending
        frequency order to get large bars on top and small bars at the bottom.
        When an umbrella id is provided, the bars use that umbrella's pie-chart
        color, progressing from light shades for low counts to darker shades
        for higher counts.
        """
        if not labels or not values:
            return

        if umbrella is not None:
            colors = self.umbrella_gradient_colors(umbrella, values)
        else:
            colors = self.juliet_gradient_colors(len(values))

        self.plot_barh(
            labels=labels,
            values=values,
            title=title,
            xlabel=xlabel,
            filename=filename,
            annotation_labels=[f"{int(v):,}" for v in values],
            wrap_width=wrap_width,
            figsize=figsize or (12, 7),
            colors=colors,
            invert_yaxis=True,
        )


    def plot_multi_ranked_panels(
        self,
        panels: Sequence[Dict[str, object]],
        title: str,
        filename: str,
        xlabel: str = "Frequency",
        wrap_width: int = 18,
    ) -> None:
        """Save one merged multi-panel chart instead of many separate umbrella charts."""
        if not panels:
            return

        n_cols = 2
        n_rows = math.ceil(len(panels) / n_cols)
        fig, axes = plt.subplots(n_rows, n_cols, figsize=(15, 4.8 * n_rows), squeeze=False)
        axes_flat = axes.flatten()

        for ax, panel in zip(axes_flat, panels):
            umbrella = str(panel.get("umbrella", ""))
            labels = list(panel.get("labels", []))
            values = list(panel.get("values", []))
            annotations = list(panel.get("annotations", [])) or [str(v) for v in values]

            shaped_labels = [self.wrap_arabic_label(str(x), width=wrap_width) for x in labels]
            colors = self.umbrella_gradient_colors(umbrella, values) if umbrella else self.dataset_chart_colors(len(values))
            ax.barh(shaped_labels, values, color=colors)
            ax.invert_yaxis()
            ax.set_title(self.get_report_chart_title(umbrella=umbrella), fontsize=12, fontweight="bold")
            ax.set_xlabel(self.shape_arabic_text(xlabel), fontsize=10)
            ax.grid(axis="x", alpha=0.20)

            max_val = max(values) if values else 0
            offset = max_val * 0.01 if max_val else 0.1
            for patch, txt in zip(ax.patches, annotations):
                ax.text(
                    patch.get_width() + offset,
                    patch.get_y() + patch.get_height() / 2,
                    str(txt),
                    va="center",
                    ha="left",
                    fontsize=8,
                )

            for spine in ["top", "right"]:
                ax.spines[spine].set_visible(False)

        for ax in axes_flat[len(panels):]:
            ax.axis("off")

        fig.suptitle("Arabic Dataset", fontsize=16, fontweight="bold", y=1.01)
        plt.tight_layout()
        plt.savefig(os.path.join(self.charts_dir, filename), bbox_inches="tight")
        plt.close()

    def plot_question_length_distribution_by_umbrella(self) -> None:
        """Create Juliet-style histogram panels for question length by umbrella."""
        assert self.df is not None

        counts = self.df["umbrella"].value_counts()
        umbrellas = counts.index.tolist()
        if not umbrellas:
            return

        n_cols = 2
        n_rows = math.ceil(len(umbrellas) / n_cols)
        fig, axes = plt.subplots(n_rows, n_cols, figsize=(14, 4.2 * n_rows), squeeze=False)
        axes_flat = axes.flatten()

        max_words = int(self.df["word_count"].max()) if len(self.df) else 1
        bin_count = min(40, max(10, int(math.sqrt(max(len(self.df), 1)))))
        bins = list(range(0, max_words + max(2, math.ceil(max_words / bin_count)), max(2, math.ceil(max_words / bin_count))))

        for ax, umbrella in zip(axes_flat, umbrellas):
            subset = self.df[self.df["umbrella"] == umbrella]["word_count"].dropna()
            color = self.get_umbrella_color(umbrella)

            ax.hist(
                subset,
                bins=bins,
                color=color,
                alpha=0.85,
                edgecolor="white",
                linewidth=0.8,
            )

            median_value = float(subset.median()) if len(subset) else 0
            ax.axvline(
                median_value,
                color="#333333",
                linestyle="--",
                linewidth=1.5,
                label=f"Median = {median_value:.0f}",
            )
            ax.set_title(self.get_report_chart_title(umbrella=umbrella), fontsize=12, fontweight="bold")
            ax.set_xlabel("Question Length (words)")
            ax.set_ylabel("Frequency")
            ax.grid(axis="y", alpha=0.20)
            ax.legend(loc="upper right", frameon=True, fontsize=9)

            for spine in ["top", "right"]:
                ax.spines[spine].set_visible(False)

        for ax in axes_flat[len(umbrellas):]:
            ax.axis("off")
        fig.suptitle("Arabic Dataset", fontsize=16, fontweight="bold", y=1.01)
        plt.tight_layout()
        plt.savefig(
            os.path.join(self.charts_dir, "question_length_distribution_by_umbrella.png"),
            bbox_inches="tight",
        )
        plt.close()

    def save_frequency_table_png(
        self,
        table: pd.DataFrame,
        title: str,
        filename: str,
    ) -> None:
        """Save a simple frequency/normalized table as a PNG image."""
        if table.empty:
            table = pd.DataFrame({"Item": ["None found"], "Count": [0]})

        max_rows = min(len(table), 35)
        display_table = table.head(max_rows).copy()

        for col in display_table.columns:
            if col.lower() in {"count", "total_question_length_words", "total_questions"}:
                display_table[col] = display_table[col].map(lambda x: f"{int(x):,}")
            elif "per_1000" in col.lower() or "normalized" in col.lower() or "ratio" in col.lower():
                display_table[col] = display_table[col].map(lambda x: f"{float(x):.4f}")

        fig_height = max(2.8, 0.42 * len(display_table) + 0.8)
        fig_width = max(8.5, 1.8 * len(display_table.columns))
        fig, ax = plt.subplots(figsize=(fig_width, fig_height))
        ax.axis("off")
        ax.set_title(self.get_report_chart_title(title), fontsize=15, fontweight="bold", pad=14)

        tbl = ax.table(
            cellText=display_table.values,
            colLabels=display_table.columns,
            cellLoc="center",
            colLoc="center",
            loc="center",
        )
        tbl.auto_set_font_size(False)
        tbl.set_fontsize(9)
        tbl.scale(1.0, 1.35)

        for (row, col), cell in tbl.get_celld().items():
            cell.set_edgecolor("#DDDDDD")
            if row == 0:
                cell.set_facecolor("#F2F2F2")
                cell.set_text_props(weight="bold")
            else:
                cell.set_facecolor("white")

        plt.tight_layout()
        plt.savefig(os.path.join(self.charts_dir, filename), bbox_inches="tight")
        plt.close()

    def save_comparison_matrix_table_png(
        self,
        matrix_df: pd.DataFrame,
        filename: str,
        row_color_lookup: Optional[Dict[str, str]] = None,
    ) -> None:
        """Save a common comparison matrix table styled like the reference image.

        The first column contains label names with umbrella-specific colors, the
        header row uses a blue band, and the numeric cells contain normalized
        decimal values rounded to four places.
        """
        if matrix_df.empty:
            matrix_df = pd.DataFrame({"Label": ["None"], "Value": [0.0]})

        display_df = matrix_df.copy()
        for col in display_df.columns[1:]:
            display_df[col] = display_df[col].map(lambda x: f"{float(x):.4f}")

        fig_height = max(2.8, 0.62 * len(display_df) + 1.2)
        fig_width = max(10.0, 2.2 * len(display_df.columns) + 2.2)
        fig, ax = plt.subplots(figsize=(fig_width, fig_height))
        ax.axis("off")
        ax.set_title("Arabic Dataset", fontsize=15, fontweight="bold", pad=14)

        tbl = ax.table(
            cellText=display_df.values,
            colLabels=display_df.columns,
            cellLoc="center",
            colLoc="center",
            loc="center",
        )
        tbl.auto_set_font_size(False)
        tbl.set_fontsize(10)
        tbl.scale(1.0, 1.6)

        header_color = "#4A76C2"
        edge_color = "#333333"

        for (row, col), cell in tbl.get_celld().items():
            cell.set_edgecolor(edge_color)
            cell.set_linewidth(1.0)

            if row == 0:
                cell.set_facecolor(header_color)
                cell.set_text_props(weight="bold", color="white")
            else:
                if col == 0:
                    label = str(display_df.iloc[row - 1, 0])
                    face = row_color_lookup.get(label, "#EFEFEF") if row_color_lookup else "#EFEFEF"
                    cell.set_facecolor(face)
                    cell.set_text_props(weight="bold", color="white", ha="left")
                else:
                    cell.set_facecolor("white")
                    cell.set_text_props(color="#333333")

        plt.tight_layout()
        plt.savefig(os.path.join(self.charts_dir, filename), bbox_inches="tight")
        plt.close()

    def dataset_summary(self) -> None:
        """Compute and export high level dataset summary statistics."""
        assert self.df is not None

        summary = {
            "rows_kept": len(self.df),
            "columns_total": self.df.shape[1],
            "unique_raw_labels": int(self.df[self.config.label_column].nunique()),
            "unique_clean_labels": int(self.df["label_clean"].nunique()),
            "unique_umbrellas": int(self.df["umbrella"].nunique()),
            "duplicate_clean_rows": int(self.df["is_duplicate_clean"].sum()),
            "duplicate_analysis_rows": int(self.df["is_duplicate_analysis_text"].sum()),
            "very_short_rows_token_count_le_2": int(self.df["is_very_short_clean"].sum()),
            "avg_word_count": round(self.df["word_count"].mean(), 2),
            "median_word_count": round(self.df["word_count"].median(), 2),
            "avg_clean_token_count": round(self.df["token_count_clean"].mean(), 2),
            "median_clean_token_count": round(self.df["token_count_clean"].median(), 2),
        }

        pd.DataFrame([summary]).to_csv(
            os.path.join(self.tables_dir, "dataset_summary.csv"),
            index=False,
            encoding="utf-8-sig",
        )

    def umbrella_distribution(self) -> None:
        """Export umbrella distribution tables and corresponding charts."""
        assert self.df is not None

        counts = self.df["umbrella"].value_counts()
        perc = (counts / counts.sum() * 100).round(1)

        table = pd.DataFrame({
            "umbrella": counts.index,
            "umbrella_display": [self.umbrella_display[x] for x in counts.index],
            "umbrella_display_en": [self.get_umbrella_english_name(x) for x in counts.index],
            "count": counts.values,
            "percentage": perc.values
        })

        table.to_csv(
            os.path.join(self.tables_dir, "umbrella_distribution.csv"),
            index=False,
            encoding="utf-8-sig",
        )

        labels = table["umbrella_display"].tolist()
        english_labels = table["umbrella_display_en"].tolist()
        values = table["count"].tolist()
        ann = [f"{c:,} ({p:.1f}%)" for c, p in zip(table["count"], table["percentage"])]

        self.plot_barh(
            labels=labels,
            values=values,
            title="Distribution of Diagnostic Umbrellas",
            xlabel="Number of Questions",
            filename="umbrella_distribution_bar.png",
            annotation_labels=ann,
            wrap_width=24,
            figsize=(13, 7),
            colors=[self.get_umbrella_color(x) for x in table["umbrella"].tolist()],
        )

        # Pie chart removed to reduce chart quantity; the colored bar chart is the main distribution visual.

    def top_sublabels_within_each_umbrella(self) -> None:
        """Export the top sublabels and their charts for each umbrella."""
        assert self.df is not None

        for umbrella, group in self.df.groupby("umbrella"):
            counts = group["label_clean"].value_counts().head(self.config.top_n_sublabels)
            total = counts.sum()
            perc = (counts / total * 100).round(1)

            out = pd.DataFrame({
                "umbrella": umbrella,
                "umbrella_display": self.umbrella_display[umbrella],
                "sub_label": counts.index,
                "count": counts.values,
                "percentage_within_selected_top": perc.values,
            })

            out.to_csv(
                os.path.join(self.tables_dir, f"top_sublabels_{self.safe_filename(umbrella)}.csv"),
                index=False,
                encoding="utf-8-sig",
            )

            labels = counts.index.tolist()
            values = counts.values.tolist()
            ann = [f"{c:,} ({p:.1f}%)" for c, p in zip(values, perc.values)]

            self.plot_barh(
                labels=labels,
                values=values,
                title=f"Most Common Sublabels within {self.umbrella_display[umbrella]}",
                xlabel="Number of Questions",
                filename=f"top_sublabels_bar_{self.safe_filename(umbrella)}.png",
                annotation_labels=ann,
                wrap_width=30,
                figsize=(13, 7),
                colors=self.umbrella_gradient_colors(umbrella, values),
            )

            # Per-umbrella sublabel pie charts removed to reduce chart quantity.
            # The combined all-sublabel chart and per-umbrella bars carry the same information more clearly.

    def combined_common_rare_sublabel_distribution(self) -> None:
        """Export one combined unstacked vertical chart for all sublabels in Section 4.3.

        This keeps all old per-umbrella outputs, but replaces the previous
        common/rare summary with a clearer all-sublabel view. Each diagnostic
        umbrella appears once on the x-axis, and every sublabel inside that
        umbrella is drawn as a separate unstacked vertical bar. The y-axis shows
        the percentage of that sublabel within its own umbrella.

        Because Arabic sublabel names are long, the chart uses compact codes
        above the bars (S1, S2, S3, ...). The full code-to-sublabel mapping is
        exported as a CSV key so the graph stays readable and presentable.
        """
        assert self.df is not None

        rows = []

        # Keep the same umbrella order as the main umbrella distribution.
        umbrella_order = self.df["umbrella"].value_counts().index.tolist()

        for umbrella in umbrella_order:
            group = self.df[self.df["umbrella"] == umbrella]
            counts = group["label_clean"].value_counts()

            if counts.empty:
                continue

            total_in_umbrella = int(counts.sum())

            # Export every sublabel, not only common/rare selections.
            for rank, (sub_label, count) in enumerate(counts.items(), start=1):
                rows.append({
                    "umbrella": umbrella,
                    "umbrella_display": self.umbrella_display[umbrella],
                    "sublabel_rank": rank,
                    "chart_code": f"S{rank}",
                    "sub_label": sub_label,
                    "count": int(count),
                    "percentage_within_umbrella": round(count / total_in_umbrella * 100, 2),
                })

        if not rows:
            return

        combined = pd.DataFrame(rows)

        # Main table containing all sublabels, their counts, and percentages.
        combined.to_csv(
            os.path.join(self.tables_dir, "combined_all_sublabels_by_umbrella.csv"),
            index=False,
            encoding="utf-8-sig",
        )

        # Keep the older filename too so previous report links do not break.
        combined.to_csv(
            os.path.join(self.tables_dir, "combined_common_rare_sublabels.csv"),
            index=False,
            encoding="utf-8-sig",
        )

        # Chart key: use this table beside/under the figure in the report if needed.
        key_columns = [
            "umbrella_display",
            "chart_code",
            "sublabel_rank",
            "sub_label",
            "count",
            "percentage_within_umbrella",
        ]
        combined[key_columns].to_csv(
            os.path.join(self.tables_dir, "combined_all_sublabels_chart_key.csv"),
            index=False,
            encoding="utf-8-sig",
        )

        # Keep the older key filename too for compatibility.
        combined[key_columns].to_csv(
            os.path.join(self.tables_dir, "combined_common_rare_sublabels_chart_key.csv"),
            index=False,
            encoding="utf-8-sig",
        )

        def draw_all_sublabels_vertical(filename: str) -> None:
            """Draw and save the all-sublabel unstacked vertical chart."""
            max_bars_in_group = int(combined.groupby("umbrella").size().max())
            fig_width = max(18, 2.8 * len(umbrella_order) + 0.20 * max_bars_in_group)
            plt.figure(figsize=(fig_width, 9))
            ax = plt.gca()

            x_centers = list(range(len(umbrella_order)))
            cluster_width = 0.86
            max_y = 0.0

            for x_center, umbrella in zip(x_centers, umbrella_order):
                subset = combined[combined["umbrella"] == umbrella].copy()
                if subset.empty:
                    continue

                subset = subset.sort_values("sublabel_rank")
                n_bars = len(subset)
                bar_width = cluster_width / max(n_bars, 1)
                start_x = x_center - cluster_width / 2 + bar_width / 2

                for idx, (_, row) in enumerate(subset.iterrows()):
                    x_pos = start_x + idx * bar_width
                    value = float(row["percentage_within_umbrella"])
                    max_y = max(max_y, value)

                    ax.bar(x_pos, value, width=bar_width * 0.90, color=self.get_umbrella_color(umbrella))

                    # Large bars get code + percentage; tiny bars get only the code
                    # to avoid clutter while keeping the full details in the CSV key.
                    if value >= 3:
                        ax.text(
                            x_pos,
                            value + 0.7,
                            f"{row['chart_code']}\n{value:.1f}%",
                            ha="center",
                            va="bottom",
                            fontsize=8,
                        )
                    else:
                        ax.text(
                            x_pos,
                            value + 0.5,
                            row["chart_code"],
                            ha="center",
                            va="bottom",
                            fontsize=7,
                            rotation=90,
                        )

            x_labels = [self.wrap_arabic_label(self.umbrella_display[u], width=18) for u in umbrella_order]
            ax.set_xticks(x_centers)
            ax.set_xticklabels(x_labels, rotation=0)
            ax.set_title("Arabic Dataset", fontsize=16, fontweight="bold", pad=14)
            ax.set_ylabel(self.shape_arabic_text("Percentage within Umbrella (%)"), fontsize=12)
            ax.set_xlabel(self.shape_arabic_text("Diagnostic Umbrella"), fontsize=12)
            ax.set_ylim(0, max(105, max_y + 10))
            ax.grid(axis="y", alpha=0.25)

            ax.text(
                0.99,
                0.98,
                "Each bar is one sublabel. Full S-code names are exported in combined_all_sublabels_chart_key.csv",
                transform=ax.transAxes,
                ha="right",
                va="top",
                fontsize=9,
                bbox={"boxstyle": "round", "facecolor": "white", "alpha": 0.75},
            )

            self._save_plot(filename)

        # One combined chart is enough; older duplicate chart exports were removed
        # to reduce chart quantity. The CSV keys remain available for details.
        draw_all_sublabels_vertical("combined_all_sublabels_vertical_unstacked.png")

    def length_profile_by_umbrella(self) -> None:
        """Compute structural length profile and export Juliet-style length charts."""
        assert self.df is not None

        prof = (
            self.df.groupby("umbrella")
            .agg(
                count=("raw_text", "size"),
                avg_words=("word_count", "mean"),
                median_words=("word_count", "median"),
                avg_clean_tokens=("token_count_clean", "mean"),
                median_clean_tokens=("token_count_clean", "median"),
                avg_chars=("char_count", "mean"),
                avg_punct=("punct_count", "mean"),
            )
            .round(2)
            .sort_values("count", ascending=False)
        )

        prof["umbrella_display"] = [self.umbrella_display[x] for x in prof.index]
        prof["umbrella_display_en"] = [self.get_umbrella_english_name(x) for x in prof.index]

        prof.to_csv(
            os.path.join(self.tables_dir, "length_profile_by_umbrella.csv"),
            encoding="utf-8-sig",
        )

        # Replaces the old average-only bar charts with a Juliet-style
        # distribution view using question length in words.
        self.plot_question_length_distribution_by_umbrella()

    def top_words_by_umbrella(self) -> None:
        """Export the most common cleaned words for each umbrella and one merged chart."""
        assert self.df is not None

        all_rows = []
        panels: List[Dict[str, object]] = []

        for umbrella, group in self.df.groupby("umbrella"):
            counter = Counter()

            for tokens in group["tokens"]:
                counter.update(tokens)

            common = [(w, c) for w, c in counter.most_common(self.config.top_n_words) if c > 1]

            out = pd.DataFrame(common, columns=["word", "count"])
            out["umbrella"] = umbrella
            out["umbrella_display"] = self.umbrella_display[umbrella]
            all_rows.append(out)

            if common:
                panels.append({
                    "umbrella": umbrella,
                    "labels": [w for w, _ in common],
                    "values": [c for _, c in common],
                    "annotations": [f"{c:,}" for _, c in common],
                })

        if all_rows:
            pd.concat(all_rows, ignore_index=True).to_csv(
                os.path.join(self.tables_dir, "top_words_by_umbrella.csv"),
                index=False,
                encoding="utf-8-sig",
            )

        # One merged chart replaces many separate per-umbrella word charts.
        self.plot_multi_ranked_panels(
            panels=panels,
            title="Top Words by Umbrella",
            filename="top_words_by_umbrella_merged.png",
            xlabel="Frequency",
            wrap_width=16,
        )

    def top_ngrams_by_umbrella(self) -> None:
        """Export common unigrams, bigrams, and trigrams and merged charts."""
        assert self.df is not None

        ngram_specs = [
            (1, "unigrams", "Unigrams"),
            (2, "bigrams", "Bigrams"),
            (3, "trigrams", "Trigrams"),
        ]

        for n, ngram_name, display_name in ngram_specs:
            all_rows = []
            panels: List[Dict[str, object]] = []

            for umbrella, group in self.df.groupby("umbrella"):
                counter = Counter()

                for ngrams_list in group[ngram_name]:
                    counter.update(ngrams_list)

                common = [(gram, c) for gram, c in counter.most_common(self.config.top_n_ngrams) if c > 1]

                out = pd.DataFrame(common, columns=["ngram", "count"])
                out["n"] = n
                out["ngram_type"] = ngram_name
                out["umbrella"] = umbrella
                out["umbrella_display"] = self.umbrella_display[umbrella]
                out["umbrella_display_en"] = self.get_umbrella_english_name(umbrella)
                all_rows.append(out)

                if common:
                    panels.append({
                        "umbrella": umbrella,
                        "labels": [g for g, _ in common],
                        "values": [c for _, c in common],
                        "annotations": [f"{c:,}" for _, c in common],
                    })

            if all_rows:
                pd.concat(all_rows, ignore_index=True).to_csv(
                    os.path.join(self.tables_dir, f"top_{ngram_name}_by_umbrella.csv"),
                    index=False,
                    encoding="utf-8-sig",
                )

            # One merged chart per n-gram type replaces many per-umbrella charts.
            self.plot_multi_ranked_panels(
                panels=panels,
                title=f"Top {display_name} by Umbrella",
                filename=f"top_{ngram_name}_by_umbrella_merged.png",
                xlabel="Frequency",
                wrap_width=22,
            )

    def distinctive_words_by_umbrella(self) -> None:
        """Identify distinctive words and export one merged chart."""
        assert self.df is not None

        global_counter = Counter()
        for tokens in self.df["tokens"]:
            global_counter.update(tokens)

        rows = []
        panels: List[Dict[str, object]] = []

        for umbrella, group in self.df.groupby("umbrella"):
            class_counter = Counter()
            for tokens in group["tokens"]:
                class_counter.update(tokens)

            other_counter = global_counter - class_counter
            scored = []

            for term, cls_count in class_counter.items():
                if cls_count < 5:
                    continue

                other_count = other_counter.get(term, 0)

                cls_rate = (cls_count + 1) / (sum(class_counter.values()) + 1)
                other_rate = (other_count + 1) / (sum(other_counter.values()) + 1)
                score = math.log(cls_rate / other_rate)

                if term in self.stopwords:
                    continue

                scored.append((term, score, cls_count, other_count))

            scored = sorted(scored, key=lambda x: (x[1], x[2]), reverse=True)[:self.config.top_n_words]

            for term, score, cls_count, other_count in scored:
                rows.append({
                    "umbrella": umbrella,
                    "umbrella_display": self.umbrella_display[umbrella],
                    "word": term,
                    "distinctiveness_score": round(score, 4),
                    "class_count": cls_count,
                    "other_count": other_count,
                })

            if scored:
                panels.append({
                    "umbrella": umbrella,
                    "labels": [x[0] for x in scored],
                    "values": [round(x[1], 2) for x in scored],
                    "annotations": [f"{round(x[1], 2):.2f}" for x in scored],
                })

        if rows:
            pd.DataFrame(rows).to_csv(
                os.path.join(self.tables_dir, "distinctive_words_by_umbrella.csv"),
                index=False,
                encoding="utf-8-sig",
            )

        # One merged chart replaces separate per-umbrella distinctive-word charts.
        self.plot_multi_ranked_panels(
            panels=panels,
            title="Most Distinctive Words by Umbrella",
            filename="distinctive_words_by_umbrella_merged.png",
            xlabel="Distinctiveness Score",
            wrap_width=16,
        )

    def duplicates_quality_report(self) -> None:
        """Measure semantic duplication and export summary visuals."""
        assert self.df is not None

        dup_count = int(self.df["is_duplicate_clean"].sum())
        non_dup = len(self.df) - dup_count
        dup_pct = round((dup_count / len(self.df)) * 100, 2) if len(self.df) else 0

        dup_analysis_count = int(self.df["is_duplicate_analysis_text"].sum())
        dup_analysis_pct = round((dup_analysis_count / len(self.df)) * 100, 2) if len(self.df) else 0

        pd.DataFrame([{
            "duplicate_rows_clean_text": dup_count,
            "duplicate_rate_clean_text_pct": dup_pct,
            "duplicate_rows_analysis_text": dup_analysis_count,
            "duplicate_rate_analysis_text_pct": dup_analysis_pct,
            "non_duplicate_rows": non_dup,
        }]).to_csv(
            os.path.join(self.tables_dir, "duplicate_summary.csv"),
            index=False,
            encoding="utf-8-sig",
        )

        labels = ["أسئلة مكررة دلاليًا", "أسئلة غير مكررة"]
        values = [dup_count, non_dup]

        # Two-category quality summary: vertical chart matches the report orientation rule.
        self.plot_barv(
            labels=labels,
            values=values,
            title="Data Quality: Semantic Duplication",
            ylabel="Number of Records",
            filename="duplicate_quality_vertical.png",
            annotation_labels=[
                f"{dup_count:,} ({dup_pct:.1f}%)",
                f"{non_dup:,} ({100 - dup_pct:.1f}%)"
            ],
            wrap_width=16,
            figsize=(10, 6),
            rotation=0,
        )

        # Duplicate pie chart removed to reduce chart quantity; the vertical bar remains.

    def punctuation_emoji_profile(self) -> None:
        """Export common normalized comparison tables for punctuation, emoji, and emoticons.

        Each table is structured like the reference image:
        - one common matrix table;
        - rows are umbrella labels;
        - columns are punctuation/emoji/emoticon types;
        - values are normalized decimals computed as count / total word count.
        """
        assert self.df is not None

        umbrella_order = self.df["umbrella"].value_counts().index.tolist()
        row_labels = [self.get_umbrella_english_name(u) for u in umbrella_order]
        row_color_lookup = {self.get_umbrella_english_name(u): self.get_umbrella_color(u) for u in umbrella_order}

        def build_matrix(category_specs: List[Tuple[str, List[str]]]) -> pd.DataFrame:
            rows = []
            for umbrella in umbrella_order:
                group = self.df[self.df["umbrella"] == umbrella]
                total_words = float(group["word_count"].sum())
                joined_text = " ".join(group["raw_text"].fillna("").astype(str).tolist())

                row = {"Label": self.get_umbrella_english_name(umbrella)}
                for display_name, raw_variants in category_specs:
                    count = 0
                    for variant in raw_variants:
                        if variant == "...":
                            count += joined_text.count("...")
                        else:
                            count += joined_text.count(variant)
                    row[display_name] = round((count / total_words), 4) if total_words else 0.0
                rows.append(row)

            return pd.DataFrame(rows)

        punctuation_specs = [
            ("Question", ["؟", "?"]),
            ("Exclamation", ["!"]),
            ("Ellipsis", ["…", "..."]),
            ("Comma", ["،", ","]),
            ("Period", ["."]),
            ("Colon", [":"]),
            ("Semicolon", ["؛", ";"]),
        ]

        punctuation_matrix = build_matrix(punctuation_specs)
        punctuation_matrix.to_csv(
            os.path.join(self.tables_dir, "punctuation_usage_matrix_normalized.csv"),
            index=False,
            encoding="utf-8-sig",
        )
        self.save_comparison_matrix_table_png(
            punctuation_matrix,
            filename="punctuation_usage_matrix_normalized.png",
            row_color_lookup=row_color_lookup,
        )

        def build_top_symbol_matrix(pattern: re.Pattern, label_col: str, filename_stub: str, top_n: int = 8) -> None:
            overall_counter = Counter()
            per_umbrella_counters = {}

            for umbrella in umbrella_order:
                group = self.df[self.df["umbrella"] == umbrella]
                joined_text = " ".join(group["raw_text"].fillna("").astype(str).tolist())
                matches = pattern.findall(joined_text)
                counter = Counter(matches)
                per_umbrella_counters[umbrella] = counter
                overall_counter.update(counter)

            top_symbols = [sym for sym, _ in overall_counter.most_common(top_n)]
            rows = []
            for umbrella in umbrella_order:
                group = self.df[self.df["umbrella"] == umbrella]
                total_words = float(group["word_count"].sum())
                row = {"Label": self.get_umbrella_english_name(umbrella)}
                counter = per_umbrella_counters.get(umbrella, Counter())
                for sym in top_symbols:
                    count = counter.get(sym, 0)
                    row[sym] = round((count / total_words), 4) if total_words else 0.0
                rows.append(row)

            matrix_df = pd.DataFrame(rows)
            matrix_df.to_csv(
                os.path.join(self.tables_dir, f"{filename_stub}_usage_matrix_normalized.csv"),
                index=False,
                encoding="utf-8-sig",
            )
            self.save_comparison_matrix_table_png(
                matrix_df,
                filename=f"{filename_stub}_usage_matrix_normalized.png",
                row_color_lookup=row_color_lookup,
            )

        build_top_symbol_matrix(self.emoji_pattern, "Emoji", "emoji")
        build_top_symbol_matrix(self.emoticon_pattern, "Emoticon", "emoticon")

    def export_final_modeling_dataset(self) -> None:
        """Export the finalized cleaned dataset used by the machine learning phase.

        The modeling dataset is intentionally compact and traceable. It keeps the
        patient-written question text in its raw and cleaned forms, the final
        analysis-ready text used for text vectorization, and the umbrella target
        label used for supervised classification. Doctor answers are not exported
        as predictive inputs because the downstream model should learn from the
        patient question only.
        """
        assert self.df is not None

        # Build a stable row identifier after all preprocessing/filtering steps.
        modeling_df = self.df.reset_index(drop=True).copy()
        modeling_df.insert(0, "record_id", range(1, len(modeling_df) + 1))

        # Keep only the columns needed for ML plus label-traceability columns.
        export_columns = [
            "record_id",
            "raw_text",
            "clean_text",
            "analysis_text",
            "umbrella",
            "label_clean",
            "label_normalized",
        ]

        missing = [c for c in export_columns if c not in modeling_df.columns]
        if missing:
            raise ValueError(f"Cannot export modeling dataset. Missing columns: {missing}")

        final_modeling_df = modeling_df[export_columns].copy()

        # Ensure the final ML input is not empty.
        final_modeling_df = final_modeling_df[
            final_modeling_df["analysis_text"].fillna("").astype(str).str.strip().ne("")
        ].copy()

        # CSV is lightweight and easy to load in Python ML scripts.
        csv_path = os.path.join(self.modeling_dir, "final_modeling_dataset.csv")
        final_modeling_df.to_csv(csv_path, index=False, encoding="utf-8-sig")

        # XLSX is useful for manual inspection and supervisor review.
        xlsx_path = os.path.join(self.modeling_dir, "final_modeling_dataset.xlsx")
        try:
            final_modeling_df.to_excel(xlsx_path, index=False)
        except Exception as exc:
            # Keep the pipeline usable even if the local Excel writer is missing.
            print(
                "Warning: final_modeling_dataset.xlsx could not be written. "
                "The CSV export was still created successfully. "
                f"Excel export error: {exc}"
            )

        # Export a tiny modeling summary to confirm row and class counts.
        summary = final_modeling_df["umbrella"].value_counts().reset_index()
        summary.columns = ["umbrella", "row_count"]
        summary["percentage"] = (summary["row_count"] / len(final_modeling_df) * 100).round(2)
        summary.to_csv(
            os.path.join(self.modeling_dir, "final_modeling_dataset_class_summary.csv"),
            index=False,
            encoding="utf-8-sig",
        )

        print(f"Final modeling dataset exported: {len(final_modeling_df):,} rows")
        print(f"CSV:  {Path(csv_path).resolve()}")
        print(f"XLSX: {Path(xlsx_path).resolve()}")

    def concise_report_text(self) -> None:
        """Write a compact text report summarizing the exported outputs."""
        assert self.df is not None

        counts = self.df["umbrella"].value_counts()
        total = counts.sum()
        top_umbrella = counts.index[0]
        top_pct = counts.iloc[0] / total * 100

        report_lines = [
            "Quick Summary of Results",
            "========================",
            f"Total analyzed records: {len(self.df):,}",
            f"Number of final umbrellas: {self.df['umbrella'].nunique()}",
            f"Largest umbrella: {self.umbrella_display[top_umbrella]} at {top_pct:.1f}%",
            "",
            "Main improvements in this updated version:",
            "- conservative clitic cleanup",
            "- controlled definite article normalization",
            "- expanded phrase-level filtering for forum/discourse noise",
            "- tokenization audit export for debugging",
            "- repeated Arabic character normalization",
            "- preservation of negation terms",
            "- short-text token-count inspection",
            "- patient/doctor/mixed/unclear question POV analysis",
            "- experimental stemming audit without changing the main pipeline",
            "- comparison table against prior cleaning strategies",
            "- chart orientation guide and corrected vertical charts for compact distributions",
            "- combined all-sublabel unstacked visual for Section 4.3",
            "- extra inline comments explaining the full pipeline logic",
            "",
            "Important files to review:",
            "- tables/label_mapping_audit.csv",
            "- tables/tokenization_audit_sample.csv",
            "- tables/umbrella_distribution.csv",
            "- tables/length_profile_by_umbrella.csv",
            "- charts/question_length_distribution_by_umbrella.png",
            "- tables/top_words_by_umbrella.csv",
            "- tables/combined_all_sublabels_by_umbrella.csv",
            "- tables/combined_all_sublabels_chart_key.csv",
            "- charts/combined_all_sublabels_vertical_unstacked.png",
            "- charts/combined_common_rare_sublabels_vertical_unstacked.png",
            "- tables/top_unigrams_by_umbrella.csv",
            "- tables/top_bigrams_by_umbrella.csv",
            "- tables/top_trigrams_by_umbrella.csv",
            "- charts/punctuation_frequency_table.png",
            "- charts/emoji_frequency_table.png",
            "- charts/emoticon_frequency_table.png",
            "- tables/short_text_token_count_summary.csv",
            "- tables/short_text_samples.csv",
            "- tables/question_pov_distribution.csv",
            "- tables/question_pov_by_umbrella.csv",
            "- tables/experimental_stemming_lemmatization_audit.csv",
            "- tables/cleaning_comparison_with_prior_work.csv",
            "- tables/chart_orientation_guide.csv",
            "- tables/cleaning_before_after_row_counts.csv",
            "- modeling/final_modeling_dataset.csv",
            "- modeling/final_modeling_dataset.xlsx",
            "- modeling/final_modeling_dataset_class_summary.csv",
            "",
            "Note:",
            "If you want to refine class assignment, edit only map_to_umbrella and rerun.",
            "If you want to refine noise filtering, edit only _setup_stopwords and rerun.",
            "If you want to use stemming in modeling, compare model performance first.",
        ]

        with open(os.path.join(self.reports_dir, "summary_report.txt"), "w", encoding="utf-8") as f:
            f.write("\n".join(report_lines))

    def run_all(self) -> None:
        """Execute the full EDA pipeline from loading to export generation."""
        self.load_data()
        self.validate_columns()
        self.preprocess_dataframe()

        self.dataset_summary()
        self.umbrella_distribution()
        self.top_sublabels_within_each_umbrella()
        self.combined_common_rare_sublabel_distribution()
        self.length_profile_by_umbrella()
        self.top_words_by_umbrella()
        self.top_ngrams_by_umbrella()
        self.distinctive_words_by_umbrella()
        self.duplicates_quality_report()
        self.punctuation_emoji_profile()
        self.concise_report_text()

        print(f"Done. Outputs saved under: {Path(self.config.output_dir).resolve()}")


if __name__ == "__main__":
    CSV_PATH = "Neuropsychological_Conditions.csv"

    config = EDAConfig(
        csv_path=CSV_PATH,
        output_dir="umbrella_eda_output_report_outputs_v6",
        text_column="Question",
        label_column="Hierarchical Diagnosis",
        title_column="Question Title",
        answer_column="Answer",
        doctor_column="Doctor Name",
        consultation_column="Consultation Number",
        date_column="Date of Answer",
        top_n_words=12,
        top_n_sublabels=8,
        top_n_common_sublabels=3,
        bottom_n_rare_sublabels=2,
        top_n_ngrams=12,
        min_token_length=2,
        min_class_size=1,
        keep_other_class=True,
        normalize_definite_article_for_analysis=True,
        enable_clitic_cleanup=True,
        normalize_repeated_chars=True,
        max_repeated_chars=2,
        preserve_negation_terms=True,
        enable_question_pov_analysis=True,
        enable_short_text_analysis=True,
        enable_experimental_stemming_audit=True,
    )

    eda = ArabicMentalHealthUmbrellaEDA(config)
    eda.run_all()
