#!/usr/bin/env python3
"""
Pass 1: arXiv Large-Scale PDF Stylometric, Structural & Reference Extractor
===========================================================================

Streams PDFs directly from multi-TB .tar.gz / .tgz archives (zero disk
extraction), computes comprehensive readability, lexical diversity, entropy,
burstiness, stance, abstract-only, confounder/noise, authorship, structural,
and offline reference metrics into a CSV, and dumps extracted references into
a crash-safe .jsonl.gz file for decoupled Pass 2 API validation.

Tuned for resource-constrained hardware (quad-core i5-6500T, 8 GB RAM).

Dependencies:
    pip install pymupdf tqdm psutil

Next steps after running this script:
    Pass 2 (Reference Validation): python validate_references.py --input arxiv_references.jsonl.gz
    Pass 3 (Trend Plots & Stats):  python plot_trends.py --metrics-csv arxiv_metrics.csv --refs-csv ref_validation.csv --cutoff-date 2022-11
"""

import argparse
import atexit
import collections
import csv
import gc
import gzip
import json
import logging
import math
import multiprocessing as mp
import os
import re
import shutil
import signal
import sqlite3
import statistics
import sys
import tarfile
import tempfile
import time
from datetime import datetime
from logging.handlers import RotatingFileHandler
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

try:
    import fitz  # PyMuPDF
except ImportError:
    fitz = None

try:
    from tqdm import tqdm
except ImportError:
    tqdm = None

try:
    import psutil
except ImportError:
    psutil = None


# ---------------------------------------------------------------------------
# Constants, Lexicons & CSV Schema
# ---------------------------------------------------------------------------

CSV_COLUMNS = [
    # Identification & arXiv stamp confounders
    "archive_name",
    "member_name",
    "arxiv_id",
    "arxiv_version",
    "arxiv_category",
    "arxiv_primary_group",
    "pub_date",
    "exact_pub_date",
    "pub_year",
    "pub_month",
    "status",
    "pdf_creator",
    "pdf_producer",
    # Volume & extraction-noise confounders
    "page_count",
    "char_count",
    "word_count",
    "sentence_count",
    "alpha_char_ratio",
    "digit_char_ratio",
    "symbol_char_ratio",
    "non_ascii_ratio",
    "single_char_token_ratio",
    # Body readability & sentence complexity / burstiness
    "avg_sentence_length",
    "sentence_length_std",
    "sentence_length_median",
    "sentence_length_iqr",
    "avg_word_length",
    "avg_syllables_per_word",
    "complex_word_ratio",
    "fre",
    "fkgl",
    "cli",
    "fog",
    # Lexical diversity & information density
    "mattr_100",
    "yules_k",
    "hapax_ratio",
    "shannon_char_entropy",
    "shannon_word_entropy",
    # Voice, stance, punctuation & excess-word markers (rates per 1,000 words)
    "passive_sentence_count",
    "passive_voice_ratio",
    "first_person_plural_per_1k",
    "first_person_singular_per_1k",
    "hedging_per_1k",
    "booster_per_1k",
    "dash_per_1k",
    "semicolon_per_1k",
    "colon_per_1k",
    "paren_per_1k",
    "llm_excess_words_per_1k",
    "control_words_per_1k",
    # Abstract-only parallel metrics (clean prose control)
    "abstract_word_count",
    "abstract_sentence_count",
    "abstract_avg_sentence_length",
    "abstract_fre",
    "abstract_fkgl",
    "abstract_mattr_100",
    "abstract_passive_ratio",
    # Authorship & document structure signals
    "author_count",
    "author_source",
    "has_code_url",
    "code_url_count",
    "has_appendix",
    "has_acknowledgments",
    "has_limitations_or_ethics",
    "figure_mention_count",
    "table_mention_count",
    # Offline reference metrics & recency (live validation runs in Pass 2)
    "ref_count",
    "doi_count",
    "arxiv_ref_count",
    "ref_future_year_flags",
    "ref_malformed_doi_flags",
    "ref_min_year",
    "ref_median_year",
    "ref_max_year",
    "ref_mean_age",
    "error_message",
]

IRREGULAR_PAST_PARTICIPLES = frozenset({
    "arisen", "awoken", "beaten", "become", "begun", "bent", "bet", "bitten",
    "bled", "blown", "broken", "brought", "broadcast", "built", "burnt",
    "burst", "bought", "cast", "caught", "chosen", "clung", "come", "cost",
    "crept", "cut", "dealt", "dug", "done", "drawn", "dreamt", "driven",
    "drunk", "eaten", "fallen", "fed", "felt", "fought", "found", "fled",
    "flung", "flown", "forbidden", "forgotten", "forgiven", "frozen", "got",
    "gotten", "given", "gone", "ground", "grown", "hung", "had", "heard",
    "hidden", "hit", "held", "hurt", "kept", "knelt", "known", "laid", "led",
    "leant", "leapt", "learnt", "left", "lent", "let", "lain", "lit", "lost",
    "made", "meant", "met", "misunderstood", "mown", "overcome", "overdone",
    "overtaken", "overthrown", "paid", "pled", "proven", "put", "quit", "read",
    "rebuilt", "redone", "repaid", "rerun", "reset", "retold", "rewritten",
    "ridden", "rung", "risen", "run", "said", "seen", "sought", "sold", "sent",
    "set", "sewn", "shaken", "shorn", "shed", "shone", "shot", "shown",
    "shrunk", "shut", "sung", "sunk", "sat", "slain", "slept", "slid", "slung",
    "slit", "smelt", "sown", "spoken", "sped", "spelt", "spent", "spilt",
    "spun", "spat", "split", "spoilt", "spread", "sprung", "stood", "stolen",
    "stuck", "stung", "stunk", "stridden", "struck", "strung", "striven",
    "sworn", "swept", "swollen", "swum", "swung", "taken", "taught", "torn",
    "told", "thought", "thrown", "thrust", "trodden", "understood", "undergone",
    "undertaken", "undone", "unwound", "upheld", "upset", "woken", "worn",
    "woven", "wept", "won", "wound", "withdrawn", "withheld", "withstood",
    "wrung", "written",
})

PASSIVE_ADJECTIVE_EXCLUSIONS = frozenset({
    "advanced", "aged", "alleged", "amazed", "amused", "annoyed", "ashamed",
    "balanced", "biased", "bored", "bounded", "closed", "colored", "complicated",
    "concerned", "confused", "connected", "convinced", "crowded", "curved",
    "dedicated", "delighted", "depressed", "detailed", "determined", "devoted",
    "disappointed", "distributed", "embarrassed", "excited", "exhausted",
    "experienced", "fixed", "frightened", "generalized", "interested",
    "isolated", "layered", "limited", "localized", "married", "mixed",
    "motivated", "nested", "normalized", "organized", "pleased", "polarized",
    "prepared", "qualified", "related", "relaxed", "required", "reserved",
    "restricted", "satisfied", "scared", "shocked", "skilled", "sophisticated",
    "specialized", "standardized", "structured", "suited", "surprised",
    "tired", "unbalanced", "unbiased", "unbounded", "unexpected", "unified",
    "united", "unrelated", "unstructured", "upset", "weighted", "worried",
})

FIRST_PERSON_PLURAL = frozenset({"we", "our", "us", "ours", "ourselves"})
FIRST_PERSON_SINGULAR = frozenset({"i", "my", "me", "mine", "myself"})

HEDGING_WORDS = frozenset({
    "may", "might", "could", "would", "suggest", "suggests", "suggested",
    "appear", "appears", "appeared", "seem", "seems", "seemed", "likely",
    "unlikely", "possibly", "probably", "perhaps", "approximately", "roughly",
    "arguably", "potentially", "indicate", "indicates", "tend", "tends",
    "plausible", "tentative", "assume", "assumes", "estimated", "unclear",
})

BOOSTER_WORDS = frozenset({
    "clearly", "obviously", "undoubtedly", "certainly", "definitely",
    "crucially", "significantly", "substantially", "dramatically", "strongly",
    "demonstrate", "demonstrates", "demonstrated", "prove", "proves", "proved",
    "proven", "confirm", "confirms", "confirmed", "establish", "establishes",
    "undeniably", "remarkably", "notably", "decisively", "essential", "vital",
})

# Transparent fixed lexicon of documented post-2022 excess academic words
LLM_EXCESS_WORDS = frozenset({
    "delve", "delves", "delving", "delved", "intricate", "intricacies",
    "underscore", "underscores", "underscoring", "underscored", "pivotal",
    "meticulous", "meticulously", "tapestry", "realm", "realms", "showcase",
    "showcases", "showcasing", "multifaceted", "commendable", "foster",
    "fosters", "fostering", "paramount", "nuanced", "holistic", "transformative",
    "seamless", "seamlessly", "elucidate", "elucidates", "bolster", "bolsters",
    "testament", "vibrant", "catalyst", "encompass", "encompasses",
    "comprehensive", "noteworthy", "streamline", "streamlines", "boast", "boasts",
})

# Neutral baseline academic control lexicon (should remain relatively stable)
CONTROL_ACADEMIC_WORDS = frozenset({
    "method", "methods", "result", "results", "table", "figure", "section",
    "model", "models", "data", "value", "values", "system", "systems",
    "sample", "samples", "error", "errors", "parameter", "parameters",
    "function", "functions", "distribution", "experiment", "experiments",
    "analysis", "matrix", "vector", "equation", "algorithm",
})

MONTH_MAP = {
    "jan": 1, "feb": 2, "mar": 3, "apr": 4, "may": 5, "jun": 6,
    "jul": 7, "aug": 8, "sep": 9, "sept": 9, "oct": 10, "nov": 11, "dec": 12,
}

# ---------------------------------------------------------------------------
# Logging, Temp Cleanup, and Hardware Helpers
# ---------------------------------------------------------------------------

TEMP_DIR_TRACKER: Optional[Path] = None


def setup_logging(log_file: str, verbose: bool = False) -> logging.Logger:
    """Configure rotating file logging plus quiet/verbose stderr logging."""
    logger = logging.getLogger("arxiv_analyzer")
    logger.setLevel(logging.DEBUG)
    logger.handlers.clear()

    log_path = Path(log_file)
    log_path.parent.mkdir(parents=True, exist_ok=True)

    file_handler = RotatingFileHandler(
        log_path, maxBytes=20 * 1024 * 1024, backupCount=5, encoding="utf-8"
    )
    file_handler.setLevel(logging.DEBUG)
    file_fmt = logging.Formatter(
        "%(asctime)s | %(levelname)-7s | %(processName)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    file_handler.setFormatter(file_fmt)
    logger.addHandler(file_handler)

    console_handler = logging.StreamHandler(sys.stderr)
    console_handler.setLevel(logging.INFO if verbose else logging.WARNING)
    console_fmt = logging.Formatter("%(levelname)s: %(message)s")
    console_handler.setFormatter(console_fmt)
    logger.addHandler(console_handler)

    return logger


def init_safe_temp_dir(base_tmp: Optional[str] = None) -> Path:
    """Isolated temp-directory safeguard that cleans itself up on exit."""
    global TEMP_DIR_TRACKER
    tmp_path = Path(tempfile.mkdtemp(prefix="arxiv_analyzer_", dir=base_tmp))
    TEMP_DIR_TRACKER = tmp_path
    atexit.register(cleanup_temp_dir)
    return tmp_path


def cleanup_temp_dir() -> None:
    """Remove any scratch files left in the session temp directory."""
    global TEMP_DIR_TRACKER
    if TEMP_DIR_TRACKER and TEMP_DIR_TRACKER.exists():
        try:
            shutil.rmtree(TEMP_DIR_TRACKER, ignore_errors=True)
        except Exception:
            pass
        TEMP_DIR_TRACKER = None


def get_memory_usage_mb() -> float:
    """Return approximate resident memory (MB) across main + child workers."""
    if psutil is not None:
        try:
            proc = psutil.Process(os.getpid())
            rss = proc.memory_info().rss
            for child in proc.children(recursive=True):
                try:
                    rss += child.memory_info().rss
                except Exception:
                    pass
            return rss / (1024 * 1024)
        except Exception:
            pass
    try:
        with open("/proc/self/statm", "r", encoding="utf-8") as f:
            pages = int(f.read().split()[1])
            return (pages * os.sysconf("SC_PAGE_SIZE")) / (1024 * 1024)
    except Exception:
        return 0.0


def parse_arxiv_id_and_date(
    member_name: str, archive_name: str
) -> Tuple[str, str, str, Optional[int], Optional[int]]:
    """
    Extract (arxiv_id, version, pub_date YYYY-MM, pub_year, pub_month)
    from modern or legacy arXiv member/archive filenames.
    """
    basename = os.path.basename(member_name)
    stem = re.sub(r"\.pdf$", "", basename, flags=re.IGNORECASE)

    # Modern arXiv ID: 2305.12345 or 2305.12345v2
    m_mod = re.search(r"(\d{4}\.\d{4,5})(v\d+)?", stem)
    if m_mod:
        arxiv_id = m_mod.group(1)
        version = m_mod.group(2) or ""
        yy = int(arxiv_id[:2])
        mm = int(arxiv_id[2:4])
        if 1 <= mm <= 12:
            year = 2000 + yy if yy < 90 else 1900 + yy
            return arxiv_id, version, f"{year:04d}-{mm:02d}", year, mm

    # Legacy arXiv ID: hep-th9901001 or hep-th/9901001v1
    m_leg = re.search(
        r"([a-z\-]+/?)(\d{2})(\d{2})\d{3}(v\d+)?", member_name, re.IGNORECASE
    )
    if m_leg:
        yy = int(m_leg.group(2))
        mm = int(m_leg.group(3))
        version = m_leg.group(4) or ""
        if 1 <= mm <= 12:
            year = 2000 + yy if yy < 90 else 1900 + yy
            return stem, version, f"{year:04d}-{mm:02d}", year, mm

    # Fallback: bulk archive naming like arXiv_pdf_2103_001.tar.gz
    m_arc = re.search(r"(?:_|/|-)(\d{2})(\d{2})(?:_|\.|-)", archive_name)
    if m_arc:
        yy = int(m_arc.group(1))
        mm = int(m_arc.group(2))
        if 1 <= mm <= 12:
            year = 2000 + yy if yy < 90 else 1900 + yy
            return stem, "", f"{year:04d}-{mm:02d}", year, mm

    m_iso = re.search(r"(19\d{2}|20\d{2})[-_]?([01]\d)", archive_name)
    if m_iso:
        year = int(m_iso.group(1))
        mm = int(m_iso.group(2))
        if 1 <= mm <= 12:
            return stem, "", f"{year:04d}-{mm:02d}", year, mm

    return stem, "", "", None, None


# Page-1 arXiv margin stamp: e.g. "arXiv:2305.12345v2 [cs.CL] 18 May 2023"
RE_ARXIV_STAMP = re.compile(
    r"arXiv\s*:\s*([a-z\-]+/\d{7}|\d{4}\.\d{4,5})(v\d+)?\s*"
    r"(?:\[\s*([A-Za-z\-]+(?:\.[A-Za-z\-]+)?)\s*\])?\s*"
    r"(?:(\d{1,2})\s+([A-Za-z]{3,9})\s+(19\d{2}|20\d{2}))?",
    re.IGNORECASE,
)


def parse_arxiv_page1_stamp(first_page_text: str) -> Dict[str, Any]:
    """
    Parse the arXiv watermark stamp on page 1 for exact submission date,
    version number, and primary subject category (e.g. cs.CL -> cs).
    """
    result: Dict[str, Any] = {
        "stamp_id": "",
        "stamp_version": "",
        "arxiv_category": "",
        "arxiv_primary_group": "",
        "exact_pub_date": "",
        "stamp_year": None,
        "stamp_month": None,
    }
    if not first_page_text:
        return result

    # Stamp usually appears in the first or last 1,500 chars of page 1 text
    snippet = first_page_text[:1800] + "\n" + first_page_text[-1200:]
    m = RE_ARXIV_STAMP.search(snippet)
    if not m:
        return result

    result["stamp_id"] = m.group(1) or ""
    result["stamp_version"] = (m.group(2) or "").lower()
    cat = (m.group(3) or "").strip()
    if cat:
        result["arxiv_category"] = cat
        result["arxiv_primary_group"] = cat.split(".")[0].lower()

    day_s, mon_s, yr_s = m.group(4), m.group(5), m.group(6)
    if day_s and mon_s and yr_s:
        mon_num = MONTH_MAP.get(mon_s[:3].lower())
        if mon_num:
            try:
                day_num = int(day_s)
                yr_num = int(yr_s)
                if 1 <= day_num <= 31:
                    result["exact_pub_date"] = f"{yr_num:04d}-{mon_num:02d}-{day_num:02d}"
                    result["stamp_year"] = yr_num
                    result["stamp_month"] = mon_num
            except ValueError:
                pass

    return result

# ---------------------------------------------------------------------------
# Linguistic, Readability, Noise & Abstract Metrics (Fast, Pure-Python)
# ---------------------------------------------------------------------------

RE_HYPHEN_BREAK = re.compile(r"([A-Za-z])-\s*\n\s*([A-Za-z])")
RE_WHITESPACE = re.compile(r"\s+")
RE_WORD = re.compile(r"[A-Za-z]+(?:'[A-Za-z]+)?")
RE_VOWEL_GROUP = re.compile(r"[aeiouy]+")

RE_ABBREV = re.compile(
    r"\b(?:Fig|Figs|Eq|Eqs|Sec|Secs|Ref|Refs|Tab|Tabs|App|Vol|No|Nos|pp|"
    r"al|et|cf|vs|Dr|Prof|Mr|Mrs|Ms|St|Jr|Sr|Inc|Ltd|Dept|Univ|Jan|Feb|"
    r"Mar|Apr|Jun|Jul|Aug|Sep|Sept|Oct|Nov|Dec|e\.g|i\.e)\.",
    re.IGNORECASE,
)
RE_SENT_SPLIT = re.compile(r"(?<=[.!?])\s+(?=[A-Z0-9(\[\"'`])")

RE_PASSIVE_AUX = re.compile(
    r"\b(?:am|is|are|was|were|be|been|being|get|gets|got|gotten)\s+"
    r"(?:(?:not|never|also|often|usually|always|already|still|now|just|only|"
    r"frequently|typically|commonly|widely|generally|previously|recently|"
    r"directly|indirectly|easily|simply|well|largely|mostly|partly|fully|"
    r"further|then|thus|here|there|[a-z]+ly)\s+){0,2}"
    r"([a-z]{3,})\b",
    re.IGNORECASE,
)

RE_REF_HEADER = re.compile(
    r"(?im)^\s*(?:\d+\.?\s*)?(?:references|bibliography|works\s+cited)\s*$"
)
RE_ABSTRACT_START = re.compile(r"(?im)^\s*abstract[\s.:\-—]*$")
RE_ABSTRACT_INLINE = re.compile(r"(?im)^\s*abstract\s*[:.\-—]\s+(.+)")
RE_ABSTRACT_END = re.compile(
    r"(?im)^\s*(?:(?:1|I)\.?\s+introduction\b|introduction\s*$|keywords\b|"
    r"index\s+terms\b|ccs\s+concepts\b|1\.?\s+background\b|1\.?\s+motivation\b|"
    r"1\.?\s+overview\b)"
)
RE_DASH = re.compile(r"(?:—|–|--)")


def clean_pdf_text(raw_text: str) -> str:
    """Repair PDF line-break hyphenation and remove non-printable control chars."""
    text = RE_HYPHEN_BREAK.sub(r"\1\2", raw_text)
    text = "".join(ch if (ch >= " " or ch in "\n\t") else " " for ch in text)
    return text


def split_body_and_references(text: str) -> Tuple[str, str]:
    """Split paper text into main prose body and references section."""
    if not text:
        return "", ""
    search_start = int(len(text) * 0.45)
    matches = list(RE_REF_HEADER.finditer(text, pos=search_start))
    if not matches:
        matches = list(RE_REF_HEADER.finditer(text))
    if matches:
        last_match = matches[-1]
        return text[: last_match.start()], text[last_match.end() :]
    return text, ""


def extract_abstract_text(first_two_pages_text: str) -> str:
    """
    Extract the clean abstract block from the first 1–2 pages.
    Returns empty string if no plausible abstract block (30–700 words) is found.
    """
    if not first_two_pages_text:
        return ""

    start_pos = None
    m_head = RE_ABSTRACT_START.search(first_two_pages_text)
    if m_head:
        start_pos = m_head.end()
    else:
        m_inline = RE_ABSTRACT_INLINE.search(first_two_pages_text)
        if m_inline:
            start_pos = m_inline.start(1)

    if start_pos is None:
        return ""

    tail = first_two_pages_text[start_pos : start_pos + 5500]
    m_end = RE_ABSTRACT_END.search(tail)
    if m_end and m_end.start() > 80:
        candidate = tail[: m_end.start()]
    else:
        # Fallback: stop at first double newline after at least 250 chars
        m_para = re.search(r"\n\s*\n", tail[250:])
        candidate = tail[: 250 + m_para.start()] if m_para else tail[:2200]

    candidate = candidate.strip()
    wc = len(RE_WORD.findall(candidate))
    if 25 <= wc <= 800:
        return candidate
    return ""


def count_syllables(word: str) -> int:
    """Fast heuristic English syllable counter for FRE, FKGL, and Fog."""
    w = word.lower().strip("'")
    if len(w) <= 3:
        return 1
    if w.endswith("es") and not w.endswith(("ses", "zes", "xes", "ches", "shes")):
        w = w[:-2]
    elif w.endswith("ed") and not w.endswith(("ted", "ded")):
        w = w[:-2]
    elif w.endswith("e") and not w.endswith(("le", "ee", "ye", "oe")):
        w = w[:-1]
    groups = RE_VOWEL_GROUP.findall(w)
    count = len(groups)
    return count if count > 0 else 1


def split_sentences(text: str) -> List[str]:
    """Split cleaned prose into sentences while respecting common abbreviations."""
    collapsed = RE_WHITESPACE.sub(" ", text).strip()
    if not collapsed:
        return []
    protected = RE_ABBREV.sub(lambda m: m.group(0).replace(".", "<DOT>"), collapsed)
    protected = re.sub(r"\b([A-Z])\.\s+(?=[A-Z])", r"\1<DOT> ", protected)
    raw_sents = RE_SENT_SPLIT.split(protected)
    sentences = []
    for s in raw_sents:
        s_clean = s.replace("<DOT>", ".").strip()
        if len(RE_WORD.findall(s_clean)) >= 3:
            sentences.append(s_clean)
    return sentences


def is_passive_sentence(sentence: str) -> bool:
    """Lightweight regex + irregular participle passive voice detector."""
    for m in RE_PASSIVE_AUX.finditer(sentence):
        candidate = m.group(1).lower()
        if candidate in PASSIVE_ADJECTIVE_EXCLUSIONS:
            continue
        if candidate in IRREGULAR_PAST_PARTICIPLES:
            return True
        if len(candidate) >= 4 and candidate.endswith("ed"):
            return True
    return False


def compute_mattr(tokens: List[str], window: int = 100) -> float:
    """O(N) sliding-window Moving-Average Type-Token Ratio."""
    n = len(tokens)
    if n == 0:
        return 0.0
    if n <= window:
        return len(set(tokens)) / n

    counts = collections.Counter(tokens[:window])
    unique = len(counts)
    ttr_sum = unique / window
    steps = 1

    for i in range(window, n):
        outgoing = tokens[i - window]
        counts[outgoing] -= 1
        if counts[outgoing] == 0:
            del counts[outgoing]
            unique -= 1

        incoming = tokens[i]
        if counts[incoming] == 0:
            unique += 1
        counts[incoming] += 1

        ttr_sum += unique / window
        steps += 1

    return ttr_sum / steps


def compute_shannon_char_entropy(text: str) -> float:
    """Shannon character entropy H = -sum(p(c) * log2(p(c)))."""
    normalized = RE_WHITESPACE.sub(" ", text).strip()
    n = len(normalized)
    if n == 0:
        return 0.0
    counts = collections.Counter(normalized)
    return -sum((c / n) * math.log2(c / n) for c in counts.values())


def compute_shannon_word_entropy_and_diversity(
    words: List[str],
) -> Tuple[float, float, float]:
    """
    Compute (shannon_word_entropy, yules_k, hapax_ratio) from lowercase word list.
    Yule's K = 10^4 * (sum(i^2 * V_i) - N) / N^2 (lower K = richer vocabulary).
    """
    n = len(words)
    if n == 0:
        return 0.0, 0.0, 0.0
    counts = collections.Counter(words)
    word_entropy = -sum((c / n) * math.log2(c / n) for c in counts.values())

    freq_of_freq = collections.Counter(counts.values())
    m2 = sum((freq * freq) * num_types for freq, num_types in freq_of_freq.items())
    yules_k = 10000.0 * (m2 - n) / (n * n) if n > 1 else 0.0
    hapax_ratio = freq_of_freq.get(1, 0) / n
    return word_entropy, yules_k, hapax_ratio


def compute_noise_ratios(text: str, words: List[str]) -> Dict[str, float]:
    """
    Character and token composition ratios to control for math-heavy papers,
    tables, and PDF extraction / OCR noise.
    """
    non_ws = [ch for ch in text if not ch.isspace()]
    n_chars = len(non_ws)
    if n_chars == 0:
        return {
            "alpha_char_ratio": 0.0,
            "digit_char_ratio": 0.0,
            "symbol_char_ratio": 0.0,
            "non_ascii_ratio": 0.0,
            "single_char_token_ratio": 0.0,
        }

    alpha = 0
    digit = 0
    non_ascii = 0
    for ch in non_ws:
        if ch.isalpha():
            alpha += 1
        elif ch.isdigit():
            digit += 1
        if ord(ch) > 127:
            non_ascii += 1

    symbol = n_chars - alpha - digit
    single_char_tokens = sum(1 for w in words if len(w) == 1)
    single_ratio = (single_char_tokens / len(words)) if words else 0.0

    return {
        "alpha_char_ratio": round(alpha / n_chars, 5),
        "digit_char_ratio": round(digit / n_chars, 5),
        "symbol_char_ratio": round(symbol / n_chars, 5),
        "non_ascii_ratio": round(non_ascii / n_chars, 5),
        "single_char_token_ratio": round(single_ratio, 5),
    }


def _iqr(values: List[int]) -> float:
    if len(values) < 4:
        return 0.0
    qs = statistics.quantiles(values, n=4, method="inclusive")
    return float(qs[2] - qs[0])


def compute_linguistic_metrics(body_text: str) -> Dict[str, Any]:
    """Compute full readability, burstiness, entropy, stance, punctuation, and noise metrics."""
    sentences = split_sentences(body_text)
    words = [w.lower() for w in RE_WORD.findall(body_text)]

    word_count = len(words)
    sentence_count = len(sentences)
    char_count = len(body_text)

    noise = compute_noise_ratios(body_text, words)

    if word_count == 0 or sentence_count == 0:
        out = {
            "char_count": char_count,
            "word_count": word_count,
            "sentence_count": sentence_count,
            "avg_sentence_length": "",
            "sentence_length_std": "",
            "sentence_length_median": "",
            "sentence_length_iqr": "",
            "avg_word_length": "",
            "avg_syllables_per_word": "",
            "complex_word_ratio": "",
            "fre": "",
            "fkgl": "",
            "cli": "",
            "fog": "",
            "mattr_100": "",
            "yules_k": "",
            "hapax_ratio": "",
            "shannon_char_entropy": round(compute_shannon_char_entropy(body_text), 5),
            "shannon_word_entropy": "",
            "passive_sentence_count": 0,
            "passive_voice_ratio": "",
            "first_person_plural_per_1k": "",
            "first_person_singular_per_1k": "",
            "hedging_per_1k": "",
            "booster_per_1k": "",
            "dash_per_1k": "",
            "semicolon_per_1k": "",
            "colon_per_1k": "",
            "paren_per_1k": "",
            "llm_excess_words_per_1k": "",
            "control_words_per_1k": "",
        }
        out.update(noise)
        return out

    # Sentence length & burstiness
    sent_word_counts = [len(RE_WORD.findall(s)) for s in sentences]
    total_sent_words = sum(sent_word_counts)
    avg_sentence_length = total_sent_words / sentence_count
    sent_std = statistics.stdev(sent_word_counts) if sentence_count > 1 else 0.0
    sent_median = statistics.median(sent_word_counts)
    sent_iqr = _iqr(sent_word_counts)

    # Word lengths, syllables, and complex words (>= 3 syllables)
    syl_cache: Dict[str, int] = {}
    total_syllables = 0
    total_letters = 0
    complex_words = 0

    fp_plural = 0
    fp_singular = 0
    hedges = 0
    boosters = 0
    llm_excess = 0
    control_hits = 0

    for w in words:
        total_letters += len(w)
        s = syl_cache.get(w)
        if s is None:
            s = count_syllables(w)
            syl_cache[w] = s
        total_syllables += s
        if s >= 3:
            complex_words += 1

        if w in FIRST_PERSON_PLURAL:
            fp_plural += 1
        elif w in FIRST_PERSON_SINGULAR:
            fp_singular += 1
        if w in HEDGING_WORDS:
            hedges += 1
        if w in BOOSTER_WORDS:
            boosters += 1
        if w in LLM_EXCESS_WORDS:
            llm_excess += 1
        if w in CONTROL_ACADEMIC_WORDS:
            control_hits += 1

    avg_word_length = total_letters / word_count
    avg_syl_per_word = total_syllables / word_count
    complex_word_ratio = complex_words / word_count

    # Readability suite
    fre = 206.835 - 1.015 * avg_sentence_length - 84.6 * avg_syl_per_word
    fkgl = 0.39 * avg_sentence_length + 11.8 * avg_syl_per_word - 15.59
    # Coleman-Liau Index: L = letters per 100 words, S = sentences per 100 words
    cli_l = avg_word_length * 100.0
    cli_s = (sentence_count / word_count) * 100.0
    cli = 0.0588 * cli_l - 0.296 * cli_s - 15.8
    # Gunning Fog Index
    fog = 0.4 * (avg_sentence_length + 100.0 * complex_word_ratio)

    # Lexical diversity & entropies
    mattr_100 = compute_mattr(words, window=100)
    shannon_char_ent = compute_shannon_char_entropy(body_text)
    shannon_word_ent, yules_k, hapax_ratio = compute_shannon_word_entropy_and_diversity(words)

    # Passive voice
    passive_count = sum(1 for s in sentences if is_passive_sentence(s))
    passive_ratio = passive_count / sentence_count

    # Punctuation & lexicon rates per 1,000 words
    scale_1k = 1000.0 / word_count
    dash_cnt = len(RE_DASH.findall(body_text))
    semicolon_cnt = body_text.count(";")
    colon_cnt = body_text.count(":")
    paren_cnt = body_text.count("(")

    out = {
        "char_count": char_count,
        "word_count": word_count,
        "sentence_count": sentence_count,
        "avg_sentence_length": round(avg_sentence_length, 4),
        "sentence_length_std": round(sent_std, 4),
        "sentence_length_median": round(float(sent_median), 4),
        "sentence_length_iqr": round(sent_iqr, 4),
        "avg_word_length": round(avg_word_length, 4),
        "avg_syllables_per_word": round(avg_syl_per_word, 4),
        "complex_word_ratio": round(complex_word_ratio, 5),
        "fre": round(fre, 4),
        "fkgl": round(fkgl, 4),
        "cli": round(cli, 4),
        "fog": round(fog, 4),
        "mattr_100": round(mattr_100, 5),
        "yules_k": round(yules_k, 4),
        "hapax_ratio": round(hapax_ratio, 5),
        "shannon_char_entropy": round(shannon_char_ent, 5),
        "shannon_word_entropy": round(shannon_word_ent, 5),
        "passive_sentence_count": passive_count,
        "passive_voice_ratio": round(passive_ratio, 5),
        "first_person_plural_per_1k": round(fp_plural * scale_1k, 4),
        "first_person_singular_per_1k": round(fp_singular * scale_1k, 4),
        "hedging_per_1k": round(hedges * scale_1k, 4),
        "booster_per_1k": round(boosters * scale_1k, 4),
        "dash_per_1k": round(dash_cnt * scale_1k, 4),
        "semicolon_per_1k": round(semicolon_cnt * scale_1k, 4),
        "colon_per_1k": round(colon_cnt * scale_1k, 4),
        "paren_per_1k": round(paren_cnt * scale_1k, 4),
        "llm_excess_words_per_1k": round(llm_excess * scale_1k, 4),
        "control_words_per_1k": round(control_hits * scale_1k, 4),
    }
    out.update(noise)
    return out


def compute_abstract_metrics(abstract_text: str) -> Dict[str, Any]:
    """Compute a compact parallel set of clean-prose metrics on the abstract alone."""
    empty = {
        "abstract_word_count": 0,
        "abstract_sentence_count": 0,
        "abstract_avg_sentence_length": "",
        "abstract_fre": "",
        "abstract_fkgl": "",
        "abstract_mattr_100": "",
        "abstract_passive_ratio": "",
    }
    if not abstract_text:
        return empty

    sents = split_sentences(abstract_text)
    words = [w.lower() for w in RE_WORD.findall(abstract_text)]
    if len(words) < 20 or len(sents) == 0:
        return empty

    sent_wc = [len(RE_WORD.findall(s)) for s in sents]
    avg_sl = sum(sent_wc) / len(sents)
    total_syl = sum(count_syllables(w) for w in words)
    avg_syl = total_syl / len(words)
    fre = 206.835 - 1.015 * avg_sl - 84.6 * avg_syl
    fkgl = 0.39 * avg_sl + 11.8 * avg_syl - 15.59
    mattr = compute_mattr(words, window=100)
    passive_ratio = sum(1 for s in sents if is_passive_sentence(s)) / len(sents)

    return {
        "abstract_word_count": len(words),
        "abstract_sentence_count": len(sents),
        "abstract_avg_sentence_length": round(avg_sl, 4),
        "abstract_fre": round(fre, 4),
        "abstract_fkgl": round(fkgl, 4),
        "abstract_mattr_100": round(mattr, 5),
        "abstract_passive_ratio": round(passive_ratio, 5),
    }

# ---------------------------------------------------------------------------
# Authorship, Structural Signals & Offline Reference Extraction
# ---------------------------------------------------------------------------

RE_DOI = re.compile(r"\b(10\.\d{4,9}/[-._;()/:A-Za-z0-9]+)", re.IGNORECASE)
RE_ARXIV_REF = re.compile(
    r"\barXiv\s*[:\s]\s*(\d{4}\.\d{4,5}(?:v\d+)?|[a-z\-]+/\d{7}(?:v\d+)?)",
    re.IGNORECASE,
)
RE_REF_ITEM_BRACKET = re.compile(r"(?m)^\s*\[\d+\]\s+")
RE_REF_ITEM_NUMBER = re.compile(r"(?m)^\s*\d{1,3}\.\s+[A-Z]")
RE_YEAR_IN_REF = re.compile(r"\b(19\d{2}|20\d{2})\b")
RE_NAME_PAIR = re.compile(
    r"\b[A-Z][a-zA-Z'\-]{1,20}(?:\s+[A-Z]\.?)?\s+[A-Z][a-zA-Z'\-]{1,24}\b"
)
RE_AFFILIATION_NOISE = re.compile(
    r"\b(?:university|institute|department|laboratory|lab|college|school|"
    r"center|centre|corporation|inc|google|meta|microsoft|amazon|deepmind|"
    r"openai|nvidia|ibm|eth\s+zurich|mit|stanford|berkeley|oxford|cambridge|"
    r"abstract|keywords|preprint|submitted|accepted|published|arxiv|doi|http|www|@)\b",
    re.IGNORECASE,
)

# Structural / content signals
RE_CODE_URL = re.compile(
    r"(?:https?://)?(?:www\.)?(?:github\.com|gitlab\.com|huggingface\.co|bitbucket\.org)/[A-Za-z0-9_.\-]+(?:/[A-Za-z0-9_.\-]+)?",
    re.IGNORECASE,
)
RE_APPENDIX_HDR = re.compile(
    r"(?im)^\s*(?:[A-Z0-9]+\.?\s*)?(?:appendix|appendices|supplementary\s+material)\b"
)
RE_ACK_HDR = re.compile(
    r"(?im)^\s*(?:\d+\.?\s*)?acknowledg(?:e)?ments?\b"
)
RE_LIMITATIONS_HDR = re.compile(
    r"(?im)^\s*(?:\d+\.?\s*)?(?:limitations?|broader\s+impact|ethical\s+considerations?|ethics\s+statement|societal\s+impact)\b"
)
RE_FIG_MENTION = re.compile(r"\b(?:Fig(?:ure|s)?\.?)\s*\d+", re.IGNORECASE)
RE_TAB_MENTION = re.compile(r"\b(?:Tab(?:le|s)?\.?)\s*\d+", re.IGNORECASE)


def estimate_author_count(
    pdf_metadata: Dict[str, Any], first_page_text: str
) -> Tuple[int, str]:
    """Estimate number of authors from PDF metadata first, then page-1 heuristics."""
    meta_author = (pdf_metadata or {}).get("author") or ""
    meta_author = meta_author.strip()
    if meta_author and len(meta_author) >= 3:
        if not re.search(r"^(?:unknown|anonymous|latex|author|user)$", meta_author, re.I):
            cleaned = re.sub(r"\s+and\s+", ";", meta_author, flags=re.IGNORECASE)
            cleaned = cleaned.replace("&", ";")
            if ";" in cleaned:
                parts = [p.strip() for p in cleaned.split(";") if len(p.strip()) >= 2]
            else:
                parts = [p.strip() for p in cleaned.split(",") if len(p.strip()) >= 2]
                if len(parts) == 2 and all(" " not in p for p in parts):
                    parts = [meta_author]
            if 1 <= len(parts) <= 200:
                return len(parts), "pdf_metadata"

    if not first_page_text:
        return 0, "unknown"

    header_match = re.search(r"(?im)^\s*abstract\b", first_page_text)
    header_block = (
        first_page_text[: header_match.start()]
        if header_match
        else first_page_text[:1800]
    )

    lines = [ln.strip() for ln in header_block.splitlines() if ln.strip()]
    candidate_lines = lines[1:] if len(lines) > 3 else lines
    emails = set(re.findall(r"[A-Za-z0-9._%+\-]+@[A-Za-z0-9.\-]+\.[A-Za-z]{2,}", header_block))

    author_candidates: List[str] = []
    for ln in candidate_lines[:25]:
        if RE_AFFILIATION_NOISE.search(ln):
            continue
        ln_clean = re.sub(r"[\*\†\‡\§\¶\d]+", "", ln)
        author_candidates.extend(RE_NAME_PAIR.findall(ln_clean))

    count = max(len(set(author_candidates)), len(emails))
    if count > 0:
        return min(count, 200), "page1_heuristic"
    return 0, "unknown"


def extract_structural_signals(full_text: str, body_text: str) -> Dict[str, Any]:
    """Detect code repository URLs, appendix, acknowledgments, ethics/limitations, and fig/table mentions."""
    code_urls = {u.lower().rstrip(".,;)]}") for u in RE_CODE_URL.findall(full_text)}
    return {
        "has_code_url": 1 if code_urls else 0,
        "code_url_count": len(code_urls),
        "has_appendix": 1 if RE_APPENDIX_HDR.search(full_text) else 0,
        "has_acknowledgments": 1 if RE_ACK_HDR.search(full_text) else 0,
        "has_limitations_or_ethics": 1 if RE_LIMITATIONS_HDR.search(full_text) else 0,
        "figure_mention_count": len(RE_FIG_MENTION.findall(body_text)),
        "table_mention_count": len(RE_TAB_MENTION.findall(body_text)),
    }


def extract_references(
    ref_text: str,
    paper_year: Optional[int],
    dump_max_entries: int = 25,
) -> Dict[str, Any]:
    """
    Parse bibliography entries, extract DOIs and arXiv IDs, compute offline
    structural flags and reference recency stats, and select entries to dump
    into the decoupled .jsonl.gz file for Pass 2 validation.
    """
    empty = {
        "ref_count": 0,
        "doi_count": 0,
        "arxiv_ref_count": 0,
        "ref_future_year_flags": 0,
        "ref_malformed_doi_flags": 0,
        "ref_min_year": "",
        "ref_median_year": "",
        "ref_max_year": "",
        "ref_mean_age": "",
        "dump_payload": None,
    }
    if not ref_text or len(ref_text.strip()) < 20:
        return empty

    bracket_starts = [m.start() for m in RE_REF_ITEM_BRACKET.finditer(ref_text)]
    num_starts = [m.start() for m in RE_REF_ITEM_NUMBER.finditer(ref_text)]

    entries: List[str] = []
    if len(bracket_starts) >= 3:
        for i, st in enumerate(bracket_starts):
            end = bracket_starts[i + 1] if i + 1 < len(bracket_starts) else len(ref_text)
            entries.append(RE_WHITESPACE.sub(" ", ref_text[st:end]).strip())
    elif len(num_starts) >= 3:
        for i, st in enumerate(num_starts):
            end = num_starts[i + 1] if i + 1 < len(num_starts) else len(ref_text)
            entries.append(RE_WHITESPACE.sub(" ", ref_text[st:end]).strip())
    else:
        blocks = [RE_WHITESPACE.sub(" ", b).strip() for b in re.split(r"\n\s*\n", ref_text)]
        entries = [b for b in blocks if len(b) >= 25]

    entries = entries[:500]
    unique_dois = sorted({d.rstrip(".,;)]}") for d in RE_DOI.findall(ref_text)})
    unique_arxiv = sorted(set(RE_ARXIV_REF.findall(ref_text)))

    max_plausible_year = (paper_year + 1) if paper_year else (datetime.now().year + 1)
    future_year_flags = 0
    entry_years: List[int] = []

    for entry in entries:
        yrs = [int(y) for y in RE_YEAR_IN_REF.findall(entry)]
        if yrs:
            yr = max(yrs)
            if yr > max_plausible_year:
                future_year_flags += 1
            elif 1900 <= yr <= max_plausible_year:
                entry_years.append(yr)

    raw_doi_mentions = re.findall(r"\bdoi\s*[:=]\s*(\S+)", ref_text, re.IGNORECASE)
    malformed_doi_flags = 0
    for token in raw_doi_mentions:
        tok = token.rstrip(".,;)]}")
        if not re.match(r"^(?:https?://(?:dx\.)?doi\.org/)?10\.\d{4,9}/\S+$", tok, re.I):
            malformed_doi_flags += 1

    # Reference recency statistics
    ref_min_year: Any = min(entry_years) if entry_years else ""
    ref_max_year: Any = max(entry_years) if entry_years else ""
    ref_median_year: Any = round(float(statistics.median(entry_years)), 2) if entry_years else ""
    ref_mean_age: Any = ""
    if entry_years and paper_year:
        ages = [max(0, paper_year - y) for y in entry_years]
        ref_mean_age = round(sum(ages) / len(ages), 3)

    # Select entries to save for Pass 2 (0 = save all entries; >0 = evenly spaced sample)
    if dump_max_entries <= 0 or len(entries) <= dump_max_entries:
        sampled_entries = entries
    else:
        step = len(entries) / dump_max_entries
        sampled_entries = [entries[int(i * step)] for i in range(dump_max_entries)]

    # Truncate unusually long individual entries to keep .jsonl.gz compact
    compact_entries = [e[:450] for e in sampled_entries]

    dump_payload = {
        "ref_count": len(entries),
        "dois": unique_dois[:150],
        "arxiv_refs": unique_arxiv[:150],
        "entries": compact_entries,
    }

    return {
        "ref_count": len(entries),
        "doi_count": len(unique_dois),
        "arxiv_ref_count": len(unique_arxiv),
        "ref_future_year_flags": future_year_flags,
        "ref_malformed_doi_flags": malformed_doi_flags,
        "ref_min_year": ref_min_year,
        "ref_median_year": ref_median_year,
        "ref_max_year": ref_max_year,
        "ref_mean_age": ref_mean_age,
        "dump_payload": dump_payload,
    }

# ---------------------------------------------------------------------------
# Checkpoint / Resume State Manager (SQLite + CSV + Crash-Safe .jsonl.gz)
# ---------------------------------------------------------------------------

class StateManager:
    """
    Tracks completed archives and processed PDFs in SQLite (WAL mode),
    appends metrics to CSV in batches, and writes extracted reference records
    as concatenated RFC-1952 gzip members to a .jsonl.gz file.

    Using independent gzip-compressed batch frames appended in binary mode means:
      1. Standard `gzip.open(..., 'rt')` and `zcat` read the file seamlessly.
      2. Stopping and resuming appends cleanly without rewriting old data.
      3. Even an unexpected power loss cannot corrupt previously flushed batches.
    """

    def __init__(self, state_db: str, csv_path: str, refs_jsonl_gz: Optional[str]):
        self.state_db = state_db
        self.csv_path = Path(csv_path)
        self.csv_path.parent.mkdir(parents=True, exist_ok=True)
        Path(state_db).parent.mkdir(parents=True, exist_ok=True)

        self.conn = sqlite3.connect(self.state_db, timeout=60)
        self.conn.execute("PRAGMA journal_mode=WAL;")
        self.conn.execute("PRAGMA synchronous=NORMAL;")
        self._init_tables()

        need_header = (not self.csv_path.exists()) or (self.csv_path.stat().st_size == 0)
        self._csv_file = open(self.csv_path, "a", newline="", encoding="utf-8")
        self._csv_writer = csv.DictWriter(
            self._csv_file, fieldnames=CSV_COLUMNS, extrasaction="ignore"
        )
        if need_header:
            self._csv_writer.writeheader()
            self._csv_file.flush()

        self._refs_file = None
        if refs_jsonl_gz:
            refs_path = Path(refs_jsonl_gz)
            refs_path.parent.mkdir(parents=True, exist_ok=True)
            self._refs_file = open(refs_path, "ab")

        self._pending_rows: List[Tuple[str, str, str]] = []
        self._pending_ref_lines: List[str] = []

    def _init_tables(self) -> None:
        self.conn.executescript(
            """
            CREATE TABLE IF NOT EXISTS completed_archives (
                archive_name TEXT PRIMARY KEY,
                pdf_count    INTEGER NOT NULL,
                completed_at REAL NOT NULL
            );
            CREATE TABLE IF NOT EXISTS processed_pdfs (
                archive_name TEXT NOT NULL,
                member_name  TEXT NOT NULL,
                status       TEXT NOT NULL,
                processed_at REAL NOT NULL,
                PRIMARY KEY (archive_name, member_name)
            );
            """
        )
        self.conn.commit()

    def get_completed_archives(self) -> set:
        rows = self.conn.execute("SELECT archive_name FROM completed_archives").fetchall()
        return {r[0] for r in rows}

    def get_processed_members_for_archive(self, archive_name: str) -> set:
        rows = self.conn.execute(
            "SELECT member_name FROM processed_pdfs WHERE archive_name = ?",
            (archive_name,),
        ).fetchall()
        return {r[0] for r in rows}

    def total_processed_count(self) -> int:
        row = self.conn.execute("SELECT COUNT(*) FROM processed_pdfs").fetchone()
        return int(row[0]) if row else 0

    def record_result(
        self, row: Dict[str, Any], ref_record: Optional[Dict[str, Any]] = None
    ) -> None:
        self._csv_writer.writerow(row)
        self._pending_rows.append(
            (row["archive_name"], row["member_name"], row["status"])
        )
        if ref_record is not None and self._refs_file is not None:
            self._pending_ref_lines.append(
                json.dumps(ref_record, ensure_ascii=False, separators=(",", ":")) + "\n"
            )
        if len(self._pending_rows) >= 50:
            self.flush()

    def mark_archive_completed(self, archive_name: str, pdf_count: int) -> None:
        self.flush()
        self.conn.execute(
            """
            INSERT OR REPLACE INTO completed_archives (archive_name, pdf_count, completed_at)
            VALUES (?, ?, ?)
            """,
            (archive_name, pdf_count, time.time()),
        )
        self.conn.commit()

    def flush(self) -> None:
        if self._pending_ref_lines and self._refs_file is not None:
            raw_bytes = "".join(self._pending_ref_lines).encode("utf-8")
            self._refs_file.write(gzip.compress(raw_bytes, compresslevel=6))
            self._refs_file.flush()
            self._pending_ref_lines.clear()

        if self._pending_rows:
            now = time.time()
            self.conn.executemany(
                """
                INSERT OR REPLACE INTO processed_pdfs
                (archive_name, member_name, status, processed_at)
                VALUES (?, ?, ?, ?)
                """,
                [(a, m, s, now) for (a, m, s) in self._pending_rows],
            )
            self.conn.commit()
            self._pending_rows.clear()

        self._csv_file.flush()

    def close(self) -> None:
        try:
            self.flush()
        finally:
            for fh in (self._refs_file, self._csv_file):
                if fh is not None:
                    try:
                        fh.close()
                    except Exception:
                        pass
            try:
                self.conn.close()
            except Exception:
                pass

# ---------------------------------------------------------------------------
# Worker Function & Memory-Bounded Streaming Pipeline
# ---------------------------------------------------------------------------

_WORKER_CFG: Dict[str, Any] = {}


def _worker_init(cfg: Dict[str, Any]) -> None:
    """Initialize worker process state and ignore SIGINT (handled by main)."""
    global _WORKER_CFG
    signal.signal(signal.SIGINT, signal.SIG_IGN)
    _WORKER_CFG = cfg
    if fitz is not None:
        try:
            fitz.TOOLS.mupdf_display_errors(False)
        except Exception:
            pass


def _empty_row(
    archive_name: str,
    member_name: str,
    arxiv_id: str,
    version: str,
    pub_date: str,
    pub_year: Optional[int],
    pub_month: Optional[int],
    status: str = "ok",
    error_message: str = "",
) -> Dict[str, Any]:
    row = {col: "" for col in CSV_COLUMNS}
    row.update(
        {
            "archive_name": archive_name,
            "member_name": member_name,
            "arxiv_id": arxiv_id,
            "arxiv_version": version,
            "pub_date": pub_date,
            "pub_year": pub_year if pub_year is not None else "",
            "pub_month": pub_month if pub_month is not None else "",
            "status": status,
            "page_count": 0,
            "char_count": 0,
            "word_count": 0,
            "sentence_count": 0,
            "passive_sentence_count": 0,
            "abstract_word_count": 0,
            "abstract_sentence_count": 0,
            "author_count": 0,
            "author_source": "unknown",
            "has_code_url": 0,
            "code_url_count": 0,
            "has_appendix": 0,
            "has_acknowledgments": 0,
            "has_limitations_or_ethics": 0,
            "figure_mention_count": 0,
            "table_mention_count": 0,
            "ref_count": 0,
            "doi_count": 0,
            "arxiv_ref_count": 0,
            "ref_future_year_flags": 0,
            "ref_malformed_doi_flags": 0,
            "error_message": error_message,
        }
    )
    return row


def process_pdf_bytes(
    task: Tuple[str, str, bytes]
) -> Tuple[Dict[str, Any], Optional[Dict[str, Any]]]:
    """
    Parse one PDF completely in RAM and return (csv_row, optional_ref_record).
    Always returns cleanly even on corrupt, encrypted, or image-only PDFs.
    """
    archive_name, member_name, pdf_bytes = task
    arxiv_id, version, pub_date, pub_year, pub_month = parse_arxiv_id_and_date(
        member_name, archive_name
    )
    row = _empty_row(
        archive_name, member_name, arxiv_id, version, pub_date, pub_year, pub_month
    )
    ref_record: Optional[Dict[str, Any]] = None
    doc = None

    try:
        max_pages = int(_WORKER_CFG.get("max_pages", 150))
        dump_max_refs = int(_WORKER_CFG.get("dump_max_refs", 25))
        save_refs = bool(_WORKER_CFG.get("save_refs", True))

        doc = fitz.open(stream=pdf_bytes, filetype="pdf")
        if doc.is_encrypted:
            try:
                if not doc.authenticate(""):
                    row["status"] = "encrypted"
                    row["error_message"] = "Password-protected PDF"
                    return row, None
            except Exception:
                row["status"] = "encrypted"
                row["error_message"] = "Encrypted PDF"
                return row, None

        page_count = len(doc)
        row["page_count"] = page_count
        if page_count == 0:
            row["status"] = "empty"
            row["error_message"] = "0 pages"
            return row, None

        metadata = doc.metadata or {}
        row["pdf_creator"] = (metadata.get("creator") or "").strip()[:120]
        row["pdf_producer"] = (metadata.get("producer") or "").strip()[:120]

        first_page_text = ""
        second_page_text = ""
        page_texts: List[str] = []

        pages_to_read = min(page_count, max_pages)
        for idx in range(pages_to_read):
            try:
                txt = doc[idx].get_text("text")
            except Exception:
                txt = ""
            if idx == 0:
                first_page_text = txt
            elif idx == 1:
                second_page_text = txt
            if txt:
                page_texts.append(txt)

        if page_count > max_pages:
            for idx in range(max(pages_to_read, page_count - 5), page_count):
                try:
                    txt = doc[idx].get_text("text")
                    if txt:
                        page_texts.append(txt)
                except Exception:
                    pass

        # Parse page-1 arXiv watermark stamp for category, version, exact date
        stamp = parse_arxiv_page1_stamp(first_page_text)
        if stamp["arxiv_category"]:
            row["arxiv_category"] = stamp["arxiv_category"]
            row["arxiv_primary_group"] = stamp["arxiv_primary_group"]
        if stamp["stamp_version"] and not row["arxiv_version"]:
            row["arxiv_version"] = stamp["stamp_version"]
        if stamp["exact_pub_date"]:
            row["exact_pub_date"] = stamp["exact_pub_date"]
            if not pub_date and stamp["stamp_year"] and stamp["stamp_month"]:
                pub_year = stamp["stamp_year"]
                pub_month = stamp["stamp_month"]
                pub_date = f"{pub_year:04d}-{pub_month:02d}"
                row["pub_date"] = pub_date
                row["pub_year"] = pub_year
                row["pub_month"] = pub_month

        full_text = clean_pdf_text("\n".join(page_texts))
        del page_texts

        if len(full_text.strip()) < 100:
            row["status"] = "no_text"
            row["error_message"] = "Scanned or image-only PDF (<100 chars extracted)"
            return row, None

        body_text, ref_text = split_body_and_references(full_text)
        if len(body_text.strip()) < 100:
            body_text = full_text

        # 1. Body linguistic, readability, burstiness, stance, noise metrics
        row.update(compute_linguistic_metrics(body_text))

        # 2. Abstract-only parallel metrics
        first_two_clean = clean_pdf_text(first_page_text + "\n" + second_page_text)
        abstract_text = extract_abstract_text(first_two_clean)
        row.update(compute_abstract_metrics(abstract_text))

        # 3. Authorship & structural signals
        author_count, author_source = estimate_author_count(metadata, first_page_text)
        row["author_count"] = author_count
        row["author_source"] = author_source
        row.update(extract_structural_signals(full_text, body_text))

        # 4. Offline reference parsing, structural flags, recency & dump payload
        ref_info = extract_references(ref_text, pub_year, dump_max_entries=dump_max_refs)
        dump_payload = ref_info.pop("dump_payload", None)
        row.update(ref_info)

        if save_refs and dump_payload is not None and (
            dump_payload["entries"] or dump_payload["dois"] or dump_payload["arxiv_refs"]
        ):
            ref_record = {
                "archive_name": archive_name,
                "member_name": member_name,
                "arxiv_id": row["arxiv_id"],
                "pub_date": row["pub_date"],
                "pub_year": pub_year,
                "arxiv_category": row["arxiv_category"],
                "arxiv_primary_group": row["arxiv_primary_group"],
                "ref_count": dump_payload["ref_count"],
                "dois": dump_payload["dois"],
                "arxiv_refs": dump_payload["arxiv_refs"],
                "entries": dump_payload["entries"],
            }

    except Exception as exc:
        row["status"] = "error"
        row["error_message"] = f"{type(exc).__name__}: {str(exc)[:240]}"
    finally:
        if doc is not None:
            try:
                doc.close()
            except Exception:
                pass
        del pdf_bytes
        gc.collect()

    return row, ref_record


def discover_archives(input_dir: str) -> List[Path]:
    """Recursively find all .tar.gz and .tgz archives in sorted deterministic order."""
    root = Path(input_dir)
    if not root.exists():
        raise FileNotFoundError(f"Input directory does not exist: {input_dir}")
    return [
        p
        for p in sorted(root.rglob("*"))
        if p.is_file()
        and (p.name.lower().endswith(".tar.gz") or p.name.lower().endswith(".tgz"))
    ]


def run_pipeline(args: argparse.Namespace, logger: logging.Logger) -> None:
    """Stream PDFs from .tar.gz archives into a bounded worker pool (zero extraction)."""
    if fitz is None:
        raise RuntimeError("PyMuPDF is required. Install with: pip install pymupdf")

    init_safe_temp_dir(args.temp_dir)
    archives = discover_archives(args.input_dir)
    if not archives:
        msg = f"No .tar.gz or .tgz archives found in {args.input_dir}"
        logger.warning(msg)
        print(msg)
        return

    refs_out = None if args.no_refs_dump else args.refs_output
    state = StateManager(args.state_db, args.output, refs_out)
    completed_archives = state.get_completed_archives()
    already_total = state.total_processed_count()

    remaining_archives = [a for a in archives if a.name not in completed_archives]
    logger.info(
        "Discovered %d archives (%d completed, %d remaining). Previously processed PDFs: %d",
        len(archives),
        len(completed_archives),
        len(remaining_archives),
        already_total,
    )

    if not remaining_archives:
        print(f"All {len(archives)} archives already completed ({already_total:,} PDFs).")
        state.close()
        return

    stop_requested = {"flag": False}

    def _handle_stop(signum, _frame):
        if not stop_requested["flag"]:
            stop_requested["flag"] = True
            logger.warning("Signal %s received — draining in-flight PDFs and saving checkpoint...", signum)
            print("\n[Stopping gracefully] Finishing in-flight PDFs and flushing CSV + .jsonl.gz + SQLite...")
        else:
            logger.error("Second interrupt received — forcing immediate exit.")
            cleanup_temp_dir()
            sys.exit(130)

    signal.signal(signal.SIGINT, _handle_stop)
    signal.signal(signal.SIGTERM, _handle_stop)

    worker_cfg = {
        "max_pages": args.max_pages,
        "dump_max_refs": args.dump_max_refs,
        "save_refs": not args.no_refs_dump,
    }

    max_pdf_bytes = int(args.max_pdf_mb * 1024 * 1024)
    max_in_flight = max(2, args.workers * 2)

    session_processed = 0
    session_errors = 0
    session_skipped = 0
    start_ts = time.time()

    pbar = (
        tqdm(desc="Pass 1 PDFs", unit="pdf", dynamic_ncols=True, smoothing=0.05)
        if tqdm is not None
        else None
    )

    def _update_status(current_archive: str, arc_idx: int, arc_total: int) -> None:
        elapsed = max(time.time() - start_ts, 1e-6)
        rate = session_processed / elapsed
        mem_mb = get_memory_usage_mb()
        if pbar is not None:
            pbar.set_postfix(
                arc=f"{arc_idx}/{arc_total}",
                file=current_archive[:22],
                err=session_errors,
                ram=f"{mem_mb:.0f}MB",
                refresh=False,
            )
        elif session_processed % 50 == 0:
            print(
                f"\r[{arc_idx}/{arc_total}] {current_archive[:28]} | "
                f"PDFs: {session_processed:,} (total {already_total + session_processed:,}) | "
                f"Err: {session_errors:,} | {rate:.1f} pdf/s | RAM: {mem_mb:.0f} MB",
                end="",
                flush=True,
            )

    def _drain_one(
        pending_queue: collections.deque,
        current_archive: str,
        arc_idx: int,
        arc_total: int,
    ) -> None:
        nonlocal session_processed, session_errors
        async_res, meta_arc, meta_mem = pending_queue.popleft()
        ref_rec = None
        try:
            res_row, ref_rec = async_res.get(timeout=args.pdf_timeout)
        except mp.TimeoutError:
            aid, ver, pdate, pyr, pmo = parse_arxiv_id_and_date(meta_mem, meta_arc)
            res_row = _empty_row(
                meta_arc, meta_mem, aid, ver, pdate, pyr, pmo,
                status="timeout",
                error_message=f"Timed out after {args.pdf_timeout}s",
            )
            logger.warning("Timeout processing %s :: %s", meta_arc, meta_mem)
        except Exception as exc:
            aid, ver, pdate, pyr, pmo = parse_arxiv_id_and_date(meta_mem, meta_arc)
            res_row = _empty_row(
                meta_arc, meta_mem, aid, ver, pdate, pyr, pmo,
                status="error",
                error_message=f"Pool error: {type(exc).__name__}: {exc}",
            )
            logger.error("Worker exception on %s :: %s: %s", meta_arc, meta_mem, exc)

        state.record_result(res_row, ref_rec)
        session_processed += 1
        if res_row.get("status") != "ok":
            session_errors += 1
        if pbar is not None:
            pbar.update(1)
        _update_status(current_archive, arc_idx, arc_total)

    pool = mp.Pool(
        processes=args.workers,
        initializer=_worker_init,
        initargs=(worker_cfg,),
        maxtasksperchild=200,
    )

    try:
        total_arcs = len(remaining_archives)
        for arc_idx, archive_path in enumerate(remaining_archives, start=1):
            if stop_requested["flag"]:
                break

            archive_name = archive_path.name
            done_members = state.get_processed_members_for_archive(archive_name)
            logger.info(
                "Opening archive [%d/%d]: %s (already done: %d)",
                arc_idx,
                total_arcs,
                archive_name,
                len(done_members),
            )
            _update_status(archive_name, arc_idx, total_arcs)

            pending: collections.deque = collections.deque()
            archive_pdf_seen = 0
            archive_cleanly_finished = False

            try:
                with tarfile.open(archive_path, mode="r|gz") as tf:
                    for member in tf:
                        if stop_requested["flag"]:
                            break
                        if not member.isfile() or not member.name.lower().endswith(".pdf"):
                            continue

                        archive_pdf_seen += 1
                        if member.name in done_members:
                            session_skipped += 1
                            continue

                        if member.size > max_pdf_bytes:
                            aid, ver, pdate, pyr, pmo = parse_arxiv_id_and_date(
                                member.name, archive_name
                            )
                            skip_row = _empty_row(
                                archive_name, member.name, aid, ver, pdate, pyr, pmo,
                                status="skipped_too_large",
                                error_message=(
                                    f"Size {member.size / (1024*1024):.1f} MB "
                                    f"> {args.max_pdf_mb} MB limit"
                                ),
                            )
                            state.record_result(skip_row, None)
                            session_processed += 1
                            session_errors += 1
                            if pbar is not None:
                                pbar.update(1)
                            logger.warning(
                                "Skipped oversized PDF (%.1f MB): %s :: %s",
                                member.size / (1024 * 1024),
                                archive_name,
                                member.name,
                            )
                            continue

                        try:
                            extracted_stream = tf.extractfile(member)
                            if extracted_stream is None:
                                continue
                            pdf_bytes = extracted_stream.read()
                        except Exception as read_exc:
                            logger.error(
                                "Failed reading %s from %s: %s",
                                member.name, archive_name, read_exc,
                            )
                            aid, ver, pdate, pyr, pmo = parse_arxiv_id_and_date(
                                member.name, archive_name
                            )
                            err_row = _empty_row(
                                archive_name, member.name, aid, ver, pdate, pyr, pmo,
                                status="stream_error",
                                error_message=f"{type(read_exc).__name__}: {str(read_exc)[:200]}",
                            )
                            state.record_result(err_row, None)
                            session_processed += 1
                            session_errors += 1
                            if pbar is not None:
                                pbar.update(1)
                            continue

                        async_res = pool.apply_async(
                            process_pdf_bytes, ((archive_name, member.name, pdf_bytes),)
                        )
                        del pdf_bytes
                        pending.append((async_res, archive_name, member.name))

                        while len(pending) >= max_in_flight:
                            _drain_one(pending, archive_name, arc_idx, total_arcs)

                if not stop_requested["flag"]:
                    archive_cleanly_finished = True

            except (tarfile.TarError, EOFError, OSError) as tar_exc:
                logger.error("Corrupt or unreadable archive %s: %s", archive_name, tar_exc)
                print(f"\n[ERROR] Corrupt/unreadable archive skipped: {archive_name} ({tar_exc})")

            while pending:
                _drain_one(pending, archive_name, arc_idx, total_arcs)

            state.flush()

            if archive_cleanly_finished:
                state.mark_archive_completed(archive_name, archive_pdf_seen)
                logger.info(
                    "Completed archive %s (%d PDFs seen).", archive_name, archive_pdf_seen
                )

            gc.collect()

    finally:
        if pbar is not None:
            pbar.close()
        else:
            print()
        try:
            pool.close()
            pool.join()
        except Exception:
            pool.terminate()
        state.close()
        cleanup_temp_dir()

    elapsed = time.time() - start_ts
    summary = (
        f"Pass 1 finished ({'interrupted cleanly' if stop_requested['flag'] else 'complete'}). "
        f"Session PDFs: {session_processed:,} | Errors/Skipped: {session_errors:,} | "
        f"Resumed-skips: {session_skipped:,} | Elapsed: {elapsed/60:.1f} min | "
        f"Metrics CSV: {args.output}"
        + (f" | Refs dump: {refs_out}" if refs_out else "")
    )
    logger.info(summary)
    print(summary)

# ---------------------------------------------------------------------------
# Command-Line Interface (Pass 1)
# ---------------------------------------------------------------------------

def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Pass 1: Stream arXiv PDFs directly from .tar.gz archives (zero disk extraction), "
            "compute comprehensive stylometric/confounder/structural/reference metrics to CSV, "
            "and dump extracted references to .jsonl.gz for Pass 2 validation."
        ),
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    io_group = parser.add_argument_group("Input / Output & Checkpointing")
    io_group.add_argument(
        "--input-dir",
        type=str,
        required=True,
        help="Directory containing arXiv .tar.gz / .tgz archives (searched recursively).",
    )
    io_group.add_argument(
        "--output",
        type=str,
        default="arxiv_metrics.csv",
        help="Output CSV path (safely appended across stop/resume sessions).",
    )
    io_group.add_argument(
        "--refs-output",
        type=str,
        default="arxiv_references.jsonl.gz",
        help="Crash-safe gzipped JSONL file storing extracted references for Pass 2 validation.",
    )
    io_group.add_argument(
        "--dump-max-refs",
        type=int,
        default=25,
        help="Max evenly spaced reference strings saved per paper in --refs-output (0 = save all).",
    )
    io_group.add_argument(
        "--no-refs-dump",
        action="store_true",
        help="Disable writing --refs-output .jsonl.gz (still computes offline ref stats in CSV).",
    )
    io_group.add_argument(
        "--state-db",
        type=str,
        default="arxiv_state.sqlite",
        help="SQLite checkpoint database for stop/resume.",
    )
    io_group.add_argument(
        "--log-file",
        type=str,
        default="arxiv_analyzer.log",
        help="Rotating log file path (20 MB x 5 backups).",
    )
    io_group.add_argument(
        "--temp-dir",
        type=str,
        default=None,
        help="Optional base directory for isolated session temp safeguard.",
    )

    perf_group = parser.add_argument_group("Hardware & Memory Guardrails (i5-6500T / 8 GB RAM)")
    perf_group.add_argument(
        "--workers",
        type=int,
        default=3,
        help="Worker processes (3 leaves 1 core on a quad-core for gzip decompression & I/O).",
    )
    perf_group.add_argument(
        "--max-pdf-mb",
        type=float,
        default=50.0,
        help="Skip individual PDFs larger than this size (MB) to protect 8 GB RAM.",
    )
    perf_group.add_argument(
        "--max-pages",
        type=int,
        default=150,
        help="Max pages read per PDF (also reads last 5 pages for references if truncated).",
    )
    perf_group.add_argument(
        "--pdf-timeout",
        type=float,
        default=45.0,
        help="Per-PDF timeout in seconds so malformed PDFs never hang the run.",
    )
    parser.add_argument(
        "-v", "--verbose",
        action="store_true",
        help="Print INFO logs to stderr in addition to the log file.",
    )
    return parser


def main() -> None:
    parser = build_arg_parser()
    args = parser.parse_args()
    logger = setup_logging(args.log_file, verbose=args.verbose)
    run_pipeline(args, logger)


if __name__ == "__main__":
    mp.freeze_support()
    main()
