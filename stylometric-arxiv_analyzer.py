#!/usr/bin/env python3
"""
arXiv Large-Scale PDF Linguistic & Reference Analyzer
=====================================================

Streams PDFs directly from multi-TB .tar.gz archives (zero disk extraction),
computes readability, lexical diversity, entropy, passive voice, authorship,
and reference validity metrics, checkpoints progress to SQLite for seamless
stop/resume, and generates before/after trend plots over time.

Tuned for resource-constrained machines (e.g. quad-core i5-6500T, 8 GB RAM).

Dependencies:
    pip install pymupdf tqdm requests pandas matplotlib

Optional (for richer RAM reporting in terminal status):
    pip install psutil

Usage examples:
    # 1. Process all .tar.gz archives in a directory (fast offline mode):
    python arxiv_analyzer.py --input-dir /mnt/arxiv_tars --output results.csv

    # 2. Enable live reference validation (CrossRef + OpenAlex, cached in SQLite):
    python arxiv_analyzer.py --input-dir /mnt/arxiv_tars --output results.csv \
        --validate-refs --ref-sample 5 --mailto you@university.edu

    # 3. Stop anytime with Ctrl+C, then run the exact same command to resume.

    # 4. Plot trends over time with a before/after cutoff date:
    python arxiv_analyzer.py --plot-only --output results.csv \
        --plots-dir ./plots --cutoff-date 2022-11
"""

import argparse
import atexit
import collections
import csv
import gc
import logging
import math
import multiprocessing as mp
import os
import re
import signal
import sqlite3
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
    import requests
except ImportError:
    requests = None

try:
    import psutil
except ImportError:
    psutil = None


# ---------------------------------------------------------------------------
# Constants & CSV Schema
# ---------------------------------------------------------------------------

CSV_COLUMNS = [
    "archive_name",
    "member_name",
    "arxiv_id",
    "pub_date",
    "pub_year",
    "pub_month",
    "status",
    "page_count",
    "char_count",
    "word_count",
    "sentence_count",
    "avg_sentence_length",
    "avg_syllables_per_word",
    "fre",
    "fkgl",
    "mattr_100",
    "shannon_char_entropy",
    "passive_sentence_count",
    "passive_voice_ratio",
    "author_count",
    "author_source",
    "ref_count",
    "doi_count",
    "arxiv_ref_count",
    "ref_future_year_flags",
    "ref_malformed_doi_flags",
    "refs_checked",
    "refs_verified",
    "refs_unverified",
    "ref_verification_rate",
    "error_message",
]

# Irregular English past participles for lightweight passive-voice detection
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

# Common -ed words that act predominantly as adjectives rather than passive verbs
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

    # Rotate at 20 MB, keep 5 backups so logs never fill the disk on 200k+ PDFs
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
    """
    Create an isolated session temp directory and register automatic cleanup.
    The main pipeline streams PDFs in RAM without extracting to disk; this
    directory is a safety net in case a library ever creates scratch files.
    """
    global TEMP_DIR_TRACKER
    tmp_path = Path(tempfile.mkdtemp(prefix="arxiv_analyzer_", dir=base_tmp))
    TEMP_DIR_TRACKER = tmp_path
    atexit.register(cleanup_temp_dir)
    return tmp_path


def cleanup_temp_dir() -> None:
    """Remove any scratch files left in the session temp directory."""
    global TEMP_DIR_TRACKER
    if TEMP_DIR_TRACKER and TEMP_DIR_TRACKER.exists():
        import shutil
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
    # Fallback on Linux/macOS without psutil
    try:
        with open("/proc/self/statm", "r", encoding="utf-8") as f:
            pages = int(f.read().split()[1])
            return (pages * os.sysconf("SC_PAGE_SIZE")) / (1024 * 1024)
    except Exception:
        return 0.0


def parse_arxiv_id_and_date(
    member_name: str, archive_name: str
) -> Tuple[str, str, Optional[int], Optional[int]]:
    """
    Extract canonical arXiv ID and publication year/month from member or archive name.
    Handles both modern IDs (YYMM.NNNNN[vN]) and legacy IDs (cat/YYMMNNN[vN]).
    """
    basename = os.path.basename(member_name)
    stem = re.sub(r"\.pdf$", "", basename, flags=re.IGNORECASE)

    # Modern arXiv ID: e.g. 2305.12345 or 2305.12345v2
    m_mod = re.search(r"(\d{4}\.\d{4,5})(v\d+)?", stem)
    if m_mod:
        arxiv_id = m_mod.group(1)
        yy = int(arxiv_id[:2])
        mm = int(arxiv_id[2:4])
        if 1 <= mm <= 12:
            year = 2000 + yy if yy < 90 else 1900 + yy
            return arxiv_id, f"{year:04d}-{mm:02d}", year, mm

    # Legacy arXiv ID: e.g. hep-th9901001 or hep-th/9901001
    m_leg = re.search(r"([a-z\-]+/?)(\d{2})(\d{2})\d{3}", member_name, re.IGNORECASE)
    if m_leg:
        yy = int(m_leg.group(2))
        mm = int(m_leg.group(3))
        if 1 <= mm <= 12:
            year = 2000 + yy if yy < 90 else 1900 + yy
            return stem, f"{year:04d}-{mm:02d}", year, mm

    # Fallback: bulk S3 tar naming like arXiv_pdf_2103_001.tar.gz or 2021-03
    m_arc = re.search(r"(?:_|/|-)(\d{2})(\d{2})(?:_|\.|-)", archive_name)
    if m_arc:
        yy = int(m_arc.group(1))
        mm = int(m_arc.group(2))
        if 1 <= mm <= 12:
            year = 2000 + yy if yy < 90 else 1900 + yy
            return stem, f"{year:04d}-{mm:02d}", year, mm

    m_iso = re.search(r"(19\d{2}|20\d{2})[-_]?([01]\d)", archive_name)
    if m_iso:
        year = int(m_iso.group(1))
        mm = int(m_iso.group(2))
        if 1 <= mm <= 12:
            return stem, f"{year:04d}-{mm:02d}", year, mm

    return stem, "", None, None

# ---------------------------------------------------------------------------
# Linguistic & Readability Metrics (Fast, Pure-Python, Low-Memory)
# ---------------------------------------------------------------------------

RE_HYPHEN_BREAK = re.compile(r"([A-Za-z])-\s*\n\s*([A-Za-z])")
RE_WHITESPACE = re.compile(r"\s+")
RE_WORD = re.compile(r"[A-Za-z]+(?:'[A-Za-z]+)?")
RE_VOWEL_GROUP = re.compile(r"[aeiouy]+")

# Protect common academic abbreviations before sentence splitting
RE_ABBREV = re.compile(
    r"\b(?:Fig|Figs|Eq|Eqs|Sec|Secs|Ref|Refs|Tab|Tabs|App|Vol|No|Nos|pp|"
    r"al|et|cf|vs|Dr|Prof|Mr|Mrs|Ms|St|Jr|Sr|Inc|Ltd|Dept|Univ|Jan|Feb|"
    r"Mar|Apr|Jun|Jul|Aug|Sep|Sept|Oct|Nov|Dec|e\.g|i\.e)\.",
    re.IGNORECASE,
)
RE_SENT_SPLIT = re.compile(r"(?<=[.!?])\s+(?=[A-Z0-9(\[\"'`])")

# Passive voice regex:
# Matches forms of "be" (or "get/got/gotten") + up to 2 optional adverbs/negations + past participle
RE_PASSIVE_AUX = re.compile(
    r"\b(?:am|is|are|was|were|be|been|being|get|gets|got|gotten)\s+"
    r"(?:(?:not|never|also|often|usually|always|already|still|now|just|only|"
    r"frequently|typically|commonly|widely|generally|previously|recently|"
    r"directly|indirectly|easily|simply|well|largely|mostly|partly|fully|"
    r"further|then|thus|here|there|[a-z]+ly)\s+){0,2}"
    r"([a-z]{3,})\b",
    re.IGNORECASE,
)

# Reference section header matcher (anchored near line start)
RE_REF_HEADER = re.compile(
    r"(?im)^\s*(?:\d+\.?\s*)?(?:references|bibliography|works\s+cited)\s*$"
)


def clean_pdf_text(raw_text: str) -> str:
    """Repair PDF line-break hyphenation and normalize whitespace."""
    text = RE_HYPHEN_BREAK.sub(r"\1\2", raw_text)
    # Remove null bytes and non-printable control chars except newline/tab
    text = "".join(ch if (ch >= " " or ch in "\n\t") else " " for ch in text)
    return text


def split_body_and_references(text: str) -> Tuple[str, str]:
    """
    Split paper text into main prose body and references section.
    Searches from the last 55% of the paper to avoid false matches in TOC/intro.
    """
    if not text:
        return "", ""
    search_start = int(len(text) * 0.45)
    matches = list(RE_REF_HEADER.finditer(text, pos=search_start))
    if not matches:
        # Try full document as fallback for short papers
        matches = list(RE_REF_HEADER.finditer(text))
    if matches:
        last_match = matches[-1]
        return text[: last_match.start()], text[last_match.end() :]
    return text, ""


def count_syllables(word: str) -> int:
    """Fast heuristic English syllable counter for FRE and FKGL."""
    w = word.lower().strip("'")
    if len(w) <= 3:
        return 1
    # Strip silent trailing 'e', 'es', 'ed' (except -le like 'table', -ted/-ded)
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
    # Temporarily replace abbreviation periods with a placeholder
    protected = RE_ABBREV.sub(lambda m: m.group(0).replace(".", "<DOT>"), collapsed)
    # Also protect single-letter initials like "J. Smith"
    protected = re.sub(r"\b([A-Z])\.\s+(?=[A-Z])", r"\1<DOT> ", protected)
    raw_sents = RE_SENT_SPLIT.split(protected)
    sentences = []
    for s in raw_sents:
        s_clean = s.replace("<DOT>", ".").strip()
        # Keep sentences that contain at least 3 alphabetic words (filters equations/headers)
        if len(RE_WORD.findall(s_clean)) >= 3:
            sentences.append(s_clean)
    return sentences


def is_passive_sentence(sentence: str) -> bool:
    """
    Lightweight regex + lexicon passive voice detector.
    Runs ~50x faster than spaCy dependency parsing and uses ~0 MB extra RAM,
    making it practical for 200k+ PDFs on an i5-6500T with 8 GB RAM.
    """
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
    """
    Moving-Average Type-Token Ratio (MATTR) with window size `window` (default 100).
    Uses O(N) sliding-window counter instead of O(N * window) recomputation.
    """
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
    """Shannon character entropy H = -sum(p(c) * log2(p(c))) over normalized text."""
    normalized = RE_WHITESPACE.sub(" ", text).strip()
    n = len(normalized)
    if n == 0:
        return 0.0
    counts = collections.Counter(normalized)
    entropy = 0.0
    for c in counts.values():
        p = c / n
        entropy -= p * math.log2(p)
    return entropy


def compute_linguistic_metrics(body_text: str) -> Dict[str, Any]:
    """Compute FRE, FKGL, MATTR-100, Shannon entropy, sentence stats, and passive voice."""
    sentences = split_sentences(body_text)
    words = [w.lower() for w in RE_WORD.findall(body_text)]

    word_count = len(words)
    sentence_count = len(sentences)
    char_count = len(body_text)

    if word_count == 0 or sentence_count == 0:
        return {
            "char_count": char_count,
            "word_count": word_count,
            "sentence_count": sentence_count,
            "avg_sentence_length": 0.0,
            "avg_syllables_per_word": 0.0,
            "fre": 0.0,
            "fkgl": 0.0,
            "mattr_100": 0.0,
            "shannon_char_entropy": round(compute_shannon_char_entropy(body_text), 5),
            "passive_sentence_count": 0,
            "passive_voice_ratio": 0.0,
        }

    # Compute sentence-based word count for accurate avg sentence length
    sent_word_counts = [len(RE_WORD.findall(s)) for s in sentences]
    total_sent_words = sum(sent_word_counts)
    avg_sentence_length = total_sent_words / sentence_count if sentence_count else 0.0

    # Syllables (cache per unique word for speed)
    syl_cache: Dict[str, int] = {}
    total_syllables = 0
    for w in words:
        s = syl_cache.get(w)
        if s is None:
            s = count_syllables(w)
            syl_cache[w] = s
        total_syllables += s

    avg_syl_per_word = total_syllables / word_count
    # Standard Flesch Reading Ease & Flesch-Kincaid Grade Level
    fre = 206.835 - 1.015 * avg_sentence_length - 84.6 * avg_syl_per_word
    fkgl = 0.39 * avg_sentence_length + 11.8 * avg_syl_per_word - 15.59

    mattr_100 = compute_mattr(words, window=100)
    shannon_entropy = compute_shannon_char_entropy(body_text)

    passive_count = sum(1 for s in sentences if is_passive_sentence(s))
    passive_ratio = passive_count / sentence_count if sentence_count else 0.0

    return {
        "char_count": char_count,
        "word_count": word_count,
        "sentence_count": sentence_count,
        "avg_sentence_length": round(avg_sentence_length, 4),
        "avg_syllables_per_word": round(avg_syl_per_word, 4),
        "fre": round(fre, 4),
        "fkgl": round(fkgl, 4),
        "mattr_100": round(mattr_100, 5),
        "shannon_char_entropy": round(shannon_entropy, 5),
        "passive_sentence_count": passive_count,
        "passive_voice_ratio": round(passive_ratio, 5),
    }

# ---------------------------------------------------------------------------
# Author Count & Reference Extraction
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


def estimate_author_count(
    pdf_metadata: Dict[str, Any], first_page_text: str
) -> Tuple[int, str]:
    """
    Estimate number of authors from PDF metadata first, then fall back to
    first-page header heuristics (between title and Abstract).
    """
    meta_author = (pdf_metadata or {}).get("author") or ""
    meta_author = meta_author.strip()
    if meta_author and len(meta_author) >= 3:
        # Ignore placeholder generator strings
        if not re.search(r"^(?:unknown|anonymous|latex|author|user)$", meta_author, re.I):
            # Split on semicolons, ' and ', '&', or commas (when not 'Last, First')
            cleaned = re.sub(r"\s+and\s+", ";", meta_author, flags=re.IGNORECASE)
            cleaned = cleaned.replace("&", ";")
            if ";" in cleaned:
                parts = [p.strip() for p in cleaned.split(";") if len(p.strip()) >= 2]
            else:
                parts = [p.strip() for p in cleaned.split(",") if len(p.strip()) >= 2]
                # Detect "Last, First" single author pattern (2 single-word tokens)
                if len(parts) == 2 and all(" " not in p for p in parts):
                    parts = [meta_author]
            if 1 <= len(parts) <= 200:
                return len(parts), "pdf_metadata"

    # Fallback: parse first page above "Abstract"
    if not first_page_text:
        return 0, "unknown"

    header_match = re.search(r"(?im)^\s*abstract\b", first_page_text)
    header_block = (
        first_page_text[: header_match.start()]
        if header_match
        else first_page_text[:1800]
    )

    lines = [ln.strip() for ln in header_block.splitlines() if ln.strip()]
    # Skip the first 1-2 lines (typically title) when enough lines exist
    candidate_lines = lines[1:] if len(lines) > 3 else lines

    # Count distinct emails as a strong floor if multiple authors list emails
    emails = set(re.findall(r"[A-Za-z0-9._%+\-]+@[A-Za-z0-9.\-]+\.[A-Za-z]{2,}", header_block))

    author_candidates: List[str] = []
    for ln in candidate_lines[:25]:
        if RE_AFFILIATION_NOISE.search(ln):
            continue
        # Strip footnote symbols/digits attached to names
        ln_clean = re.sub(r"[\*\†\‡\§\¶\d]+", "", ln)
        matches = RE_NAME_PAIR.findall(ln_clean)
        author_candidates.extend(matches)

    count = max(len(set(author_candidates)), len(emails))
    if count > 0:
        return min(count, 200), "page1_heuristic"
    return 0, "unknown"


def extract_references(
    ref_text: str, paper_year: Optional[int]
) -> Dict[str, Any]:
    """
    Parse the references section into individual entries, extract DOIs and arXiv IDs,
    and run fast offline sanity checks (future publication years, malformed DOIs).
    """
    if not ref_text or len(ref_text.strip()) < 20:
        return {
            "ref_count": 0,
            "doi_count": 0,
            "arxiv_ref_count": 0,
            "ref_future_year_flags": 0,
            "ref_malformed_doi_flags": 0,
            "entries": [],
        }

    # Try bracketed style [1], [2]... first, then numbered 1. 2., then blank-line blocks
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
        # Author-year style: split on double newlines or lines ending with a year/period
        blocks = [RE_WHITESPACE.sub(" ", b).strip() for b in re.split(r"\n\s*\n", ref_text)]
        entries = [b for b in blocks if len(b) >= 25]

    # Cap pathological parses on malformed PDFs
    entries = entries[:500]

    dois = [d.rstrip(".,;)]}") for d in RE_DOI.findall(ref_text)]
    arxiv_refs = RE_ARXIV_REF.findall(ref_text)

    # Structural red-flag heuristics (zero network calls)
    max_plausible_year = (paper_year + 1) if paper_year else (datetime.now().year + 1)
    future_year_flags = 0
    for entry in entries:
        years = [int(y) for y in RE_YEAR_IN_REF.findall(entry)]
        if years and max(years) > max_plausible_year:
            future_year_flags += 1

    # Detect malformed DOI mentions (e.g. "doi:" followed by invalid syntax)
    raw_doi_mentions = re.findall(r"\bdoi\s*[:=]\s*(\S+)", ref_text, re.IGNORECASE)
    malformed_doi_flags = 0
    for token in raw_doi_mentions:
        tok = token.rstrip(".,;)]}")
        if not re.match(r"^(?:https?://(?:dx\.)?doi\.org/)?10\.\d{4,9}/\S+$", tok, re.I):
            malformed_doi_flags += 1

    return {
        "ref_count": len(entries),
        "doi_count": len(set(dois)),
        "arxiv_ref_count": len(set(arxiv_refs)),
        "ref_future_year_flags": future_year_flags,
        "ref_malformed_doi_flags": malformed_doi_flags,
        "entries": entries,
    }

# ---------------------------------------------------------------------------
# Reference Validator (CrossRef + OpenAlex + Persistent SQLite Cache)
# ---------------------------------------------------------------------------

class ReferenceValidator:
    """
    Validates extracted references against CrossRef (by DOI or bibliographic query)
    and OpenAlex to flag potentially hallucinated or unverified citations.

    Uses a persistent SQLite cache so citations shared across 200,000+ papers are
    only queried once over the network.
    """

    def __init__(
        self,
        cache_db: str,
        mailto: str = "",
        timeout: float = 6.0,
        rate_limit_delay: float = 0.08,
        logger: Optional[logging.Logger] = None,
    ):
        self.cache_db = cache_db
        self.mailto = mailto.strip()
        self.timeout = timeout
        self.rate_limit_delay = rate_limit_delay
        self.logger = logger or logging.getLogger("arxiv_analyzer")
        self._last_req_ts = 0.0

        self.session = requests.Session() if requests is not None else None
        if self.session is not None:
            ua = f"ArxivLinguisticAnalyzer/1.0 (mailto:{self.mailto or 'research@localhost'})"
            self.session.headers.update({"User-Agent": ua})

        self._init_cache()

    def _init_cache(self) -> None:
        Path(self.cache_db).parent.mkdir(parents=True, exist_ok=True)
        with sqlite3.connect(self.cache_db, timeout=30) as conn:
            conn.execute("PRAGMA journal_mode=WAL;")
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS ref_cache (
                    cache_key TEXT PRIMARY KEY,
                    verified  INTEGER NOT NULL,
                    source    TEXT NOT NULL,
                    checked_at REAL NOT NULL
                )
                """
            )
            conn.commit()

    def _cache_get(self, key: str) -> Optional[bool]:
        with sqlite3.connect(self.cache_db, timeout=30) as conn:
            row = conn.execute(
                "SELECT verified FROM ref_cache WHERE cache_key = ?", (key,)
            ).fetchone()
            return bool(row[0]) if row is not None else None

    def _cache_set(self, key: str, verified: bool, source: str) -> None:
        with sqlite3.connect(self.cache_db, timeout=30) as conn:
            conn.execute(
                """
                INSERT OR REPLACE INTO ref_cache (cache_key, verified, source, checked_at)
                VALUES (?, ?, ?, ?)
                """,
                (key, 1 if verified else 0, source, time.time()),
            )
            conn.commit()

    def _throttle(self) -> None:
        elapsed = time.time() - self._last_req_ts
        if elapsed < self.rate_limit_delay:
            time.sleep(self.rate_limit_delay - elapsed)
        self._last_req_ts = time.time()

    @staticmethod
    def _token_jaccard(a: str, b: str) -> float:
        ta = {w for w in RE_WORD.findall(a.lower()) if len(w) > 2}
        tb = {w for w in RE_WORD.findall(b.lower()) if len(w) > 2}
        if not ta or not tb:
            return 0.0
        return len(ta & tb) / len(ta | tb)

    def _verify_doi(self, doi: str) -> Optional[bool]:
        key = f"doi:{doi.lower()}"
        cached = self._cache_get(key)
        if cached is not None:
            return cached
        if self.session is None:
            return None

        self._throttle()
        url = f"https://api.crossref.org/works/{requests.utils.quote(doi, safe='')}"
        params = {"mailto": self.mailto} if self.mailto else {}
        try:
            resp = self.session.get(url, params=params, timeout=self.timeout)
            if resp.status_code == 200:
                self._cache_set(key, True, "crossref_doi")
                return True
            if resp.status_code == 404:
                self._cache_set(key, False, "crossref_doi_404")
                return False
        except Exception as exc:
            self.logger.debug("CrossRef DOI lookup failed for %s: %s", doi, exc)
        return None

    def _verify_arxiv_ref(self, arxiv_id: str) -> Optional[bool]:
        clean_id = re.sub(r"v\d+$", "", arxiv_id.strip())
        key = f"arxiv:{clean_id.lower()}"
        cached = self._cache_get(key)
        if cached is not None:
            return cached
        if self.session is None:
            return None

        self._throttle()
        url = "https://export.arxiv.org/api/query"
        try:
            resp = self.session.get(
                url, params={"id_list": clean_id}, timeout=self.timeout
            )
            if resp.status_code == 200:
                # Valid arXiv entries contain a <title> inside <entry>
                has_entry = "<entry>" in resp.text and "<id>http" in resp.text
                is_error = "Error" in resp.text and "incorrect id format" in resp.text.lower()
                ok = has_entry and not is_error
                self._cache_set(key, ok, "arxiv_api")
                return ok
        except Exception as exc:
            self.logger.debug("arXiv API lookup failed for %s: %s", clean_id, exc)
        return None

    def _verify_bibliographic_text(self, entry: str) -> Optional[bool]:
        """Query CrossRef bibliographic search and check title/year overlap."""
        # Normalize query string to compact form
        query = re.sub(r"^\s*(?:\[\d+\]|\d+\.)\s*", "", entry).strip()[:220]
        if len(query) < 25:
            return None

        key = f"bib:{re.sub(r'[^a-z0-9]+', ' ', query.lower()).strip()[:140]}"
        cached = self._cache_get(key)
        if cached is not None:
            return cached
        if self.session is None:
            return None

        self._throttle()
        params = {"query.bibliographic": query, "rows": 1}
        if self.mailto:
            params["mailto"] = self.mailto

        try:
            resp = self.session.get(
                "https://api.crossref.org/works", params=params, timeout=self.timeout
            )
            if resp.status_code == 200:
                items = resp.json().get("message", {}).get("items", [])
                if not items:
                    self._cache_set(key, False, "crossref_bib_empty")
                    return False
                top = items[0]
                titles = top.get("title") or []
                top_title = titles[0] if titles else ""
                # Compare top candidate title words against the citation string
                title_tokens = {w for w in RE_WORD.findall(top_title.lower()) if len(w) > 2}
                entry_tokens = {w for w in RE_WORD.findall(entry.lower()) if len(w) > 2}
                if title_tokens:
                    recall = len(title_tokens & entry_tokens) / len(title_tokens)
                else:
                    recall = 0.0
                verified = recall >= 0.60
                self._cache_set(key, verified, f"crossref_bib_{recall:.2f}")
                return verified
        except Exception as exc:
            self.logger.debug("CrossRef bib lookup failed: %s", exc)
        return None

    def validate_entries(
        self, entries: List[str], max_checks: int = 5
    ) -> Tuple[int, int, int, Optional[float]]:
        """
        Validate up to `max_checks` evenly spaced references from a paper.
        Returns (checked, verified, unverified, verification_rate).
        """
        if not entries or max_checks <= 0:
            return 0, 0, 0, None

        # Deterministic, evenly spaced sampling across the reference list
        if len(entries) <= max_checks:
            sample = entries
        else:
            step = len(entries) / max_checks
            sample = [entries[int(i * step)] for i in range(max_checks)]

        checked = 0
        verified = 0
        unverified = 0

        for entry in sample:
            result: Optional[bool] = None
            m_doi = RE_DOI.search(entry)
            m_arxiv = RE_ARXIV_REF.search(entry)

            if m_doi:
                result = self._verify_doi(m_doi.group(1).rstrip(".,;)]}"))
            elif m_arxiv:
                result = self._verify_arxiv_ref(m_arxiv.group(1))
            else:
                result = self._verify_bibliographic_text(entry)

            if result is True:
                checked += 1
                verified += 1
            elif result is False:
                checked += 1
                unverified += 1
            # None indicates a transient network timeout; don't penalize the paper

        rate = round(verified / checked, 4) if checked > 0 else None
        return checked, verified, unverified, rate

# ---------------------------------------------------------------------------
# Checkpoint / Resume State Manager (SQLite + Append-Safe CSV Writer)
# ---------------------------------------------------------------------------

class StateManager:
    """
    Tracks completed archives and individual processed PDFs in SQLite (WAL mode)
    so the script can be stopped with Ctrl+C and resumed seamlessly.

    - Completely finished .tar.gz archives are skipped immediately on restart
      without re-decompressing them.
    - Partially finished archives skip already-processed PDFs while streaming.
    - Writes results to CSV in batched flushes alongside SQLite transactions.
    """

    def __init__(self, state_db: str, csv_path: str):
        self.state_db = state_db
        self.csv_path = Path(csv_path)
        self.csv_path.parent.mkdir(parents=True, exist_ok=True)
        Path(state_db).parent.mkdir(parents=True, exist_ok=True)

        self.conn = sqlite3.connect(self.state_db, timeout=60)
        self.conn.execute("PRAGMA journal_mode=WAL;")
        self.conn.execute("PRAGMA synchronous=NORMAL;")
        self._init_tables()

        # Open CSV in append mode; write header only if file is new/empty
        need_header = (not self.csv_path.exists()) or (self.csv_path.stat().st_size == 0)
        self._csv_file = open(self.csv_path, "a", newline="", encoding="utf-8")
        self._csv_writer = csv.DictWriter(
            self._csv_file, fieldnames=CSV_COLUMNS, extrasaction="ignore"
        )
        if need_header:
            self._csv_writer.writeheader()
            self._csv_file.flush()

        self._pending_rows: List[Tuple[str, str, str]] = []

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

    def record_result(self, row: Dict[str, Any]) -> None:
        self._csv_writer.writerow(row)
        self._pending_rows.append(
            (row["archive_name"], row["member_name"], row["status"])
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
            try:
                self._csv_file.close()
            except Exception:
                pass
            try:
                self.conn.close()
            except Exception:
                pass

# ---------------------------------------------------------------------------
# Worker Function & Memory-Bounded Streaming Pipeline
# ---------------------------------------------------------------------------

# Per-worker global config initialized once when a worker spawns
_WORKER_CFG: Dict[str, Any] = {}
_WORKER_VALIDATOR: Optional[ReferenceValidator] = None


def _worker_init(cfg: Dict[str, Any]) -> None:
    """Initialize worker process state and ignore SIGINT (handled by main process)."""
    global _WORKER_CFG, _WORKER_VALIDATOR
    signal.signal(signal.SIGINT, signal.SIG_IGN)
    _WORKER_CFG = cfg

    # Suppress verbose MuPDF C-level stderr warnings on malformed PDFs
    if fitz is not None:
        try:
            fitz.TOOLS.mupdf_display_errors(False)
        except Exception:
            pass

    if cfg.get("validate_refs"):
        _WORKER_VALIDATOR = ReferenceValidator(
            cache_db=cfg["ref_cache_db"],
            mailto=cfg.get("mailto", ""),
            timeout=cfg.get("ref_timeout", 6.0),
            rate_limit_delay=cfg.get("ref_delay", 0.08),
        )


def process_pdf_bytes(task: Tuple[str, str, bytes]) -> Dict[str, Any]:
    """
    Parse one PDF completely in memory (no disk extraction) and compute all metrics.
    Always returns a complete CSV row dictionary even on corrupt/encrypted PDFs.
    """
    archive_name, member_name, pdf_bytes = task
    arxiv_id, pub_date, pub_year, pub_month = parse_arxiv_id_and_date(
        member_name, archive_name
    )

    row: Dict[str, Any] = {
        "archive_name": archive_name,
        "member_name": member_name,
        "arxiv_id": arxiv_id,
        "pub_date": pub_date,
        "pub_year": pub_year if pub_year is not None else "",
        "pub_month": pub_month if pub_month is not None else "",
        "status": "ok",
        "page_count": 0,
        "char_count": 0,
        "word_count": 0,
        "sentence_count": 0,
        "avg_sentence_length": "",
        "avg_syllables_per_word": "",
        "fre": "",
        "fkgl": "",
        "mattr_100": "",
        "shannon_char_entropy": "",
        "passive_sentence_count": 0,
        "passive_voice_ratio": "",
        "author_count": 0,
        "author_source": "unknown",
        "ref_count": 0,
        "doi_count": 0,
        "arxiv_ref_count": 0,
        "ref_future_year_flags": 0,
        "ref_malformed_doi_flags": 0,
        "refs_checked": 0,
        "refs_verified": 0,
        "refs_unverified": 0,
        "ref_verification_rate": "",
        "error_message": "",
    }

    doc = None
    try:
        max_pages = int(_WORKER_CFG.get("max_pages", 150))
        doc = fitz.open(stream=pdf_bytes, filetype="pdf")

        if doc.is_encrypted:
            try:
                if not doc.authenticate(""):
                    row["status"] = "encrypted"
                    row["error_message"] = "Password-protected PDF"
                    return row
            except Exception:
                row["status"] = "encrypted"
                row["error_message"] = "Encrypted PDF"
                return row

        page_count = len(doc)
        row["page_count"] = page_count
        if page_count == 0:
            row["status"] = "empty"
            row["error_message"] = "0 pages"
            return row

        metadata = doc.metadata or {}
        first_page_text = ""
        page_texts: List[str] = []

        pages_to_read = min(page_count, max_pages)
        for idx in range(pages_to_read):
            try:
                txt = doc[idx].get_text("text")
            except Exception:
                txt = ""
            if idx == 0:
                first_page_text = txt
            if txt:
                page_texts.append(txt)

        # If truncated on an ultra-long thesis/book, still read last 5 pages for references
        if page_count > max_pages:
            for idx in range(max(pages_to_read, page_count - 5), page_count):
                try:
                    txt = doc[idx].get_text("text")
                    if txt:
                        page_texts.append(txt)
                except Exception:
                    pass

        full_text = clean_pdf_text("\n".join(page_texts))
        del page_texts

        if len(full_text.strip()) < 100:
            row["status"] = "no_text"
            row["error_message"] = "Scanned or image-only PDF (<100 chars extracted)"
            return row

        # Split main prose body from bibliography so references don't skew FRE/MATTR/passive
        body_text, ref_text = split_body_and_references(full_text)
        if len(body_text.strip()) < 100:
            body_text = full_text

        # 1. Linguistic & readability metrics
        metrics = compute_linguistic_metrics(body_text)
        row.update(metrics)

        # 2. Author count
        author_count, author_source = estimate_author_count(metadata, first_page_text)
        row["author_count"] = author_count
        row["author_source"] = author_source

        # 3. References extraction + offline structural checks
        ref_info = extract_references(ref_text, pub_year)
        row["ref_count"] = ref_info["ref_count"]
        row["doi_count"] = ref_info["doi_count"]
        row["arxiv_ref_count"] = ref_info["arxiv_ref_count"]
        row["ref_future_year_flags"] = ref_info["ref_future_year_flags"]
        row["ref_malformed_doi_flags"] = ref_info["ref_malformed_doi_flags"]

        # 4. Optional live reference validation (CrossRef / arXiv API + SQLite cache)
        if _WORKER_VALIDATOR is not None and ref_info["entries"]:
            sample_n = int(_WORKER_CFG.get("ref_sample", 5))
            checked, verified, unverified, rate = _WORKER_VALIDATOR.validate_entries(
                ref_info["entries"], max_checks=sample_n
            )
            row["refs_checked"] = checked
            row["refs_verified"] = verified
            row["refs_unverified"] = unverified
            row["ref_verification_rate"] = "" if rate is None else rate

    except Exception as exc:
        row["status"] = "error"
        row["error_message"] = f"{type(exc).__name__}: {str(exc)[:240]}"
    finally:
        if doc is not None:
            try:
                doc.close()
            except Exception:
                pass
        # Help keep worker RSS low on 8 GB RAM systems
        del pdf_bytes
        gc.collect()

    return row


def discover_archives(input_dir: str) -> List[Path]:
    """Recursively find all .tar.gz and .tgz archives in sorted deterministic order."""
    root = Path(input_dir)
    if not root.exists():
        raise FileNotFoundError(f"Input directory does not exist: {input_dir}")
    archives: List[Path] = []
    for p in sorted(root.rglob("*")):
        if p.is_file() and (
            p.name.lower().endswith(".tar.gz") or p.name.lower().endswith(".tgz")
        ):
            archives.append(p)
    return archives


def run_pipeline(args: argparse.Namespace, logger: logging.Logger) -> None:
    """
    Stream PDFs from every .tar.gz archive into a bounded worker pool without
    extracting files to disk. Supports Ctrl+C graceful stop and resume.
    """
    if fitz is None:
        raise RuntimeError(
            "PyMuPDF is required. Install it with: pip install pymupdf"
        )
    if args.validate_refs and requests is None:
        raise RuntimeError(
            "requests is required for --validate-refs. Install with: pip install requests"
        )

    init_safe_temp_dir(args.temp_dir)
    archives = discover_archives(args.input_dir)
    if not archives:
        logger.warning("No .tar.gz or .tgz archives found in %s", args.input_dir)
        print(f"No .tar.gz or .tgz archives found in {args.input_dir}")
        return

    state = StateManager(args.state_db, args.output)
    completed_archives = state.get_completed_archives()
    already_total = state.total_processed_count()

    remaining_archives = [
        a for a in archives if a.name not in completed_archives
    ]
    logger.info(
        "Discovered %d archives (%d already completed, %d remaining). "
        "Previously processed PDFs: %d",
        len(archives),
        len(completed_archives),
        len(remaining_archives),
        already_total,
    )

    if not remaining_archives:
        print(f"All {len(archives)} archives are already completed ({already_total:,} PDFs).")
        state.close()
        return

    # Graceful shutdown flag on Ctrl+C / SIGTERM
    stop_requested = {"flag": False}

    def _handle_stop(signum, _frame):
        if not stop_requested["flag"]:
            stop_requested["flag"] = True
            logger.warning("Signal %s received — finishing in-flight PDFs and saving checkpoint...", signum)
            print("\n[Stopping gracefully] Waiting for in-flight PDFs to finish and flushing checkpoint...")
        else:
            logger.error("Second interrupt received — forcing exit.")
            cleanup_temp_dir()
            sys.exit(130)

    signal.signal(signal.SIGINT, _handle_stop)
    signal.signal(signal.SIGTERM, _handle_stop)

    worker_cfg = {
        "max_pages": args.max_pages,
        "validate_refs": args.validate_refs,
        "ref_cache_db": args.ref_cache_db,
        "ref_sample": args.ref_sample,
        "ref_timeout": args.ref_timeout,
        "ref_delay": args.ref_delay,
        "mailto": args.mailto or "",
    }

    max_pdf_bytes = int(args.max_pdf_mb * 1024 * 1024)
    # Keep only 2 tasks per worker in RAM at once so 8 GB RAM is never exhausted
    max_in_flight = max(2, args.workers * 2)

    session_processed = 0
    session_errors = 0
    session_skipped = 0
    start_ts = time.time()

    pbar = None
    if tqdm is not None:
        pbar = tqdm(
            desc="PDFs processed",
            unit="pdf",
            dynamic_ncols=True,
            smoothing=0.05,
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

    def _drain_one(pending_queue: collections.deque, current_archive: str, arc_idx: int, arc_total: int) -> None:
        nonlocal session_processed, session_errors
        async_res, meta_arc, meta_mem = pending_queue.popleft()
        try:
            res_row = async_res.get(timeout=args.pdf_timeout)
        except mp.TimeoutError:
            arxiv_id, pub_date, pub_year, pub_month = parse_arxiv_id_and_date(
                meta_mem, meta_arc
            )
            res_row = {col: "" for col in CSV_COLUMNS}
            res_row.update(
                {
                    "archive_name": meta_arc,
                    "member_name": meta_mem,
                    "arxiv_id": arxiv_id,
                    "pub_date": pub_date,
                    "pub_year": pub_year if pub_year is not None else "",
                    "pub_month": pub_month if pub_month is not None else "",
                    "status": "timeout",
                    "error_message": f"Timed out after {args.pdf_timeout}s",
                }
            )
            logger.warning("Timeout processing %s :: %s", meta_arc, meta_mem)
        except Exception as exc:
            arxiv_id, pub_date, pub_year, pub_month = parse_arxiv_id_and_date(
                meta_mem, meta_arc
            )
            res_row = {col: "" for col in CSV_COLUMNS}
            res_row.update(
                {
                    "archive_name": meta_arc,
                    "member_name": meta_mem,
                    "arxiv_id": arxiv_id,
                    "pub_date": pub_date,
                    "pub_year": pub_year if pub_year is not None else "",
                    "pub_month": pub_month if pub_month is not None else "",
                    "status": "error",
                    "error_message": f"Pool error: {type(exc).__name__}: {exc}",
                }
            )
            logger.error("Worker exception on %s :: %s: %s", meta_arc, meta_mem, exc)

        state.record_result(res_row)
        session_processed += 1
        if res_row.get("status") != "ok":
            session_errors += 1
        if pbar is not None:
            pbar.update(1)
        _update_status(current_archive, arc_idx, arc_total)

    # Recycle worker processes every 200 PDFs to guarantee zero C-level memory leaks
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
                "Opening archive [%d/%d]: %s (already done in archive: %d)",
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
                # Streaming mode 'r|gz': never loads tar index into RAM, never extracts to disk
                with tarfile.open(archive_path, mode="r|gz") as tf:
                    for member in tf:
                        if stop_requested["flag"]:
                            break
                        if not member.isfile():
                            continue
                        if not member.name.lower().endswith(".pdf"):
                            continue

                        archive_pdf_seen += 1

                        if member.name in done_members:
                            session_skipped += 1
                            continue

                        # Guard against oversized PDFs that could exhaust 8 GB RAM
                        if member.size > max_pdf_bytes:
                            arxiv_id, pub_date, pub_year, pub_month = parse_arxiv_id_and_date(
                                member.name, archive_name
                            )
                            skip_row = {col: "" for col in CSV_COLUMNS}
                            skip_row.update(
                                {
                                    "archive_name": archive_name,
                                    "member_name": member.name,
                                    "arxiv_id": arxiv_id,
                                    "pub_date": pub_date,
                                    "pub_year": pub_year if pub_year is not None else "",
                                    "pub_month": pub_month if pub_month is not None else "",
                                    "status": "skipped_too_large",
                                    "error_message": f"Size {member.size / (1024*1024):.1f} MB > {args.max_pdf_mb} MB limit",
                                }
                            )
                            state.record_result(skip_row)
                            session_processed += 1
                            session_errors += 1
                            if pbar is not None:
                                pbar.update(1)
                            logger.warning(
                                "Skipped oversized PDF (%0.1f MB): %s :: %s",
                                member.size / (1024 * 1024),
                                archive_name,
                                member.name,
                            )
                            continue

                        # Read raw PDF bytes directly from gzip stream into RAM
                        try:
                            extracted_stream = tf.extractfile(member)
                            if extracted_stream is None:
                                continue
                            pdf_bytes = extracted_stream.read()
                        except Exception as read_exc:
                            logger.error(
                                "Failed reading member %s from %s: %s",
                                member.name,
                                archive_name,
                                read_exc,
                            )
                            arxiv_id, pub_date, pub_year, pub_month = parse_arxiv_id_and_date(
                                member.name, archive_name
                            )
                            err_row = {col: "" for col in CSV_COLUMNS}
                            err_row.update(
                                {
                                    "archive_name": archive_name,
                                    "member_name": member.name,
                                    "arxiv_id": arxiv_id,
                                    "pub_date": pub_date,
                                    "pub_year": pub_year if pub_year is not None else "",
                                    "pub_month": pub_month if pub_month is not None else "",
                                    "status": "stream_error",
                                    "error_message": f"{type(read_exc).__name__}: {str(read_exc)[:200]}",
                                }
                            )
                            state.record_result(err_row)
                            session_processed += 1
                            session_errors += 1
                            if pbar is not None:
                                pbar.update(1)
                            continue

                        # Submit to worker pool with strict backpressure
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
                logger.error(
                    "Corrupt or unreadable archive %s: %s", archive_name, tar_exc
                )
                print(f"\n[ERROR] Corrupt/unreadable archive skipped: {archive_name} ({tar_exc})")

            # Drain all remaining in-flight PDFs from this archive
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
        f"Run finished ({'interrupted cleanly' if stop_requested['flag'] else 'complete'}). "
        f"Session PDFs: {session_processed:,} | Errors/Skipped: {session_errors:,} | "
        f"Resumed-skips: {session_skipped:,} | Elapsed: {elapsed/60:.1f} min | "
        f"Output: {args.output}"
    )
    logger.info(summary)
    print(summary)

# ---------------------------------------------------------------------------
# Trend Plotting Over Time (Before/After Comparison)
# ---------------------------------------------------------------------------

def generate_trend_plots(
    csv_path: str,
    plots_dir: str,
    cutoff_date: Optional[str] = None,
    freq: str = "M",
    logger: Optional[logging.Logger] = None,
) -> None:
    """
    Aggregate the output CSV by month (or quarter/year) and generate publication-ready
    trend plots plus a before/after summary CSV when --cutoff-date (YYYY-MM) is given.
    """
    try:
        import pandas as pd
        import matplotlib
        matplotlib.use("Agg")  # Headless safe for servers / terminals
        import matplotlib.pyplot as plt
        import matplotlib.dates as mdates
    except ImportError as exc:
        raise RuntimeError(
            "Plotting requires pandas and matplotlib. Install with: "
            "pip install pandas matplotlib"
        ) from exc

    log = logger or logging.getLogger("arxiv_analyzer")
    out_dir = Path(plots_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Read CSV in chunks to keep RAM footprint tiny even with 200k-500k rows
    usecols = [
        "pub_date", "status", "fre", "fkgl", "mattr_100",
        "shannon_char_entropy", "passive_voice_ratio",
        "avg_sentence_length", "author_count", "ref_count",
        "ref_verification_rate", "ref_future_year_flags",
        "ref_malformed_doi_flags", "word_count",
    ]
    chunks = []
    for chunk in pd.read_csv(csv_path, usecols=lambda c: c in usecols, chunksize=50_000, low_memory=False):
        chunk = chunk[chunk["status"] == "ok"]
        chunk = chunk[chunk["pub_date"].notna() & (chunk["pub_date"].astype(str).str.len() >= 7)]
        if not chunk.empty:
            chunks.append(chunk)

    if not chunks:
        msg = f"No valid 'ok' rows with publication dates found in {csv_path}."
        log.warning(msg)
        print(msg)
        return

    df = pd.concat(chunks, ignore_index=True)
    del chunks
    gc.collect()

    df["dt"] = pd.to_datetime(df["pub_date"].astype(str).str.slice(0, 7) + "-01", errors="coerce")
    df = df.dropna(subset=["dt"])

    numeric_cols = [
        "fre", "fkgl", "mattr_100", "shannon_char_entropy",
        "passive_voice_ratio", "avg_sentence_length", "author_count",
        "ref_count", "ref_verification_rate", "ref_future_year_flags",
        "ref_malformed_doi_flags", "word_count",
    ]
    for col in numeric_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    # Resample period (M=monthly, Q=quarterly, Y=yearly)
    period_code = {"M": "ME", "Q": "QE", "Y": "YE"}.get(freq.upper(), "ME")
    try:
        grouped = df.set_index("dt").resample(period_code)
    except ValueError:
        # Older pandas compatibility ('M', 'Q', 'Y')
        grouped = df.set_index("dt").resample(freq.upper())

    agg_mean = grouped[numeric_cols].mean()
    agg_std = grouped[numeric_cols].std()
    agg_n = grouped["word_count"].count().rename("paper_count")

    # Drop sparse periods with fewer than 5 papers to avoid noisy spikes
    valid_mask = agg_n >= 5
    if valid_mask.sum() >= 3:
        agg_mean = agg_mean[valid_mask]
        agg_std = agg_std[valid_mask]
        agg_n = agg_n[valid_mask]

    # Save aggregated time-series table for downstream statistical analysis
    ts_table = agg_mean.copy()
    ts_table.insert(0, "paper_count", agg_n)
    ts_csv = out_dir / "trends_timeseries.csv"
    ts_table.to_csv(ts_csv)

    cutoff_ts = None
    if cutoff_date:
        cutoff_ts = pd.to_datetime(cutoff_date + ("-01" if len(cutoff_date) == 7 else ""), errors="coerce")
        if pd.notna(cutoff_ts):
            before = df[df["dt"] < cutoff_ts]
            after = df[df["dt"] >= cutoff_ts]
            summary_rows = []
            for col in numeric_cols:
                if col in df.columns and df[col].notna().any():
                    b_mean = before[col].mean()
                    a_mean = after[col].mean()
                    delta = a_mean - b_mean if (pd.notna(b_mean) and pd.notna(a_mean)) else float("nan")
                    summary_rows.append(
                        {
                            "metric": col,
                            "before_n": int(before[col].notna().sum()),
                            "before_mean": round(b_mean, 4) if pd.notna(b_mean) else "",
                            "before_std": round(before[col].std(), 4) if pd.notna(before[col].std()) else "",
                            "after_n": int(after[col].notna().sum()),
                            "after_mean": round(a_mean, 4) if pd.notna(a_mean) else "",
                            "after_std": round(after[col].std(), 4) if pd.notna(after[col].std()) else "",
                            "delta_after_minus_before": round(delta, 4) if pd.notna(delta) else "",
                        }
                    )
            summary_df = pd.DataFrame(summary_rows)
            summary_csv = out_dir / "before_after_summary.csv"
            summary_df.to_csv(summary_csv, index=False)
            print(f"\nBefore vs. After ({cutoff_date}) summary saved to: {summary_csv}")
            print(summary_df.to_string(index=False))

    plot_specs = [
        ("fre", "Flesch Reading Ease (FRE)", "Score"),
        ("fkgl", "Flesch-Kincaid Grade Level (FKGL)", "Grade Level"),
        ("mattr_100", "MATTR-100 (Lexical Diversity)", "MATTR-100"),
        ("shannon_char_entropy", "Shannon Character Entropy", "Bits / Char"),
        ("passive_voice_ratio", "Passive Voice Ratio", "Fraction of Sentences"),
        ("avg_sentence_length", "Average Sentence Length", "Words / Sentence"),
        ("author_count", "Average Number of Authors", "Authors / Paper"),
        ("ref_count", "Average References per Paper", "References"),
        ("ref_verification_rate", "Reference Verification Rate", "Verified Fraction"),
    ]

    # Filter to metrics that actually have data
    active_specs = [
        spec for spec in plot_specs
        if spec[0] in agg_mean.columns and agg_mean[spec[0]].notna().sum() >= 2
    ]

    # 1. Combined multi-panel dashboard
    if active_specs:
        n_plots = len(active_specs)
        ncols = 3
        nrows = math.ceil(n_plots / ncols)
        fig, axes = plt.subplots(nrows, ncols, figsize=(16, 4.2 * nrows), squeeze=False)
        fig.suptitle(
            f"arXiv Linguistic & Reference Trends Over Time (n = {len(df):,} papers)",
            fontsize=15,
            fontweight="bold",
            y=0.995,
        )

        for idx, (col, title, ylabel) in enumerate(active_specs):
            ax = axes[idx // ncols][idx % ncols]
            series = agg_mean[col].dropna()
            std_s = agg_std[col].reindex(series.index).fillna(0.0)
            n_s = agg_n.reindex(series.index).clip(lower=1)
            se = std_s / (n_s ** 0.5)

            ax.plot(series.index, series.values, linewidth=2.0, color="#1f5a96")
            ax.fill_between(
                series.index,
                series.values - 1.96 * se.values,
                series.values + 1.96 * se.values,
                alpha=0.2,
                color="#1f5a96",
                label="95% CI",
            )
            if cutoff_ts is not None and pd.notna(cutoff_ts):
                ax.axvline(cutoff_ts, color="#c0392b", linestyle="--", linewidth=1.4, label=f"Cutoff ({cutoff_date})")
            ax.set_title(title, fontsize=11, fontweight="bold")
            ax.set_ylabel(ylabel, fontsize=9)
            ax.grid(True, linestyle=":", alpha=0.5)
            ax.xaxis.set_major_locator(mdates.AutoDateLocator(maxticks=6))
            ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m"))
            for label in ax.get_xticklabels():
                label.set_rotation(30)
                label.set_ha("right")

            # Also save individual high-res plot per metric
            fig_single, ax_s = plt.subplots(figsize=(9, 4.8))
            ax_s.plot(series.index, series.values, linewidth=2.2, color="#1f5a96", label="Mean")
            ax_s.fill_between(
                series.index,
                series.values - 1.96 * se.values,
                series.values + 1.96 * se.values,
                alpha=0.2,
                color="#1f5a96",
                label="95% CI",
            )
            if cutoff_ts is not None and pd.notna(cutoff_ts):
                ax_s.axvline(cutoff_ts, color="#c0392b", linestyle="--", linewidth=1.6, label=f"Cutoff ({cutoff_date})")
            ax_s.set_title(title, fontsize=13, fontweight="bold")
            ax_s.set_xlabel("Publication Date")
            ax_s.set_ylabel(ylabel)
            ax_s.grid(True, linestyle=":", alpha=0.5)
            ax_s.legend()
            fig_single.tight_layout()
            fig_single.savefig(out_dir / f"trend_{col}.png", dpi=180)
            plt.close(fig_single)

        # Hide unused subplots
        for idx in range(n_plots, nrows * ncols):
            axes[idx // ncols][idx % ncols].axis("off")

        fig.tight_layout(rect=[0, 0, 1, 0.97])
        dashboard_path = out_dir / "trends_dashboard.png"
        fig.savefig(dashboard_path, dpi=200)
        plt.close(fig)

        print(f"Dashboard plot saved to: {dashboard_path}")
        print(f"Time-series table saved to: {ts_csv}")
        log.info("Generated %d trend plots in %s", len(active_specs), out_dir)

# ---------------------------------------------------------------------------
# Command-Line Interface
# ---------------------------------------------------------------------------

def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Stream arXiv PDFs directly from .tar.gz archives (no disk extraction), "
            "compute linguistic/readability/reference metrics into CSV with full "
            "stop/resume checkpointing, and plot before/after trends."
        ),
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    io_group = parser.add_argument_group("Input / Output & Checkpointing")
    io_group.add_argument(
        "--input-dir",
        type=str,
        help="Directory containing arXiv .tar.gz / .tgz archives (searched recursively).",
    )
    io_group.add_argument(
        "--output",
        type=str,
        default="arxiv_metrics.csv",
        help="Output CSV path (appended safely across stop/resume sessions).",
    )
    io_group.add_argument(
        "--state-db",
        type=str,
        default="arxiv_state.sqlite",
        help="SQLite database tracking completed archives and processed PDFs.",
    )
    io_group.add_argument(
        "--log-file",
        type=str,
        default="arxiv_analyzer.log",
        help="Rotating log file path (max 20 MB x 5 backups).",
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
        help="Number of worker processes (3 leaves 1 core on a quad-core for tar.gz decompression & I/O).",
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
        help="Max pages to read per PDF (also reads final 5 pages for references if truncated).",
    )
    perf_group.add_argument(
        "--pdf-timeout",
        type=float,
        default=45.0,
        help="Per-PDF timeout in seconds so malformed PDFs never hang the pipeline.",
    )

    ref_group = parser.add_argument_group("Reference Validation (Optional Live API)")
    ref_group.add_argument(
        "--validate-refs",
        action="store_true",
        help="Enable live reference validation via CrossRef & arXiv API (cached in SQLite).",
    )
    ref_group.add_argument(
        "--ref-cache-db",
        type=str,
        default="ref_validation_cache.sqlite",
        help="Persistent SQLite cache so repeated references are only checked once.",
    )
    ref_group.add_argument(
        "--ref-sample",
        type=int,
        default=5,
        help="Max references sampled per paper for live API validation.",
    )
    ref_group.add_argument(
        "--ref-timeout",
        type=float,
        default=6.0,
        help="HTTP timeout (seconds) per reference lookup.",
    )
    ref_group.add_argument(
        "--ref-delay",
        type=float,
        default=0.08,
        help="Minimum delay (seconds) between API requests per worker.",
    )
    ref_group.add_argument(
        "--mailto",
        type=str,
        default="",
        help="Contact email for CrossRef polite pool (faster, more reliable rate limits).",
    )

    plot_group = parser.add_argument_group("Trend Plotting")
    plot_group.add_argument(
        "--plot",
        action="store_true",
        help="Generate trend plots after processing finishes.",
    )
    plot_group.add_argument(
        "--plot-only",
        action="store_true",
        help="Skip PDF processing and only generate trend plots from --output CSV.",
    )
    plot_group.add_argument(
        "--plots-dir",
        type=str,
        default="plots",
        help="Directory where trend PNGs and before/after summary CSV will be saved.",
    )
    plot_group.add_argument(
        "--cutoff-date",
        type=str,
        default=None,
        help="Optional before/after cutoff date (YYYY-MM, e.g. 2022-11) drawn on plots and summarized.",
    )
    plot_group.add_argument(
        "--plot-freq",
        type=str,
        default="M",
        choices=["M", "Q", "Y"],
        help="Time aggregation frequency for plots: M (monthly), Q (quarterly), Y (yearly).",
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

    if not args.plot_only:
        if not args.input_dir:
            parser.error("--input-dir is required unless --plot-only is specified.")
        run_pipeline(args, logger)

    if args.plot or args.plot_only:
        generate_trend_plots(
            csv_path=args.output,
            plots_dir=args.plots_dir,
            cutoff_date=args.cutoff_date,
            freq=args.plot_freq,
            logger=logger,
        )


if __name__ == "__main__":
    # 'spawn' or 'fork': keep safe cross-platform entry point
    mp.freeze_support()
    main()
