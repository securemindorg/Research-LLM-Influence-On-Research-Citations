#!/usr/bin/env python3
"""
Pass 2: Decoupled arXiv Reference Validator (CrossRef + arXiv + OpenAlex)
=========================================================================

Reads the crash-safe .jsonl.gz file dumped by Pass 1 (arxiv_analyzer.py),
validates extracted DOIs, arXiv IDs, and bibliographic reference strings
against live scholarly APIs with a persistent SQLite cache, and writes
paper-level validity statistics to a CSV (plus an optional per-reference
audit .jsonl.gz file).

Key features:
  - Streams .jsonl.gz directly (handles concatenated RFC-1952 gzip frames).
  - Persistent SQLite API cache (WAL mode): shared citations across papers
    are queried at most once across all runs.
  - Stop / resume checkpointing in SQLite + append-safe CSV output.
  - Optional stratified sampling (e.g., N papers per pub_month) so you can
    get statistically rigorous trend curves quickly before a full run.
  - Strict token-bucket rate limiting and polite-pool User-Agent headers.
  - Network/API errors are tracked separately and excluded from the validity
    denominator so outages never inflate invalid/hallucination rates.
  - Low memory footprint (<120 MB RAM), ideal for an 8 GB RAM machine.

Dependencies:
    pip install requests tqdm psutil

Usage:
    python validate_references.py \\
        --input arxiv_references.jsonl.gz \\
        --output ref_validation.csv \\
        --email you@university.edu

    # Fast monthly stratified sample (e.g. 200 papers/month) first:
    python validate_references.py \\
        --input arxiv_references.jsonl.gz \\
        --output ref_validation_sample.csv \\
        --per-month-cap 200 \\
        --email you@university.edu
"""

import argparse
import csv
import gzip
import hashlib
import json
import logging
import math
import os
import random
import re
import signal
import sqlite3
import sys
import threading
import time
import urllib.parse
import xml.etree.ElementTree as ET
from collections import defaultdict
from logging.handlers import RotatingFileHandler
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

try:
    import requests
except ImportError:
    requests = None

try:
    from tqdm import tqdm
except ImportError:
    tqdm = None

try:
    import psutil
except ImportError:
    psutil = None


# ---------------------------------------------------------------------------
# Constants & Output CSV Schema
# ---------------------------------------------------------------------------

CSV_COLUMNS = [
    "archive_name",
    "member_name",
    "arxiv_id",
    "pub_date",
    "pub_year",
    "arxiv_category",
    "arxiv_primary_group",
    "ref_count",
    "dois_checked",
    "dois_valid",
    "dois_invalid",
    "dois_error",
    "doi_valid_rate",
    "arxiv_refs_checked",
    "arxiv_refs_valid",
    "arxiv_refs_invalid",
    "arxiv_refs_error",
    "arxiv_valid_rate",
    "entries_checked",
    "entries_verified",
    "entries_unverified",
    "entries_error",
    "entry_verified_rate",
    "total_checked",
    "total_valid",
    "total_invalid",
    "total_error",
    "ref_validity_rate",
    "ref_invalid_rate",
    "has_any_invalid_ref",
    "has_multiple_invalid_refs",
    "mean_match_score",
    "status",
    "error_message",
]

# Section placeholders filled below
# ---------------------------------------------------------------------------
# Logging, Text Normalization & Persistent SQLite Cache
# ---------------------------------------------------------------------------

ARXIV_ATOM_NS = {"atom": "http://www.w3.org/2005/Atom"}

STOPWORDS = frozenset({
    "a", "an", "and", "are", "as", "at", "be", "by", "for", "from", "in",
    "into", "is", "it", "of", "on", "or", "that", "the", "to", "using",
    "via", "with", "over", "under", "based", "toward", "towards", "study",
    "analysis", "approach", "method", "methods", "new", "novel",
})

DOI_CLEAN_RE = re.compile(r"[^A-Za-z0-9./_;()<>:-]+")
ARXIV_ID_NORM_RE = re.compile(
    r"^(?:arXiv:)?(\d{4}\.\d{4,5}|[a-z\-]+(?:\.[A-Z]{2})?/\d{7})(?:v\d+)?$",
    re.IGNORECASE,
)
YEAR_RE = re.compile(r"\b((?:19|20)\d{2})\b")
QUOTED_TITLE_RE = re.compile(r'["“”]([^"“”]{12,260})["“”]')
WORD_TOKEN_RE = re.compile(r"[a-z0-9]{2,}")


def setup_logging(log_file: str, verbose: bool = False) -> logging.Logger:
    logger = logging.getLogger("ref_validator")
    logger.setLevel(logging.DEBUG if verbose else logging.INFO)
    logger.handlers.clear()

    Path(log_file).parent.mkdir(parents=True, exist_ok=True)
    fh = RotatingFileHandler(
        log_file, maxBytes=10 * 1024 * 1024, backupCount=3, encoding="utf-8"
    )
    fh.setLevel(logging.DEBUG)
    fh.setFormatter(
        logging.Formatter("%(asctime)s [%(levelname)s] %(message)s", "%Y-%m-%d %H:%M:%S")
    )
    logger.addHandler(fh)

    ch = logging.StreamHandler(sys.stderr)
    ch.setLevel(logging.WARNING if not verbose else logging.INFO)
    ch.setFormatter(logging.Formatter("[%(levelname)s] %(message)s"))
    logger.addHandler(ch)
    return logger


def get_memory_usage_mb() -> float:
    if psutil is not None:
        try:
            return psutil.Process(os.getpid()).memory_info().rss / (1024 * 1024)
        except Exception:
            pass
    return 0.0


def normalize_doi(raw: str) -> str:
    if not raw:
        return ""
    s = raw.strip().lower()
    s = re.sub(r"^https?://(?:dx\.)?doi\.org/", "", s)
    s = re.sub(r"^doi:\s*", "", s)
    s = s.rstrip(".,;:)]}>\"'")
    # Strip trailing unmatched parenthesis
    while s.endswith(")") and s.count(")") > s.count("("):
        s = s[:-1]
    return s.strip()


def normalize_arxiv_id(raw: str) -> str:
    if not raw:
        return ""
    s = raw.strip()
    s = re.sub(r"^https?://arxiv\.org/(?:abs|pdf)/", "", s, flags=re.IGNORECASE)
    s = re.sub(r"\.pdf$", "", s, flags=re.IGNORECASE)
    s = s.rstrip(".,;:)]}>\"'")
    m = ARXIV_ID_NORM_RE.match(s)
    if not m:
        return ""
    core = m.group(1)
    # New-style numeric IDs stay as-is; old-style archive prefix lowercased
    if "/" in core:
        arch, num = core.split("/", 1)
        return f"{arch.lower()}/{num}"
    return core


def normalize_text_tokens(text: str) -> List[str]:
    if not text:
        return []
    tokens = WORD_TOKEN_RE.findall(text.lower())
    return [t for t in tokens if t not in STOPWORDS]


def extract_query_features(entry: str) -> Dict[str, Any]:
    """
    Extract normalized bibliographic query text, likely title snippet,
    candidate years, and content tokens from a raw reference string.
    """
    clean = re.sub(r"^\s*(?:\[\d+\]|\(\d+\)|\d+[.)])\s*", "", entry or "").strip()
    clean = re.sub(r"\s+", " ", clean)

    years = [int(y) for y in YEAR_RE.findall(clean) if 1900 <= int(y) <= 2030]
    quoted = QUOTED_TITLE_RE.search(clean)
    title_hint = quoted.group(1).strip() if quoted else ""

    if not title_hint:
        # Heuristic: split on periods and pick the longest non-author-like clause
        parts = [p.strip() for p in re.split(r"\.\s+", clean) if len(p.strip()) >= 12]
        if len(parts) >= 2:
            # Usually part 0 is authors, part 1 is title (or year then title)
            candidates = parts[1:4]
            title_hint = max(candidates, key=lambda p: len(normalize_text_tokens(p)), default="")
        elif parts:
            title_hint = parts[0]

    # Build concise query string for CrossRef / OpenAlex bibliographic search
    query_str = clean[:260]
    norm_key = hashlib.sha256(
        " ".join(normalize_text_tokens(clean)[:28]).encode("utf-8")
    ).hexdigest()

    return {
        "clean": clean,
        "query": query_str,
        "title_hint": title_hint[:200],
        "years": years,
        "tokens": set(normalize_text_tokens(clean)),
        "title_tokens": set(normalize_text_tokens(title_hint)),
        "cache_key": norm_key,
    }


def score_bibliographic_match(
    feat: Dict[str, Any],
    cand_title: str,
    cand_year: Optional[int],
    cand_authors: List[str],
) -> Tuple[float, Dict[str, Any]]:
    """
    Compute a transparent, deterministic [0, 1] similarity score between an
    extracted reference entry and a candidate scholarly record.

    Weighted components:
      - Title token overlap (recall of candidate title tokens in entry,
        and overlap with title_hint if available): 0.65
      - Author surname match: 0.20
      - Publication year match (+/- 1 year tolerance for preprint->journal lag): 0.15
    """
    cand_title_tokens = set(normalize_text_tokens(cand_title))
    if not cand_title_tokens:
        return 0.0, {"title_overlap": 0.0, "author_match": 0.0, "year_match": 0.0}

    entry_tokens = feat["tokens"]
    title_hint_tokens = feat["title_tokens"]

    # Recall of candidate title words inside the full reference string
    cand_in_entry = len(cand_title_tokens & entry_tokens) / max(1, len(cand_title_tokens))

    # Symmetric F1 against title_hint if we isolated a decent title segment
    if len(title_hint_tokens) >= 3:
        inter = len(cand_title_tokens & title_hint_tokens)
        prec = inter / len(title_hint_tokens)
        rec = inter / len(cand_title_tokens)
        f1 = (2 * prec * rec / (prec + rec)) if (prec + rec) > 0 else 0.0
        title_score = max(cand_in_entry, f1)
    else:
        title_score = cand_in_entry

    # Penalize very short candidate titles (<=2 content tokens) to avoid false matches
    if len(cand_title_tokens) <= 2:
        title_score *= 0.65

    # Author surname check
    author_score = 0.0
    if cand_authors:
        surnames = []
        for a in cand_authors[:8]:
            parts = WORD_TOKEN_RE.findall(a.lower())
            if parts:
                surnames.append(parts[-1])
        surnames = [s for s in surnames if len(s) >= 3]
        if surnames:
            matched_authors = sum(1 for s in surnames if s in entry_tokens)
            author_score = min(1.0, matched_authors / min(2, len(surnames)))

    # Year match (+/- 1 year allowed for arXiv-to-proceedings/journal delay)
    year_score = 0.0
    if cand_year is not None and feat["years"]:
        min_diff = min(abs(cand_year - y) for y in feat["years"])
        if min_diff == 0:
            year_score = 1.0
        elif min_diff == 1:
            year_score = 0.7
        elif min_diff == 2:
            year_score = 0.3

    # If no author metadata returned, redistribute author weight onto title
    if cand_authors:
        total = 0.65 * title_score + 0.20 * author_score + 0.15 * year_score
    else:
        total = 0.82 * title_score + 0.18 * year_score

    # Guardrail: high author+year alone must NEVER validate a mismatched title
    if title_score < 0.45:
        total = min(total, 0.44)

    return round(total, 4), {
        "title_overlap": round(title_score, 3),
        "author_match": round(author_score, 3),
        "year_match": round(year_score, 3),
    }


class RateLimiter:
    """Thread-safe minimum-interval rate limiter."""

    def __init__(self, min_interval_sec: float):
        self.min_interval = max(0.0, float(min_interval_sec))
        self._lock = threading.Lock()
        self._next_allowed = 0.0

    def wait(self) -> None:
        if self.min_interval <= 0:
            return
        with self._lock:
            now = time.monotonic()
            if now < self._next_allowed:
                sleep_for = self._next_allowed - now
                self._next_allowed += self.min_interval
            else:
                sleep_for = 0.0
                self._next_allowed = now + self.min_interval
        if sleep_for > 0:
            time.sleep(sleep_for)


class ReferenceCache:
    """
    Persistent SQLite cache (WAL mode) storing validation outcomes for:
      - DOIs       (kind='doi')
      - arXiv IDs  (kind='arxiv')
      - Ref texts  (kind='entry')

    Statuses stored: 'valid', 'invalid'. Transient network/API 'error'
    outcomes are NOT permanently cached so a later retry can resolve them.
    """

    def __init__(self, db_path: str):
        self.db_path = db_path
        Path(db_path).parent.mkdir(parents=True, exist_ok=True)
        self.conn = sqlite3.connect(db_path, timeout=60)
        self.conn.execute("PRAGMA journal_mode=WAL;")
        self.conn.execute("PRAGMA synchronous=NORMAL;")
        self.conn.execute(
            """
            CREATE TABLE IF NOT EXISTS ref_cache (
                kind TEXT NOT NULL,
                key TEXT NOT NULL,
                status TEXT NOT NULL,
                source TEXT,
                score REAL,
                matched_title TEXT,
                matched_year INTEGER,
                checked_at REAL,
                PRIMARY KEY (kind, key)
            )
            """
        )
        self.conn.commit()
        self._pending: List[Tuple] = []
        self.hits = 0
        self.misses = 0

    def get(self, kind: str, key: str) -> Optional[Dict[str, Any]]:
        row = self.conn.execute(
            """
            SELECT status, source, score, matched_title, matched_year
            FROM ref_cache WHERE kind = ? AND key = ?
            """,
            (kind, key),
        ).fetchone()
        if row is None:
            self.misses += 1
            return None
        self.hits += 1
        return {
            "status": row[0],
            "source": row[1] or "",
            "score": float(row[2]) if row[2] is not None else 0.0,
            "matched_title": row[3] or "",
            "matched_year": row[4],
            "cached": True,
        }

    def put(
        self,
        kind: str,
        key: str,
        status: str,
        source: str,
        score: float = 0.0,
        matched_title: str = "",
        matched_year: Optional[int] = None,
    ) -> None:
        if status not in ("valid", "invalid"):
            return
        self._pending.append(
            (kind, key, status, source, score, (matched_title or "")[:240], matched_year, time.time())
        )
        if len(self._pending) >= 100:
            self.flush()

    def flush(self) -> None:
        if not self._pending:
            return
        self.conn.executemany(
            """
            INSERT OR REPLACE INTO ref_cache
            (kind, key, status, source, score, matched_title, matched_year, checked_at)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """,
            self._pending,
        )
        self.conn.commit()
        self._pending.clear()

    def count(self) -> int:
        row = self.conn.execute("SELECT COUNT(*) FROM ref_cache").fetchone()
        return int(row[0]) if row else 0

    def close(self) -> None:
        try:
            self.flush()
        finally:
            try:
                self.conn.close()
            except Exception:
                pass

# ---------------------------------------------------------------------------
# Scholarly API Client (CrossRef + arXiv + OpenAlex)
# ---------------------------------------------------------------------------

class ScholarlyAPIClient:
    """
    Polite, rate-limited HTTP client for CrossRef, arXiv, and OpenAlex APIs.
    Uses persistent HTTP keep-alive connections and exponential backoff on
    429 / 5xx responses.
    """

    CROSSREF_BASE = "https://api.crossref.org/works"
    OPENALEX_BASE = "https://api.openalex.org/works"
    ARXIV_BASE = "https://export.arxiv.org/api/query"

    def __init__(
        self,
        email: str = "",
        crossref_delay: float = 0.12,
        openalex_delay: float = 0.12,
        arxiv_delay: float = 3.1,
        timeout: float = 15.0,
        max_retries: int = 3,
        use_openalex: bool = True,
        match_threshold: float = 0.62,
        logger: Optional[logging.Logger] = None,
    ):
        if requests is None:
            raise RuntimeError("`requests` is required. Install with: pip install requests")
        self.email = (email or "").strip()
        self.timeout = timeout
        self.max_retries = max_retries
        self.use_openalex = use_openalex
        self.match_threshold = match_threshold
        self.logger = logger or logging.getLogger("ref_validator")

        ua_contact = f" (mailto:{self.email})" if self.email else ""
        self.session = requests.Session()
        self.session.headers.update(
            {
                "User-Agent": f"arXivStylometryRefValidator/2.0{ua_contact}",
                "Accept": "application/json, application/atom+xml;q=0.9",
            }
        )

        self.crossref_limiter = RateLimiter(crossref_delay)
        self.openalex_limiter = RateLimiter(openalex_delay)
        self.arxiv_limiter = RateLimiter(arxiv_delay)

        self.api_calls = {"crossref": 0, "openalex": 0, "arxiv": 0}
        self.api_errors = {"crossref": 0, "openalex": 0, "arxiv": 0}

    def close(self) -> None:
        try:
            self.session.close()
        except Exception:
            pass

    def _get(
        self,
        url: str,
        params: Optional[Dict[str, Any]],
        limiter: RateLimiter,
        service: str,
    ) -> Optional[requests.Response]:
        backoff = 1.5
        for attempt in range(self.max_retries):
            limiter.wait()
            try:
                self.api_calls[service] += 1
                resp = self.session.get(url, params=params, timeout=self.timeout)
                if resp.status_code == 429 or 500 <= resp.status_code < 600:
                    retry_after = resp.headers.get("Retry-After")
                    wait_s = backoff * (2 ** attempt)
                    if retry_after:
                        try:
                            wait_s = max(wait_s, min(30.0, float(retry_after)))
                        except ValueError:
                            pass
                    self.logger.debug(
                        "%s HTTP %s on attempt %d — sleeping %.1fs",
                        service, resp.status_code, attempt + 1, wait_s,
                    )
                    time.sleep(wait_s)
                    continue
                return resp
            except requests.RequestException as exc:
                self.logger.debug("%s request exception (attempt %d): %s", service, attempt + 1, exc)
                if attempt + 1 < self.max_retries:
                    time.sleep(backoff * (2 ** attempt))
        self.api_errors[service] += 1
        return None

    # ---- DOI Validation ----------------------------------------------------

    def verify_doi(self, doi: str) -> Dict[str, Any]:
        """
        Verify a normalized DOI via CrossRef, falling back to OpenAlex
        (which covers DataCite, Zenodo, arXiv DOIs, etc.) on 404.
        """
        encoded = urllib.parse.quote(doi, safe="")
        params = {"mailto": self.email} if self.email else None

        # 1. CrossRef exact DOI lookup
        resp = self._get(f"{self.CROSSREF_BASE}/{encoded}", params, self.crossref_limiter, "crossref")
        crossref_404 = False
        if resp is not None:
            if resp.status_code == 200:
                try:
                    msg = resp.json().get("message", {})
                    titles = msg.get("title") or []
                    title = titles[0] if titles else ""
                    year = self._extract_crossref_year(msg)
                    return {
                        "status": "valid",
                        "source": "crossref_doi",
                        "score": 1.0,
                        "matched_title": title,
                        "matched_year": year,
                    }
                except Exception:
                    return {
                        "status": "valid",
                        "source": "crossref_doi",
                        "score": 1.0,
                        "matched_title": "",
                        "matched_year": None,
                    }
            elif resp.status_code == 404:
                crossref_404 = True

        # 2. OpenAlex fallback (catches DataCite / Zenodo / arXiv / non-CrossRef DOIs)
        if self.use_openalex:
            oa_url = f"{self.OPENALEX_BASE}/https://doi.org/{encoded}"
            oa_params = {"select": "id,title,publication_year"}
            if self.email:
                oa_params["mailto"] = self.email
            oa_resp = self._get(oa_url, oa_params, self.openalex_limiter, "openalex")
            if oa_resp is not None:
                if oa_resp.status_code == 200:
                    try:
                        data = oa_resp.json()
                        return {
                            "status": "valid",
                            "source": "openalex_doi",
                            "score": 1.0,
                            "matched_title": data.get("title") or "",
                            "matched_year": data.get("publication_year"),
                        }
                    except Exception:
                        return {
                            "status": "valid",
                            "source": "openalex_doi",
                            "score": 1.0,
                            "matched_title": "",
                            "matched_year": None,
                        }
                elif oa_resp.status_code == 404 and crossref_404:
                    return {
                        "status": "invalid",
                        "source": "crossref+openalex_404",
                        "score": 0.0,
                        "matched_title": "",
                        "matched_year": None,
                    }
        elif crossref_404:
            return {
                "status": "invalid",
                "source": "crossref_404",
                "score": 0.0,
                "matched_title": "",
                "matched_year": None,
            }

        return {
            "status": "error",
            "source": "doi_api_error",
            "score": 0.0,
            "matched_title": "",
            "matched_year": None,
        }

    # ---- arXiv ID Batch Validation -----------------------------------------

    def verify_arxiv_batch(self, arxiv_ids: List[str]) -> Dict[str, Dict[str, Any]]:
        """
        Verify a batch of normalized arXiv IDs in a single polite arXiv API call
        (recommended batch size <= 25).
        """
        results: Dict[str, Dict[str, Any]] = {}
        if not arxiv_ids:
            return results

        params = {
            "id_list": ",".join(arxiv_ids),
            "max_results": len(arxiv_ids) + 5,
        }
        resp = self._get(self.ARXIV_BASE, params, self.arxiv_limiter, "arxiv")
        if resp is None or resp.status_code != 200:
            for aid in arxiv_ids:
                results[aid] = {
                    "status": "error",
                    "source": "arxiv_api_error",
                    "score": 0.0,
                    "matched_title": "",
                    "matched_year": None,
                }
            return results

        found: Dict[str, Tuple[str, Optional[int]]] = {}
        try:
            root = ET.fromstring(resp.text)
            for entry in root.findall("atom:entry", ARXIV_ATOM_NS):
                id_el = entry.find("atom:id", ARXIV_ATOM_NS)
                title_el = entry.find("atom:title", ARXIV_ATOM_NS)
                pub_el = entry.find("atom:published", ARXIV_ATOM_NS)

                if id_el is None or not id_el.text:
                    continue
                raw_id = id_el.text.strip()
                title = re.sub(r"\s+", " ", (title_el.text or "").strip()) if title_el is not None else ""
                # arXiv returns a synthetic "Error" entry when an ID is malformed/nonexistent
                if title.lower() == "error" or "/api/errors" in raw_id:
                    continue

                norm = normalize_arxiv_id(raw_id)
                year = None
                if pub_el is not None and pub_el.text and len(pub_el.text) >= 4:
                    try:
                        year = int(pub_el.text[:4])
                    except ValueError:
                        year = None
                if norm:
                    found[norm] = (title, year)
        except ET.ParseError as exc:
            self.logger.debug("arXiv XML parse error: %s", exc)
            for aid in arxiv_ids:
                results[aid] = {
                    "status": "error",
                    "source": "arxiv_xml_error",
                    "score": 0.0,
                    "matched_title": "",
                    "matched_year": None,
                }
            return results

        for aid in arxiv_ids:
            if aid in found:
                title, year = found[aid]
                results[aid] = {
                    "status": "valid",
                    "source": "arxiv_api",
                    "score": 1.0,
                    "matched_title": title,
                    "matched_year": year,
                }
            else:
                results[aid] = {
                    "status": "invalid",
                    "source": "arxiv_not_found",
                    "score": 0.0,
                    "matched_title": "",
                    "matched_year": None,
                }
        return results

    # ---- Bibliographic Text Reference Validation ---------------------------

    def verify_entry_text(self, feat: Dict[str, Any]) -> Dict[str, Any]:
        """
        Validate a free-text bibliographic reference string against CrossRef
        bibliographic search, with OpenAlex fallback if below threshold.
        """
        best_score = 0.0
        best_title = ""
        best_year: Optional[int] = None
        best_source = "unverified"
        any_api_ok = False

        # 1. CrossRef bibliographic search
        cr_params: Dict[str, Any] = {
            "query.bibliographic": feat["query"],
            "rows": 3,
            "select": "DOI,title,author,issued,published-print,published-online,created",
        }
        if self.email:
            cr_params["mailto"] = self.email

        resp = self._get(self.CROSSREF_BASE, cr_params, self.crossref_limiter, "crossref")
        if resp is not None and resp.status_code == 200:
            any_api_ok = True
            try:
                items = resp.json().get("message", {}).get("items") or []
                for item in items:
                    titles = item.get("title") or []
                    cand_title = titles[0] if titles else ""
                    cand_year = self._extract_crossref_year(item)
                    cand_authors = self._extract_crossref_authors(item)
                    sc, _ = score_bibliographic_match(feat, cand_title, cand_year, cand_authors)
                    if sc > best_score:
                        best_score = sc
                        best_title = cand_title
                        best_year = cand_year
                        best_source = "crossref_bib"
            except Exception as exc:
                self.logger.debug("CrossRef JSON parse error: %s", exc)

        if best_score >= self.match_threshold:
            return {
                "status": "valid",
                "source": best_source,
                "score": best_score,
                "matched_title": best_title,
                "matched_year": best_year,
            }

        # 2. OpenAlex fallback (strong coverage of CS conference papers & preprints)
        if self.use_openalex:
            search_q = feat["title_hint"] if len(feat["title_tokens"]) >= 4 else feat["query"][:180]
            oa_params: Dict[str, Any] = {
                "search": search_q,
                "per-page": 3,
                "select": "id,title,publication_year,authorships",
            }
            if self.email:
                oa_params["mailto"] = self.email
            oa_resp = self._get(self.OPENALEX_BASE, oa_params, self.openalex_limiter, "openalex")
            if oa_resp is not None and oa_resp.status_code == 200:
                any_api_ok = True
                try:
                    items = oa_resp.json().get("results") or []
                    for item in items:
                        cand_title = item.get("title") or ""
                        cand_year = item.get("publication_year")
                        cand_authors = [
                            (a.get("author") or {}).get("display_name", "")
                            for a in (item.get("authorships") or [])[:8]
                        ]
                        sc, _ = score_bibliographic_match(feat, cand_title, cand_year, cand_authors)
                        if sc > best_score:
                            best_score = sc
                            best_title = cand_title
                            best_year = cand_year
                            best_source = "openalex_bib"
                except Exception as exc:
                    self.logger.debug("OpenAlex JSON parse error: %s", exc)

        if not any_api_ok:
            return {
                "status": "error",
                "source": "bib_api_error",
                "score": 0.0,
                "matched_title": "",
                "matched_year": None,
            }

        if best_score >= self.match_threshold:
            return {
                "status": "valid",
                "source": best_source,
                "score": best_score,
                "matched_title": best_title,
                "matched_year": best_year,
            }

        return {
            "status": "invalid",
            "source": "unverified",
            "score": best_score,
            "matched_title": best_title,
            "matched_year": best_year,
        }

    @staticmethod
    def _extract_crossref_year(item: Dict[str, Any]) -> Optional[int]:
        for field in ("issued", "published-print", "published-online", "created"):
            val = item.get(field)
            if isinstance(val, dict):
                parts = val.get("date-parts")
                if parts and isinstance(parts[0], list) and parts[0]:
                    try:
                        return int(parts[0][0])
                    except (TypeError, ValueError):
                        pass
        return None

    @staticmethod
    def _extract_crossref_authors(item: Dict[str, Any]) -> List[str]:
        out: List[str] = []
        for a in (item.get("author") or [])[:8]:
            fam = (a.get("family") or "").strip()
            giv = (a.get("given") or "").strip()
            if fam:
                out.append(f"{giv} {fam}".strip())
        return out

# ---------------------------------------------------------------------------
# Paper Validator & Checkpoint State Manager
# ---------------------------------------------------------------------------

class ValidationStateManager:
    """
    Tracks validated papers in SQLite (WAL mode) for seamless stop/resume,
    writes paper-level rows to an append-safe CSV, and optionally writes
    per-reference audit records to a crash-safe .jsonl.gz file.
    """

    def __init__(
        self,
        state_db: str,
        csv_path: str,
        audit_jsonl_gz: Optional[str] = None,
    ):
        self.state_db = state_db
        self.csv_path = Path(csv_path)
        self.csv_path.parent.mkdir(parents=True, exist_ok=True)
        Path(state_db).parent.mkdir(parents=True, exist_ok=True)

        self.conn = sqlite3.connect(state_db, timeout=60)
        self.conn.execute("PRAGMA journal_mode=WAL;")
        self.conn.execute("PRAGMA synchronous=NORMAL;")
        self.conn.execute(
            """
            CREATE TABLE IF NOT EXISTS validated_papers (
                archive_name TEXT NOT NULL,
                member_name TEXT NOT NULL,
                arxiv_id TEXT,
                pub_date TEXT,
                status TEXT,
                validated_at REAL,
                PRIMARY KEY (archive_name, member_name)
            )
            """
        )
        self.conn.commit()

        need_header = (not self.csv_path.exists()) or (self.csv_path.stat().st_size == 0)
        self._csv_file = open(self.csv_path, "a", newline="", encoding="utf-8")
        self._csv_writer = csv.DictWriter(
            self._csv_file, fieldnames=CSV_COLUMNS, extrasaction="ignore"
        )
        if need_header:
            self._csv_writer.writeheader()
            self._csv_file.flush()

        self._audit_file = None
        if audit_jsonl_gz:
            ap = Path(audit_jsonl_gz)
            ap.parent.mkdir(parents=True, exist_ok=True)
            self._audit_file = open(ap, "ab")

        self._pending_papers: List[Tuple[str, str, str, str, str]] = []
        self._pending_audit_lines: List[str] = []

    def load_done_keys(self) -> Tuple[set, Dict[str, int]]:
        """Return set of (archive_name, member_name) already validated and per-month counts."""
        done = set()
        month_counts: Dict[str, int] = defaultdict(int)
        for arch, memb, pub_date in self.conn.execute(
            "SELECT archive_name, member_name, pub_date FROM validated_papers"
        ):
            done.add((arch, memb))
            if pub_date:
                month_counts[pub_date] += 1
        return done, month_counts

    def record_paper(
        self,
        row: Dict[str, Any],
        audit_items: Optional[List[Dict[str, Any]]] = None,
    ) -> None:
        self._csv_writer.writerow(row)
        self._pending_papers.append(
            (
                row["archive_name"],
                row["member_name"],
                row["arxiv_id"],
                row["pub_date"],
                row["status"],
            )
        )
        if audit_items and self._audit_file is not None:
            record = {
                "arxiv_id": row["arxiv_id"],
                "pub_date": row["pub_date"],
                "arxiv_primary_group": row["arxiv_primary_group"],
                "checks": audit_items,
            }
            self._pending_audit_lines.append(
                json.dumps(record, ensure_ascii=False, separators=(",", ":")) + "\n"
            )
        if len(self._pending_papers) >= 20:
            self.flush()

    def flush(self) -> None:
        if self._pending_audit_lines and self._audit_file is not None:
            raw = "".join(self._pending_audit_lines).encode("utf-8")
            self._audit_file.write(gzip.compress(raw, compresslevel=6))
            self._audit_file.flush()
            self._pending_audit_lines.clear()

        if self._pending_papers:
            now = time.time()
            self.conn.executemany(
                """
                INSERT OR REPLACE INTO validated_papers
                (archive_name, member_name, arxiv_id, pub_date, status, validated_at)
                VALUES (?, ?, ?, ?, ?, ?)
                """,
                [(a, m, aid, pd, st, now) for (a, m, aid, pd, st) in self._pending_papers],
            )
            self.conn.commit()
            self._pending_papers.clear()

        self._csv_file.flush()

    def close(self) -> None:
        try:
            self.flush()
        finally:
            for fh in (self._audit_file, self._csv_file):
                if fh is not None:
                    try:
                        fh.close()
                    except Exception:
                        pass
            try:
                self.conn.close()
            except Exception:
                pass


def _evenly_spaced_sample(items: List[Any], cap: int) -> List[Any]:
    if cap <= 0 or len(items) <= cap:
        return items
    step = len(items) / cap
    return [items[int(i * step)] for i in range(cap)]


def validate_paper_record(
    rec: Dict[str, Any],
    api: ScholarlyAPIClient,
    cache: ReferenceCache,
    mode: str = "all",
    max_dois: int = 10,
    max_arxiv: int = 10,
    max_entries: int = 5,
    arxiv_batch_size: int = 20,
    keep_audit: bool = False,
) -> Tuple[Dict[str, Any], List[Dict[str, Any]]]:
    """
    Validate references for a single Pass 1 JSONL record using the cache and API client.
    Returns (csv_row, audit_items).
    """
    row: Dict[str, Any] = {col: "" for col in CSV_COLUMNS}
    row.update(
        {
            "archive_name": rec.get("archive_name", ""),
            "member_name": rec.get("member_name", ""),
            "arxiv_id": rec.get("arxiv_id", ""),
            "pub_date": rec.get("pub_date", ""),
            "pub_year": rec.get("pub_year", ""),
            "arxiv_category": rec.get("arxiv_category", ""),
            "arxiv_primary_group": rec.get("arxiv_primary_group", ""),
            "ref_count": rec.get("ref_count", 0),
            "status": "ok",
            "error_message": "",
        }
    )

    audit_items: List[Dict[str, Any]] = []
    match_scores: List[float] = []

    dois_checked = dois_valid = dois_invalid = dois_error = 0
    arxiv_checked = arxiv_valid = arxiv_invalid = arxiv_error = 0
    entries_checked = entries_verified = entries_unverified = entries_error = 0

    checked_dois_norm: List[str] = []
    checked_arxiv_norm: List[str] = []

    # 1. Validate DOIs
    if mode in ("all", "identifiers-only"):
        raw_dois = rec.get("dois") or []
        norm_dois: List[str] = []
        seen_d = set()
        for d in raw_dois:
            nd = normalize_doi(str(d))
            if nd and nd.startswith("10.") and "/" in nd and nd not in seen_d:
                seen_d.add(nd)
                norm_dois.append(nd)
        sampled_dois = _evenly_spaced_sample(norm_dois, max_dois)
        checked_dois_norm = sampled_dois

        for doi in sampled_dois:
            res = cache.get("doi", doi)
            if res is None:
                res = api.verify_doi(doi)
                if res["status"] in ("valid", "invalid"):
                    cache.put(
                        "doi",
                        doi,
                        res["status"],
                        res["source"],
                        res["score"],
                        res["matched_title"],
                        res["matched_year"],
                    )
            st = res["status"]
            if st == "valid":
                dois_checked += 1
                dois_valid += 1
                match_scores.append(1.0)
            elif st == "invalid":
                dois_checked += 1
                dois_invalid += 1
                match_scores.append(0.0)
            else:
                dois_error += 1

            if keep_audit:
                audit_items.append(
                    {
                        "kind": "doi",
                        "ref": doi,
                        "status": st,
                        "source": res["source"],
                        "score": res["score"],
                    }
                )

    # 2. Validate arXiv IDs (batched for speed & arXiv API courtesy)
    if mode in ("all", "identifiers-only"):
        raw_arxiv = rec.get("arxiv_refs") or []
        norm_arxiv: List[str] = []
        seen_a = set()
        self_arxiv = normalize_arxiv_id(str(rec.get("arxiv_id", "")))
        for a in raw_arxiv:
            na = normalize_arxiv_id(str(a))
            if na and na != self_arxiv and na not in seen_a:
                seen_a.add(na)
                norm_arxiv.append(na)
        sampled_arxiv = _evenly_spaced_sample(norm_arxiv, max_arxiv)
        checked_arxiv_norm = sampled_arxiv

        uncached_arxiv: List[str] = []
        arxiv_results: Dict[str, Dict[str, Any]] = {}
        for aid in sampled_arxiv:
            cached = cache.get("arxiv", aid)
            if cached is not None:
                arxiv_results[aid] = cached
            else:
                uncached_arxiv.append(aid)

        for i in range(0, len(uncached_arxiv), max(1, arxiv_batch_size)):
            chunk = uncached_arxiv[i : i + arxiv_batch_size]
            batch_res = api.verify_arxiv_batch(chunk)
            for aid, res in batch_res.items():
                arxiv_results[aid] = res
                if res["status"] in ("valid", "invalid"):
                    cache.put(
                        "arxiv",
                        aid,
                        res["status"],
                        res["source"],
                        res["score"],
                        res["matched_title"],
                        res["matched_year"],
                    )

        for aid in sampled_arxiv:
            res = arxiv_results.get(aid, {"status": "error", "source": "missing", "score": 0.0})
            st = res["status"]
            if st == "valid":
                arxiv_checked += 1
                arxiv_valid += 1
                match_scores.append(1.0)
            elif st == "invalid":
                arxiv_checked += 1
                arxiv_invalid += 1
                match_scores.append(0.0)
            else:
                arxiv_error += 1

            if keep_audit:
                audit_items.append(
                    {
                        "kind": "arxiv",
                        "ref": aid,
                        "status": st,
                        "source": res["source"],
                        "score": res["score"],
                    }
                )

    # 3. Validate free-text bibliographic entries (skip entries whose DOI/arXiv was already checked)
    if mode in ("all", "entries-only") and max_entries != 0:
        raw_entries = rec.get("entries") or []
        candidate_entries: List[str] = []
        for entry in raw_entries:
            if not isinstance(entry, str) or len(entry.strip()) < 25:
                continue
            low = entry.lower()
            if mode == "all":
                if any(d in low for d in checked_dois_norm):
                    continue
                if any(a.lower() in low for a in checked_arxiv_norm):
                    continue
            candidate_entries.append(entry)

        # Cap = -1 means unlimited; >0 means evenly spaced sample
        sampled_entries = (
            candidate_entries if max_entries < 0 else _evenly_spaced_sample(candidate_entries, max_entries)
        )

        for entry in sampled_entries:
            feat = extract_query_features(entry)
            if len(feat["tokens"]) < 4:
                continue

            res = cache.get("entry", feat["cache_key"])
            if res is None:
                res = api.verify_entry_text(feat)
                if res["status"] in ("valid", "invalid"):
                    cache.put(
                        "entry",
                        feat["cache_key"],
                        res["status"],
                        res["source"],
                        res["score"],
                        res["matched_title"],
                        res["matched_year"],
                    )

            st = res["status"]
            if st == "valid":
                entries_checked += 1
                entries_verified += 1
                match_scores.append(float(res["score"]))
            elif st == "invalid":
                entries_checked += 1
                entries_unverified += 1
                match_scores.append(float(res["score"]))
            else:
                entries_error += 1

            if keep_audit:
                audit_items.append(
                    {
                        "kind": "entry",
                        "ref": feat["clean"][:220],
                        "status": st,
                        "source": res["source"],
                        "score": res["score"],
                        "matched_title": res.get("matched_title", "")[:140],
                    }
                )

    # Aggregate paper-level totals (excluding API errors from denominators)
    total_checked = dois_checked + arxiv_checked + entries_checked
    total_valid = dois_valid + arxiv_valid + entries_verified
    total_invalid = dois_invalid + arxiv_invalid + entries_unverified
    total_error = dois_error + arxiv_error + entries_error

    row.update(
        {
            "dois_checked": dois_checked,
            "dois_valid": dois_valid,
            "dois_invalid": dois_invalid,
            "dois_error": dois_error,
            "doi_valid_rate": round(dois_valid / dois_checked, 4) if dois_checked else "",
            "arxiv_refs_checked": arxiv_checked,
            "arxiv_refs_valid": arxiv_valid,
            "arxiv_refs_invalid": arxiv_invalid,
            "arxiv_refs_error": arxiv_error,
            "arxiv_valid_rate": round(arxiv_valid / arxiv_checked, 4) if arxiv_checked else "",
            "entries_checked": entries_checked,
            "entries_verified": entries_verified,
            "entries_unverified": entries_unverified,
            "entries_error": entries_error,
            "entry_verified_rate": round(entries_verified / entries_checked, 4) if entries_checked else "",
            "total_checked": total_checked,
            "total_valid": total_valid,
            "total_invalid": total_invalid,
            "total_error": total_error,
            "ref_validity_rate": round(total_valid / total_checked, 4) if total_checked else "",
            "ref_invalid_rate": round(total_invalid / total_checked, 4) if total_checked else "",
            "has_any_invalid_ref": int(total_invalid > 0) if total_checked else "",
            "has_multiple_invalid_refs": int(total_invalid >= 2) if total_checked else "",
            "mean_match_score": (
                round(sum(match_scores) / len(match_scores), 4) if match_scores else ""
            ),
            "status": "ok" if total_checked > 0 else ("api_error" if total_error > 0 else "no_checkable_refs"),
        }
    )

    return row, audit_items

# ---------------------------------------------------------------------------
# Streaming .jsonl.gz Reader, Pipeline & CLI
# ---------------------------------------------------------------------------

def iter_jsonl_gz(path: Path, logger: logging.Logger) -> Iterable[Dict[str, Any]]:
    """
    Stream JSON objects line by line from a .jsonl.gz (or plain .jsonl) file.
    Handles concatenated RFC-1952 gzip frames seamlessly and tolerates an
    unexpected truncated tail frame if Pass 1 was killed mid-write.
    """
    is_gz = path.name.lower().endswith(".gz")
    opener = (lambda p: gzip.open(p, "rt", encoding="utf-8", errors="replace")) if is_gz else (
        lambda p: open(p, "r", encoding="utf-8", errors="replace")
    )
    line_num = 0
    try:
        with opener(path) as fh:
            for line in fh:
                line_num += 1
                s = line.strip()
                if not s:
                    continue
                try:
                    yield json.loads(s)
                except json.JSONDecodeError as exc:
                    logger.warning("Skipping malformed JSON at line %d: %s", line_num, exc)
    except (EOFError, OSError, gzip.BadGzipFile) as exc:
        logger.warning(
            "Reached truncated/corrupted gzip tail after line %d (%s); "
            "all prior complete frames were processed safely.",
            line_num,
            exc,
        )


def should_include_record(
    rec: Dict[str, Any],
    args: argparse.Namespace,
    month_counts: Dict[str, int],
) -> bool:
    """Apply optional date, category, deterministic sampling, and per-month cap filters."""
    pub_year = rec.get("pub_year")
    if args.year_min is not None:
        if pub_year is None or pub_year == "" or int(pub_year) < args.year_min:
            return False
    if args.year_max is not None:
        if pub_year is None or pub_year == "" or int(pub_year) > args.year_max:
            return False

    if args.category_group:
        grp = (rec.get("arxiv_primary_group") or "").strip().lower()
        if grp not in args.category_group:
            return False

    pub_date = rec.get("pub_date") or ""
    if args.per_month_cap > 0 and pub_date:
        if month_counts[pub_date] >= args.per_month_cap:
            return False

    if 0.0 < args.sample_rate < 1.0:
        # Deterministic hash sampling so stop/resume selects the exact same papers
        key = f"{args.seed}:{rec.get('archive_name', '')}:{rec.get('member_name', '')}"
        digest = int(hashlib.md5(key.encode("utf-8")).hexdigest()[:8], 16)
        if (digest / 0xFFFFFFFF) >= args.sample_rate:
            return False

    return True


def run_validation(args: argparse.Namespace, logger: logging.Logger) -> None:
    input_path = Path(args.input)
    if not input_path.exists():
        raise FileNotFoundError(f"Input references file not found: {input_path}")

    cache = ReferenceCache(args.cache_db)
    state = ValidationStateManager(
        state_db=args.state_db,
        csv_path=args.output,
        audit_jsonl_gz=args.audit_output if args.audit_output else None,
    )
    api = ScholarlyAPIClient(
        email=args.email,
        crossref_delay=args.crossref_delay,
        openalex_delay=args.openalex_delay,
        arxiv_delay=args.arxiv_delay,
        timeout=args.timeout,
        max_retries=args.max_retries,
        use_openalex=not args.no_openalex,
        match_threshold=args.match_threshold,
        logger=logger,
    )

    done_keys, month_counts = state.load_done_keys()
    already_done = len(done_keys)

    stop_requested = {"flag": False}

    def _handle_stop(signum, _frame):
        if not stop_requested["flag"]:
            stop_requested["flag"] = True
            logger.warning("Signal %s received — flushing CSV, cache, and checkpoint...", signum)
            print("\n[Stopping gracefully] Finishing current paper and saving state...")
        else:
            logger.error("Second interrupt received — forcing exit.")
            sys.exit(130)

    signal.signal(signal.SIGINT, _handle_stop)
    signal.signal(signal.SIGTERM, _handle_stop)

    print(
        f"Pass 2 Reference Validator\n"
        f"  Input:          {input_path}\n"
        f"  Output CSV:     {args.output}\n"
        f"  Cache DB:       {args.cache_db} ({cache.count():,} cached lookups)\n"
        f"  Already done:   {already_done:,} papers\n"
        f"  Mode:           {args.mode} (DOIs<={args.max_dois}, arXiv<={args.max_arxiv}, entries<={args.max_entries})\n"
        f"  Polite email:   {args.email or '(none set — consider passing --email for faster CrossRef polite pool)'}\n"
        f"  Sampling:       rate={args.sample_rate}, per_month_cap={args.per_month_cap or 'unlimited'}"
    )

    pbar = None
    if tqdm is not None:
        pbar = tqdm(
            desc="Validating papers",
            unit="paper",
            dynamic_ncols=True,
            initial=0,
            total=args.max_papers if args.max_papers > 0 else None,
        )

    session_papers = 0
    scanned = 0
    skipped_done = 0
    skipped_filter = 0
    running_valid = 0
    running_checked = 0
    t0 = time.monotonic()

    try:
        for rec in iter_jsonl_gz(input_path, logger):
            if stop_requested["flag"]:
                break
            if args.max_papers > 0 and session_papers >= args.max_papers:
                break

            scanned += 1
            arch = rec.get("archive_name", "")
            memb = rec.get("member_name", "")
            if (arch, memb) in done_keys:
                skipped_done += 1
                continue

            if not should_include_record(rec, args, month_counts):
                skipped_filter += 1
                continue

            try:
                row, audit_items = validate_paper_record(
                    rec=rec,
                    api=api,
                    cache=cache,
                    mode=args.mode,
                    max_dois=args.max_dois,
                    max_arxiv=args.max_arxiv,
                    max_entries=args.max_entries,
                    arxiv_batch_size=args.arxiv_batch_size,
                    keep_audit=bool(args.audit_output),
                )
            except Exception as exc:
                logger.exception("Unexpected error validating %s/%s: %s", arch, memb, exc)
                row = {col: "" for col in CSV_COLUMNS}
                row.update(
                    {
                        "archive_name": arch,
                        "member_name": memb,
                        "arxiv_id": rec.get("arxiv_id", ""),
                        "pub_date": rec.get("pub_date", ""),
                        "pub_year": rec.get("pub_year", ""),
                        "arxiv_category": rec.get("arxiv_category", ""),
                        "arxiv_primary_group": rec.get("arxiv_primary_group", ""),
                        "ref_count": rec.get("ref_count", 0),
                        "status": "error",
                        "error_message": f"{type(exc).__name__}: {str(exc)[:200]}",
                    }
                )
                audit_items = []

            state.record_paper(row, audit_items if args.audit_output else None)
            done_keys.add((arch, memb))
            if row["pub_date"]:
                month_counts[row["pub_date"]] += 1

            session_papers += 1
            running_checked += int(row["total_checked"] or 0)
            running_valid += int(row["total_valid"] or 0)

            if session_papers % 25 == 0:
                cache.flush()

            val_pct = (100.0 * running_valid / running_checked) if running_checked else 0.0
            total_lookups = cache.hits + cache.misses
            hit_pct = (100.0 * cache.hits / total_lookups) if total_lookups else 0.0

            if pbar is not None:
                pbar.update(1)
                if session_papers % 5 == 0 or session_papers == 1:
                    pbar.set_postfix_str(
                        f"valid={val_pct:.1f}% | cache={hit_pct:.0f}% | "
                        f"CR={api.api_calls['crossref']} OA={api.api_calls['openalex']} "
                        f"arX={api.api_calls['arxiv']} | RAM={get_memory_usage_mb():.0f}MB"
                    )
            elif session_papers % 25 == 0:
                elapsed = max(1e-3, time.monotonic() - t0)
                print(
                    f"[Progress] papers={session_papers:,} ({session_papers/elapsed:.2f}/s) | "
                    f"valid={val_pct:.1f}% | cache_hit={hit_pct:.1f}% | "
                    f"API calls={api.api_calls}"
                )

    finally:
        if pbar is not None:
            pbar.close()
        state.close()
        cache.close()
        api.close()

    elapsed = max(1e-3, time.monotonic() - t0)
    val_pct = (100.0 * running_valid / running_checked) if running_checked else 0.0
    total_lookups = cache.hits + cache.misses
    hit_pct = (100.0 * cache.hits / total_lookups) if total_lookups else 0.0

    summary = (
        f"\nPass 2 Validation {'Paused' if stop_requested['flag'] else 'Finished'} in {elapsed:.1f}s\n"
        f"  Scanned records:    {scanned:,} (skipped done={skipped_done:,}, filtered={skipped_filter:,})\n"
        f"  Validated session:  {session_papers:,} papers (total done: {already_done + session_papers:,})\n"
        f"  Refs checked:       {running_checked:,} (valid={running_valid:,}, rate={val_pct:.2f}%)\n"
        f"  Cache hit rate:     {hit_pct:.1f}% (hits={cache.hits:,}, misses={cache.misses:,})\n"
        f"  API calls:          CrossRef={api.api_calls['crossref']:,}, "
        f"OpenAlex={api.api_calls['openalex']:,}, arXiv={api.api_calls['arxiv']:,}\n"
        f"  API errors:         CrossRef={api.api_errors['crossref']:,}, "
        f"OpenAlex={api.api_errors['openalex']:,}, arXiv={api.api_errors['arxiv']:,}\n"
        f"  Output CSV:         {args.output}"
    )
    logger.info(summary)
    print(summary)


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Pass 2: Read Pass 1 arxiv_references.jsonl.gz and validate DOIs, "
            "arXiv IDs, and bibliographic entries against CrossRef, arXiv, and "
            "OpenAlex APIs with persistent SQLite caching and stop/resume."
        ),
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    io_g = parser.add_argument_group("Input / Output & Checkpointing")
    io_g.add_argument(
        "--input",
        type=str,
        default="arxiv_references.jsonl.gz",
        help="Path to Pass 1 extracted references .jsonl.gz file.",
    )
    io_g.add_argument(
        "--output",
        type=str,
        default="ref_validation.csv",
        help="Output paper-level validation CSV (append-safe across resumes).",
    )
    io_g.add_argument(
        "--audit-output",
        type=str,
        default="",
        help="Optional crash-safe .jsonl.gz saving individual ref check outcomes for manual audit.",
    )
    io_g.add_argument(
        "--state-db",
        type=str,
        default="ref_validation_state.sqlite",
        help="SQLite checkpoint DB tracking already-validated papers.",
    )
    io_g.add_argument(
        "--cache-db",
        type=str,
        default="ref_api_cache.sqlite",
        help="Persistent SQLite cache shared across runs so repeated citations are queried once.",
    )
    io_g.add_argument(
        "--log-file",
        type=str,
        default="ref_validator.log",
        help="Rotating log file path.",
    )

    val_g = parser.add_argument_group("Validation Scope & Matching")
    val_g.add_argument(
        "--mode",
        choices=["all", "identifiers-only", "entries-only"],
        default="all",
        help=(
            "'identifiers-only' checks DOIs + arXiv IDs (fastest, highest precision); "
            "'all' also verifies sampled free-text bibliography entries."
        ),
    )
    val_g.add_argument(
        "--max-dois",
        type=int,
        default=10,
        help="Max DOIs checked per paper (0 = unlimited).",
    )
    val_g.add_argument(
        "--max-arxiv",
        type=int,
        default=10,
        help="Max arXiv references checked per paper (0 = unlimited).",
    )
    val_g.add_argument(
        "--max-entries",
        type=int,
        default=5,
        help="Max free-text bibliographic entries checked per paper (0 = skip text entries; -1 = all).",
    )
    val_g.add_argument(
        "--match-threshold",
        type=float,
        default=0.62,
        help="Minimum bibliographic similarity score [0-1] to mark a free-text reference verified.",
    )
    val_g.add_argument(
        "--no-openalex",
        action="store_true",
        help="Disable OpenAlex fallback (use CrossRef and arXiv APIs only).",
    )

    samp_g = parser.add_argument_group("Sampling & Filtering (Useful for Rapid Pilot Curves)")
    samp_g.add_argument(
        "--per-month-cap",
        type=int,
        default=0,
        help="Validate at most N papers per YYYY-MM month (0 = no cap; e.g., 200 for fast stratified curve).",
    )
    samp_g.add_argument(
        "--sample-rate",
        type=float,
        default=1.0,
        help="Deterministic random fraction of papers to validate (0.0-1.0).",
    )
    samp_g.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Seed for deterministic hash sampling.",
    )
    samp_g.add_argument(
        "--max-papers",
        type=int,
        default=0,
        help="Stop after validating N papers in this session (0 = unlimited).",
    )
    samp_g.add_argument(
        "--year-min",
        type=int,
        default=None,
        help="Only validate papers with pub_year >= YEAR_MIN.",
    )
    samp_g.add_argument(
        "--year-max",
        type=int,
        default=None,
        help="Only validate papers with pub_year <= YEAR_MAX.",
    )
    samp_g.add_argument(
        "--category-group",
        type=str,
        default="",
        help="Comma-separated primary discipline groups to include (e.g. 'cs,physics,math').",
    )

    api_g = parser.add_argument_group("API Rate Limits & Networking")
    api_g.add_argument(
        "--email",
        type=str,
        default="",
        help="Contact email for CrossRef/OpenAlex polite pool (strongly recommended).",
    )
    api_g.add_argument(
        "--crossref-delay",
        type=float,
        default=0.12,
        help="Minimum seconds between CrossRef API requests.",
    )
    api_g.add_argument(
        "--openalex-delay",
        type=float,
        default=0.12,
        help="Minimum seconds between OpenAlex API requests.",
    )
    api_g.add_argument(
        "--arxiv-delay",
        type=float,
        default=3.1,
        help="Minimum seconds between batched arXiv API requests (arXiv guideline: >=3s).",
    )
    api_g.add_argument(
        "--arxiv-batch-size",
        type=int,
        default=20,
        help="Number of arXiv IDs checked per single arXiv API request.",
    )
    api_g.add_argument(
        "--timeout",
        type=float,
        default=15.0,
        help="HTTP request timeout in seconds.",
    )
    api_g.add_argument(
        "--max-retries",
        type=int,
        default=3,
        help="Retry attempts on HTTP 429 / 5xx / transient network failures.",
    )
    api_g.add_argument(
        "--verbose",
        action="store_true",
        help="Enable verbose debug logging.",
    )

    return parser


def main() -> None:
    parser = build_arg_parser()
    args = parser.parse_args()

    if args.category_group:
        args.category_group = {
            c.strip().lower() for c in args.category_group.split(",") if c.strip()
        }
    else:
        args.category_group = set()

    logger = setup_logging(args.log_file, verbose=args.verbose)
    try:
        run_validation(args, logger)
    except KeyboardInterrupt:
        print("\nInterrupted.")
        sys.exit(130)
    except Exception as exc:
        logger.exception("Fatal error in Pass 2 validator: %s", exc)
        print(f"\nFatal error: {exc}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()

