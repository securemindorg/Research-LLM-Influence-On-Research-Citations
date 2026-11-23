"""
Bulk-download CS papers from arXiv, with metadata, PDFs, and source archives.

Key fixes & improvements over the original draft:
1. Calls `r.source_url()` as a method instead of assigning the bound method object.
2. Sends a descriptive `User-Agent` header and pauses between `/pdf/` and `/src/`
   downloads to avoid triggering HTTP 429 / temporary blocks on file endpoints.
3. Logs download errors clearly (and records `status:error` in CSV) while still
   skipping already-recorded `entry_id`s on resume.
4. Supports optional `MAX_PER_BUCKET` so you can grab a balanced sample across
   `before` and `after` windows without downloading tens of thousands of files.

Usage:
    python download_cs_ai_arxiv_dataset.py

Requirements:
    pip install arxiv
"""

import csv
import os
import time
from datetime import datetime, timezone
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

import arxiv

# ---------------------------------------------------------------------------
# CONFIG
# ---------------------------------------------------------------------------
CATEGORIES = ["cs.AI"]

# Set your contact email in User-Agent per arXiv etiquette.
USER_AGENT = "cs-ai-citation-validity-study/1.0 (mailto:researcher@example.com)"

# Time windows. Dates are INCLUSIVE and interpreted as UTC.
BEFORE_START = datetime(2015, 1, 1, tzinfo=timezone.utc)
BEFORE_END = datetime(2020, 12, 31, 23, 59, 59, tzinfo=timezone.utc)

AFTER_START = datetime(2023, 1, 1, tzinfo=timezone.utc)
AFTER_END = datetime(2026, 12, 31, 23, 59, 59, tzinfo=timezone.utc)

# Optional cap per bucket ("before" / "after"). Set to None for unlimited.
MAX_PER_BUCKET = 500

# Toggle whether to download PDFs in addition to source archives.
DOWNLOAD_PDFS = False
DOWNLOAD_SOURCES = True

# Polite delay (seconds) after each /pdf/ or /src/ file request.
FILE_DELAY_SECONDS = 1.5
MAX_FILE_RETRIES = 3

# Sort order for the API fetch.
SORT_CRITERION = arxiv.SortCriterion.SubmittedDate

# Output paths
PDF_DIR = "pdfs"
SRC_DIR = "src"
META_CSV = "arxiv_metadata.csv"

FIELDNAMES = [
    "entry_id",
    "bucket",
    "submitted",
    "updated",
    "title",
    "authors",
    "primary_category",
    "categories",
    "doi",
    "journal_ref",
    "pdf_url",
    "source_url",
    "pdf_path",
    "source_path",
]


def build_query(categories):
    """Build an arXiv search query for `cat:` filters OR'd together."""
    return " OR ".join(f"cat:{c}" for c in categories)


def sanitize_filename(value):
    """Turn an arbitrary ID into a safe filename."""
    return "".join(ch if ch.isalnum() or ch in "._-" else "_" for ch in value)


def load_existing_state(csv_path):
    """Load already-recorded entry_ids and existing counts per bucket."""
    done_ids = set()
    bucket_counts = {"before": 0, "after": 0}

    if not os.path.exists(csv_path):
        return done_ids, bucket_counts

    with open(csv_path, "r", newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            entry_id = row.get("entry_id", "").strip()
            bucket = row.get("bucket", "").strip()
            if entry_id:
                done_ids.add(entry_id)
            if bucket in bucket_counts:
                bucket_counts[bucket] += 1

    return done_ids, bucket_counts


def in_window(dt, start, end):
    """Return True if dt falls within [start, end], treating None as open."""
    if start is not None and dt < start:
        return False
    if end is not None and dt > end:
        return False
    return True


def download_with_retry(url, dest_path, label, entry_id):
    """Download a URL to `dest_path` with retries and backoff for 429/5xx."""
    headers = {"User-Agent": USER_AGENT}
    last_error = None

    for attempt in range(1, MAX_FILE_RETRIES + 1):
        try:
            req = Request(url, headers=headers)
            with urlopen(req, timeout=60) as resp, open(dest_path, "wb") as out:
                data = resp.read()
                if not data:
                    raise ValueError("Empty response body")
                out.write(data)
            time.sleep(FILE_DELAY_SECONDS)
            return dest_path
        except HTTPError as e:
            last_error = e
            # Retry rate limits (429) and transient server errors (5xx).
            if e.code == 429 or e.code >= 500:
                backoff = FILE_DELAY_SECONDS * (2 ** attempt)
                print(
                    f"[{label} HTTP {e.code}] {entry_id} (attempt {attempt}/{MAX_FILE_RETRIES}), "
                    f"sleeping {backoff:.1f}s..."
                )
                time.sleep(backoff)
                continue
            break
        except (URLError, TimeoutError, OSError, ValueError) as e:
            last_error = e
            backoff = FILE_DELAY_SECONDS * (2 ** attempt)
            print(
                f"[{label} ERR] {entry_id}: {e} (attempt {attempt}/{MAX_FILE_RETRIES}), "
                f"sleeping {backoff:.1f}s..."
            )
            time.sleep(backoff)

    # Remove partial file if created before failure.
    if os.path.exists(dest_path):
        try:
            os.remove(dest_path)
        except OSError:
            pass

    print(f"[{label} FAILED] {entry_id} ({url}): {last_error}")
    return f"ERROR: {last_error}"


def buckets_complete(bucket_counts):
    """Return True if all active buckets have hit MAX_PER_BUCKET."""
    if MAX_PER_BUCKET is None:
        return False

    has_before = BEFORE_START is not None or BEFORE_END is not None
    has_after = AFTER_START is not None or AFTER_END is not None

    before_done = (not has_before) or (bucket_counts["before"] >= MAX_PER_BUCKET)
    after_done = (not has_after) or (bucket_counts["after"] >= MAX_PER_BUCKET)
    return before_done and after_done


def main():
    if DOWNLOAD_PDFS:
        os.makedirs(PDF_DIR, exist_ok=True)
    if DOWNLOAD_SOURCES:
        os.makedirs(SRC_DIR, exist_ok=True)

    done_ids, bucket_counts = load_existing_state(META_CSV)
    if done_ids:
        print(
            f"Resuming: {len(done_ids)} paper(s) already in {META_CSV} "
            f"(before={bucket_counts['before']}, after={bucket_counts['after']})."
        )

    query = build_query(CATEGORIES)
    print(f"Query: {query}")

    client = arxiv.Client(page_size=500, delay_seconds=3.0, num_retries=3)
    search = arxiv.Search(
        query=query,
        max_results=None,
        sort_by=SORT_CRITERION,
        sort_order=arxiv.SortOrder.Descending,
    )

    file_exists_nonempty = os.path.exists(META_CSV) and os.path.getsize(META_CSV) > 0
    processed_this_run = 0

    with open(META_CSV, "a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=FIELDNAMES)
        if not file_exists_nonempty:
            writer.writeheader()
            f.flush()

        for r in client.results(search):
            if buckets_complete(bucket_counts):
                print("Reached MAX_PER_BUCKET for target windows. Stopping.")
                break

            if r.entry_id in done_ids:
                continue

            submitted = r.published
            if submitted is None:
                continue

            in_before = (BEFORE_START is not None or BEFORE_END is not None) and in_window(
                submitted, BEFORE_START, BEFORE_END
            )
            in_after = (AFTER_START is not None or AFTER_END is not None) and in_window(
                submitted, AFTER_START, AFTER_END
            )
            if not (in_before or in_after):
                continue

            bucket = "before" if in_before else "after"
            if MAX_PER_BUCKET is not None and bucket_counts[bucket] >= MAX_PER_BUCKET:
                continue

            safe_id = sanitize_filename(r.entry_id)
            pdf_url = r.pdf_url or ""
            # IMPORTANT: r.source_url is a method on arxiv.Result, so call it.
            src_url = r.source_url() or ""

            if DOWNLOAD_PDFS and pdf_url:
                pdf_target = os.path.join(PDF_DIR, f"{safe_id}.pdf")
                pdf_path = download_with_retry(pdf_url, pdf_target, "PDF", r.entry_id)
            else:
                pdf_path = ""

            if DOWNLOAD_SOURCES and src_url:
                # Note: arXiv source archives are typically gzipped/tarballs, though
                # occasional single-file submissions can vary in internal compression.
                src_target = os.path.join(SRC_DIR, f"{safe_id}.tar.gz")
                src_path = download_with_retry(src_url, src_target, "SRC", r.entry_id)
            else:
                src_path = ""

            row = {
                "entry_id": r.entry_id,
                "bucket": bucket,
                "submitted": r.published.isoformat() if r.published else "",
                "updated": r.updated.isoformat() if r.updated else "",
                "title": " ".join((r.title or "").split()),
                "authors": "; ".join(a.name for a in r.authors),
                "primary_category": r.primary_category or "",
                "categories": "; ".join(r.categories or []),
                "doi": r.doi or "",
                "journal_ref": r.journal_ref or "",
                "pdf_url": pdf_url,
                "source_url": src_url,
                "pdf_path": pdf_path,
                "source_path": src_path,
            }
            writer.writerow(row)
            done_ids.add(r.entry_id)
            bucket_counts[bucket] += 1
            processed_this_run += 1

            if processed_this_run % 50 == 0:
                print(
                    f"Processed {processed_this_run} new paper(s) this run "
                    f"(total before={bucket_counts['before']}, after={bucket_counts['after']})..."
                )
                f.flush()

    print(
        f"Done. {processed_this_run} new paper(s) processed this run "
        f"(total before={bucket_counts['before']}, after={bucket_counts['after']}). "
        f"Metadata saved to {META_CSV}."
    )


if __name__ == "__main__":
    main()
