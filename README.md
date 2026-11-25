# arXiv Large-Scale PDF Linguistic & Reference Analyzer

A memory-bounded Python pipeline for analyzing hundreds of thousands of arXiv research articles stored inside multi-terabyte `.tar.gz` archives **without extracting PDFs to disk**.

Built to run reliably over long multi-day sessions on modest hardware (such as a quad-core Intel i5-6500T with 8 GB RAM), with full SQLite checkpointing for stop/resume, rotating logs, live terminal progress, optional CrossRef/arXiv reference validation, and before/after trend plotting.

---

## Features

- **Zero Disk Extraction:** Streams `.pdf` files directly from `.tar.gz` / `.tgz` archives using `tarfile` streaming mode (`r|gz`) straight into `PyMuPDF` in memory. Includes an isolated temp-directory safeguard that cleans itself up automatically on exit.
- **Tuned for 4 Cores & 8 GB RAM:**
  - Defaults to `--workers 3`, leaving 1 core free for gzip stream decompression, CSV flushing, and OS I/O.
  - Strict backpressure queue (`2 × workers` PDFs in memory at once).
  - Automatic worker recycling every 200 PDFs (`maxtasksperchild=200`) to prevent C-level memory leaks over 200,000+ documents.
  - Configurable size cap (`--max-pdf-mb`), page cap (`--max-pages`), and per-PDF timeout (`--pdf-timeout`).
- **Crash-Safe Stop & Resume:** Tracks completed archives and individual PDFs in a WAL-mode SQLite database (`arxiv_state.sqlite`) while appending to the output CSV in batches. Press `Ctrl+C` anytime to flush in-flight work cleanly, and re-run the same command to pick up right where you left off.
- **Linguistic & Structural Metrics:** Separates main prose from the bibliography before computing:
  - **FRE** (Flesch Reading Ease)
  - **FKGL** (Flesch-Kincaid Grade Level)
  - **MATTR-100** (Moving-Average Type-Token Ratio, 100-word sliding window in `O(N)` time)
  - **Shannon Character Entropy** (bits/character)
  - **Passive Voice Ratio** (fast regex + irregular participle lexicon detector)
  - **Average Sentence Length**, word/sentence/page counts, and average syllables per word
  - **Author Count** (PDF metadata first, falling back to page-1 header heuristics)
- **Reference Extraction & Validation:**
  - **Offline (always on):** Counts extracted references, DOIs, and arXiv citations, and flags structural anomalies (impossible future publication years, malformed DOI strings).
  - **Live API (`--validate-refs`):** Deterministically samples citations per paper and verifies them against CrossRef (by DOI or bibliographic query) and the arXiv API, backed by a persistent SQLite cache (`ref_validation_cache.sqlite`) so citations shared across papers are only queried once.
- **Before/After Trend Plotting:** Reads the output CSV in memory-safe chunks and outputs a combined dashboard PNG, individual high-res metric PNGs with 95% confidence bands, a time-series CSV, and a before/after summary table for any cutoff date.

---

## Requirements

- Python 3.9+
- Dependencies:

```bash
pip install pymupdf tqdm requests pandas matplotlib psutil
```

> **Notes:**
> - `pymupdf` (`fitz`) is required for PDF parsing.
> - `tqdm` and `psutil` are optional but strongly recommended for the live progress bar and RAM monitor.
> - `requests` is only required when using `--validate-refs`.
> - `pandas` and `matplotlib` are only required when using `--plot` or `--plot-only`.

---

## Quick Start

### 1. Fast Offline Run (Recommended First Pass)

Process all `.tar.gz` / `.tgz` files recursively inside a directory:

```bash
python arxiv_analyzer.py \
  --input-dir /mnt/arxiv_tars \
  --output arxiv_metrics.csv
```

Stop at any time with `Ctrl+C`. Wait a few seconds for in-flight PDFs to flush, then run the exact same command later to resume. Finished archives are skipped instantly without being re-decompressed.

### 2. Run With Live Reference Validation

Check a sample of references per paper against CrossRef and the arXiv API (provide `--mailto` to use CrossRef's faster polite pool):

```bash
python arxiv_analyzer.py \
  --input-dir /mnt/arxiv_tars \
  --output arxiv_metrics.csv \
  --validate-refs \
  --ref-sample 5 \
  --mailto you@university.edu
```

> **Throughput tip:** Because live API checks are network-bound, validating 5 references across 200,000 papers will take much longer than a pure local pass—even with the SQLite cache deduplicating popular papers. Test on a small folder of 1–2 archives first to gauge your preferred `--ref-sample` size.

### 3. Generate Before / After Trend Plots

Run plotting at any time, even on a partially completed CSV while collecting data, without re-reading any archives:

```bash
python arxiv_analyzer.py \
  --plot-only \
  --output arxiv_metrics.csv \
  --plots-dir ./plots \
  --cutoff-date 2022-11 \
  --plot-freq M
```

This creates inside `./plots/`:
- `trends_dashboard.png` — multi-panel overview of all metrics over time with 95% confidence intervals and a vertical cutoff line
- `trend_<metric>.png` — individual high-resolution plots for each metric
- `trends_timeseries.csv` — aggregated monthly/quarterly/yearly means and paper counts
- `before_after_summary.csv` — before vs. after sample sizes, means, standard deviations, and deltas

---

## CLI Reference

| Flag | Default | Description |
| :--- | :--- | :--- |
| `--input-dir` | *(required unless `--plot-only`)* | Root directory searched recursively for `.tar.gz` and `.tgz` files |
| `--output` | `arxiv_metrics.csv` | Output CSV path (safely appended across sessions) |
| `--state-db` | `arxiv_state.sqlite` | SQLite checkpoint DB for completed archives and processed PDFs |
| `--log-file` | `arxiv_analyzer.log` | Rotating log file (20 MB per file, up to 5 backups) |
| `--temp-dir` | OS default | Optional base directory for isolated session temp safeguard |
| `--workers` | `3` | Number of worker processes (3 is optimal for a 4-core CPU) |
| `--max-pdf-mb` | `50.0` | Skip individual PDFs larger than this threshold (MB) |
| `--max-pages` | `150` | Max pages parsed per PDF (always reads final 5 pages for references if truncated) |
| `--pdf-timeout` | `45.0` | Timeout in seconds per PDF before marking as `timeout` and continuing |
| `--validate-refs` | `False` | Enable live CrossRef / arXiv API reference verification |
| `--ref-cache-db` | `ref_validation_cache.sqlite` | Persistent SQLite cache for verified/unverified citations |
| `--ref-sample` | `5` | Max evenly-spaced references checked per paper |
| `--ref-timeout` | `6.0` | HTTP request timeout per reference lookup (seconds) |
| `--ref-delay` | `0.08` | Minimum delay between API requests per worker (seconds) |
| `--mailto` | `""` | Contact email sent to CrossRef polite pool |
| `--plot` | `False` | Automatically generate trend plots when processing finishes |
| `--plot-only` | `False` | Skip PDF processing and plot directly from `--output` |
| `--plots-dir` | `plots` | Directory for generated PNGs and summary CSVs |
| `--cutoff-date` | `None` | Optional `YYYY-MM` cutoff date for before/after comparison |
| `--plot-freq` | `M` | Aggregation frequency: `M` (monthly), `Q` (quarterly), `Y` (yearly) |
| `-v`, `--verbose` | `False` | Also print `INFO` logs to stderr |

---

## Output CSV Schema

Every PDF encountered produces one row in `--output` (including skipped, encrypted, scanned, or errored files, distinguished via the `status` column):

| Column | Description |
| :--- | :--- |
| `archive_name`, `member_name` | Source `.tar.gz` filename and internal member path |
| `arxiv_id` | Canonical arXiv identifier parsed from the filename |
| `pub_date`, `pub_year`, `pub_month` | Derived `YYYY-MM` date, integer year, and integer month |
| `status` | `ok`, `no_text`, `encrypted`, `empty`, `skipped_too_large`, `timeout`, `stream_error`, or `error` |
| `page_count`, `char_count`, `word_count`, `sentence_count` | Basic document volume statistics (computed on body prose) |
| `avg_sentence_length`, `avg_syllables_per_word` | Core readability inputs |
| `fre`, `fkgl` | Flesch Reading Ease and Flesch-Kincaid Grade Level |
| `mattr_100` | Moving-Average Type-Token Ratio (100-word window) |
| `shannon_char_entropy` | Shannon character entropy in bits per character |
| `passive_sentence_count`, `passive_voice_ratio` | Count and fraction of body sentences with passive constructions |
| `author_count`, `author_source` | Estimated author count and source (`pdf_metadata`, `page1_heuristic`, or `unknown`) |
| `ref_count`, `doi_count`, `arxiv_ref_count` | Parsed bibliography entries, unique DOIs, and unique arXiv citations |
| `ref_future_year_flags`, `ref_malformed_doi_flags` | Offline structural red-flag counts |
| `refs_checked`, `refs_verified`, `refs_unverified`, `ref_verification_rate` | Live API validation results (populated when `--validate-refs` is enabled) |
| `error_message` | Diagnostic message if `status != "ok"` |

---

## Methodological Notes & Limitations

- **Passive Voice Detection:** Uses a regex + irregular past-participle lexicon with common participial adjective exclusions rather than full neural dependency parsing (e.g., spaCy). This keeps RAM usage tiny and speed ~50× faster across 200k+ papers, with consistent bias across before/after cohorts.
- **Author Counts:** PDF metadata is sometimes blank or unformatted on older LaTeX submissions; the page-1 heuristic provides a fallback, and `author_source` lets you filter or validate against the official arXiv metadata dataset if desired.
- **Reference Validation:** An `unverified` citation does not automatically prove hallucination—books, technical reports, non-DOI conference papers, OCR glitches, or unusual citation formats can fail automated CrossRef lookups. Treat `ref_verification_rate` alongside `ref_future_year_flags` and `ref_malformed_doi_flags` as comparative signals across time periods, and inspect flagged outliers manually.
- **Duplicate Rows on Hard Power Loss:** Normal `Ctrl+C` stops flush SQLite and CSV together. If the machine loses power mid-batch (at most 50 rows), a resumed run could re-append up to 50 rows; deduplicating on `(archive_name, member_name)` before final statistical modeling is good practice.

---

## Running Overnight / Long Sessions

For multi-day runs on Linux/macOS, run inside `tmux` or `screen` so terminal disconnects do not interrupt the job:

```bash
tmux new -s arxiv
python arxiv_analyzer.py --input-dir /mnt/arxiv_tars --output arxiv_metrics.csv
# Detach with Ctrl+B, then D
# Reattach anytime with: tmux attach -t arxiv
```

Monitor detailed progress or errors in another terminal with:

```bash
tail -f arxiv_analyzer.log
```
