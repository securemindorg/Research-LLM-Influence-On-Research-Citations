# arXiv Large-Scale PDF Stylometric & Reference Analysis Pipeline

A three-stage, memory-bounded Python pipeline for analyzing hundreds of thousands of arXiv research articles stored inside multi-terabyte `.tar.gz` archives—**without extracting PDFs to disk**.

Built to run reliably over multi-day sessions on modest hardware (such as a quad-core Intel i5-6500T with 8 GB RAM), with WAL-mode SQLite checkpointing for stop/resume, rotating logs, live terminal progress, decoupled API reference verification, and a full before/after longitudinal plotting suite.

---

## Architecture: Three Decoupled Passes

Separating local PDF parsing from network-bound reference lookups ensures Pass 1 runs at full CPU/disk speed (never stalling on CrossRef/arXiv rate limits), while Pass 2 can be sampled, resumed, or re-run without ever decompressing the `.tar.gz` archives again.

```text
  [.tar.gz archives]
          │
          ▼  Pass 1: arxiv_analyzer.py (100% offline, CPU-bound)
   ├── arxiv_metrics.csv        (75+ stylometric, structural, confounder & offline ref columns)
   └── arxiv_references.jsonl.gz (crash-safe compressed dump of DOIs, arXiv IDs & ref entries)
          │
          ▼  Pass 2: validate_references.py (network-bound, cached, stop/resume)
   ├── ref_validation.csv       (paper-level validity/error rates & match scores)
   └── ref_audit.jsonl.gz       (optional per-citation outcomes for manual audit)
          │
          ▼  Pass 3: plot_trends.py (joins Pass 1 + Pass 2, or runs on either alone)
   └── plots/                   (15 longitudinal figures + before_after_stats.csv + summary)
```

---

## Requirements

- Python 3.9+
- Install dependencies:

```bash
# Pass 1 (PDF streaming & stylometry)
pip install pymupdf tqdm psutil

# Pass 2 (Live CrossRef / OpenAlex / arXiv validation)
pip install requests tqdm psutil

# Pass 3 (Graphs & statistical tests)
pip install pandas numpy matplotlib scipy
```

Or install everything at once:

```bash
pip install pymupdf tqdm psutil requests pandas numpy matplotlib scipy
```

---

## Quick Start

### Pass 1 — Stream PDFs, Compute All Metrics & Dump References (Offline)

```bash
python arxiv_analyzer.py \
  --input-dir /mnt/arxiv_tars \
  --output arxiv_metrics.csv \
  --refs-output arxiv_references.jsonl.gz \
  --workers 3
```

- Press `Ctrl+C` anytime to flush in-flight batches cleanly; re-run the exact same command later to resume. Finished archives are skipped immediately without re-decompressing.
- Each flush appends an independent gzip frame to `arxiv_references.jsonl.gz`, so a sudden power loss cannot corrupt previously flushed papers.

### Pass 2 — Validate Extracted References (Live APIs + Persistent SQLite Cache)

For a **fast stratified pilot curve** (e.g., up to 200 papers per month across the full timeline):

```bash
python validate_references.py \
  --input arxiv_references.jsonl.gz \
  --output ref_validation.csv \
  --audit-output ref_audit.jsonl.gz \
  --per-month-cap 200 \
  --max-dois 10 \
  --max-arxiv 10 \
  --max-entries 5 \
  --email you@university.edu
```

For a **full validation run** (remove `--per-month-cap`; adjust `--mode` to `identifiers-only` for much faster DOI + arXiv checks without free-text CrossRef queries):

```bash
python validate_references.py \
  --input arxiv_references.jsonl.gz \
  --output ref_validation.csv \
  --email you@university.edu
```

- Providing `--email` places requests in the CrossRef and OpenAlex polite pools (faster and more reliable).
- Verified citations are cached permanently in `ref_api_cache.sqlite` so popular papers cited thousands of times are only queried once.
- Stop with `Ctrl+C` and re-run the exact same command to resume; deterministic hash sampling guarantees identical paper selection across restarts.

### Pass 3 — Generate All Graphs & Before/After Statistics

Run with both Pass 1 and Pass 2 outputs (or supply only `--metrics-csv` or only `--refs-csv` while the other pass is still running):

```bash
python plot_trends.py \
  --metrics-csv arxiv_metrics.csv \
  --refs-csv ref_validation.csv \
  --cutoff-date 2022-11 \
  --output-dir plots \
  --freq M \
  --rolling 3
```

Optional quality/discipline filtering:

```bash
python plot_trends.py \
  --metrics-csv arxiv_metrics.csv \
  --refs-csv ref_validation.csv \
  --cutoff-date 2022-11 \
  --start-date 2018-01 \
  --min-alpha-ratio 0.55 \
  --groups cs,stat,math,physics \
  --output-dir plots_clean
```

---

## What Pass 1 Computes (`arxiv_metrics.csv`)

Every PDF produces one row (including skipped/encrypted/errored files, distinguished via `status`):

- **Identification & Confounders:** `archive_name`, `member_name`, `arxiv_id`, `arxiv_version`, `arxiv_category`, `arxiv_primary_group` (parsed from page-1 arXiv stamp), `pub_date`, `exact_pub_date`, `pub_year`, `pub_month`, `pdf_creator`, `pdf_producer`.
- **Volume & Extraction-Noise Controls:** `page_count`, `char_count`, `word_count`, `sentence_count`, `alpha_char_ratio`, `digit_char_ratio`, `symbol_char_ratio`, `non_ascii_ratio`, `single_char_token_ratio` (lets you filter math-heavy or poor-OCR PDFs cleanly).
- **Readability & Sentence Burstiness:** `fre`, `fkgl`, `cli` (Coleman-Liau), `fog` (Gunning Fog), `avg_sentence_length`, `sentence_length_std`, `sentence_length_median`, `sentence_length_iqr`, `avg_word_length`, `avg_syllables_per_word`, `complex_word_ratio`.
- **Lexical Diversity & Entropy:** `mattr_100` (100-word sliding window in `O(N)` time), `yules_k`, `hapax_ratio`, `shannon_char_entropy`, `shannon_word_entropy`.
- **Voice, Stance, Punctuation & Marker Lexicons (per 1,000 words):** `passive_sentence_count`, `passive_voice_ratio`, `first_person_plural_per_1k`, `first_person_singular_per_1k`, `hedging_per_1k`, `booster_per_1k`, `dash_per_1k`, `semicolon_per_1k`, `colon_per_1k`, `paren_per_1k`, `llm_excess_words_per_1k` (transparent fixed lexicon), `control_words_per_1k` (neutral academic baseline).
- **Abstract-Only Parallel Control:** `abstract_word_count`, `abstract_sentence_count`, `abstract_avg_sentence_length`, `abstract_fre`, `abstract_fkgl`, `abstract_mattr_100`, `abstract_passive_ratio` (clean prose unaffected by equations/tables).
- **Authorship & Structure:** `author_count`, `author_source`, `has_code_url`, `code_url_count`, `has_appendix`, `has_acknowledgments`, `has_limitations_or_ethics`, `figure_mention_count`, `table_mention_count`.
- **Offline Reference Stats & Recency:** `ref_count`, `doi_count`, `arxiv_ref_count`, `ref_future_year_flags`, `ref_malformed_doi_flags`, `ref_min_year`, `ref_median_year`, `ref_max_year`, `ref_mean_age`.

---

## What Pass 2 Validates (`ref_validation.csv`)

Reads `arxiv_references.jsonl.gz` line-by-line and checks three citation layers against live registries:

1. **DOIs** → CrossRef REST API with automatic OpenAlex fallback (catches DataCite, Zenodo, and arXiv DOIs absent from CrossRef).
2. **arXiv IDs** → Batched arXiv Atom API queries (respecting the ≥3 s arXiv rate limit).
3. **Free-text reference entries** → CrossRef bibliographic search scored deterministically on `[0, 1]` using title token overlap (`0.65`), author surname match (`0.20`), and publication year match (`0.15`, ±1 year tolerance).

Outputs per-paper checked/valid/invalid/error counts for each layer plus `ref_validity_rate`, `ref_invalid_rate`, `has_any_invalid_ref`, `has_multiple_invalid_refs`, and `mean_match_score`.

---

## What Pass 3 Generates (`plots/`)

### Figures (up to 15 publication-ready PNG/PDF/SVG files)
1. `01_overview_dashboard` — 6-panel executive summary (FRE, MATTR-100, char entropy, passive voice, excess words, live ref validity)
2. `02_readability_trends` — FRE, FKGL, Coleman-Liau, Gunning Fog
3. `03_lexical_diversity_and_entropy` — MATTR-100, Yule's K, hapax ratio, char/word entropy, word length
4. `04_sentence_structure_and_burstiness` — sentence length mean, std dev, IQR, complex word ratio
5. `05_voice_stance_and_punctuation` — passive voice, 1st-person plural, hedging, boosters, dashes, semicolons
6. `06_llm_excess_vs_control_words` — raw + indexed (`Pre = 100`) comparison of LLM excess vs neutral academic control words
7. `07_body_vs_abstract_control` — body vs abstract-only parallel overlay
8. `08_authorship_and_structure` — author count, pages, word count, code URL share, appendix, limitations/ethics
9. `09_confounders_and_discipline_mix` — monthly volume, stacked discipline share, alphabetic ratio, single-char token ratio
10. `10_discipline_stratified_trends` — top-5 arXiv disciplines across 4 core metrics
11. `11_offline_references_and_recency` — reference count, mean age, DOI/arXiv counts, future-year & malformed-DOI flags
12. `12_live_reference_validation` — overall, DOI, arXiv, and free-text verification rates + paper-level invalid-ref prevalence
13. `13_ref_validation_by_discipline` — Pass 2 validity rates stratified by discipline
14. `14_stylometry_vs_ref_validity` — binned excess-word density vs ref invalidity (pre/post) + Spearman correlation chart
15. `15_before_after_effect_sizes` — Cohen's d forest plot with 95% confidence intervals across all key metrics

### Data & Statistical Summaries
- `before_after_stats.csv` — pre/post means, medians, SDs, % change, Welch's t-test, Mann-Whitney U, Cohen's d (±95% CI), and pre/post annual trend slopes
- `timeseries_aggregated.csv` — monthly/quarterly counts, means, SDs, 95% CIs, medians, and IQRs for every metric
- `summary_report.txt` — terminal-printed overview of sample sizes and top 15 shifts by |d|

---

## Key CLI Reference

### `arxiv_analyzer.py` (Pass 1)
| Flag | Default | Description |
| :--- | :--- | :--- |
| `--input-dir` | *(required)* | Root directory searched recursively for `.tar.gz` / `.tgz` files |
| `--output` | `arxiv_metrics.csv` | Output CSV path (append-safe across resumes) |
| `--refs-output` | `arxiv_references.jsonl.gz` | Compressed JSONL dump of extracted references for Pass 2 |
| `--dump-max-refs` | `25` | Max evenly spaced reference strings saved per paper (`0` = all) |
| `--no-refs-dump` | `False` | Skip writing `.jsonl.gz` (still records offline ref metrics in CSV) |
| `--state-db` | `arxiv_state.sqlite` | SQLite checkpoint DB for completed archives and PDFs |
| `--workers` | `3` | Worker processes (3 leaves 1 core for gzip I/O on a quad-core CPU) |
| `--max-pdf-mb` | `50.0` | Skip individual PDFs larger than this threshold (MB) |
| `--max-pages` | `150` | Max pages parsed per PDF (always reads final 5 pages for references) |
| `--pdf-timeout` | `45.0` | Per-PDF timeout in seconds |

### `validate_references.py` (Pass 2)
| Flag | Default | Description |
| :--- | :--- | :--- |
| `--input` | `arxiv_references.jsonl.gz` | Pass 1 reference dump `.jsonl.gz` |
| `--output` | `ref_validation.csv` | Output paper-level validation CSV |
| `--audit-output` | `""` | Optional `.jsonl.gz` storing per-reference check details for manual review |
| `--state-db` | `ref_validation_state.sqlite` | SQLite checkpoint DB for stop/resume |
| `--cache-db` | `ref_api_cache.sqlite` | Persistent citation cache shared across runs |
| `--mode` | `all` | `all`, `identifiers-only`, or `entries-only` |
| `--max-dois` / `--max-arxiv` / `--max-entries` | `10` / `10` / `5` | Per-paper verification caps (`0` = unlimited for DOIs/arXiv, skip for entries) |
| `--match-threshold` | `0.62` | Minimum bibliographic similarity score `[0–1]` for free-text match |
| `--per-month-cap` | `0` | Cap validated papers per `YYYY-MM` month (`0` = unlimited) |
| `--sample-rate` | `1.0` | Deterministic hash-sampled fraction of papers `(0.0–1.0]` |
| `--year-min` / `--year-max` / `--category-group` | `None` | Optional subset filters |
| `--email` | `""` | Contact email for CrossRef / OpenAlex polite pool |

### `plot_trends.py` (Pass 3)
| Flag | Default | Description |
| :--- | :--- | :--- |
| `--metrics-csv` | `None` | Pass 1 CSV (`arxiv_metrics.csv`) |
| `--refs-csv` | `None` | Pass 2 CSV (`ref_validation.csv`) |
| `--output-dir` | `plots` | Output directory for figures and CSV summaries |
| `--cutoff-date` | `2022-11` | Before/after cutoff month (`YYYY-MM`) |
| `--start-date` / `--end-date` | `None` | Optional date window bounds (`YYYY-MM`) |
| `--freq` | `MS` | Binning frequency: `M`/`MS` (monthly) or `Q`/`QS` (quarterly) |
| `--rolling` | `3` | Centered rolling window in bins (`1` = raw) |
| `--min-per-bin` | `10` | Minimum papers required to plot a time bin |
| `--groups` | `None` | Comma-separated arXiv groups (e.g., `cs,math,physics,stat`) |
| `--min-words` / `--min-alpha-ratio` | `300` / `0.0` | Pass 1 quality filters |
| `--format` / `--dpi` | `png` / `180` | Output format (`png`, `pdf`, `svg`) and DPI |

---

## Running Multi-Day Jobs Safely

Run long Pass 1 or Pass 2 sessions inside `tmux` or `screen` so SSH/terminal disconnects never interrupt progress:

```bash
tmux new -s arxiv
python arxiv_analyzer.py --input-dir /mnt/arxiv_tars --output arxiv_metrics.csv
# Detach: Ctrl+B then D  |  Reattach: tmux attach -t arxiv
```

Monitor logs in another pane:

```bash
tail -f arxiv_analyzer.log
```

---

## Methodological Notes

- **Confounder Control:** Overall arXiv trends can shift simply because computer science (`cs`) grows as a share of submissions. Use `09_confounders_and_discipline_mix`, `10_discipline_stratified_trends`, and `--groups` / `--min-alpha-ratio` in Pass 3 to separate genuine writing shifts from field-mix or OCR artifacts.
- **Unverified ≠ Proven Hallucination:** Books, workshops, technical reports, non-DOI venues, or PDF line-wrap artifacts can fail automated lookups. Use `ref_validity_rate` comparatively across time cohorts alongside `--audit-output` to inspect flagged citations manually.
- **Hard Power Loss Deduplication:** Normal `Ctrl+C` exits flush SQLite and CSV together. On an abrupt power cut, at most one batch (≤50 rows) could be re-appended upon resume; `plot_trends.py` automatically deduplicates on `(archive_name, member_name)`.
