#!/usr/bin/env python3
"""
Pass 3: Trend Plotting, Confounder Diagnostics & Before/After Statistics
========================================================================

Generates publication-ready longitudinal graphs and statistical summaries
from the outputs of Pass 1 (arxiv_metrics.csv) and Pass 2 (ref_validation.csv).

Works seamlessly when:
  - Both Pass 1 CSV and Pass 2 CSV are supplied (full 15-figure suite + joined correlations),
  - Only Pass 1 CSV is supplied (generates all stylometric/confounder/offline-ref figures), or
  - Only Pass 2 CSV is supplied (generates all reference-validation figures).

Memory-friendly for 200,000+ papers on 8 GB RAM machines: reads CSVs with
explicit compact dtypes and optional quality/date filters.

Dependencies:
    pip install pandas numpy matplotlib scipy

Usage:
    python plot_trends.py \\
        --metrics-csv arxiv_metrics.csv \\
        --refs-csv ref_validation.csv \\
        --cutoff-date 2022-11 \\
        --output-dir plots
"""

import argparse
import logging
import math
import os
import sys
import warnings
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

# Headless-safe Matplotlib backend
import matplotlib
matplotlib.use("Agg")
import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker

try:
    from scipy import stats as scipy_stats
except ImportError:
    scipy_stats = None

warnings.filterwarnings("ignore", category=RuntimeWarning)


# ---------------------------------------------------------------------------
# Plot Styling & Metric Registry
# ---------------------------------------------------------------------------

PALETTE = {
    "primary": "#1f5a99",
    "secondary": "#d95f02",
    "tertiary": "#1b9e77",
    "quaternary": "#7570b3",
    "accent": "#e7298a",
    "muted": "#666666",
    "cutoff": "#c0392b",
    "pre": "#2c7bb6",
    "post": "#d7191c",
}

DISCIPLINE_COLORS = [
    "#1f77b4", "#d95f02", "#1b9e77", "#7570b3",
    "#e7298a", "#66a61e", "#e6ab02", "#a6761d",
]

# Human-readable labels for metrics
METRIC_LABELS: Dict[str, str] = {
    # Readability
    "fre": "Flesch Reading Ease (FRE)",
    "fkgl": "Flesch-Kincaid Grade Level (FKGL)",
    "cli": "Coleman-Liau Index (CLI)",
    "fog": "Gunning Fog Index",
    # Lexical diversity & entropy
    "mattr_100": "Lexical Diversity (MATTR-100)",
    "yules_k": "Yule's K (Lower = Richer Vocab)",
    "hapax_ratio": "Hapax Legomena Ratio",
    "shannon_char_entropy": "Shannon Character Entropy (bits)",
    "shannon_word_entropy": "Shannon Word Entropy (bits)",
    # Sentence & burstiness
    "avg_sentence_length": "Avg Sentence Length (words)",
    "sentence_length_std": "Sentence Length Std Dev (Burstiness)",
    "sentence_length_median": "Median Sentence Length (words)",
    "sentence_length_iqr": "Sentence Length IQR (words)",
    "avg_word_length": "Avg Word Length (chars)",
    "avg_syllables_per_word": "Avg Syllables / Word",
    "complex_word_ratio": "Complex Word Ratio (>=3 syl)",
    # Voice, stance, punctuation & markers
    "passive_voice_ratio": "Passive Voice Ratio",
    "first_person_plural_per_1k": "1st-Person Plural ('we/our') / 1k words",
    "first_person_singular_per_1k": "1st-Person Singular ('I/my') / 1k words",
    "hedging_per_1k": "Hedging Words / 1k words",
    "booster_per_1k": "Booster Words / 1k words",
    "dash_per_1k": "Em/En Dashes / 1k words",
    "semicolon_per_1k": "Semicolons / 1k words",
    "colon_per_1k": "Colons / 1k words",
    "paren_per_1k": "Parentheses / 1k words",
    "llm_excess_words_per_1k": "LLM Excess Words / 1k words",
    "control_words_per_1k": "Neutral Academic Control Words / 1k words",
    # Abstract-only
    "abstract_word_count": "Abstract Word Count",
    "abstract_avg_sentence_length": "Abstract Avg Sentence Length",
    "abstract_fre": "Abstract FRE",
    "abstract_fkgl": "Abstract FKGL",
    "abstract_mattr_100": "Abstract MATTR-100",
    "abstract_passive_ratio": "Abstract Passive Voice Ratio",
    # Volume & structure
    "page_count": "Page Count",
    "word_count": "Body Word Count",
    "author_count": "Estimated Author Count",
    "has_code_url": "Papers with Code URL (Share)",
    "has_appendix": "Papers with Appendix (Share)",
    "has_acknowledgments": "Papers with Acknowledgments (Share)",
    "has_limitations_or_ethics": "Papers with Limitations/Ethics (Share)",
    "figure_mention_count": "Figure Mentions / Paper",
    "table_mention_count": "Table Mentions / Paper",
    # Noise confounders
    "alpha_char_ratio": "Alphabetic Char Ratio",
    "digit_char_ratio": "Digit Char Ratio",
    "symbol_char_ratio": "Symbol Char Ratio",
    "non_ascii_ratio": "Non-ASCII Char Ratio",
    "single_char_token_ratio": "Single-Char Token Ratio (Math/OCR proxy)",
    # Offline reference stats
    "ref_count": "Reference Count / Paper",
    "doi_count": "DOIs / Paper",
    "arxiv_ref_count": "arXiv Citations / Paper",
    "ref_future_year_flags": "Future-Year Ref Flags / Paper",
    "ref_malformed_doi_flags": "Malformed DOI Flags / Paper",
    "ref_median_year": "Median Cited Year",
    "ref_mean_age": "Mean Citation Age (years)",
    # Pass 2 live reference validation
    "ref_validity_rate": "Overall Reference Validity Rate",
    "ref_invalid_rate": "Overall Reference Invalid/Unverified Rate",
    "doi_valid_rate": "DOI Validity Rate (CrossRef/OpenAlex)",
    "arxiv_valid_rate": "arXiv Citation Validity Rate",
    "entry_verified_rate": "Free-Text Entry Verification Rate",
    "has_any_invalid_ref": "Papers with >=1 Unverified/Invalid Ref",
    "has_multiple_invalid_refs": "Papers with >=2 Unverified/Invalid Refs",
    "mean_match_score": "Mean Bibliographic Match Score",
}


def configure_matplotlib(dpi: int = 180) -> None:
    plt.rcParams.update(
        {
            "figure.dpi": dpi,
            "savefig.dpi": dpi,
            "figure.facecolor": "white",
            "axes.facecolor": "#fbfcfd",
            "axes.edgecolor": "#cccccc",
            "axes.linewidth": 0.8,
            "axes.grid": True,
            "grid.color": "#e5e8ec",
            "grid.linestyle": "-",
            "grid.linewidth": 0.6,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "font.size": 10,
            "axes.titlesize": 11.5,
            "axes.titleweight": "bold",
            "axes.labelsize": 10,
            "legend.fontsize": 8.8,
            "legend.frameon": True,
            "legend.facecolor": "white",
            "legend.edgecolor": "#dddddd",
            "xtick.labelsize": 8.8,
            "ytick.labelsize": 8.8,
        }
    )


# Section placeholders filled below
# ---------------------------------------------------------------------------
# Data Loading, Filtering, Aggregation & Statistical Testing
# ---------------------------------------------------------------------------

NUMERIC_METRICS_PASS1 = [
    "pub_year", "pub_month", "page_count", "char_count", "word_count",
    "sentence_count", "alpha_char_ratio", "digit_char_ratio", "symbol_char_ratio",
    "non_ascii_ratio", "single_char_token_ratio", "avg_sentence_length",
    "sentence_length_std", "sentence_length_median", "sentence_length_iqr",
    "avg_word_length", "avg_syllables_per_word", "complex_word_ratio",
    "fre", "fkgl", "cli", "fog", "mattr_100", "yules_k", "hapax_ratio",
    "shannon_char_entropy", "shannon_word_entropy", "passive_sentence_count",
    "passive_voice_ratio", "first_person_plural_per_1k",
    "first_person_singular_per_1k", "hedging_per_1k", "booster_per_1k",
    "dash_per_1k", "semicolon_per_1k", "colon_per_1k", "paren_per_1k",
    "llm_excess_words_per_1k", "control_words_per_1k",
    "abstract_word_count", "abstract_sentence_count",
    "abstract_avg_sentence_length", "abstract_fre", "abstract_fkgl",
    "abstract_mattr_100", "abstract_passive_ratio",
    "author_count", "has_code_url", "code_url_count", "has_appendix",
    "has_acknowledgments", "has_limitations_or_ethics",
    "figure_mention_count", "table_mention_count",
    "ref_count", "doi_count", "arxiv_ref_count",
    "ref_future_year_flags", "ref_malformed_doi_flags",
    "ref_min_year", "ref_median_year", "ref_max_year", "ref_mean_age",
]

NUMERIC_METRICS_PASS2 = [
    "pub_year", "ref_count", "dois_checked", "dois_valid", "dois_invalid",
    "dois_error", "doi_valid_rate", "arxiv_refs_checked", "arxiv_refs_valid",
    "arxiv_refs_invalid", "arxiv_refs_error", "arxiv_valid_rate",
    "entries_checked", "entries_verified", "entries_unverified",
    "entries_error", "entry_verified_rate", "total_checked", "total_valid",
    "total_invalid", "total_error", "ref_validity_rate", "ref_invalid_rate",
    "has_any_invalid_ref", "has_multiple_invalid_refs", "mean_match_score",
]


def parse_pub_month_ts(series: pd.Series) -> pd.Series:
    """Convert YYYY-MM strings to pandas Timestamp (1st of each month)."""
    return pd.to_datetime(series.astype(str).str.slice(0, 7), format="%Y-%m", errors="coerce")


def _clean_numeric_columns(df: pd.DataFrame, cols: Sequence[str]) -> pd.DataFrame:
    for col in cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce").astype("float32")
    return df


def _apply_date_and_group_filters(
    df: pd.DataFrame,
    start_date: Optional[str],
    end_date: Optional[str],
    groups: Optional[set],
) -> pd.DataFrame:
    df = df.dropna(subset=["month_ts"]).copy()
    if start_date:
        start_ts = pd.to_datetime(start_date[:7], format="%Y-%m")
        df = df[df["month_ts"] >= start_ts]
    if end_date:
        end_ts = pd.to_datetime(end_date[:7], format="%Y-%m")
        df = df[df["month_ts"] <= end_ts]
    if groups and "arxiv_primary_group" in df.columns:
        df = df[df["arxiv_primary_group"].astype(str).str.lower().isin(groups)]
    return df


def load_pass1_csv(
    path: str,
    min_words: int = 300,
    min_alpha_ratio: float = 0.0,
    start_date: Optional[str] = None,
    end_date: Optional[str] = None,
    groups: Optional[set] = None,
    deduplicate: bool = True,
) -> pd.DataFrame:
    print(f"Loading Pass 1 metrics CSV: {path} ...")
    df = pd.read_csv(path, low_memory=False)
    n_raw = len(df)

    if deduplicate and {"archive_name", "member_name"}.issubset(df.columns):
        df = df.drop_duplicates(subset=["archive_name", "member_name"], keep="last")

    if "status" in df.columns:
        df = df[df["status"] == "ok"].copy()

    df = _clean_numeric_columns(df, NUMERIC_METRICS_PASS1)

    if "word_count" in df.columns and min_words > 0:
        df = df[df["word_count"] >= min_words]
    if "alpha_char_ratio" in df.columns and min_alpha_ratio > 0:
        df = df[df["alpha_char_ratio"] >= min_alpha_ratio]

    # Normalize discipline group labels
    if "arxiv_primary_group" in df.columns:
        df["arxiv_primary_group"] = (
            df["arxiv_primary_group"]
            .fillna("unknown")
            .astype(str)
            .str.strip()
            .replace({"": "unknown", "nan": "unknown"})
        )
    else:
        df["arxiv_primary_group"] = "unknown"

    df["month_ts"] = parse_pub_month_ts(df["pub_date"])
    df = _apply_date_and_group_filters(df, start_date, end_date, groups)

    # Guard against extreme OCR/garbage outliers in readability metrics while preserving real values
    if "fre" in df.columns:
        df.loc[(df["fre"] < -150) | (df["fre"] > 130), "fre"] = np.nan
    if "fkgl" in df.columns:
        df.loc[(df["fkgl"] < 0) | (df["fkgl"] > 45), "fkgl"] = np.nan
    if "fog" in df.columns:
        df.loc[(df["fog"] < 0) | (df["fog"] > 50), "fog"] = np.nan
    if "cli" in df.columns:
        df.loc[(df["cli"] < 0) | (df["cli"] > 45), "cli"] = np.nan
    if "yules_k" in df.columns:
        df.loc[(df["yules_k"] < 0) | (df["yules_k"] > 1000), "yules_k"] = np.nan

    print(f"  Pass 1 loaded: {n_raw:,} raw rows -> {len(df):,} clean papers kept.")
    return df


def load_pass2_csv(
    path: str,
    min_refs_checked: int = 1,
    start_date: Optional[str] = None,
    end_date: Optional[str] = None,
    groups: Optional[set] = None,
    deduplicate: bool = True,
) -> pd.DataFrame:
    print(f"Loading Pass 2 reference validation CSV: {path} ...")
    df = pd.read_csv(path, low_memory=False)
    n_raw = len(df)

    if deduplicate and {"archive_name", "member_name"}.issubset(df.columns):
        df = df.drop_duplicates(subset=["archive_name", "member_name"], keep="last")

    if "status" in df.columns:
        df = df[df["status"] == "ok"].copy()

    df = _clean_numeric_columns(df, NUMERIC_METRICS_PASS2)
    if "total_checked" in df.columns and min_refs_checked > 0:
        df = df[df["total_checked"] >= min_refs_checked]

    if "arxiv_primary_group" in df.columns:
        df["arxiv_primary_group"] = (
            df["arxiv_primary_group"]
            .fillna("unknown")
            .astype(str)
            .str.strip()
            .replace({"": "unknown", "nan": "unknown"})
        )
    else:
        df["arxiv_primary_group"] = "unknown"

    df["month_ts"] = parse_pub_month_ts(df["pub_date"])
    df = _apply_date_and_group_filters(df, start_date, end_date, groups)

    print(f"  Pass 2 loaded: {n_raw:,} raw rows -> {len(df):,} validated papers kept.")
    return df


def join_pass1_and_pass2(df1: pd.DataFrame, df2: pd.DataFrame) -> pd.DataFrame:
    """Inner-join Pass 1 and Pass 2 DataFrames on (archive_name, member_name) or arxiv_id."""
    p2_unique_cols = [c for c in df2.columns if c not in df1.columns]
    if {"archive_name", "member_name"}.issubset(df1.columns) and {
        "archive_name", "member_name"
    }.issubset(df2.columns):
        merged = pd.merge(
            df1,
            df2[["archive_name", "member_name"] + p2_unique_cols],
            on=["archive_name", "member_name"],
            how="inner",
        )
    elif "arxiv_id" in df1.columns and "arxiv_id" in df2.columns:
        merged = pd.merge(
            df1,
            df2[["arxiv_id"] + p2_unique_cols],
            on="arxiv_id",
            how="inner",
        )
    else:
        return pd.DataFrame()
    print(f"  Joined Pass 1 + Pass 2: {len(merged):,} matched papers.")
    return merged


def aggregate_time_series(
    df: pd.DataFrame,
    metric: str,
    freq: str = "MS",
    min_per_bin: int = 5,
) -> pd.DataFrame:
    """
    Aggregate a metric by month ('MS') or quarter ('QS').
    Returns DataFrame indexed by bin timestamp with columns:
    count, mean, std, sem, ci95, median, q25, q75.
    """
    if metric not in df.columns:
        return pd.DataFrame()
    sub = df[["month_ts", metric]].dropna()
    if sub.empty:
        return pd.DataFrame()

    if freq == "QS":
        period = sub["month_ts"].dt.to_period("Q").dt.to_timestamp()
    else:
        period = sub["month_ts"]

    grp = sub.groupby(period)[metric]
    agg = pd.DataFrame(
        {
            "count": grp.count(),
            "mean": grp.mean(),
            "std": grp.std(),
            "median": grp.median(),
            "q25": grp.quantile(0.25),
            "q75": grp.quantile(0.75),
        }
    )
    agg = agg[agg["count"] >= min_per_bin].sort_index()
    agg["sem"] = agg["std"] / np.sqrt(agg["count"].clip(lower=1))
    agg["ci95"] = 1.96 * agg["sem"].fillna(0.0)
    return agg


def compute_before_after_stats(
    df: pd.DataFrame,
    metrics: Sequence[str],
    cutoff_ts: pd.Timestamp,
    source_label: str,
) -> pd.DataFrame:
    """
    Compute pre- vs post-cutoff descriptive stats, Welch's t-test,
    Mann-Whitney U test, Cohen's d (with 95% CI), and monthly slope shift.
    """
    rows: List[Dict[str, Any]] = []
    pre_mask = df["month_ts"] < cutoff_ts
    post_mask = df["month_ts"] >= cutoff_ts

    for metric in metrics:
        if metric not in df.columns:
            continue
        pre = df.loc[pre_mask, metric].dropna().to_numpy(dtype=np.float64)
        post = df.loc[post_mask, metric].dropna().to_numpy(dtype=np.float64)
        n_pre, n_post = len(pre), len(post)
        if n_pre < 10 or n_post < 10:
            continue

        mean_pre, mean_post = float(np.mean(pre)), float(np.mean(post))
        sd_pre, sd_post = float(np.std(pre, ddof=1)), float(np.std(post, ddof=1))
        med_pre, med_post = float(np.median(pre)), float(np.median(post))
        diff = mean_post - mean_pre
        pct_change = (100.0 * diff / abs(mean_pre)) if abs(mean_pre) > 1e-9 else np.nan

        # Pooled SD and Cohen's d
        pooled_var = (((n_pre - 1) * sd_pre ** 2) + ((n_post - 1) * sd_post ** 2)) / max(
            1, n_pre + n_post - 2
        )
        pooled_sd = math.sqrt(max(pooled_var, 1e-12))
        cohens_d = diff / pooled_sd
        se_d = math.sqrt((n_pre + n_post) / (n_pre * n_post) + (cohens_d ** 2) / (2 * (n_pre + n_post)))
        d_ci_low = cohens_d - 1.96 * se_d
        d_ci_high = cohens_d + 1.96 * se_d

        # Welch's t-test & Mann-Whitney U
        t_stat = p_welch = u_p = np.nan
        if scipy_stats is not None:
            try:
                t_res = scipy_stats.ttest_ind(post, pre, equal_var=False)
                t_stat = float(t_res.statistic)
                p_welch = float(t_res.pvalue)
            except Exception:
                pass
            try:
                # Subsample very large arrays for Mann-Whitney speed while preserving precision
                rng = np.random.default_rng(42)
                pre_s = pre if n_pre <= 50000 else rng.choice(pre, 50000, replace=False)
                post_s = post if n_post <= 50000 else rng.choice(post, 50000, replace=False)
                u_res = scipy_stats.mannwhitneyu(post_s, pre_s, alternative="two-sided")
                u_p = float(u_res.pvalue)
            except Exception:
                pass

        # Pre vs Post monthly linear trend slopes (units per year)
        slope_pre = slope_post = np.nan
        agg = aggregate_time_series(df, metric, freq="MS", min_per_bin=5)
        if not agg.empty:
            for side, is_post in (("pre", False), ("post", True)):
                sub_a = agg[agg.index >= cutoff_ts] if is_post else agg[agg.index < cutoff_ts]
                if len(sub_a) >= 4:
                    x_years = (sub_a.index - cutoff_ts).days.to_numpy(dtype=np.float64) / 365.25
                    y_vals = sub_a["mean"].to_numpy(dtype=np.float64)
                    try:
                        slope = float(np.polyfit(x_years, y_vals, 1)[0])
                        if is_post:
                            slope_post = slope
                        else:
                            slope_pre = slope
                    except Exception:
                        pass

        rows.append(
            {
                "source": source_label,
                "metric": metric,
                "label": METRIC_LABELS.get(metric, metric),
                "n_pre": n_pre,
                "n_post": n_post,
                "mean_pre": round(mean_pre, 5),
                "sd_pre": round(sd_pre, 5),
                "median_pre": round(med_pre, 5),
                "mean_post": round(mean_post, 5),
                "sd_post": round(sd_post, 5),
                "median_post": round(med_post, 5),
                "diff_post_minus_pre": round(diff, 5),
                "pct_change": round(pct_change, 3) if not math.isnan(pct_change) else np.nan,
                "cohens_d": round(cohens_d, 4),
                "cohens_d_ci95_low": round(d_ci_low, 4),
                "cohens_d_ci95_high": round(d_ci_high, 4),
                "welch_t": round(t_stat, 4) if not math.isnan(t_stat) else np.nan,
                "welch_p": p_welch,
                "mannwhitney_p": u_p,
                "slope_pre_per_yr": round(slope_pre, 5) if not math.isnan(slope_pre) else np.nan,
                "slope_post_per_yr": round(slope_post, 5) if not math.isnan(slope_post) else np.nan,
            }
        )

    return pd.DataFrame(rows)

# ---------------------------------------------------------------------------
# Reusable Plotting Helpers & Stylometric Figures (01 - 07)
# ---------------------------------------------------------------------------

def _format_date_axis(ax: plt.Axes) -> None:
    locator = mdates.AutoDateLocator(minticks=4, maxticks=10)
    ax.xaxis.set_major_locator(locator)
    ax.xaxis.set_major_formatter(mdates.ConciseDateFormatter(locator))


def _draw_cutoff(ax: plt.Axes, cutoff_ts: pd.Timestamp, label: bool = True) -> None:
    ax.axvline(
        cutoff_ts,
        color=PALETTE["cutoff"],
        linestyle="--",
        linewidth=1.4,
        alpha=0.9,
        zorder=5,
        label=f"Cutoff ({cutoff_ts.strftime('%Y-%m')})" if label else None,
    )
    xlim = ax.get_xlim()
    cutoff_num = mdates.date2num(cutoff_ts)
    if xlim[1] > cutoff_num:
        ax.axvspan(cutoff_num, xlim[1], color=PALETTE["cutoff"], alpha=0.045, zorder=0)


def plot_metric_panel(
    ax: plt.Axes,
    df: pd.DataFrame,
    metric: str,
    cutoff_ts: pd.Timestamp,
    freq: str = "MS",
    min_per_bin: int = 5,
    rolling_window: int = 3,
    color: str = PALETTE["primary"],
    show_trend_lines: bool = True,
    title: Optional[str] = None,
    ylabel: Optional[str] = None,
    as_percent: bool = False,
) -> bool:
    """Plot monthly/quarterly mean with 95% CI ribbon, rolling smooth, and segmented trend."""
    agg = aggregate_time_series(df, metric, freq=freq, min_per_bin=min_per_bin)
    if agg.empty:
        ax.text(
            0.5, 0.5, f"No sufficient data\nfor '{metric}'",
            ha="center", va="center", transform=ax.transAxes, color="#888888",
        )
        ax.set_title(title or METRIC_LABELS.get(metric, metric))
        return False

    scale = 100.0 if as_percent else 1.0
    x = agg.index
    y = agg["mean"] * scale
    ci = agg["ci95"] * scale

    ax.fill_between(x, y - ci, y + ci, color=color, alpha=0.16, linewidth=0, label="95% CI")
    ax.plot(x, y, color=color, alpha=0.45, linewidth=1.1, marker="o", markersize=2.5, label="Monthly mean")

    if rolling_window > 1 and len(agg) >= rolling_window:
        smooth = y.rolling(rolling_window, center=True, min_periods=max(1, rolling_window // 2)).mean()
        ax.plot(x, smooth, color=color, linewidth=2.2, label=f"{rolling_window}-bin rolling")

    # Pre / Post segmented linear fits (interrupted time-series visual check)
    if show_trend_lines:
        for is_post, lcolor in ((False, PALETTE["pre"]), (True, PALETTE["post"])):
            sub = agg[agg.index >= cutoff_ts] if is_post else agg[agg.index < cutoff_ts]
            if len(sub) >= 4:
                x_num = mdates.date2num(sub.index)
                try:
                    coeffs = np.polyfit(x_num, sub["mean"].to_numpy() * scale, 1)
                    ax.plot(
                        sub.index,
                        np.polyval(coeffs, x_num),
                        color=lcolor,
                        linestyle=":",
                        linewidth=1.5,
                        alpha=0.85,
                    )
                except Exception:
                    pass

    _draw_cutoff(ax, cutoff_ts, label=False)
    _format_date_axis(ax)

    ax.set_title(title or METRIC_LABELS.get(metric, metric))
    ax.set_ylabel(ylabel or ("%" if as_percent else METRIC_LABELS.get(metric, metric)))
    if as_percent:
        ax.yaxis.set_major_formatter(mticker.FormatStrFormatter("%.1f%%"))

    # Compact pre vs post mean annotation box
    pre_vals = df.loc[df["month_ts"] < cutoff_ts, metric].dropna() * scale
    post_vals = df.loc[df["month_ts"] >= cutoff_ts, metric].dropna() * scale
    if len(pre_vals) >= 5 and len(post_vals) >= 5:
        delta = post_vals.mean() - pre_vals.mean()
        sign = "+" if delta >= 0 else ""
        unit = "%" if as_percent else ""
        note = f"Pre: {pre_vals.mean():.2f}{unit} | Post: {post_vals.mean():.2f}{unit} ({sign}{delta:.2f}{unit})"
        ax.text(
            0.02, 0.96, note,
            transform=ax.transAxes,
            fontsize=8.0,
            va="top",
            ha="left",
            bbox={"boxstyle": "round,pad=0.25", "facecolor": "white", "edgecolor": "#dddddd", "alpha": 0.88},
        )
    return True


def _save_fig(fig: plt.Figure, out_dir: Path, filename: str, fmt: str = "png") -> Path:
    out_path = out_dir / f"{filename}.{fmt}"
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved {out_path.name}")
    return out_path


def plot_01_overview_dashboard(
    df1: Optional[pd.DataFrame],
    df2: Optional[pd.DataFrame],
    cutoff_ts: pd.Timestamp,
    out_dir: Path,
    args: argparse.Namespace,
) -> None:
    if df1 is None or df1.empty:
        return
    fig, axes = plt.subplots(2, 3, figsize=(17, 9.5))
    fig.suptitle(
        f"arXiv Longitudinal Overview — Pre vs Post {cutoff_ts.strftime('%Y-%m')} "
        f"(N = {len(df1):,} papers)",
        fontsize=14,
        fontweight="bold",
        y=1.01,
    )

    specs = [
        (axes[0, 0], df1, "fre", PALETTE["primary"], False, None),
        (axes[0, 1], df1, "mattr_100", PALETTE["tertiary"], False, None),
        (axes[0, 2], df1, "shannon_char_entropy", PALETTE["quaternary"], False, None),
        (axes[1, 0], df1, "passive_voice_ratio", PALETTE["secondary"], True, None),
        (axes[1, 1], df1, "llm_excess_words_per_1k", PALETTE["accent"], False, None),
    ]
    for ax, data, col, clr, pct, custom_title in specs:
        plot_metric_panel(
            ax, data, col, cutoff_ts,
            freq=args.freq, min_per_bin=args.min_per_bin,
            rolling_window=args.rolling, color=clr, as_percent=pct, title=custom_title,
        )

    # Bottom-right: live reference validity if Pass 2 exists, otherwise offline ref count
    if df2 is not None and not df2.empty and "ref_validity_rate" in df2.columns:
        plot_metric_panel(
            axes[1, 2], df2, "ref_validity_rate", cutoff_ts,
            freq=args.freq, min_per_bin=max(3, args.min_per_bin // 2),
            rolling_window=args.rolling, color="#2e7d32", as_percent=True,
            title=f"Live Reference Validity Rate (Pass 2, N={len(df2):,})",
        )
    else:
        plot_metric_panel(
            axes[1, 2], df1, "ref_count", cutoff_ts,
            freq=args.freq, min_per_bin=args.min_per_bin,
            rolling_window=args.rolling, color="#2e7d32", as_percent=False,
        )

    fig.tight_layout()
    _save_fig(fig, out_dir, "01_overview_dashboard", args.format)


def plot_02_readability(
    df1: pd.DataFrame, cutoff_ts: pd.Timestamp, out_dir: Path, args: argparse.Namespace
) -> None:
    fig, axes = plt.subplots(2, 2, figsize=(14, 9))
    fig.suptitle("Readability Indices Over Time (Body Text)", fontsize=13.5, fontweight="bold", y=1.01)
    metrics = [
        ("fre", PALETTE["primary"]),
        ("fkgl", PALETTE["secondary"]),
        ("cli", PALETTE["tertiary"]),
        ("fog", PALETTE["quaternary"]),
    ]
    for ax, (m, clr) in zip(axes.flat, metrics):
        plot_metric_panel(
            ax, df1, m, cutoff_ts, freq=args.freq,
            min_per_bin=args.min_per_bin, rolling_window=args.rolling, color=clr,
        )
    fig.tight_layout()
    _save_fig(fig, out_dir, "02_readability_trends", args.format)


def plot_03_lexical_diversity_and_entropy(
    df1: pd.DataFrame, cutoff_ts: pd.Timestamp, out_dir: Path, args: argparse.Namespace
) -> None:
    fig, axes = plt.subplots(2, 3, figsize=(16.5, 9))
    fig.suptitle("Lexical Diversity & Information Density Over Time", fontsize=13.5, fontweight="bold", y=1.01)
    metrics = [
        ("mattr_100", PALETTE["primary"], False),
        ("yules_k", PALETTE["secondary"], False),
        ("hapax_ratio", PALETTE["tertiary"], True),
        ("shannon_char_entropy", PALETTE["quaternary"], False),
        ("shannon_word_entropy", PALETTE["accent"], False),
        ("avg_word_length", "#455a64", False),
    ]
    for ax, (m, clr, pct) in zip(axes.flat, metrics):
        plot_metric_panel(
            ax, df1, m, cutoff_ts, freq=args.freq,
            min_per_bin=args.min_per_bin, rolling_window=args.rolling,
            color=clr, as_percent=pct,
        )
    fig.tight_layout()
    _save_fig(fig, out_dir, "03_lexical_diversity_and_entropy", args.format)


def plot_04_sentence_and_burstiness(
    df1: pd.DataFrame, cutoff_ts: pd.Timestamp, out_dir: Path, args: argparse.Namespace
) -> None:
    fig, axes = plt.subplots(2, 2, figsize=(14, 9))
    fig.suptitle("Sentence Structure, Complexity & Burstiness Over Time", fontsize=13.5, fontweight="bold", y=1.01)
    metrics = [
        ("avg_sentence_length", PALETTE["primary"], False),
        ("sentence_length_std", PALETTE["secondary"], False),
        ("sentence_length_iqr", PALETTE["tertiary"], False),
        ("complex_word_ratio", PALETTE["quaternary"], True),
    ]
    for ax, (m, clr, pct) in zip(axes.flat, metrics):
        plot_metric_panel(
            ax, df1, m, cutoff_ts, freq=args.freq,
            min_per_bin=args.min_per_bin, rolling_window=args.rolling,
            color=clr, as_percent=pct,
        )
    fig.tight_layout()
    _save_fig(fig, out_dir, "04_sentence_structure_and_burstiness", args.format)


def plot_05_voice_stance_punctuation(
    df1: pd.DataFrame, cutoff_ts: pd.Timestamp, out_dir: Path, args: argparse.Namespace
) -> None:
    fig, axes = plt.subplots(2, 3, figsize=(16.5, 9.2))
    fig.suptitle("Voice, Epistemic Stance & Punctuation Habits Over Time", fontsize=13.5, fontweight="bold", y=1.01)
    metrics = [
        ("passive_voice_ratio", PALETTE["primary"], True),
        ("first_person_plural_per_1k", PALETTE["tertiary"], False),
        ("hedging_per_1k", PALETTE["quaternary"], False),
        ("booster_per_1k", PALETTE["secondary"], False),
        ("dash_per_1k", PALETTE["accent"], False),
        ("semicolon_per_1k", "#5d4037", False),
    ]
    for ax, (m, clr, pct) in zip(axes.flat, metrics):
        plot_metric_panel(
            ax, df1, m, cutoff_ts, freq=args.freq,
            min_per_bin=args.min_per_bin, rolling_window=args.rolling,
            color=clr, as_percent=pct,
        )
    fig.tight_layout()
    _save_fig(fig, out_dir, "05_voice_stance_and_punctuation", args.format)


def plot_06_excess_vs_control_words(
    df1: pd.DataFrame, cutoff_ts: pd.Timestamp, out_dir: Path, args: argparse.Namespace
) -> None:
    if not {"llm_excess_words_per_1k", "control_words_per_1k"}.issubset(df1.columns):
        return
    agg_ex = aggregate_time_series(df1, "llm_excess_words_per_1k", freq=args.freq, min_per_bin=args.min_per_bin)
    agg_ct = aggregate_time_series(df1, "control_words_per_1k", freq=args.freq, min_per_bin=args.min_per_bin)
    if agg_ex.empty or agg_ct.empty:
        return

    fig, axes = plt.subplots(1, 3, figsize=(17, 5.2))
    fig.suptitle(
        "LLM Excess-Word Lexicon vs Neutral Academic Control Lexicon (Placebo Check)",
        fontsize=13,
        fontweight="bold",
        y=1.02,
    )

    # Panel A: Excess words raw rate
    plot_metric_panel(
        axes[0], df1, "llm_excess_words_per_1k", cutoff_ts,
        freq=args.freq, min_per_bin=args.min_per_bin,
        rolling_window=args.rolling, color=PALETTE["accent"],
    )

    # Panel B: Control words raw rate
    plot_metric_panel(
        axes[1], df1, "control_words_per_1k", cutoff_ts,
        freq=args.freq, min_per_bin=args.min_per_bin,
        rolling_window=args.rolling, color=PALETTE["primary"],
    )

    # Panel C: Both indexed to pre-cutoff mean = 100
    ax = axes[2]
    pre_ex = agg_ex.loc[agg_ex.index < cutoff_ts, "mean"].mean()
    pre_ct = agg_ct.loc[agg_ct.index < cutoff_ts, "mean"].mean()
    if pd.notna(pre_ex) and pre_ex > 0 and pd.notna(pre_ct) and pre_ct > 0:
        idx_ex = 100.0 * agg_ex["mean"] / pre_ex
        idx_ct = 100.0 * agg_ct["mean"] / pre_ct
        ax.plot(agg_ex.index, idx_ex, color=PALETTE["accent"], linewidth=2.1, label="LLM Excess Lexicon (Pre=100)")
        ax.plot(agg_ct.index, idx_ct, color=PALETTE["primary"], linewidth=2.1, label="Neutral Control Lexicon (Pre=100)")
        ax.axhline(100.0, color="#888888", linestyle=":", linewidth=1.0)
        _draw_cutoff(ax, cutoff_ts, label=True)
        _format_date_axis(ax)
        ax.set_title("Indexed Trajectory (Pre-Cutoff Mean = 100)")
        ax.set_ylabel("Index (Pre-Cutoff = 100)")
        ax.legend(loc="upper left")
    else:
        ax.text(0.5, 0.5, "Insufficient pre-cutoff baseline", ha="center", va="center", transform=ax.transAxes)

    fig.tight_layout()
    _save_fig(fig, out_dir, "06_llm_excess_vs_control_words", args.format)


def plot_07_body_vs_abstract_control(
    df1: pd.DataFrame, cutoff_ts: pd.Timestamp, out_dir: Path, args: argparse.Namespace
) -> None:
    pairs = [
        ("fre", "abstract_fre", "Flesch Reading Ease (FRE)", False),
        ("fkgl", "abstract_fkgl", "Flesch-Kincaid Grade Level (FKGL)", False),
        ("mattr_100", "abstract_mattr_100", "Lexical Diversity (MATTR-100)", False),
        ("passive_voice_ratio", "abstract_passive_ratio", "Passive Voice Ratio", True),
    ]
    fig, axes = plt.subplots(2, 2, figsize=(14.5, 9))
    fig.suptitle(
        "Clean-Prose Robustness Check: Full Body vs Abstract-Only Parallel Metrics",
        fontsize=13.5,
        fontweight="bold",
        y=1.01,
    )

    for ax, (body_col, abs_col, title, as_pct) in zip(axes.flat, pairs):
        scale = 100.0 if as_pct else 1.0
        agg_b = aggregate_time_series(df1, body_col, freq=args.freq, min_per_bin=args.min_per_bin)
        agg_a = aggregate_time_series(df1, abs_col, freq=args.freq, min_per_bin=args.min_per_bin)

        if not agg_b.empty:
            yb = (agg_b["mean"] * scale).rolling(args.rolling, center=True, min_periods=1).mean()
            ax.plot(agg_b.index, yb, color=PALETTE["primary"], linewidth=2.0, label="Full Body")
            ax.fill_between(
                agg_b.index,
                agg_b["mean"] * scale - agg_b["ci95"] * scale,
                agg_b["mean"] * scale + agg_b["ci95"] * scale,
                color=PALETTE["primary"], alpha=0.12, linewidth=0,
            )
        if not agg_a.empty:
            ya = (agg_a["mean"] * scale).rolling(args.rolling, center=True, min_periods=1).mean()
            ax.plot(agg_a.index, ya, color=PALETTE["secondary"], linewidth=2.0, linestyle="-.", label="Abstract Only")
            ax.fill_between(
                agg_a.index,
                agg_a["mean"] * scale - agg_a["ci95"] * scale,
                agg_a["mean"] * scale + agg_a["ci95"] * scale,
                color=PALETTE["secondary"], alpha=0.12, linewidth=0,
            )

        _draw_cutoff(ax, cutoff_ts, label=False)
        _format_date_axis(ax)
        ax.set_title(title)
        ax.set_ylabel("%" if as_pct else title)
        if as_pct:
            ax.yaxis.set_major_formatter(mticker.FormatStrFormatter("%.1f%%"))
        ax.legend(loc="best")

    fig.tight_layout()
    _save_fig(fig, out_dir, "07_body_vs_abstract_control", args.format)

# ---------------------------------------------------------------------------
# Structure, Confounders, Discipline Stratification, Pass 2 & Effect Plots
# ---------------------------------------------------------------------------

def _top_discipline_groups(df: pd.DataFrame, max_groups: int = 6) -> List[str]:
    if "arxiv_primary_group" not in df.columns:
        return []
    counts = df["arxiv_primary_group"].value_counts()
    counts = counts[counts.index != "unknown"]
    return list(counts.head(max_groups).index)


def plot_08_authorship_and_structure(
    df1: pd.DataFrame, cutoff_ts: pd.Timestamp, out_dir: Path, args: argparse.Namespace
) -> None:
    fig, axes = plt.subplots(2, 3, figsize=(16.5, 9.2))
    fig.suptitle("Authorship, Document Length & Structural Signals Over Time", fontsize=13.5, fontweight="bold", y=1.01)
    specs = [
        ("author_count", PALETTE["primary"], False),
        ("page_count", PALETTE["secondary"], False),
        ("word_count", PALETTE["tertiary"], False),
        ("has_code_url", "#00897b", True),
        ("has_appendix", PALETTE["quaternary"], True),
        ("has_limitations_or_ethics", PALETTE["accent"], True),
    ]
    for ax, (m, clr, pct) in zip(axes.flat, specs):
        plot_metric_panel(
            ax, df1, m, cutoff_ts, freq=args.freq,
            min_per_bin=args.min_per_bin, rolling_window=args.rolling,
            color=clr, as_percent=pct,
        )
    fig.tight_layout()
    _save_fig(fig, out_dir, "08_authorship_and_structure", args.format)


def plot_09_confounders_and_discipline_mix(
    df1: pd.DataFrame, cutoff_ts: pd.Timestamp, out_dir: Path, args: argparse.Namespace
) -> None:
    fig, axes = plt.subplots(2, 2, figsize=(15, 9.2))
    fig.suptitle(
        "Confounder & Corpus-Health Diagnostics (Volume, Field Mix & Extraction Noise)",
        fontsize=13.5,
        fontweight="bold",
        y=1.01,
    )

    # Panel 1: Monthly paper counts
    ax0 = axes[0, 0]
    counts = df1.groupby("month_ts").size().sort_index()
    ax0.bar(counts.index, counts.values, width=22, color=PALETTE["primary"], alpha=0.75)
    _draw_cutoff(ax0, cutoff_ts, label=True)
    _format_date_axis(ax0)
    ax0.set_title("Monthly Analyzed Paper Count")
    ax0.set_ylabel("Papers / Month")
    ax0.legend(loc="upper left")

    # Panel 2: Discipline mix over time (stacked share)
    ax1 = axes[0, 1]
    top_groups = _top_discipline_groups(df1, max_groups=6)
    if top_groups:
        tmp = df1[["month_ts", "arxiv_primary_group"]].copy()
        tmp["grp_clean"] = np.where(tmp["arxiv_primary_group"].isin(top_groups), tmp["arxiv_primary_group"], "other")
        ct = pd.crosstab(tmp["month_ts"], tmp["grp_clean"])
        row_sums = ct.sum(axis=1)
        ct = ct[row_sums >= args.min_per_bin]
        shares = ct.div(ct.sum(axis=1), axis=0) * 100.0
        cols_order = [g for g in top_groups if g in shares.columns] + (["other"] if "other" in shares.columns else [])
        shares = shares[cols_order]
        ax1.stackplot(
            shares.index,
            [shares[c].to_numpy() for c in shares.columns],
            labels=shares.columns,
            colors=DISCIPLINE_COLORS[: len(shares.columns)],
            alpha=0.82,
        )
        _draw_cutoff(ax1, cutoff_ts, label=False)
        _format_date_axis(ax1)
        ax1.set_ylim(0, 100)
        ax1.set_title("Discipline Share Over Time (Field-Mix Confounder)")
        ax1.set_ylabel("Share of Papers (%)")
        ax1.legend(loc="upper left", ncol=2, fontsize=8)
    else:
        ax1.text(0.5, 0.5, "No arXiv category stamps parsed", ha="center", va="center", transform=ax1.transAxes)
        ax1.set_title("Discipline Share Over Time")

    # Panel 3: Alpha char ratio
    plot_metric_panel(
        axes[1, 0], df1, "alpha_char_ratio", cutoff_ts,
        freq=args.freq, min_per_bin=args.min_per_bin,
        rolling_window=args.rolling, color=PALETTE["tertiary"], as_percent=True,
    )

    # Panel 4: Single-char token ratio (math/OCR noise proxy)
    plot_metric_panel(
        axes[1, 1], df1, "single_char_token_ratio", cutoff_ts,
        freq=args.freq, min_per_bin=args.min_per_bin,
        rolling_window=args.rolling, color=PALETTE["secondary"], as_percent=True,
    )

    fig.tight_layout()
    _save_fig(fig, out_dir, "09_confounders_and_discipline_mix", args.format)


def plot_10_discipline_stratified_trends(
    df1: pd.DataFrame, cutoff_ts: pd.Timestamp, out_dir: Path, args: argparse.Namespace
) -> None:
    top_groups = _top_discipline_groups(df1, max_groups=5)
    if len(top_groups) < 2:
        return

    metrics = [
        ("fre", "Flesch Reading Ease (FRE)", False),
        ("mattr_100", "Lexical Diversity (MATTR-100)", False),
        ("llm_excess_words_per_1k", "LLM Excess Words / 1k words", False),
        ("passive_voice_ratio", "Passive Voice Ratio (%)", True),
    ]
    fig, axes = plt.subplots(2, 2, figsize=(15, 9.2))
    fig.suptitle(
        "Field-Mix Control: Key Stylometric Trends Stratified by Primary arXiv Discipline",
        fontsize=13.5,
        fontweight="bold",
        y=1.01,
    )

    for ax, (metric, title, as_pct) in zip(axes.flat, metrics):
        scale = 100.0 if as_pct else 1.0
        plotted = False
        for idx, grp in enumerate(top_groups):
            sub = df1[df1["arxiv_primary_group"] == grp]
            agg = aggregate_time_series(sub, metric, freq=args.freq, min_per_bin=max(3, args.min_per_bin // 2))
            if agg.empty:
                continue
            y = (agg["mean"] * scale).rolling(args.rolling, center=True, min_periods=1).mean()
            ax.plot(
                agg.index, y,
                color=DISCIPLINE_COLORS[idx % len(DISCIPLINE_COLORS)],
                linewidth=2.0,
                label=f"{grp} (n={len(sub):,})",
            )
            plotted = True

        if plotted:
            _draw_cutoff(ax, cutoff_ts, label=False)
            _format_date_axis(ax)
            ax.legend(loc="best", fontsize=8.2)
            if as_pct:
                ax.yaxis.set_major_formatter(mticker.FormatStrFormatter("%.1f%%"))
        else:
            ax.text(0.5, 0.5, "Insufficient stratified data", ha="center", va="center", transform=ax.transAxes)
        ax.set_title(title)
        ax.set_ylabel("%" if as_pct else title)

    fig.tight_layout()
    _save_fig(fig, out_dir, "10_discipline_stratified_trends", args.format)


def plot_11_offline_references(
    df1: pd.DataFrame, cutoff_ts: pd.Timestamp, out_dir: Path, args: argparse.Namespace
) -> None:
    fig, axes = plt.subplots(2, 3, figsize=(16.5, 9.2))
    fig.suptitle("Offline Reference Volume, Recency & Structural Flags (Pass 1)", fontsize=13.5, fontweight="bold", y=1.01)
    specs = [
        ("ref_count", PALETTE["primary"], False),
        ("ref_mean_age", PALETTE["secondary"], False),
        ("doi_count", PALETTE["tertiary"], False),
        ("arxiv_ref_count", PALETTE["quaternary"], False),
        ("ref_future_year_flags", PALETTE["cutoff"], False),
        ("ref_malformed_doi_flags", PALETTE["accent"], False),
    ]
    for ax, (m, clr, pct) in zip(axes.flat, specs):
        plot_metric_panel(
            ax, df1, m, cutoff_ts, freq=args.freq,
            min_per_bin=args.min_per_bin, rolling_window=args.rolling,
            color=clr, as_percent=pct,
        )
    fig.tight_layout()
    _save_fig(fig, out_dir, "11_offline_references_and_recency", args.format)


def plot_12_live_reference_validation(
    df2: pd.DataFrame, cutoff_ts: pd.Timestamp, out_dir: Path, args: argparse.Namespace
) -> None:
    fig, axes = plt.subplots(2, 3, figsize=(16.5, 9.2))
    fig.suptitle(
        f"Live Reference Validity Over Time (Pass 2: CrossRef + arXiv + OpenAlex, N = {len(df2):,})",
        fontsize=13.5,
        fontweight="bold",
        y=1.01,
    )
    min_bin = max(3, args.min_per_bin // 2)
    specs = [
        ("ref_validity_rate", "#2e7d32", True),
        ("doi_valid_rate", PALETTE["primary"], True),
        ("arxiv_valid_rate", PALETTE["quaternary"], True),
        ("entry_verified_rate", PALETTE["tertiary"], True),
        ("has_any_invalid_ref", PALETTE["secondary"], True),
        ("has_multiple_invalid_refs", PALETTE["cutoff"], True),
    ]
    for ax, (m, clr, pct) in zip(axes.flat, specs):
        plot_metric_panel(
            ax, df2, m, cutoff_ts, freq=args.freq,
            min_per_bin=min_bin, rolling_window=args.rolling,
            color=clr, as_percent=pct,
        )
    fig.tight_layout()
    _save_fig(fig, out_dir, "12_live_reference_validation", args.format)


def plot_13_reference_validation_by_discipline(
    df2: pd.DataFrame, cutoff_ts: pd.Timestamp, out_dir: Path, args: argparse.Namespace
) -> None:
    top_groups = _top_discipline_groups(df2, max_groups=5)
    if len(top_groups) < 2:
        return

    fig, axes = plt.subplots(1, 2, figsize=(15, 5.5))
    fig.suptitle("Live Reference Validity Stratified by Discipline (Pass 2)", fontsize=13.5, fontweight="bold", y=1.02)

    panels = [
        (axes[0], "ref_validity_rate", "Overall Reference Validity Rate (%)"),
        (axes[1], "has_any_invalid_ref", "Share of Papers with >=1 Unverified/Invalid Ref (%)"),
    ]
    for ax, metric, title in panels:
        plotted = False
        for idx, grp in enumerate(top_groups):
            sub = df2[df2["arxiv_primary_group"] == grp]
            agg = aggregate_time_series(sub, metric, freq=args.freq, min_per_bin=3)
            if agg.empty:
                continue
            y = (agg["mean"] * 100.0).rolling(args.rolling, center=True, min_periods=1).mean()
            ax.plot(
                agg.index, y,
                color=DISCIPLINE_COLORS[idx % len(DISCIPLINE_COLORS)],
                linewidth=2.0,
                label=f"{grp} (n={len(sub):,})",
            )
            plotted = True
        if plotted:
            _draw_cutoff(ax, cutoff_ts, label=False)
            _format_date_axis(ax)
            ax.yaxis.set_major_formatter(mticker.FormatStrFormatter("%.1f%%"))
            ax.legend(loc="best")
        else:
            ax.text(0.5, 0.5, "Insufficient stratified Pass 2 data", ha="center", va="center", transform=ax.transAxes)
        ax.set_title(title)
        ax.set_ylabel("%")

    fig.tight_layout()
    _save_fig(fig, out_dir, "13_ref_validation_by_discipline", args.format)


def plot_14_joined_stylometry_vs_refs(
    merged: pd.DataFrame, cutoff_ts: pd.Timestamp, out_dir: Path, args: argparse.Namespace
) -> None:
    """
    Cross-pass diagnostic joining Pass 1 stylometrics with Pass 2 reference validity:
      1. Invalid reference rate across quantiles of LLM excess-word density (Pre vs Post).
      2. Correlation bar chart of Pass 1 stylometrics with ref_invalid_rate.
    """
    if merged is None or len(merged) < 30:
        return

    fig, axes = plt.subplots(1, 2, figsize=(15, 5.6))
    fig.suptitle(
        f"Cross-Pass Analysis: Stylometric Signals vs Live Reference Validity (Joined N = {len(merged):,})",
        fontsize=13,
        fontweight="bold",
        y=1.02,
    )

    # Panel 1: Binned LLM excess words vs ref_invalid_rate (Pre vs Post)
    ax0 = axes[0]
    if {"llm_excess_words_per_1k", "ref_invalid_rate"}.issubset(merged.columns):
        sub = merged[["month_ts", "llm_excess_words_per_1k", "ref_invalid_rate"]].dropna().copy()
        if len(sub) >= 30:
            bins = [-0.01, 0.0, 0.5, 1.5, 3.0, 1e9]
            labels = ["0", "0-0.5", "0.5-1.5", "1.5-3.0", ">3.0"]
            sub["excess_bin"] = pd.cut(sub["llm_excess_words_per_1k"], bins=bins, labels=labels)

            x_pos = np.arange(len(labels))
            width = 0.36
            for offset, label_era, mask, clr in (
                (-width / 2, f"Pre {cutoff_ts.strftime('%Y-%m')}", sub["month_ts"] < cutoff_ts, PALETTE["pre"]),
                (width / 2, f"Post {cutoff_ts.strftime('%Y-%m')}", sub["month_ts"] >= cutoff_ts, PALETTE["post"]),
            ):
                era = sub[mask]
                means, errs = [], []
                for b in labels:
                    vals = era.loc[era["excess_bin"] == b, "ref_invalid_rate"] * 100.0
                    if len(vals) >= 5:
                        means.append(vals.mean())
                        errs.append(1.96 * vals.std() / math.sqrt(len(vals)))
                    else:
                        means.append(np.nan)
                        errs.append(0.0)
                ax0.bar(x_pos + offset, means, width=width, yerr=errs, capsize=3, color=clr, alpha=0.82, label=label_era)

            ax0.set_xticks(x_pos)
            ax0.set_xticklabels(labels)
            ax0.set_xlabel("LLM Excess Words per 1,000 Body Words")
            ax0.set_ylabel("Invalid / Unverified Reference Rate (%)")
            ax0.set_title("Reference Invalidity Rate by Excess-Word Density")
            ax0.yaxis.set_major_formatter(mticker.FormatStrFormatter("%.1f%%"))
            ax0.legend(loc="best")
        else:
            ax0.text(0.5, 0.5, "Insufficient joined rows", ha="center", va="center", transform=ax0.transAxes)

    # Panel 2: Spearman correlations with ref_invalid_rate
    ax1 = axes[1]
    candidate_cols = [
        "llm_excess_words_per_1k", "fre", "fkgl", "mattr_100", "shannon_char_entropy",
        "passive_voice_ratio", "sentence_length_std", "dash_per_1k", "hedging_per_1k",
        "booster_per_1k", "author_count", "page_count", "ref_count", "ref_mean_age",
    ]
    corrs: List[Tuple[str, float]] = []
    if "ref_invalid_rate" in merged.columns:
        for col in candidate_cols:
            if col not in merged.columns:
                continue
            pair = merged[[col, "ref_invalid_rate"]].dropna()
            if len(pair) < 20 or pair[col].nunique() < 2 or pair["ref_invalid_rate"].nunique() < 2:
                continue
            r = pair[col].corr(pair["ref_invalid_rate"], method="spearman")
            if pd.notna(r):
                corrs.append((METRIC_LABELS.get(col, col), float(r)))

    if corrs:
        corrs.sort(key=lambda t: t[1])
        names = [c[0] for c in corrs]
        vals = [c[1] for c in corrs]
        colors = [PALETTE["post"] if v > 0 else PALETTE["pre"] for v in vals]
        y_pos = np.arange(len(names))
        ax1.barh(y_pos, vals, color=colors, alpha=0.85)
        ax1.axvline(0.0, color="#444444", linewidth=1.0)
        ax1.set_yticks(y_pos)
        ax1.set_yticklabels(names, fontsize=8.4)
        ax1.set_xlabel("Spearman Correlation (rho) with ref_invalid_rate")
        ax1.set_title("Which Paper Features Correlate with Invalid Refs?")
    else:
        ax1.text(0.5, 0.5, "Insufficient variance for correlations", ha="center", va="center", transform=ax1.transAxes)

    fig.tight_layout()
    _save_fig(fig, out_dir, "14_stylometry_vs_ref_validity", args.format)


def plot_15_effect_size_forest(
    stats_df: pd.DataFrame, cutoff_ts: pd.Timestamp, out_dir: Path, args: argparse.Namespace
) -> None:
    """Horizontal forest plot of standardized Pre vs Post shifts (Cohen's d +/- 95% CI)."""
    if stats_df is None or stats_df.empty:
        return

    featured = [
        "llm_excess_words_per_1k", "control_words_per_1k",
        "fre", "fkgl", "cli", "fog",
        "mattr_100", "yules_k", "shannon_char_entropy", "shannon_word_entropy",
        "avg_sentence_length", "sentence_length_std",
        "passive_voice_ratio", "first_person_plural_per_1k",
        "hedging_per_1k", "booster_per_1k", "dash_per_1k", "semicolon_per_1k",
        "abstract_fre", "abstract_mattr_100",
        "author_count", "page_count", "has_code_url", "has_limitations_or_ethics",
        "ref_count", "ref_mean_age",
        "ref_validity_rate", "doi_valid_rate", "arxiv_valid_rate",
        "entry_verified_rate", "has_any_invalid_ref",
    ]
    sub = stats_df[stats_df["metric"].isin(featured)].copy()
    if sub.empty:
        return

    sub = sub.sort_values("cohens_d", ascending=True).reset_index(drop=True)
    height = max(6.0, 0.34 * len(sub) + 1.8)
    fig, ax = plt.subplots(figsize=(12.5, height))

    y = np.arange(len(sub))
    d_vals = sub["cohens_d"].to_numpy()
    err_low = d_vals - sub["cohens_d_ci95_low"].to_numpy()
    err_high = sub["cohens_d_ci95_high"].to_numpy() - d_vals
    colors = [PALETTE["post"] if d >= 0 else PALETTE["pre"] for d in d_vals]

    ax.axvline(0.0, color="#333333", linewidth=1.1, linestyle="-")
    for thresh in (-0.2, 0.2, -0.5, 0.5):
        ax.axvline(thresh, color="#cccccc", linewidth=0.7, linestyle=":")

    ax.errorbar(
        d_vals, y, xerr=[err_low, err_high],
        fmt="none", ecolor="#555555", elinewidth=1.2, capsize=2.5, zorder=3,
    )
    ax.scatter(d_vals, y, c=colors, s=48, zorder=4, edgecolors="white", linewidths=0.5)

    ax.set_yticks(y)
    ax.set_yticklabels(sub["label"], fontsize=8.8)
    ax.set_xlabel(f"Standardized Effect Size: Cohen's d (Post minus Pre {cutoff_ts.strftime('%Y-%m')}, +/- 95% CI)")
    ax.set_title(f"Before vs After {cutoff_ts.strftime('%Y-%m')} Shift Magnitude Across All Key Metrics")

    # Annotate exact d values on the right
    x_min, x_max = ax.get_xlim()
    pad = (x_max - x_min) * 0.02
    for yi, dv in zip(y, d_vals):
        sign = "+" if dv >= 0 else ""
        ha = "left" if dv >= 0 else "right"
        xpos = dv + pad if dv >= 0 else dv - pad
        ax.text(xpos, yi, f"{sign}{dv:.2f}", va="center", ha=ha, fontsize=7.8, color="#333333")

    ax.set_xlim(x_min - pad * 4, x_max + pad * 4)
    fig.tight_layout()
    _save_fig(fig, out_dir, "15_before_after_effect_sizes", args.format)

# ---------------------------------------------------------------------------
# Time-Series Export, Summary Report & CLI
# ---------------------------------------------------------------------------

def export_aggregated_timeseries(
    df1: Optional[pd.DataFrame],
    df2: Optional[pd.DataFrame],
    freq: str,
    min_per_bin: int,
    out_dir: Path,
) -> Path:
    """Export aggregated time-series table for all key metrics to CSV."""
    frames: List[pd.DataFrame] = []

    if df1 is not None and not df1.empty:
        for metric in NUMERIC_METRICS_PASS1:
            if metric in ("pub_year", "pub_month"):
                continue
            agg = aggregate_time_series(df1, metric, freq=freq, min_per_bin=min_per_bin)
            if agg.empty:
                continue
            tmp = agg.reset_index().rename(columns={"index": "period", "month_ts": "period"})
            if "period" not in tmp.columns:
                tmp = tmp.rename(columns={tmp.columns[0]: "period"})
            tmp["source"] = "pass1"
            tmp["metric"] = metric
            tmp["label"] = METRIC_LABELS.get(metric, metric)
            frames.append(tmp)

    if df2 is not None and not df2.empty:
        p2_min = max(3, min_per_bin // 2)
        for metric in NUMERIC_METRICS_PASS2:
            if metric == "pub_year":
                continue
            agg = aggregate_time_series(df2, metric, freq=freq, min_per_bin=p2_min)
            if agg.empty:
                continue
            tmp = agg.reset_index().rename(columns={"index": "period", "month_ts": "period"})
            if "period" not in tmp.columns:
                tmp = tmp.rename(columns={tmp.columns[0]: "period"})
            tmp["source"] = "pass2"
            tmp["metric"] = metric
            tmp["label"] = METRIC_LABELS.get(metric, metric)
            frames.append(tmp)

    out_path = out_dir / "timeseries_aggregated.csv"
    if frames:
        combined = pd.concat(frames, ignore_index=True)
        combined["period"] = pd.to_datetime(combined["period"]).dt.strftime("%Y-%m-%d")
        cols = ["source", "metric", "label", "period", "count", "mean", "std", "sem", "ci95", "median", "q25", "q75"]
        combined[[c for c in cols if c in combined.columns]].to_csv(out_path, index=False)
    return out_path


def write_summary_report(
    df1: Optional[pd.DataFrame],
    df2: Optional[pd.DataFrame],
    merged: Optional[pd.DataFrame],
    stats_df: pd.DataFrame,
    cutoff_ts: pd.Timestamp,
    out_dir: Path,
) -> Path:
    """Write a concise human-readable summary of sample sizes and largest shifts."""
    lines: List[str] = []
    lines.append("=" * 78)
    lines.append("arXiv Stylometric & Reference Validation Trend Report")
    lines.append("=" * 78)
    lines.append(f"Cutoff date : {cutoff_ts.strftime('%Y-%m')}")
    if df1 is not None:
        d_min = df1["month_ts"].min()
        d_max = df1["month_ts"].max()
        lines.append(
            f"Pass 1 rows : {len(df1):,} clean papers "
            f"({d_min.strftime('%Y-%m') if pd.notna(d_min) else '?'} to "
            f"{d_max.strftime('%Y-%m') if pd.notna(d_max) else '?'})"
        )
    if df2 is not None:
        d_min = df2["month_ts"].min()
        d_max = df2["month_ts"].max()
        lines.append(
            f"Pass 2 rows : {len(df2):,} validated papers "
            f"({d_min.strftime('%Y-%m') if pd.notna(d_min) else '?'} to "
            f"{d_max.strftime('%Y-%m') if pd.notna(d_max) else '?'})"
        )
    if merged is not None and not merged.empty:
        lines.append(f"Joined rows : {len(merged):,} papers with both Pass 1 & Pass 2 data")

    if stats_df is not None and not stats_df.empty:
        lines.append("")
        lines.append("Top 15 Largest Pre- vs Post-Cutoff Shifts (by |Cohen's d|):")
        lines.append("-" * 78)
        top = stats_df.reindex(stats_df["cohens_d"].abs().sort_values(ascending=False).index).head(15)
        for _, row in top.iterrows():
            sign = "+" if row["cohens_d"] >= 0 else ""
            pct = f"{row['pct_change']:+.1f}%" if pd.notna(row.get("pct_change")) else "n/a"
            p_val = row.get("p_welch", np.nan)
            p_str = f"p={p_val:.2e}" if pd.notna(p_val) else "p=n/a"
            lines.append(
                f"  {row['metric']:<30s} pre={row['mean_pre']:8.3f} -> post={row['mean_post']:8.3f} "
                f"({pct:>7s})  d={sign}{row['cohens_d']:.3f}  [{p_str}]"
            )

    lines.append("=" * 78)
    report_text = "\n".join(lines) + "\n"
    report_path = out_dir / "summary_report.txt"
    report_path.write_text(report_text, encoding="utf-8")
    print("\n" + report_text)
    return report_path


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Pass 3: Generate longitudinal graphs and before/after statistics "
            "from Pass 1 (arxiv_metrics.csv) and Pass 2 (ref_validation.csv)."
        ),
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    io_g = parser.add_argument_group("Inputs & Output")
    io_g.add_argument(
        "--metrics-csv",
        type=str,
        default=None,
        help="Path to Pass 1 output CSV (arxiv_metrics.csv).",
    )
    io_g.add_argument(
        "--refs-csv",
        type=str,
        default=None,
        help="Path to Pass 2 reference validation CSV (ref_validation.csv).",
    )
    io_g.add_argument(
        "--output-dir",
        type=str,
        default="plots",
        help="Directory where all figures, stats CSVs, and summary report are saved.",
    )
    io_g.add_argument(
        "--format",
        choices=["png", "pdf", "svg"],
        default="png",
        help="Output image format for generated figures.",
    )
    io_g.add_argument(
        "--dpi",
        type=int,
        default=180,
        help="Figure resolution in DPI (for raster formats like png).",
    )

    time_g = parser.add_argument_group("Time Window, Cutoff & Aggregation")
    time_g.add_argument(
        "--cutoff-date",
        type=str,
        default="2022-11",
        help="Before/after cutoff date in YYYY-MM format (default: 2022-11, ChatGPT release).",
    )
    time_g.add_argument(
        "--start-date",
        type=str,
        default=None,
        help="Optional earliest YYYY-MM date to include.",
    )
    time_g.add_argument(
        "--end-date",
        type=str,
        default=None,
        help="Optional latest YYYY-MM date to include.",
    )
    time_g.add_argument(
        "--freq",
        choices=["M", "MS", "Q", "QS"],
        default="MS",
        help="Time aggregation frequency: M/MS (monthly) or Q/QS (quarterly).",
    )
    time_g.add_argument(
        "--rolling",
        type=int,
        default=3,
        help="Centered rolling smoothing window in bins (1 = no smoothing).",
    )
    time_g.add_argument(
        "--min-per-bin",
        type=int,
        default=10,
        help="Minimum number of papers required for a time bin to be plotted.",
    )

    filt_g = parser.add_argument_group("Quality & Discipline Filters")
    filt_g.add_argument(
        "--groups",
        type=str,
        default=None,
        help="Optional comma-separated arXiv discipline groups to keep (e.g., cs,math,physics,stat).",
    )
    filt_g.add_argument(
        "--min-words",
        type=int,
        default=300,
        help="Pass 1 quality filter: minimum body word count.",
    )
    filt_g.add_argument(
        "--min-alpha-ratio",
        type=float,
        default=0.0,
        help="Pass 1 quality filter: minimum alphabetic character ratio (e.g., 0.55 to drop heavy-OCR/math noise).",
    )
    filt_g.add_argument(
        "--min-refs-checked",
        type=int,
        default=1,
        help="Pass 2 quality filter: minimum checked references per paper.",
    )
    filt_g.add_argument(
        "--no-dedup",
        action="store_true",
        help="Disable automatic deduplication on (archive_name, member_name).",
    )

    args = parser.parse_args(argv)
    if not args.metrics_csv and not args.refs_csv:
        parser.error("Provide at least one of --metrics-csv (Pass 1) or --refs-csv (Pass 2).")

    # Normalize frequency aliases
    if args.freq == "M":
        args.freq = "MS"
    elif args.freq == "Q":
        args.freq = "QS"

    return args


def main(argv: Optional[Sequence[str]] = None) -> None:
    args = parse_args(argv)
    configure_matplotlib(dpi=args.dpi)

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    cutoff_ts = pd.to_datetime(args.cutoff_date[:7], format="%Y-%m")
    groups_set = None
    if args.groups:
        groups_set = {g.strip().lower() for g in args.groups.split(",") if g.strip()}

    dedup = not args.no_dedup

    df1: Optional[pd.DataFrame] = None
    df2: Optional[pd.DataFrame] = None
    merged: Optional[pd.DataFrame] = None

    if args.metrics_csv:
        df1 = load_pass1_csv(
            args.metrics_csv,
            min_words=args.min_words,
            min_alpha_ratio=args.min_alpha_ratio,
            start_date=args.start_date,
            end_date=args.end_date,
            groups=groups_set,
            deduplicate=dedup,
        )

    if args.refs_csv:
        df2 = load_pass2_csv(
            args.refs_csv,
            min_refs_checked=args.min_refs_checked,
            start_date=args.start_date,
            end_date=args.end_date,
            groups=groups_set,
            deduplicate=dedup,
        )

    if df1 is not None and not df1.empty and df2 is not None and not df2.empty:
        merged = join_pass1_and_pass2(df1, df2)

    # 1. Compute before/after statistical tables
    stats_parts: List[pd.DataFrame] = []
    if df1 is not None and not df1.empty:
        p1_metrics = [m for m in NUMERIC_METRICS_PASS1 if m not in ("pub_year", "pub_month")]
        s1 = compute_before_after_stats(df1, p1_metrics, cutoff_ts, source_label="pass1")
        if not s1.empty:
            stats_parts.append(s1)
    if df2 is not None and not df2.empty:
        p2_metrics = [m for m in NUMERIC_METRICS_PASS2 if m != "pub_year"]
        s2 = compute_before_after_stats(df2, p2_metrics, cutoff_ts, source_label="pass2")
        if not s2.empty:
            stats_parts.append(s2)

    stats_df = pd.concat(stats_parts, ignore_index=True) if stats_parts else pd.DataFrame()
    if not stats_df.empty:
        stats_path = out_dir / "before_after_stats.csv"
        stats_df.to_csv(stats_path, index=False)
        print(f"Saved statistical summary: {stats_path}")

    # 2. Export aggregated monthly/quarterly time series
    ts_path = export_aggregated_timeseries(df1, df2, args.freq, args.min_per_bin, out_dir)
    print(f"Saved aggregated time series: {ts_path}")

    # 3. Generate figures
    print(f"Generating figures in {out_dir.resolve()} ...")
    if df1 is not None and not df1.empty:
        plot_01_overview_dashboard(df1, df2, cutoff_ts, out_dir, args)
        plot_02_readability(df1, cutoff_ts, out_dir, args)
        plot_03_lexical_diversity_and_entropy(df1, cutoff_ts, out_dir, args)
        plot_04_sentence_and_burstiness(df1, cutoff_ts, out_dir, args)
        plot_05_voice_stance_punctuation(df1, cutoff_ts, out_dir, args)
        plot_06_excess_vs_control_words(df1, cutoff_ts, out_dir, args)
        plot_07_body_vs_abstract_control(df1, cutoff_ts, out_dir, args)
        plot_08_authorship_and_structure(df1, cutoff_ts, out_dir, args)
        plot_09_confounders_and_discipline_mix(df1, cutoff_ts, out_dir, args)
        plot_10_discipline_stratified_trends(df1, cutoff_ts, out_dir, args)
        plot_11_offline_references(df1, cutoff_ts, out_dir, args)

    if df2 is not None and not df2.empty:
        plot_12_live_reference_validation(df2, cutoff_ts, out_dir, args)
        plot_13_reference_validation_by_discipline(df2, cutoff_ts, out_dir, args)

    if merged is not None and not merged.empty:
        plot_14_joined_stylometry_vs_refs(merged, cutoff_ts, out_dir, args)

    if not stats_df.empty:
        plot_15_effect_size_forest(stats_df, cutoff_ts, out_dir, args)

    # 4. Write & print concise summary report
    write_summary_report(df1, df2, merged, stats_df, cutoff_ts, out_dir)
    print(f"Done. All outputs written to: {out_dir.resolve()}")


if __name__ == "__main__":
    main()

