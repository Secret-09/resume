"""
ML scoring engine.

Implements:
- TF-IDF vectorisation (scikit-learn)
- Cosine similarity between resume and job description
- Keyword overlap scoring
- Weighted final score calculation
- Multi-resume ranking
"""
from __future__ import annotations

import logging
import re
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from config import settings
from utils.nlp_processor import (
    extract_experience_years,
    extract_skills,
    preprocess_text,
    tokenize,
)

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Data containers
# ---------------------------------------------------------------------------


@dataclass
class ScoringResult:
    """Raw scores and detailed breakdown for one resume–JD pair."""

    # Scores (0.0 – 1.0 internally; multiply by 100 for frontend)
    semantic_score: float = 0.0
    keyword_score: float = 0.0
    final_score: float = 0.0

    # Keyword breakdown
    matched_keywords: List[str] = field(default_factory=list)
    missing_keywords: List[str] = field(default_factory=list)
    top_resume_keywords: List[str] = field(default_factory=list)

    # Skills breakdown
    matched_skills: List[str] = field(default_factory=list)
    missing_skills: List[str] = field(default_factory=list)
    extra_skills: List[str] = field(default_factory=list)
    skills_match_pct: float = 0.0

    # Experience
    years_required: Optional[float] = None
    years_found: Optional[float] = None
    meets_experience: Optional[bool] = None
    experience_note: str = ""


# ---------------------------------------------------------------------------
# TF-IDF helpers
# ---------------------------------------------------------------------------


def _build_tfidf_vectorizer() -> TfidfVectorizer:
    """Return a configured TF-IDF vectorizer."""
    return TfidfVectorizer(
        ngram_range=(1, 2),          # unigrams + bigrams
        max_features=5_000,
        sublinear_tf=True,           # log normalization for term frequencies
        min_df=1,
        strip_accents="unicode",
        analyzer="word",
    )


def compute_tfidf_cosine(resume_text: str, jd_text: str) -> float:
    """
    Compute cosine similarity between two preprocessed texts using TF-IDF.

    Returns a float in [0.0, 1.0].
    """
    if not resume_text.strip() or not jd_text.strip():
        return 0.0

    vectorizer = _build_tfidf_vectorizer()
    try:
        tfidf_matrix = vectorizer.fit_transform([jd_text, resume_text])
        score = cosine_similarity(tfidf_matrix[0:1], tfidf_matrix[1:2])[0][0]
        return float(np.clip(score, 0.0, 1.0))
    except Exception as exc:
        logger.warning("TF-IDF cosine failed: %s", exc)
        return 0.0


def get_top_keywords(text: str, n: int = 15) -> List[str]:
    """
    Extract the top-n TF-IDF keywords from a single document.

    Returns a list of keyword strings sorted by TF-IDF score (desc).
    """
    if not text.strip():
        return []

    vectorizer = _build_tfidf_vectorizer()
    try:
        tfidf_matrix = vectorizer.fit_transform([text])
        feature_names = vectorizer.get_feature_names_out()
        scores = tfidf_matrix.toarray()[0]
        top_indices = np.argsort(scores)[::-1][:n]
        return [feature_names[i] for i in top_indices if scores[i] > 0]
    except Exception as exc:
        logger.warning("Top keyword extraction failed: %s", exc)
        return []


# ---------------------------------------------------------------------------
# Keyword matching
# ---------------------------------------------------------------------------


def compute_keyword_score(
    resume_tokens: List[str], jd_tokens: List[str]
) -> Tuple[float, List[str], List[str]]:
    """
    Compute keyword overlap score.

    Args:
        resume_tokens: Pre-processed tokens from the resume.
        jd_tokens:     Pre-processed tokens from the job description.

    Returns:
        Tuple of (score, matched_keywords, missing_keywords).
        score is in [0.0, 1.0].
    """
    if not jd_tokens:
        return 0.0, [], []

    resume_set = set(resume_tokens)
    jd_set = set(jd_tokens)

    matched = sorted(jd_set & resume_set)
    missing = sorted(jd_set - resume_set)

    score = len(matched) / len(jd_set) if jd_set else 0.0
    return float(np.clip(score, 0.0, 1.0)), matched, missing


# ---------------------------------------------------------------------------
# Skills matching
# ---------------------------------------------------------------------------


def compute_skills_match(
    resume_skills: List[str], jd_skills: List[str]
) -> Tuple[float, List[str], List[str], List[str]]:
    """
    Compute skills overlap.

    Returns:
        (match_pct, matched_skills, missing_skills, extra_skills)
        match_pct in [0.0, 100.0]
    """
    resume_set = set(resume_skills)
    jd_set = set(jd_skills)

    matched = sorted(resume_set & jd_set)
    missing = sorted(jd_set - resume_set)
    extra = sorted(resume_set - jd_set)

    pct = (len(matched) / len(jd_set) * 100) if jd_set else 0.0
    return float(pct), matched, missing, extra


# ---------------------------------------------------------------------------
# Experience matching
# ---------------------------------------------------------------------------


def compute_experience_match(
    resume_raw: str, jd_raw: str
) -> Tuple[Optional[float], Optional[float], Optional[bool], str]:
    """
    Detect and compare years of experience.

    Returns:
        (years_required, years_found, meets_requirement, note)
    """
    years_req = extract_experience_years(jd_raw)
    years_found = extract_experience_years(resume_raw)

    if years_req is None:
        return None, years_found, None, "No experience requirement detected in JD."

    if years_found is None:
        return (
            years_req,
            None,
            None,
            f"JD requires {years_req}+ years; could not detect years in resume.",
        )

    meets = years_found >= years_req
    note = (
        f"Resume shows ~{years_found} year(s); JD requires {years_req}+ year(s). "
        + ("✓ Requirement met." if meets else "✗ Below requirement.")
    )
    return years_req, years_found, meets, note


# ---------------------------------------------------------------------------
# Main scoring function
# ---------------------------------------------------------------------------


def score_resume_against_jd(
    resume_raw: str,
    resume_cleaned: str,
    resume_skills: List[str],
    jd_raw: str,
    jd_cleaned: str,
    jd_skills: List[str],
) -> ScoringResult:
    """
    Produce a full ScoringResult for one resume vs one job description.

    Uses:
    - TF-IDF cosine similarity (semantic score)
    - Keyword token overlap (keyword score)
    - Weighted combination for final score
    """
    result = ScoringResult()

    # 1. Preprocess for TF-IDF
    resume_pp = preprocess_text(resume_raw)
    jd_pp = preprocess_text(jd_raw)

    # 2. Semantic score (TF-IDF cosine)
    result.semantic_score = compute_tfidf_cosine(resume_pp, jd_pp)

    # 3. Keyword score
    resume_tokens = tokenize(resume_raw)
    jd_tokens = tokenize(jd_raw)
    kw_score, matched_kw, missing_kw = compute_keyword_score(resume_tokens, jd_tokens)
    result.keyword_score = kw_score
    result.matched_keywords = matched_kw[:30]   # cap for response size
    result.missing_keywords = missing_kw[:30]

    # 4. Top TF-IDF keywords from resume
    result.top_resume_keywords = get_top_keywords(resume_pp, n=15)

    # 5. Skills matching
    pct, matched_sk, missing_sk, extra_sk = compute_skills_match(
        resume_skills, jd_skills
    )
    result.matched_skills = matched_sk
    result.missing_skills = missing_sk
    result.extra_skills = extra_sk
    result.skills_match_pct = pct

    # 6. Experience matching
    (
        result.years_required,
        result.years_found,
        result.meets_experience,
        result.experience_note,
    ) = compute_experience_match(resume_raw, jd_raw)

    # 7. Weighted final score (kept in [0.0, 1.0])
    result.final_score = float(
        np.clip(
            settings.semantic_weight * result.semantic_score
            + settings.keyword_weight * result.keyword_score,
            0.0,
            1.0,
        )
    )

    logger.debug(
        "Score: semantic=%.3f keyword=%.3f final=%.3f",
        result.semantic_score,
        result.keyword_score,
        result.final_score,
    )
    return result


# ---------------------------------------------------------------------------
# Multi-resume ranking
# ---------------------------------------------------------------------------


def rank_resumes(
    scored_results: List[Tuple[int, ScoringResult]]
) -> List[Tuple[int, ScoringResult, int]]:
    """
    Given a list of (resume_id, ScoringResult) tuples, return them sorted
    by final_score descending with an integer rank (1 = best).

    Returns:
        List of (resume_id, ScoringResult, rank) tuples.
    """
    sorted_results = sorted(
        scored_results, key=lambda x: x[1].final_score, reverse=True
    )
    return [(rid, res, rank + 1) for rank, (rid, res) in enumerate(sorted_results)]
