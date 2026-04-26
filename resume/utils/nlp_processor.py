"""
NLP preprocessing pipeline.

Steps:
1. Unicode normalisation & whitespace cleanup
2. Tokenisation (spaCy)
3. Stopword removal (spaCy + NLTK)
4. Lemmatisation (spaCy)
5. POS-filtered token extraction (nouns, proper nouns, adjectives, verbs)

Also exposes helper functions used by the scoring service:
- extract_skills()        – match against a curated tech-skills vocabulary
- extract_experience_years() – regex-based years-of-experience detection
- extract_contact_info()  – name / email / phone heuristics
"""
from __future__ import annotations

import logging
import re
import unicodedata
from functools import lru_cache
from typing import Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Known tech / soft skills vocabulary
# (extend this list or load from a DB / config file in production)
# ---------------------------------------------------------------------------

TECH_SKILLS: set[str] = {
    # Languages
    "python", "java", "javascript", "typescript", "c", "c++", "c#", "go",
    "rust", "ruby", "php", "swift", "kotlin", "scala", "r", "matlab",
    "bash", "shell", "perl", "dart", "lua",
    # Web
    "html", "css", "react", "angular", "vue", "nextjs", "nuxt", "svelte",
    "jquery", "bootstrap", "tailwind", "webpack", "vite",
    # Backend / frameworks
    "fastapi", "flask", "django", "express", "nestjs", "spring", "rails",
    "laravel", "asp.net", "nodejs", "graphql", "rest", "grpc",
    # Data / ML
    "pandas", "numpy", "scipy", "matplotlib", "seaborn", "sklearn",
    "scikit-learn", "tensorflow", "keras", "pytorch", "xgboost", "lightgbm",
    "spacy", "nltk", "transformers", "huggingface", "opencv",
    # Cloud / DevOps
    "aws", "gcp", "azure", "docker", "kubernetes", "terraform", "ansible",
    "jenkins", "github actions", "circleci", "gitlab ci", "helm",
    "linux", "nginx", "apache",
    # Databases
    "mysql", "postgresql", "mongodb", "redis", "elasticsearch", "cassandra",
    "sqlite", "oracle", "dynamodb", "firebase", "supabase",
    # Tools
    "git", "github", "gitlab", "jira", "confluence", "figma", "postman",
    # Soft skills
    "communication", "leadership", "teamwork", "problem solving",
    "critical thinking", "agile", "scrum", "kanban",
}

# Stopwords that spaCy/NLTK miss in resume context
EXTRA_STOPWORDS: set[str] = {
    "experience", "work", "responsibilities", "job", "role", "position",
    "company", "team", "project", "year", "month", "day", "time",
    "etc", "eg", "ie", "ref", "references",
}

# Regex patterns
_EMAIL_RE = re.compile(r"[a-zA-Z0-9._%+\-]+@[a-zA-Z0-9.\-]+\.[a-zA-Z]{2,}")
_PHONE_RE = re.compile(
    r"(\+?\d{1,3}[\s\-.]?)?\(?\d{2,4}\)?[\s\-.]?\d{3,4}[\s\-.]?\d{3,4}"
)
_EXP_RE = re.compile(
    r"(\d+(?:\.\d+)?)\s*\+?\s*(?:years?|yrs?)\s*(?:of\s+)?(?:experience|exp)?",
    re.IGNORECASE,
)
_NAME_LINE_RE = re.compile(r"^([A-Z][a-z]+(?:\s+[A-Z][a-z]+){1,3})$", re.MULTILINE)


# ---------------------------------------------------------------------------
# spaCy model loader (cached)
# ---------------------------------------------------------------------------


@lru_cache(maxsize=1)
def _get_nlp():  # noqa: ANN202
    """Load and cache the spaCy model. Falls back to blank if model missing."""
    import spacy

    from config import settings

    model_name = settings.spacy_model
    try:
        nlp = spacy.load(model_name, disable=["parser", "ner"])
        logger.info("Loaded spaCy model: %s", model_name)
        return nlp
    except OSError:
        logger.warning(
            "spaCy model '%s' not found. Run: python -m spacy download %s",
            model_name,
            model_name,
        )
        nlp = spacy.blank("en")
        logger.warning("Falling back to spacy.blank('en') — NLP quality reduced.")
        return nlp


@lru_cache(maxsize=1)
def _get_nltk_stopwords() -> set[str]:
    """Return NLTK English stopword set, downloading if necessary."""
    import nltk

    try:
        from nltk.corpus import stopwords as sw

        return set(sw.words("english"))
    except LookupError:
        nltk.download("stopwords", quiet=True)
        from nltk.corpus import stopwords as sw

        return set(sw.words("english"))


# ---------------------------------------------------------------------------
# Text cleaning
# ---------------------------------------------------------------------------


def clean_text(text: str) -> str:
    """
    Normalise raw extracted text:
    1. Unicode NFKC normalisation
    2. Strip non-printable characters
    3. Collapse multiple whitespace / newlines
    4. Lower-case
    """
    # Normalise unicode (handles ligatures, em-dashes, etc.)
    text = unicodedata.normalize("NFKC", text)
    # Remove non-printable chars (keep newlines and tabs)
    text = re.sub(r"[^\x09\x0A\x0D\x20-\x7E\u00A0-\uFFFF]", " ", text)
    # Collapse multiple blank lines
    text = re.sub(r"\n{3,}", "\n\n", text)
    # Collapse multiple spaces
    text = re.sub(r"[ \t]{2,}", " ", text)
    return text.strip().lower()


# ---------------------------------------------------------------------------
# Core preprocessing pipeline
# ---------------------------------------------------------------------------


def preprocess_text(raw_text: str) -> str:
    """
    Full NLP preprocessing pipeline.

    Returns a cleaned, tokenised, stopword-removed, lemmatised string
    suitable for TF-IDF vectorisation.
    """
    cleaned = clean_text(raw_text)
    nlp = _get_nlp()
    nltk_sw = _get_nltk_stopwords()
    combined_sw = nltk_sw | EXTRA_STOPWORDS

    doc = nlp(cleaned)

    tokens: List[str] = []
    for token in doc:
        # Skip punctuation, spaces, numbers-only tokens, very short tokens
        if token.is_punct or token.is_space:
            continue
        if token.is_digit or len(token.text) < 2:
            continue

        lemma = token.lemma_.lower().strip()

        # Skip stopwords
        if lemma in combined_sw or token.text.lower() in combined_sw:
            continue

        # Keep nouns, proper nouns, adjectives, verbs (skip adverbs, det, etc.)
        if token.pos_ in {"NOUN", "PROPN", "ADJ", "VERB", "X"}:
            tokens.append(lemma)

    return " ".join(tokens)


def tokenize(text: str) -> List[str]:
    """Return a list of meaningful tokens from text (post-cleaning)."""
    return preprocess_text(text).split()


# ---------------------------------------------------------------------------
# Skill extraction
# ---------------------------------------------------------------------------


def extract_skills(text: str) -> List[str]:
    """
    Extract known tech/soft skills from text using the TECH_SKILLS vocabulary.

    Returns a deduplicated, sorted list of matched skills.
    """
    lower_text = text.lower()
    found: set[str] = set()

    for skill in TECH_SKILLS:
        # Use word-boundary-like matching to avoid "c" matching "access"
        pattern = r"\b" + re.escape(skill) + r"\b"
        if re.search(pattern, lower_text):
            found.add(skill)

    return sorted(found)


# ---------------------------------------------------------------------------
# Experience detection
# ---------------------------------------------------------------------------


def extract_experience_years(text: str) -> Optional[float]:
    """
    Detect the maximum years of experience mentioned in the text.

    Returns the largest value found, or None if not detected.
    """
    matches = _EXP_RE.findall(text)
    if not matches:
        return None
    try:
        return max(float(m) for m in matches)
    except ValueError:
        return None


# ---------------------------------------------------------------------------
# Contact info extraction
# ---------------------------------------------------------------------------


def extract_contact_info(raw_text: str) -> Dict[str, Optional[str]]:
    """
    Heuristically extract name, email, and phone from raw resume text.

    Returns a dict with keys: name, email, phone (each may be None).
    """
    info: Dict[str, Optional[str]] = {"name": None, "email": None, "phone": None}

    # Email
    email_match = _EMAIL_RE.search(raw_text)
    if email_match:
        info["email"] = email_match.group(0).lower()

    # Phone
    phone_match = _PHONE_RE.search(raw_text)
    if phone_match:
        # Normalise whitespace in phone number
        info["phone"] = re.sub(r"\s+", " ", phone_match.group(0)).strip()

    # Name — first line of the resume is usually the candidate's name
    lines = [ln.strip() for ln in raw_text.split("\n") if ln.strip()]
    for line in lines[:5]:  # check first 5 non-empty lines
        name_match = _NAME_LINE_RE.match(line)
        if name_match and len(line.split()) <= 4:
            info["name"] = name_match.group(1)
            break

    return info
