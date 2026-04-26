"""
Pydantic v2 schemas for API request validation and response serialization.
All response models are structured for direct consumption by a Next.js frontend.
"""
from datetime import datetime
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field, field_validator


# ---------------------------------------------------------------------------
# Shared / base schemas
# ---------------------------------------------------------------------------


class SkillsMatch(BaseModel):
    matched: List[str] = Field(default_factory=list, description="Skills present in both resume and JD")
    missing: List[str] = Field(default_factory=list, description="Skills required by JD but absent from resume")
    extra: List[str] = Field(default_factory=list, description="Skills on resume not mentioned in JD")
    match_percentage: float = Field(0.0, ge=0, le=100, description="Percentage of required skills matched")


class ExperienceMatch(BaseModel):
    years_required: Optional[float] = Field(None, description="Years of experience required by JD")
    years_found: Optional[float] = Field(None, description="Years of experience detected in resume")
    meets_requirement: Optional[bool] = Field(None, description="Whether resume meets the experience requirement")
    note: str = Field("", description="Human-readable note about experience match")


# ---------------------------------------------------------------------------
# Resume schemas
# ---------------------------------------------------------------------------


class ResumeUploadResponse(BaseModel):
    resume_id: int
    filename: str
    candidate_name: Optional[str] = None
    email: Optional[str] = None
    phone: Optional[str] = None
    skills: List[str] = Field(default_factory=list)
    word_count: int
    message: str = "Resume uploaded and processed successfully"


class ResumeDetail(BaseModel):
    resume_id: int
    filename: str
    candidate_name: Optional[str] = None
    email: Optional[str] = None
    skills: List[str] = Field(default_factory=list)
    word_count: int
    created_at: datetime

    model_config = {"from_attributes": True}


# ---------------------------------------------------------------------------
# Job Description schemas
# ---------------------------------------------------------------------------


class JobDescriptionRequest(BaseModel):
    title: Optional[str] = Field(None, max_length=255, examples=["Senior Python Developer"])
    company: Optional[str] = Field(None, max_length=255, examples=["Acme Corp"])
    text: str = Field(..., min_length=50, description="Full job description text")

    @field_validator("text")
    @classmethod
    def text_must_not_be_empty(cls, v: str) -> str:
        if not v.strip():
            raise ValueError("Job description text cannot be empty")
        return v.strip()


class JobDescriptionResponse(BaseModel):
    job_description_id: int
    title: Optional[str] = None
    company: Optional[str] = None
    required_skills: List[str] = Field(default_factory=list)
    word_count: int
    message: str = "Job description processed successfully"


# ---------------------------------------------------------------------------
# Analysis schemas
# ---------------------------------------------------------------------------


class AnalyzeRequest(BaseModel):
    resume_ids: List[int] = Field(..., min_length=1, description="One or more resume IDs to analyze")
    job_description_id: int = Field(..., description="Target job description ID")

    @field_validator("resume_ids")
    @classmethod
    def no_duplicate_ids(cls, v: List[int]) -> List[int]:
        if len(v) != len(set(v)):
            raise ValueError("resume_ids must not contain duplicates")
        return v


class SingleAnalysisResult(BaseModel):
    """Result for one resume vs one job description."""

    analysis_id: int
    resume_id: int
    candidate_name: Optional[str] = None
    filename: str

    # Scores
    score: float = Field(..., ge=0, le=100, description="Weighted final score (0–100)")
    semantic_score: float = Field(..., ge=0, le=100, description="Cosine similarity score (0–100)")
    keyword_score: float = Field(..., ge=0, le=100, description="Keyword match score (0–100)")

    # Breakdown
    skills_match: SkillsMatch
    experience_match: ExperienceMatch
    keywords: Dict[str, Any] = Field(
        default_factory=dict,
        description="Keyword analysis: matched, missing, top_resume_keywords",
    )

    # Ranking (populated when multiple resumes are analyzed together)
    ranking: Optional[int] = Field(None, description="Rank among all analyzed resumes (1 = best)")

    model_config = {"from_attributes": True}


class AnalyzeResponse(BaseModel):
    job_description_id: int
    total_resumes: int
    results: List[SingleAnalysisResult]
    message: str = "Analysis completed successfully"


# ---------------------------------------------------------------------------
# Results / GET schemas
# ---------------------------------------------------------------------------


class ResultsFilter(BaseModel):
    job_description_id: Optional[int] = None
    resume_id: Optional[int] = None
    min_score: float = Field(0.0, ge=0, le=100)
    limit: int = Field(50, ge=1, le=200)
    offset: int = Field(0, ge=0)


class ResultsResponse(BaseModel):
    total: int
    results: List[SingleAnalysisResult]


# ---------------------------------------------------------------------------
# Health check
# ---------------------------------------------------------------------------


class HealthResponse(BaseModel):
    status: str = "ok"
    version: str
    database: str = "connected"
    nlp_model: str


# ---------------------------------------------------------------------------
# Error schema
# ---------------------------------------------------------------------------


class ErrorResponse(BaseModel):
    error: str
    detail: Optional[str] = None
    code: int
