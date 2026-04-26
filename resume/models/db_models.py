"""
SQLAlchemy ORM models for the Resume Analyzer database.
Tables: resumes, job_descriptions, analysis_results
"""
from datetime import datetime
from typing import Optional

from sqlalchemy import (
    Column,
    DateTime,
    Float,
    ForeignKey,
    Integer,
    JSON,
    String,
    Text,
    func,
)
from sqlalchemy.orm import DeclarativeBase, Mapped, relationship


class Base(DeclarativeBase):
    pass


class Resume(Base):
    """Stores uploaded resumes and their extracted text."""

    __tablename__ = "resumes"

    id: Mapped[int] = Column(Integer, primary_key=True, autoincrement=True)
    filename: Mapped[str] = Column(String(255), nullable=False)
    original_filename: Mapped[str] = Column(String(255), nullable=False)
    file_type: Mapped[str] = Column(String(10), nullable=False)  # pdf | docx
    file_path: Mapped[str] = Column(String(512), nullable=False)
    raw_text: Mapped[str] = Column(Text, nullable=False)
    cleaned_text: Mapped[str] = Column(Text, nullable=False)
    candidate_name: Mapped[Optional[str]] = Column(String(255), nullable=True)
    email: Mapped[Optional[str]] = Column(String(255), nullable=True)
    phone: Mapped[Optional[str]] = Column(String(50), nullable=True)
    skills: Mapped[Optional[dict]] = Column(JSON, nullable=True)  # list of skills
    word_count: Mapped[int] = Column(Integer, default=0)
    created_at: Mapped[datetime] = Column(
        DateTime, server_default=func.now(), nullable=False
    )
    updated_at: Mapped[datetime] = Column(
        DateTime, server_default=func.now(), onupdate=func.now(), nullable=False
    )

    # Relationships
    analysis_results = relationship(
        "AnalysisResult", back_populates="resume", cascade="all, delete-orphan"
    )

    def __repr__(self) -> str:
        return f"<Resume id={self.id} filename={self.original_filename!r}>"


class JobDescription(Base):
    """Stores job descriptions submitted for analysis."""

    __tablename__ = "job_descriptions"

    id: Mapped[int] = Column(Integer, primary_key=True, autoincrement=True)
    title: Mapped[Optional[str]] = Column(String(255), nullable=True)
    company: Mapped[Optional[str]] = Column(String(255), nullable=True)
    raw_text: Mapped[str] = Column(Text, nullable=False)
    cleaned_text: Mapped[str] = Column(Text, nullable=False)
    required_skills: Mapped[Optional[dict]] = Column(JSON, nullable=True)
    word_count: Mapped[int] = Column(Integer, default=0)
    created_at: Mapped[datetime] = Column(
        DateTime, server_default=func.now(), nullable=False
    )

    # Relationships
    analysis_results = relationship(
        "AnalysisResult", back_populates="job_description", cascade="all, delete-orphan"
    )

    def __repr__(self) -> str:
        return f"<JobDescription id={self.id} title={self.title!r}>"


class AnalysisResult(Base):
    """Stores analysis scores linking a resume to a job description."""

    __tablename__ = "analysis_results"

    id: Mapped[int] = Column(Integer, primary_key=True, autoincrement=True)
    resume_id: Mapped[int] = Column(
        Integer, ForeignKey("resumes.id", ondelete="CASCADE"), nullable=False
    )
    job_description_id: Mapped[int] = Column(
        Integer,
        ForeignKey("job_descriptions.id", ondelete="CASCADE"),
        nullable=False,
    )

    # Core scores (0.0 – 1.0)
    semantic_score: Mapped[float] = Column(Float, nullable=False, default=0.0)
    keyword_score: Mapped[float] = Column(Float, nullable=False, default=0.0)
    final_score: Mapped[float] = Column(Float, nullable=False, default=0.0)

    # Detailed breakdown
    skills_match: Mapped[Optional[dict]] = Column(JSON, nullable=True)
    matched_keywords: Mapped[Optional[dict]] = Column(JSON, nullable=True)
    missing_keywords: Mapped[Optional[dict]] = Column(JSON, nullable=True)
    experience_match: Mapped[Optional[dict]] = Column(JSON, nullable=True)

    # Ranking within a job description batch
    rank: Mapped[Optional[int]] = Column(Integer, nullable=True)

    created_at: Mapped[datetime] = Column(
        DateTime, server_default=func.now(), nullable=False
    )

    # Relationships
    resume = relationship("Resume", back_populates="analysis_results")
    job_description = relationship(
        "JobDescription", back_populates="analysis_results"
    )

    def __repr__(self) -> str:
        return (
            f"<AnalysisResult resume={self.resume_id} "
            f"jd={self.job_description_id} score={self.final_score:.2f}>"
        )
