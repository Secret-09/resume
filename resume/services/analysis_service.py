"""
Analysis service: orchestrates resume–JD scoring, ranking, and
persisting AnalysisResult records to the database.
"""
from __future__ import annotations

import logging
from typing import List

from sqlalchemy.orm import Session

from models.db_models import AnalysisResult, JobDescription, Resume
from models.schemas import (
    AnalyzeRequest,
    AnalyzeResponse,
    ExperienceMatch,
    ResultsFilter,
    ResultsResponse,
    SingleAnalysisResult,
    SkillsMatch,
)
from services.job_description_service import JobDescriptionService
from services.resume_service import ResumeService
from services.scoring_engine import rank_resumes, score_resume_against_jd

logger = logging.getLogger(__name__)

_resume_svc = ResumeService()
_jd_svc = JobDescriptionService()


class AnalysisService:
    """Orchestrates multi-resume analysis and result retrieval."""

    # ------------------------------------------------------------------
    # Analyze
    # ------------------------------------------------------------------

    def analyze(self, request: AnalyzeRequest, db: Session) -> AnalyzeResponse:
        """
        Score one or more resumes against a job description, rank them,
        and persist the results.

        Args:
            request: Contains resume_ids and job_description_id.
            db:      Active SQLAlchemy session.

        Returns:
            AnalyzeResponse with ranked results.

        Raises:
            ValueError: If any resume or JD is not found.
        """
        # --- 1. Fetch JD ---
        jd: JobDescription = _jd_svc.get_job_description(
            request.job_description_id, db
        )

        # --- 2. Fetch all resumes ---
        resumes: List[Resume] = []
        for rid in request.resume_ids:
            resumes.append(_resume_svc.get_resume(rid, db))

        logger.info(
            "Analyzing %d resume(s) against JD id=%d",
            len(resumes),
            jd.id,
        )

        # --- 3. Score each resume ---
        raw_scores = []
        for resume in resumes:
            scoring = score_resume_against_jd(
                resume_raw=resume.raw_text,
                resume_cleaned=resume.cleaned_text,
                resume_skills=resume.skills or [],
                jd_raw=jd.raw_text,
                jd_cleaned=jd.cleaned_text,
                jd_skills=jd.required_skills or [],
            )
            raw_scores.append((resume.id, scoring))

        # --- 4. Rank ---
        ranked = rank_resumes(raw_scores)  # [(resume_id, ScoringResult, rank)]

        # --- 5. Persist & build response ---
        results: List[SingleAnalysisResult] = []

        for resume_id, scoring, rank in ranked:
            resume = next(r for r in resumes if r.id == resume_id)

            # Upsert: delete existing result for same resume+jd pair
            db.query(AnalysisResult).filter_by(
                resume_id=resume_id,
                job_description_id=jd.id,
            ).delete(synchronize_session=False)

            ar = AnalysisResult(
                resume_id=resume_id,
                job_description_id=jd.id,
                semantic_score=round(scoring.semantic_score, 6),
                keyword_score=round(scoring.keyword_score, 6),
                final_score=round(scoring.final_score, 6),
                skills_match={
                    "matched": scoring.matched_skills,
                    "missing": scoring.missing_skills,
                    "extra": scoring.extra_skills,
                    "match_percentage": round(scoring.skills_match_pct, 2),
                },
                matched_keywords=scoring.matched_keywords,
                missing_keywords=scoring.missing_keywords,
                experience_match={
                    "years_required": scoring.years_required,
                    "years_found": scoring.years_found,
                    "meets_requirement": scoring.meets_experience,
                    "note": scoring.experience_note,
                },
                rank=rank,
            )
            db.add(ar)
            db.flush()  # get ar.id before commit

            results.append(
                self._build_single_result(ar, resume, scoring, rank)
            )

        db.commit()

        logger.info(
            "Analysis complete: JD id=%d, %d result(s) persisted",
            jd.id,
            len(results),
        )

        return AnalyzeResponse(
            job_description_id=jd.id,
            total_resumes=len(results),
            results=results,
        )

    # ------------------------------------------------------------------
    # Results retrieval
    # ------------------------------------------------------------------

    def get_results(
        self, filters: ResultsFilter, db: Session
    ) -> ResultsResponse:
        """
        Retrieve stored analysis results with optional filters.
        """
        query = (
            db.query(AnalysisResult)
            .join(Resume, AnalysisResult.resume_id == Resume.id)
            .join(
                JobDescription,
                AnalysisResult.job_description_id == JobDescription.id,
            )
            .filter(AnalysisResult.final_score >= filters.min_score / 100)
        )

        if filters.job_description_id is not None:
            query = query.filter(
                AnalysisResult.job_description_id == filters.job_description_id
            )
        if filters.resume_id is not None:
            query = query.filter(AnalysisResult.resume_id == filters.resume_id)

        total = query.count()
        rows = (
            query.order_by(AnalysisResult.final_score.desc())
            .offset(filters.offset)
            .limit(filters.limit)
            .all()
        )

        results = [
            SingleAnalysisResult(
                analysis_id=ar.id,
                resume_id=ar.resume_id,
                candidate_name=ar.resume.candidate_name,
                filename=ar.resume.original_filename,
                score=round(ar.final_score * 100, 2),
                semantic_score=round(ar.semantic_score * 100, 2),
                keyword_score=round(ar.keyword_score * 100, 2),
                skills_match=SkillsMatch(**(ar.skills_match or {})),
                experience_match=ExperienceMatch(**(ar.experience_match or {})),
                keywords={
                    "matched": ar.matched_keywords or [],
                    "missing": ar.missing_keywords or [],
                    "top_resume_keywords": [],
                },
                ranking=ar.rank,
            )
            for ar in rows
        ]

        return ResultsResponse(total=total, results=results)

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _build_single_result(
        ar: AnalysisResult,
        resume: Resume,
        scoring,
        rank: int,
    ) -> SingleAnalysisResult:
        return SingleAnalysisResult(
            analysis_id=ar.id,
            resume_id=resume.id,
            candidate_name=resume.candidate_name,
            filename=resume.original_filename,
            # Frontend-friendly: 0–100 scale
            score=round(scoring.final_score * 100, 2),
            semantic_score=round(scoring.semantic_score * 100, 2),
            keyword_score=round(scoring.keyword_score * 100, 2),
            skills_match=SkillsMatch(
                matched=scoring.matched_skills,
                missing=scoring.missing_skills,
                extra=scoring.extra_skills,
                match_percentage=round(scoring.skills_match_pct, 2),
            ),
            experience_match=ExperienceMatch(
                years_required=scoring.years_required,
                years_found=scoring.years_found,
                meets_requirement=scoring.meets_experience,
                note=scoring.experience_note,
            ),
            keywords={
                "matched": scoring.matched_keywords,
                "missing": scoring.missing_keywords,
                "top_resume_keywords": scoring.top_resume_keywords,
            },
            ranking=rank,
        )
