from fastapi import APIRouter
from models.schemas import AnalyzeRequest

router = APIRouter()

@router.post("/analyze")
def analyze(request: AnalyzeRequest):
    return {
        "job_description_id": request.job_description_id,
        "total_resumes": len(request.resume_ids),
        "results": [
            {
                "analysis_id": 1,
                "resume_id": rid,
                "candidate_name": "Demo User",
                "filename": "resume.pdf",
                "score": 80,
                "semantic_score": 78,
                "keyword_score": 82,
                "skills_match": {
                    "matched": ["python"],
                    "missing": ["aws"],
                    "extra": [],
                    "match_percentage": 70
                },
                "experience_match": {
                    "years_required": 2,
                    "years_found": 3,
                    "meets_requirement": True,
                    "note": "ok"
                },
                "keywords": {},
                "ranking": i + 1
            }
            for i, rid in enumerate(request.resume_ids)
        ]
    }

@router.get("/results")
def get_results():
    return {
        "total": 1,
        "results": []
    }