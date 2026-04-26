from fastapi import APIRouter, UploadFile, File

router = APIRouter(prefix="/resumes")

@router.post("/upload_resume")
async def upload_resume(file: UploadFile = File(...)):
    return {
        "resume_id": 1,
        "filename": file.filename,
        "message": "Uploaded successfully"
    }

@router.get("")
def list_resumes():
    return []