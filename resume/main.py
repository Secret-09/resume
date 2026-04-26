from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from routes import analysis_routes, resume_routes

app = FastAPI(
    title="AI Resume Analyzer",
    description="Simple API for resume analysis",
    version="1.0.0"
)

# CORS (allow frontend)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Routers
app.include_router(resume_routes.router, prefix="/api/v1/resumes")
app.include_router(analysis_routes.router, prefix="/api/v1")

# Root
@app.get("/")
def root():
    return {"message": "API running"}

# Health check
@app.get("/health")
def health():
    return {"status": "ok"}