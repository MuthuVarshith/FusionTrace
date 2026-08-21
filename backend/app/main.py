"""FusionTrace API: evidence-based media-authenticity analysis."""

from contextlib import asynccontextmanager
from pathlib import Path
from uuid import UUID
from typing import Optional

from fastapi import BackgroundTasks, FastAPI, File, HTTPException, UploadFile
from pydantic import BaseModel, Field
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from starlette.requests import Request

from .config import ARTIFACT_DIR, BASE_DIR
from .service import ScanService

service = ScanService()


@asynccontextmanager
async def lifespan(_: FastAPI):
    service.initialise()
    yield


app = FastAPI(title="FusionTrace", version="1.0.0", lifespan=lifespan)
app.add_middleware(
    CORSMiddleware,
    allow_origins=[origin for origin in __import__("os").getenv("FUSIONTRACE_CORS_ORIGINS", "http://localhost:8000").split(",")],
    allow_credentials=False,
    allow_methods=["GET", "POST"],
    allow_headers=["Content-Type"],
)

templates = Jinja2Templates(directory=str(BASE_DIR / "templates"))
app.mount("/static", StaticFiles(directory=str(BASE_DIR / "templates" / "static")), name="static")


@app.get("/", response_class=HTMLResponse)
async def home(request: Request):
    return templates.TemplateResponse(request, "index.html")


@app.get("/health")
async def health():
    return service.health()


@app.post("/api/scans", status_code=202)
async def create_scan(background_tasks: BackgroundTasks, file: UploadFile = File(...)):
    try:
        scan = await service.create_scan(file)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    background_tasks.add_task(service.process_scan, scan["id"])
    return scan


@app.get("/api/scans/{scan_id}")
async def get_scan(scan_id: UUID):
    scan = service.get_scan(str(scan_id))
    if scan is None:
        raise HTTPException(status_code=404, detail="Scan not found")
    return scan


class FeedbackRequest(BaseModel):
    label: str = Field(..., description="The feedback label for the scan.")
    notes: Optional[str] = Field(None, max_length=2000, description="Optional reviewer notes.")


@app.get("/api/scans")
async def list_scans(limit: int = 50):
    return service.get_recent_scans(limit=limit)


@app.post("/api/scans/{scan_id}/feedback", status_code=201)
async def submit_feedback(scan_id: UUID, request: FeedbackRequest):
    valid_labels = {"confirmed_real", "confirmed_fake", "needs_review"}
    if request.label not in valid_labels:
        raise HTTPException(status_code=400, detail=f"Invalid label. Must be one of: {', '.join(valid_labels)}")
    
    try:
        service.add_feedback(str(scan_id), request.label, request.notes)
    except ValueError as exc:
        if str(exc) == "Scan not found":
            raise HTTPException(status_code=404, detail="Scan not found")
        raise HTTPException(status_code=400, detail=str(exc))
    
    return {"status": "success"}


@app.get("/api/scans/{scan_id}/heatmap")
async def get_heatmap(scan_id: UUID):
    """Serve the GradCAM attention heatmap for an image scan (generated async after verdict)."""
    path = ARTIFACT_DIR / f"{scan_id}_heatmap.jpg"
    if not path.exists():
        raise HTTPException(status_code=404, detail="Heatmap not yet available.")
    return FileResponse(str(path), media_type="image/jpeg")


# Compatibility endpoints for the original single-request frontend/API.
@app.post("/image/detect")
async def image_detect(background_tasks: BackgroundTasks, file: UploadFile = File(...)):
    scan = await create_scan(background_tasks, file)
    return {"scan_id": scan["id"], "status": scan["status"], "detail": "Use /api/scans/{scan_id} for the report."}


@app.post("/audio/detect")
async def audio_detect(background_tasks: BackgroundTasks, file: UploadFile = File(...)):
    scan = await create_scan(background_tasks, file)
    return {"scan_id": scan["id"], "status": scan["status"], "detail": "Use /api/scans/{scan_id} for the report."}
