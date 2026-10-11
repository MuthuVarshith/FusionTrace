"""FusionTrace API: evidence-based media-authenticity analysis.

Routes
------
GET  /                              web UI
GET  /health                        liveness + detector/feature availability (no auth)
POST /api/scans                     multipart: ``file`` + optional ``generate_ai_summary`` (bool, default false)
GET  /api/scans                     the caller's recent scans
GET  /api/scans/{id}                one report
GET  /api/scans/{id}/heatmap        explanation heatmap JPEG once ``report.heatmap_ready``
POST /api/scans/{id}/feedback       human review label

All ``/api/*`` routes require ``X-API-Key`` when ``FUSIONTRACE_API_KEYS`` is set
(see ``app.auth``). Uploads and feedback are rate limited (``app.ratelimit``).
Error responses carry generic, client-safe messages; details go to the log.
"""

from __future__ import annotations

import logging
from contextlib import asynccontextmanager
from typing import Annotated, Any
from uuid import UUID

from fastapi import APIRouter, Depends, FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from pydantic import BaseModel, Field
from starlette.datastructures import FormData, UploadFile
from starlette.requests import Request
from starlette.types import Message

from . import config as settings
from .ai_summary import summaries_available
from .auth import API_KEY_HEADER, Caller, auth_enabled, require_caller
from .config import BASE_DIR
from .jobs import QueueFull, ScanQueue
from .ratelimit import RateLimiter, retry_after_header
from .service import ScanService
from .validation import FileTooLarge, UploadRejected

logger = logging.getLogger("fusiontrace")

CallerDep = Annotated[Caller, Depends(require_caller)]

service = ScanService()
scan_queue: ScanQueue | None = None
upload_limiter = RateLimiter(settings.UPLOAD_RATE_LIMIT, settings.RATE_LIMIT_WINDOW_SECONDS)
feedback_limiter = RateLimiter(settings.FEEDBACK_RATE_LIMIT, settings.RATE_LIMIT_WINDOW_SECONDS)

_FEEDBACK_LABELS = ("confirmed_real", "confirmed_fake", "needs_review")
_MULTIPART_SLACK_BYTES = 1024 * 1024  # boundaries and form fields around the file


def _process(scan_id: str) -> None:
    service.process_scan(scan_id)  # resolved at call time so tests can swap ``service``


@asynccontextmanager
async def lifespan(_: FastAPI):
    global scan_queue
    service.initialise()
    scan_queue = ScanQueue(_process, workers=settings.MAX_CONCURRENT_SCANS, max_pending=settings.MAX_QUEUED_SCANS)
    scan_queue.start()
    try:
        yield
    finally:
        scan_queue.stop()


app = FastAPI(title="FusionTrace", version="2.0.0", lifespan=lifespan)
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.CORS_ORIGINS,
    allow_credentials=False,
    allow_methods=["GET", "POST"],
    allow_headers=["Content-Type", API_KEY_HEADER],
)

templates = Jinja2Templates(directory=str(BASE_DIR / "templates"))
app.mount("/static", StaticFiles(directory=str(BASE_DIR / "templates" / "static")), name="static")


@app.exception_handler(Exception)
async def unexpected_error(request: Request, exc: Exception) -> JSONResponse:
    logger.exception("Unhandled error on %s %s", request.method, request.url.path)
    return JSONResponse({"detail": "Internal server error."}, status_code=500)


def _enforce(limiter: RateLimiter, caller: Caller) -> None:
    wait = limiter.check(caller.rate_key)
    if wait is not None:
        raise HTTPException(status_code=429, detail="Too many requests. Please wait and try again.", headers=retry_after_header(wait))


def _form_bool(value: Any) -> bool:
    return isinstance(value, str) and value.strip().lower() in {"1", "true", "yes", "on"}


async def _read_form(request: Request, limit_bytes: int) -> FormData:
    """Parse the multipart body, aborting as soon as it exceeds ``limit_bytes``."""
    declared = request.headers.get("content-length", "")
    if declared.isdigit() and int(declared) > limit_bytes:
        raise FileTooLarge(f"File exceeds the {settings.MAX_UPLOAD_MB} MB upload limit.")
    received = 0
    receive = request.receive

    async def limited_receive() -> Message:
        nonlocal received
        message = await receive()
        if message["type"] == "http.request":
            received += len(message.get("body", b""))
            if received > limit_bytes:
                raise FileTooLarge(f"File exceeds the {settings.MAX_UPLOAD_MB} MB upload limit.")
        return message

    return await Request(request.scope, limited_receive).form(max_files=1, max_fields=8)


# ── pages ───────────────────────────────────────────────────────────────────


@app.get("/", response_class=HTMLResponse)
async def home(request: Request):
    return templates.TemplateResponse(
        request,
        "index.html",
        {
            "ai_summary_available": summaries_available(),
            "auth_required": auth_enabled(),
            "retention_hours": settings.RETENTION_HOURS,
            "max_upload_mb": settings.MAX_UPLOAD_MB,
        },
    )


@app.get("/health")
async def health():
    report = service.health()
    report["auth_required"] = auth_enabled()
    report["queue"] = {"pending": scan_queue.pending(), "active": scan_queue.active()} if scan_queue else None
    return report


# ── API ─────────────────────────────────────────────────────────────────────

api = APIRouter(prefix="/api", dependencies=[Depends(require_caller)])

_UPLOAD_OPENAPI = {
    "requestBody": {
        "required": True,
        "content": {
            "multipart/form-data": {
                "schema": {
                    "type": "object",
                    "required": ["file"],
                    "properties": {
                        "file": {"type": "string", "format": "binary"},
                        "generate_ai_summary": {
                            "type": "boolean",
                            "default": False,
                            "description": "Opt in to a Gemini summary of the detector scores (never the media). Ignored when unavailable.",
                        },
                    },
                }
            }
        },
    }
}


@api.post("/scans", status_code=202, openapi_extra=_UPLOAD_OPENAPI)
async def create_scan(request: Request, caller: CallerDep):
    # Auth, rate limit and queue capacity are checked before the body is read.
    _enforce(upload_limiter, caller)
    if scan_queue is None or not scan_queue.has_capacity():
        raise HTTPException(status_code=503, detail="The server is busy. Please try again shortly.", headers={"Retry-After": "30"})
    try:
        form = await _read_form(request, settings.MAX_UPLOAD_MB * 1024 * 1024 + _MULTIPART_SLACK_BYTES)
    except UploadRejected as exc:
        raise HTTPException(status_code=exc.status_code, detail=str(exc)) from exc
    try:
        upload = form.get("file")
        if not isinstance(upload, UploadFile):
            raise HTTPException(status_code=400, detail="Attach the media file in the 'file' form field.")
        generate_ai_summary = _form_bool(form.get("generate_ai_summary"))
        try:
            scan = await service.create_scan(upload, generate_ai_summary=generate_ai_summary, owner=caller.owner)
        except UploadRejected as exc:
            raise HTTPException(status_code=exc.status_code, detail=str(exc)) from exc
    finally:
        await form.close()
    try:
        scan_queue.submit(scan["id"])
    except QueueFull as exc:
        service.fail_scan(scan["id"], "The server was busy. Please upload the file again.")
        raise HTTPException(status_code=503, detail="The server is busy. Please try again shortly.", headers={"Retry-After": "30"}) from exc
    return scan


@api.get("/scans")
async def list_scans(caller: CallerDep, limit: Annotated[int, Query(ge=1, le=200)] = 50):
    return service.get_recent_scans(limit=limit, owner=caller.owner)


@api.get("/scans/{scan_id}")
async def get_scan(scan_id: UUID, caller: CallerDep):
    scan = service.get_scan(str(scan_id), owner=caller.owner)
    if scan is None:
        raise HTTPException(status_code=404, detail="Scan not found")
    return scan


class FeedbackRequest(BaseModel):
    label: str = Field(..., description="One of confirmed_real, confirmed_fake, needs_review.")
    notes: str | None = Field(None, max_length=2000, description="Optional reviewer notes.")


@api.post("/scans/{scan_id}/feedback", status_code=201)
async def submit_feedback(scan_id: UUID, request: FeedbackRequest, caller: CallerDep):
    _enforce(feedback_limiter, caller)
    if request.label not in _FEEDBACK_LABELS:
        raise HTTPException(status_code=400, detail=f"Invalid label. Must be one of: {', '.join(_FEEDBACK_LABELS)}")
    try:
        service.add_feedback(str(scan_id), request.label, request.notes, owner=caller.owner)
    except ValueError as exc:
        if str(exc) == "Scan not found":
            raise HTTPException(status_code=404, detail="Scan not found") from exc
        raise HTTPException(status_code=400, detail="Invalid feedback.") from exc
    return {"status": "success"}


@api.get("/scans/{scan_id}/heatmap")
async def get_heatmap(scan_id: UUID, caller: CallerDep):
    """Serve the explanation heatmap for an image scan (generated after the verdict)."""
    scan = service.get_scan(str(scan_id), owner=caller.owner)
    if scan is None:
        raise HTTPException(status_code=404, detail="Scan not found")
    path = service.heatmap_path(str(scan_id))
    if not (scan.get("report") or {}).get("heatmap_ready") or not path.exists():
        raise HTTPException(status_code=404, detail="Heatmap not available.")
    return FileResponse(str(path), media_type="image/jpeg", headers={"Cache-Control": "private, no-store"})


app.include_router(api)
