from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates
from fastapi.staticfiles import StaticFiles
from starlette.requests import Request
from fastapi.middleware.cors import CORSMiddleware
import os
import uuid
from pathlib import Path
from transformers import Wav2Vec2Processor, Wav2Vec2ForSequenceClassification
from .audio_detection import predict_audio
from .image_detection import detect_image_deepfake
from .config import AUDIO_MODEL_PATH, TEST_DATA_DIR, logger

app = FastAPI()

# Add CORS middleware with explicit origins
app.add_middleware(
    CORSMiddleware,
    # allow_origins=[
    #     "http://localhost:3000",
    #     "http://127.0.0.1:3000",
    #     "http://localhost:5500",
    #     "http://127.0.0.1:5500",
    #     "http://localhost:8080",
    #     "http://localhost:8000"
    # ],
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Mount templates
templates = Jinja2Templates(directory="templates")

# Mount static files
app.mount("/static", StaticFiles(directory="templates/static"), name="static")

# Ensure test_data directory exists
Path(TEST_DATA_DIR).mkdir(parents=True, exist_ok=True)

# Load audio model and processor globally
try:
    audio_processor = Wav2Vec2Processor.from_pretrained(AUDIO_MODEL_PATH)
    audio_model = Wav2Vec2ForSequenceClassification.from_pretrained(AUDIO_MODEL_PATH)
    audio_model.eval()
    logger.info(f"Loaded audio model and processor from {AUDIO_MODEL_PATH}")
except Exception as e:
    logger.error(f"Failed to load audio model or processor from {AUDIO_MODEL_PATH}: {e}")
    raise

@app.get("/", response_class=HTMLResponse)
async def get_upload_form(request: Request):
    """
    Serve the upload HTML page.
    """
    return templates.TemplateResponse("index.html", {"request": request})

@app.post("/audio/detect")
async def audio_detect(file: UploadFile = File(...)):
    """
    Endpoint for audio deepfake detection.
    """
    logger.info(f"Received audio file: {file.filename}, type: {file.content_type}, size: {file.size}")
    if not file.filename or not file.filename.lower().endswith(('.wav', '.mp3')):
        logger.error("Invalid audio file format")
        raise HTTPException(status_code=400, detail="Invalid audio file format. Please upload a WAV or MP3 file.")

    # Save uploaded file
    file_extension = os.path.splitext(file.filename or "")[1]
    unique_filename = f"uploaded_{uuid.uuid4()}{file_extension}"
    file_path = os.path.join(TEST_DATA_DIR, unique_filename)

    try:
        with open(file_path, "wb") as f:
            f.write(await file.read())

        # Run detection using predict_audio
        logger.info(f"Processing audio file: {file_path}")
        prediction, confidence, saved_file_path = predict_audio(file_path, audio_model, audio_processor)
        logger.info(f"Audio detection completed: prediction={prediction}, confidence={confidence}")
        return {
            "prediction": prediction,
            "confidence": confidence,
            "saved_file_path": unique_filename  # Return only filename
        }
    except Exception as e:
        logger.error(f"Error processing audio file {file_path}: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error processing audio: {str(e)}")

@app.post("/image/detect")
async def image_detect(file: UploadFile = File(...)):
    """
    Endpoint for image deepfake detection.
    """
    logger.info(f"Received image file: {file.filename}, type: {file.content_type}, size: {file.size}")
    if not file.filename or not file.filename.lower().endswith(('.png', '.jpg', '.jpeg')):
        logger.error("Invalid image file format")
        raise HTTPException(status_code=400, detail="Invalid image file format. Please upload a PNG, JPG, or JPEG file.")

    # Save uploaded file
    file_extension = os.path.splitext(file.filename)[1]
    unique_filename = f"image_{uuid.uuid4()}{file_extension}"
    file_path = os.path.join(TEST_DATA_DIR, unique_filename)

    try:
        with open(file_path, "wb") as f:
            f.write(await file.read())

        # Run detection
        logger.info(f"Processing image file: {file_path}")
        result = detect_image_deepfake(file_path)
        
        # Extract prediction and confidence
        prediction = result.get("prediction", "Unknown")
        confidence = result.get("confidence", "0.00%")
        
        if prediction not in ["Real", "Fake"]:
            prediction = "Unknown"
        
        logger.info(f"Image detection completed: prediction={prediction}, confidence={confidence}")
        return {
            "prediction": prediction,
            "confidence": confidence,  # Keep as percentage string (e.g., "95.23%")
            "saved_file_path": unique_filename  # Return only filename
        }
    except Exception as e:
        logger.error(f"Error processing image file {file_path}: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error processing image: {str(e)}")