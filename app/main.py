import os
import shutil
import uuid
import logging
from fastapi import FastAPI, UploadFile, File, HTTPException, status
from fastapi.middleware.cors import CORSMiddleware
from app.config import UPLOAD_DIR
from app.services.audio import validate_audio_file
from app.services.analyzer import analyze_speech
from app.schemas.analysis import AnalysisResponse
from app.db import init_db, save_analysis, get_analyses_history, get_analysis_detail, delete_analysis

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    handlers=[
        logging.StreamHandler()
    ]
)
logger = logging.getLogger("livospeak")

app = FastAPI(
    title="LivoSpeak AI API",
    description="Intelligent Pronunciation & Speaking Coach Backend API",
    version="1.0"
)

@app.on_event("startup")
async def startup_event():
    await init_db()

# Configure CORS
# Next.js usually runs on port 3000 by default, so we allow port 3000 and any common port
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # For local development we allow all. Can be restricted in production.
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/api/health")
def health_check():
    """Health check endpoint."""
    return {"status": "healthy", "service": "livospeak-backend"}

@app.post(
    "/api/analyze",
    response_model=AnalysisResponse,
    status_code=status.HTTP_200_OK,
    summary="Upload and analyze audio file",
    description="Accepts MP3, WAV, or M4A audio. Enforces English speech of 30-45 seconds. Automatically deletes the file after processing."
)
async def analyze_audio(file: UploadFile = File(...)):
    # 1. Generate a secure, unique filename to avoid collisions
    file_id = str(uuid.uuid4())
    original_ext = os.path.splitext(file.filename)[1].lower()
    temp_filename = f"{file_id}{original_ext}"
    temp_file_path = os.path.join(UPLOAD_DIR, temp_filename)

    logger.info(f"Received file upload: {file.filename} (MIME: {file.content_type})")

    try:
        # 2. Write file content to temporary storage
        with open(temp_file_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)

        # 3. Perform audio validations (format, size, and duration via ffprobe)
        is_valid, error_msg, duration = validate_audio_file(
            file_path=temp_file_path,
            filename=file.filename,
            mime_type=file.content_type
        )

        if not is_valid:
            logger.warning(f"File validation failed: {error_msg}")
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=error_msg
            )

        # 4. Process speech (transcription, scoring, mistakes detection, LLM feedback)
        logger.info(f"Valid audio file. Duration: {duration:.2f}s. Commencing AI speech analysis.")
        analysis_result = analyze_speech(file_path=temp_file_path, duration=duration)
        logger.info("Speech analysis completed successfully.")
        
        # 4.5. Save analysis result to history database
        try:
            analysis_id = await save_analysis(analysis_result)
            analysis_result["id"] = analysis_id
        except Exception as db_save_error:
            logger.error(f"Failed to record analysis to database: {str(db_save_error)}")
        
        return analysis_result

    except HTTPException as he:
        # Re-raise HTTP exceptions to preserve details
        raise he
    except ValueError as ve:
        # User input / validation issues (e.g. non-English language detected, no speech)
        logger.warning(f"Validation error: {str(ve)}")
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(ve)
        )
    except Exception as e:
        logger.error(f"Internal error processing audio: {str(e)}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="An error occurred while processing the speech. Please try again later."
        )
    finally:
        # 5. Privacy-first compliance: Ensure file is always deleted from temp storage
        if os.path.exists(temp_file_path):
            try:
                os.remove(temp_file_path)
                logger.info(f"Privacy clean-up: Temp file {temp_file_path} deleted successfully.")
            except Exception as cleanup_error:
                logger.error(f"Failed to delete temp file {temp_file_path}: {str(cleanup_error)}")

@app.get(
    "/api/history",
    summary="Get speech analysis history list",
    description="Returns a list of the 15 most recent speech analysis runs, including overall score, WPM, and timestamp."
)
async def get_history():
    try:
        return await get_analyses_history()
    except Exception as e:
        logger.error(f"Failed to fetch history list: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to retrieve history."
        )

@app.get(
    "/api/history/{analysis_id}",
    response_model=AnalysisResponse,
    summary="Get detailed analysis result by ID",
    description="Fetches the full speech analysis details for a previous recording using its unique ID."
)
async def get_history_detail(analysis_id: str):
    try:
        detail = await get_analysis_detail(analysis_id)
        if not detail:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="Analysis record not found."
            )
        return detail
    except HTTPException as he:
        raise he
    except Exception as e:
        logger.error(f"Failed to fetch history detail for {analysis_id}: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to retrieve record detail."
        )

@app.delete(
    "/api/history/{analysis_id}",
    summary="Delete a speech analysis run from history",
    description="Deletes a specific speaking session run by ID from the MongoDB database or In-Memory store."
)
async def delete_history_item(analysis_id: str):
    try:
        success = await delete_analysis(analysis_id)
        if not success:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="Analysis record not found."
            )
        return {"success": True, "message": "Record deleted successfully."}
    except HTTPException as he:
        raise he
    except Exception as e:
        logger.error(f"Failed to delete history item ({analysis_id}): {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to delete history item."
        )
