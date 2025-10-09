import os
import re
import logging
from typing import Optional
import requests
import uuid
import torch
import shutil
import subprocess
from faster_whisper import WhisperModel
from fastapi import Body, FastAPI, UploadFile, File, HTTPException, BackgroundTasks
from fastapi.responses import FileResponse, JSONResponse


import pdfplumber
import pytesseract
from PIL import Image
import docx
import openpyxl
import csv

# Configure logging
logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

# Global configuration
MODEL_NAME = "base.en"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
MAX_FILE_SIZE = 200 * 1024 * 1024  # 200 MB
ALLOWED_EXTENSIONS = {
    # Audio
    ".mp3", ".wav", ".m4a", ".flac",
    # Video
    ".mp4", ".mov", ".avi", ".mkv",
    # Documents
    ".pdf", ".docx", ".txt", ".xlsx", ".csv",
    # Images
    ".jpg", ".jpeg", ".png", ".tiff"
}

# Load Whisper model
try:
    whisper_model = WhisperModel(
        MODEL_NAME, device=DEVICE, compute_type="int8")
except Exception as e:
    logger.error(f"Failed to load Whisper model: {e}")
    raise

app = FastAPI()


def sanitize_filename(filename: str) -> str:
    filename = os.path.basename(filename)
    filename = re.sub(r"[^\w\-\.]", "", filename)
    return filename


def validate_file(file: UploadFile):
    ext = os.path.splitext(file.filename)[1].lower()
    if ext not in ALLOWED_EXTENSIONS:
        raise HTTPException(
            status_code=400,
            detail=f"File type {ext} not allowed. Supported types: {', '.join(ALLOWED_EXTENSIONS)}",
        )
    file.file.seek(0, 2)
    file_size = file.file.tell()
    file.file.seek(0)
    if file_size > MAX_FILE_SIZE:
        raise HTTPException(
            status_code=400,
            detail=f"File too large. Max size is {MAX_FILE_SIZE / (1024*1024)} MB",
        )


def transcribe_audio(audio_path: str):
    try:
        segments, info = whisper_model.transcribe(audio_path, beam_size=5)
        transcription = " ".join([seg.text.strip()
                                 for seg in segments if seg.text.strip()])
        return transcription or "No speech detected."
    except Exception as e:
        logger.error(f"Transcription failed: {e}")
        raise HTTPException(status_code=500, detail="Transcription failed")


def extract_text_from_pdf(pdf_path: str):
    text = ""
    try:
        with pdfplumber.open(pdf_path) as pdf:
            for page in pdf.pages:
                text += (page.extract_text() or "") + "\n"
        return text.strip() if text else "No text found in PDF."
    except Exception as e:
        logger.error(f"PDF extraction failed: {e}")
        raise HTTPException(
            status_code=500, detail="PDF text extraction failed")


def extract_text_from_image(img_path: str):
    try:
        text = pytesseract.image_to_string(Image.open(img_path))
        return text.strip() if text else "No text found in image."
    except Exception as e:
        logger.error(f"OCR failed: {e}")
        raise HTTPException(status_code=500, detail="Image OCR failed")


def extract_text_from_docx(docx_path: str):
    try:
        doc = docx.Document(docx_path)
        text = "\n".join([p.text for p in doc.paragraphs])
        return text.strip() if text else "No text found in Word file."
    except Exception as e:
        logger.error(f"DOCX extraction failed: {e}")
        raise HTTPException(
            status_code=500, detail="DOCX text extraction failed")


def extract_text_from_txt(txt_path: str):
    try:
        with open(txt_path, "r", encoding="utf-8", errors="ignore") as f:
            text = f.read()
        return text.strip() if text else "No text found in TXT file."
    except Exception as e:
        logger.error(f"TXT extraction failed: {e}")
        raise HTTPException(status_code=500, detail="TXT extraction failed")


def extract_text_from_xlsx(xlsx_path: str):
    try:
        wb = openpyxl.load_workbook(xlsx_path, data_only=True)
        text = []
        for sheet in wb.worksheets:
            for row in sheet.iter_rows(values_only=True):
                row_text = " ".join([str(cell)
                                    for cell in row if cell is not None])
                if row_text.strip():
                    text.append(row_text)
        return "\n".join(text) if text else "No text found in Excel file."
    except Exception as e:
        logger.error(f"XLSX extraction failed: {e}")
        raise HTTPException(
            status_code=500, detail="XLSX text extraction failed")


def extract_text_from_csv(csv_path: str):
    try:
        text = []
        with open(csv_path, newline="", encoding="utf-8", errors="ignore") as csvfile:
            reader = csv.reader(csvfile)
            for row in reader:
                row_text = " ".join(row)
                if row_text.strip():
                    text.append(row_text)
        return "\n".join(text) if text else "No text found in CSV file."
    except Exception as e:
        logger.error(f"CSV extraction failed: {e}")
        raise HTTPException(
            status_code=500, detail="CSV text extraction failed")


@app.post("/transcribe")
async def process_file(file: UploadFile = File(...)):
    try:
        validate_file(file)
        safe_filename = sanitize_filename(file.filename)
        os.makedirs("temp", exist_ok=True)
        file_path = f"temp/{safe_filename}"
        with open(file_path, "wb") as f:
            f.write(await file.read())

        ext = os.path.splitext(file_path)[1].lower()
        transcription = ""

        if ext in {".mp3", ".wav", ".m4a", ".flac"}:  # Audio
            transcription = transcribe_audio(file_path)

        elif ext in {".mp4", ".mov", ".avi", ".mkv"}:  # Video
            audio_path = file_path + ".wav"
            subprocess.run(
                ["ffmpeg", "-i", file_path, "-vn", "-acodec",
                    "pcm_s16le", "-ar", "16000", audio_path],
                check=True
            )
            transcription = transcribe_audio(audio_path)

        elif ext == ".pdf":
            transcription = extract_text_from_pdf(file_path)

        elif ext in {".jpg", ".jpeg", ".png", ".tiff"}:
            transcription = extract_text_from_image(file_path)

        elif ext == ".docx":
            transcription = extract_text_from_docx(file_path)

        elif ext == ".txt":
            transcription = extract_text_from_txt(file_path)

        elif ext == ".xlsx":
            transcription = extract_text_from_xlsx(file_path)

        elif ext == ".csv":
            transcription = extract_text_from_csv(file_path)

        shutil.rmtree("temp")
        return JSONResponse({"filename": safe_filename, "transcription": transcription})

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Unexpected error: {e}")
        raise HTTPException(status_code=500, detail="Internal server error")

        # =================================================================================


model_cache = {}


def get_model(model_size: str = "base"):
    """Get or initialize the Whisper model"""
    if model_size not in model_cache:
        print(f"Loading Whisper model: {model_size}")
        model_cache[model_size] = WhisperModel(model_size)
        print(f"Model {model_size} loaded successfully")
    return model_cache[model_size]


# @app.post("/transcribe-audio")
# async def transcribe_audio(
#     background_tasks: BackgroundTasks,
#     audio_file: UploadFile = File(...),
#     model_size: str = "base",
#     language: Optional[str] = None,
#     task: str = "transcribe"
# ):
#     """
#     Transcribe audio file and generate subtitles in VTT format
#     """
#     # Validate file type
#     allowed_extensions = {'.wav', '.mp3', '.m4a',
#                           '.flac', '.aac', '.ogg', '.mpeg', '.webm'}
#     file_extension = os.path.splitext(audio_file.filename)[1].lower()

#     if file_extension not in allowed_extensions:
#         raise HTTPException(
#             status_code=400,
#             detail=f"Unsupported file type. Allowed types: {', '.join(allowed_extensions)}"
#         )

#     # Create temporary directory if it doesn't exist
#     temp_dir = "temp_audio"
#     os.makedirs(temp_dir, exist_ok=True)

#     # Generate unique filename
#     file_id = str(uuid.uuid4())
#     temp_audio_path = os.path.join(temp_dir, f"{file_id}{file_extension}")
#     vtt_filename = f"{file_id}.vtt"
#     vtt_path = os.path.join(temp_dir, vtt_filename)

#     try:
#         # Save uploaded file
#         with open(temp_audio_path, "wb") as buffer:
#             content = await audio_file.read()
#             buffer.write(content)

#         # Get the model
#         transcribe_model = get_model(model_size)

#         # Transcribe audio
#         segments, info = transcribe_model.transcribe(
#             temp_audio_path,
#             language=language,
#             task=task
#         )

#         # Convert segments to list to avoid generator exhaustion
#         segments_list = list(segments)

#         # Generate VTT file
#         with open(vtt_path, "w", encoding="utf-8") as vtt_file:
#             # Write VTT header
#             vtt_file.write("WEBVTT\n\n")

#             for i, segment in enumerate(segments_list, start=1):
#                 # Convert seconds to VTT time format (HH:MM:SS.mmm)
#                 start_time = format_time_vtt(segment.start)
#                 end_time = format_time_vtt(segment.end)

#                 # Write VTT entry
#                 vtt_file.write(f"{i}\n")
#                 vtt_file.write(f"{start_time} --> {end_time}\n")
#                 vtt_file.write(f"{segment.text}\n\n")

#         # Clean up audio file immediately
#         if os.path.exists(temp_audio_path):
#             os.remove(temp_audio_path)

#         # Add cleanup task for VTT file
#         background_tasks.add_task(cleanup_vtt_file, vtt_path)

#         # Return the VTT file
#         return FileResponse(
#             path=vtt_path,
#             media_type='text/vtt',
#             filename=f"transcription_{os.path.splitext(audio_file.filename)[0]}.vtt"
#         )

#     except Exception as e:
#         # Cleanup on error
#         try:
#             if os.path.exists(temp_audio_path):
#                 os.remove(temp_audio_path)
#             if os.path.exists(vtt_path):
#                 os.remove(vtt_path)
#         except:
#             pass
#         raise HTTPException(
#             status_code=500, detail=f"Transcription failed: {str(e)}")


# def cleanup_vtt_file(file_path: str):
#     """Clean up VTT file after download"""
#     try:
#         if os.path.exists(file_path):
#             os.remove(file_path)
#             print(f"Cleaned up VTT file: {file_path}")
#     except Exception as e:
#         print(f"Error cleaning up VTT file: {e}")


def format_time_vtt(seconds: float) -> str:
    """
    Convert seconds to VTT time format: HH:MM:SS.mmm
    """
    hours = int(seconds // 3600)
    minutes = int((seconds % 3600) // 60)
    seconds_remainder = seconds % 60
    milliseconds = int((seconds_remainder - int(seconds_remainder)) * 1000)

    return f"{hours:02d}:{minutes:02d}:{int(seconds_remainder):02d}.{milliseconds:03d}"


STRAPI_URL = os.getenv("STRAPI_URL")
STRAPI_TOKEN = os.getenv("STRAPI_TOKEN")


@app.post("/transcribe-audio")
async def transcribe_audio(
    background_tasks: BackgroundTasks,
    audio_url: str = Body(...),
    audio_id: str = Body(...),
    model_size: str = Body("base"),
    language: Optional[str] = Body(None),
    task: str = Body("transcribe")
):
    log_prefix = "[🎧 Subtitle Generator]"
    audio_url = audio_url
    audio_id = audio_id
    model_size = model_size
    language = language
    task = task

    print(f"{log_prefix} Started for audio_id: {audio_id}")

    temp_dir = "temp_audio"
    os.makedirs(temp_dir, exist_ok=True)

    file_id = str(uuid.uuid4())
    temp_audio_path = os.path.join(temp_dir, f"{file_id}.mp3")
    vtt_path = os.path.join(temp_dir, f"{file_id}.vtt")

    try:
        # 1️⃣ Download audio file
        print(f"{log_prefix} Downloading from {audio_url}")
        audio_res = requests.get(audio_url, stream=True, timeout=60)
        if audio_res.status_code != 200:
            raise HTTPException(
                status_code=400, detail=f"Failed to download audio: {audio_res.status_code}")

        with open(temp_audio_path, "wb") as f:
            for chunk in audio_res.iter_content(chunk_size=8192):
                f.write(chunk)

        # 2️⃣ Load Whisper model
        print(f"{log_prefix} Loading model: {model_size}")
        transcribe_model = get_model(model_size)

        # 3️⃣ Transcribe audio
        print(f"{log_prefix} Transcribing audio...")
        segments, info = transcribe_model.transcribe(
            temp_audio_path,
            language=language,
            task=task
        )

        segments_list = list(segments)

        # 4️⃣ Generate VTT file
        print(f"{log_prefix} Generating VTT...")
        with open(vtt_path, "w", encoding="utf-8") as vtt_file:
            vtt_file.write("WEBVTT\n\n")
            for i, segment in enumerate(segments_list, start=1):
                start_time = format_time_vtt(segment.start)
                end_time = format_time_vtt(segment.end)
                text = segment.text.strip()
                vtt_file.write(f"{i}\n{start_time} --> {end_time}\n{text}\n\n")

        # 5️⃣ Upload VTT to Strapi
        print(f"{log_prefix} Uploading VTT to Strapi...")
        with open(vtt_path, "rb") as file_data:
            files = {"files": (os.path.basename(
                vtt_path), file_data, "text/vtt")}
            headers = {"Authorization": f"Bearer {STRAPI_TOKEN}"}
            upload_res = requests.post(
                f"{STRAPI_URL}/api/upload", headers=headers, files=files)

        if upload_res.status_code not in (200, 201):
            raise Exception(f"Upload failed: {upload_res.text}")

        # ✅ Handle Strapi upload response
        try:
            uploaded_json = upload_res.json()
            if isinstance(uploaded_json, list) and len(uploaded_json) > 0:
                uploaded_vtt = uploaded_json[0]
            elif isinstance(uploaded_json, dict) and "id" in uploaded_json:
                uploaded_vtt = uploaded_json
            else:
                raise ValueError(
                    f"Unexpected upload response format: {uploaded_json}")
        except Exception as e:
            raise Exception(f"Upload succeeded but parsing failed: {e}")

        subtitle_file_id = uploaded_vtt["id"]
        print(f"{log_prefix} ✅ Uploaded VTT file (id={subtitle_file_id})")

        # 6️⃣ Create subtitle entry in Strapi
        print(f"{log_prefix} Creating subtitle entry in Strapi...")
        subtitle_payload = {
            "data": {
                "subtitle": subtitle_file_id,
                "audio": audio_id
            }
        }

        headers = {
            "Authorization": f"Bearer {STRAPI_TOKEN}",
            "Content-Type": "application/json"
        }

        create_res = requests.post(
            f"{STRAPI_URL}/api/subtitles",
            headers=headers,
            json=subtitle_payload
        )

        if create_res.status_code not in (200, 201):
            raise Exception(
                f"Subtitle entry creation failed: {create_res.text}")

        print(f"{log_prefix} ✅ Subtitle entry created successfully")

        # 7️⃣ Cleanup after success
        background_tasks.add_task(cleanup_files, [temp_audio_path, vtt_path])

        # 8️⃣ Return success JSON
        return JSONResponse({
            "success": True,
            "subtitle_file": uploaded_vtt,
            "subtitle_entry": create_res.json()
        })

    except Exception as e:
        print(f"{log_prefix} ❌ Error: {e}")
        cleanup_files([temp_audio_path, vtt_path])
        raise HTTPException(
            status_code=500, detail=f"Transcription failed: {e}")


def cleanup_files(file_paths: list[str]):
    for p in file_paths:
        try:
            if os.path.exists(p):
                os.remove(p)
                print(f"🧹 Cleaned up: {p}")
        except Exception as e:
            print(f"Cleanup error for {p}: {e}")
