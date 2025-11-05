import os
import re
import logging
from typing import Dict, Optional
import requests
import uuid
import torch
import shutil
import subprocess
import threading
import time
from faster_whisper import WhisperModel
from fastapi import Body, FastAPI, Request, UploadFile, File, HTTPException, BackgroundTasks
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware


import pdfplumber
import pytesseract
from PIL import Image
import docx
import openpyxl
import csv
import math


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

# ✅ Allowed origins
allowed_origins = [
    "https://www.voxento.com",
    "http://localhost:8080",
    "http://localhost:1337",
    "https://jon.voxento.com",
    "https://voxento-backend-staging-ed1cc9fc8b8e.herokuapp.com",
    "https://voxento-backend-778d4071d912.herokuapp.com",
    "http://host.docker.internal:3000",
    "https://slide-conversion.tkturners.com"
]

# ✅ Origin-specific Strapi configuration
origin_configs = {
    "https://www.voxento.com": {
        "STRAPI_URL": "https://voxento-backend-778d4071d912.herokuapp.com",
        "STRAPI_TOKEN": "a5eab6606a5a41d7d2d74004f2cd61ad3799c15d24d042286c22c324ce2ba44451b465ee0d12fbf645fcecb05744b5fff1e0948b62f340b2194dfd814ccff08df5579ed617a820ba40b84e4e65059c063ec1893595606552b200787a06448cdd8d4329f1957249c4e5c481eeff5ea95dfdbe66588411ec6cc1409c955ee786ad",
    },
    "https://voxento-backend-778d4071d912.herokuapp.com": {
        "STRAPI_URL": "https://voxento-backend-778d4071d912.herokuapp.com",
        "STRAPI_TOKEN": "a5eab6606a5a41d7d2d74004f2cd61ad3799c15d24d042286c22c324ce2ba44451b465ee0d12fbf645fcecb05744b5fff1e0948b62f340b2194dfd814ccff08df5579ed617a820ba40b84e4e65059c063ec1893595606552b200787a06448cdd8d4329f1957249c4e5c481eeff5ea95dfdbe66588411ec6cc1409c955ee786ad",
    },
    "https://slide-conversion.tkturners.com": {
        "STRAPI_URL": "https://voxento-backend-778d4071d912.herokuapp.com",
        "STRAPI_TOKEN": "a5eab6606a5a41d7d2d74004f2cd61ad3799c15d24d042286c22c324ce2ba44451b465ee0d12fbf645fcecb05744b5fff1e0948b62f340b2194dfd814ccff08df5579ed617a820ba40b84e4e65059c063ec1893595606552b200787a06448cdd8d4329f1957249c4e5c481eeff5ea95dfdbe66588411ec6cc1409c955ee786ad",
    },
    "https://jon.voxento.com": {
        "STRAPI_URL": "https://voxento-backend-staging-ed1cc9fc8b8e.herokuapp.com",
        "STRAPI_TOKEN": "05936d0fe48bd7c3ac95d982484f8f016032c49b9c915d458d6ba2e7a5274acce770bfcd2b0260b3c8051b5bc0f2bd4db1c854c9a1a239377689b39bc20a9d1ca47e8c72fe9814ca703693ee9ff288b0ef2dbae44fa921a6f7e8165699f472123ff245a780f8736aa512342cf63211d27e7993fea464d8353305b17bb3f4ea21",
    },
    "https://voxento-backend-staging-ed1cc9fc8b8e.herokuapp.com": {
        "STRAPI_URL": "https://voxento-backend-staging-ed1cc9fc8b8e.herokuapp.com",
        "STRAPI_TOKEN": "05936d0fe48bd7c3ac95d982484f8f016032c49b9c915d458d6ba2e7a5274acce770bfcd2b0260b3c8051b5bc0f2bd4db1c854c9a1a239377689b39bc20a9d1ca47e8c72fe9814ca703693ee9ff288b0ef2dbae44fa921a6f7e8165699f472123ff245a780f8736aa512342cf63211d27e7993fea464d8353305b17bb3f4ea21",
    },
    "http://localhost:8080": {
        "STRAPI_URL": os.getenv("STRAPI_URL", "http://localhost:1337"),
        "STRAPI_TOKEN": os.getenv("STRAPI_TOKEN", "5d7fa9bcabe02db85c0b4cccf84d14a378e55c2d979f3e02a357a5de01a23c8a947aa863cd0e58d3d238c1d4caa00d106d76a3c8c1534cc574a4310fcb10835e1c330002a83b21835ad9625263874efae6a659d68433a59776541fba96ef6ba666ae48d11359915c8bc11b9ed52c04e0bf261e9a114a5c83f5c430ee8a02ae71"),
    },
    "http://localhost:1337": {
        "STRAPI_URL": os.getenv("STRAPI_URL", "http://localhost:1337"),
        "STRAPI_TOKEN": os.getenv("STRAPI_TOKEN", "5d7fa9bcabe02db85c0b4cccf84d14a378e55c2d979f3e02a357a5de01a23c8a947aa863cd0e58d3d238c1d4caa00d106d76a3c8c1534cc574a4310fcb10835e1c330002a83b21835ad9625263874efae6a659d68433a59776541fba96ef6ba666ae48d11359915c8bc11b9ed52c04e0bf261e9a114a5c83f5c430ee8a02ae71"),
    },
    "http://host.docker.internal:3000": {
        "STRAPI_URL": os.getenv("STRAPI_URL", "http://localhost:1337"),
        "STRAPI_TOKEN": os.getenv("STRAPI_TOKEN", "5d7fa9bcabe02db85c0b4cccf84d14a378e55c2d979f3e02a357a5de01a23c8a947aa863cd0e58d3d238c1d4caa00d106d76a3c8c1534cc574a4310fcb10835e1c330002a83b21835ad9625263874efae6a659d68433a59776541fba96ef6ba666ae48d11359915c8bc11b9ed52c04e0bf261e9a114a5c83f5c430ee8a02ae71"),
    },
    
 
}

# Load Whisper model
try:
    whisper_model = WhisperModel(
        MODEL_NAME, device=DEVICE, compute_type="int8")
except Exception as e:
    logger.error(f"Failed to load Whisper model: {e}")
    raise

app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=allowed_origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


def get_strapi_config(origin: Optional[str]) -> Optional[Dict]:
    if origin and origin in origin_configs:
        return origin_configs[origin]

    # 🧠 Fallback for local tools (e.g., Postman / direct call)
    if not origin and os.getenv("STRAPI_URL") and os.getenv("STRAPI_TOKEN"):
        return {
            "STRAPI_URL": os.getenv("STRAPI_URL"),
            "STRAPI_TOKEN": os.getenv("STRAPI_TOKEN"),
        }

    print(f"⚠️ Unknown origin: {origin}")
    return None


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

# In-memory throttling state to avoid spamming Strapi with too many updates
_progress_lock = threading.Lock()
_progress_state: Dict[str, Dict] = {}


def _send_progress(strapi_url: str, token: str, record_id: str, percent: int, status: Optional[str] = None):
    """Sends the HTTP PUT to Strapi. Runs in a background thread to avoid blocking."""
    try:
        url = f"{strapi_url}/api/upload-records/{record_id}"
        headers = {"Authorization": f"Bearer {token}", "Content-Type": "application/json"}
        payload = {"data": {"transcriptionProgress": int(max(0, min(100, percent)))}}
        if status:
            payload["data"]["transcriptionStage"] = status
        res = requests.put(url, headers=headers, json=payload, timeout=10)
        if res.status_code not in (200, 201):
            logger.warning(f"Warning: progress update returned {res.status_code} -> {res.text}")
    except Exception as e:
        logger.warning(f"Failed to update progress to Strapi: {e}")


def update_strapi_progress(strapi_url: str, token: str, record_id: str, percent: int, status: Optional[str] = None, force: bool = False):
    """Rate-limited progress updates to Strapi.

    Behavior:
    - If required fields are missing, returns immediately.
    - Sends immediately for critical statuses (e.g., 'failed', 'completed').
    - Otherwise, only sends if percent increased by delta_percent OR min_interval seconds elapsed since last send.
    - Runs the actual HTTP request in a background thread to avoid blocking the main request loop.

    Keep this function lightweight and safe to call frequently from loops.
    """
    if not (strapi_url and token and record_id):
        return

    now = time.time()
    key = str(record_id)

    # Configurable thresholds (tweak if needed)
    delta_percent_threshold = 8      # only send if percent increased by this much
    min_interval_seconds = 5         # at most one non-critical update every N seconds

    critical_statuses = {"failed", "completed", "error"}
    is_critical = bool(status and status.lower() in critical_statuses)

    with _progress_lock:
        state = _progress_state.get(key, {"percent": -1, "status": None, "time": 0})
        last_percent = state["percent"]
        last_status = state["status"]
        last_time = state["time"]

        percent = int(max(0, min(100, percent)))

        # Decide whether to send:
        should_send = False

        # 1) Critical statuses should always be sent immediately
        if is_critical or force:
            should_send = True

        # 2) If status changed, send immediately
        elif status != last_status and status is not None:
            should_send = True

        # 3) If percent jumped by threshold
        elif percent - last_percent >= delta_percent_threshold:
            should_send = True

        # 4) If it's been a while since the last send (prevent stalling)
        elif (now - last_time) >= min_interval_seconds:
            should_send = True

        # Update in-memory state to reflect decision so subsequent calls are throttled
        if should_send:
            _progress_state[key] = {"percent": percent, "status": status, "time": now}
            # Send in background so callers don't block
            threading.Thread(target=_send_progress, args=(strapi_url, token, record_id, percent, status), daemon=True).start()
        else:
            # If we didn't send, still keep the highest percent we've seen (so we can send later)
            if percent > last_percent:
                _progress_state.setdefault(key, state)["percent"] = percent
            logger.debug(f"Throttled progress update for {record_id}: last={last_percent}% now={percent}% status={status}")

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


def format_time_vtt(seconds: float) -> str:
    """
    Convert seconds to VTT time format: HH:MM:SS.mmm
    """
    hours = int(seconds // 3600)
    minutes = int((seconds % 3600) // 60)
    seconds_remainder = seconds % 60
    milliseconds = int((seconds_remainder - int(seconds_remainder)) * 1000)

    return f"{hours:02d}:{minutes:02d}:{int(seconds_remainder):02d}.{milliseconds:03d}"


@app.post("/transcribe-audio")
async def transcribe_audio_api(
    request: Request,
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

    origin = request.headers.get("origin")
    config = get_strapi_config(origin)

    print(f"config: {config}------ origin:{origin}")

    if not config:
        raise HTTPException(
            status_code=400, detail=f"Invalid or unsupported origin: {origin}")

    STRAPI_URL = config["STRAPI_URL"]
    STRAPI_TOKEN = config["STRAPI_TOKEN"]

    print(f"{log_prefix} Started for audio_id: {audio_id} | Origin: {origin}")

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


@app.post("/transcribe-video")
async def transcribe_video(
    request: Request,
    background_tasks: BackgroundTasks,
    video_url: str = Body(...),
    upload_record_id: Optional[str] = Body(None),
):
    log_prefix = "[🎬 Video Transcription]"
    print(f"{log_prefix} video_url: {video_url}")
    origin = request.headers.get("origin")
    config = get_strapi_config(origin)

    print(f"{log_prefix} Origin: {origin}")

    if not config:
        raise HTTPException(
            status_code=400, detail=f"Invalid or unsupported origin: {origin}")

    STRAPI_URL = config["STRAPI_URL"]
    STRAPI_TOKEN = config["STRAPI_TOKEN"]

    temp_dir = "temp_video"
    os.makedirs(temp_dir, exist_ok=True)
    file_id = str(uuid.uuid4())

    temp_video_path = os.path.join(temp_dir, f"{file_id}.mp4")
    temp_audio_path = os.path.join(temp_dir, f"{file_id}.wav")
    txt_path = os.path.join(temp_dir, f"{file_id}.txt")

    try:
        # 1️⃣ Download video
        print(f"{log_prefix} Downloading from {video_url}")
        update_strapi_progress(STRAPI_URL, STRAPI_TOKEN, upload_record_id, 5, status="Starting download")

        video_res = requests.get(video_url, stream=True, timeout=60)
        if video_res.status_code != 200:
            raise HTTPException(
                status_code=400, detail=f"Failed to download video: {video_res.status_code}")

        update_strapi_progress(STRAPI_URL, STRAPI_TOKEN, upload_record_id, 10, status="Downloading video")

        # Download with progress tracking
        total_size = int(video_res.headers.get('content-length', 0))
        downloaded = 0
        chunk_size = 8192
        last_progress = 10
        
        with open(temp_video_path, "wb") as f:
            for chunk in video_res.iter_content(chunk_size=chunk_size):
                f.write(chunk)
                downloaded += len(chunk)
                
                # Update progress during download (10% to 25%)
                if total_size > 0:
                    download_progress = 10 + int((downloaded / total_size) * 15)
                    if download_progress > last_progress and download_progress <= 25:
                        update_strapi_progress(STRAPI_URL, STRAPI_TOKEN, upload_record_id, download_progress, status="Downloading video")
                        last_progress = download_progress

        update_strapi_progress(STRAPI_URL, STRAPI_TOKEN, upload_record_id, 25, status="Download complete")

        # 2️⃣ Extract audio
        print(f"{log_prefix} Extracting audio with ffmpeg...")
        update_strapi_progress(STRAPI_URL, STRAPI_TOKEN, upload_record_id, 30, status="Preparing audio extraction")

        try:
            subprocess.run(
                ["ffmpeg", "-i", temp_video_path, "-vn", "-acodec",
                    "pcm_s16le", "-ar", "16000", temp_audio_path],
                check=True,
                capture_output=True,
                text=True
            )
        except subprocess.CalledProcessError as e:
            print(f"{log_prefix} ⚠️ FFmpeg error: {e.stderr}")
            if "does not contain any stream" in e.stderr or "no audio" in e.stderr.lower():
                print(f"{log_prefix} ⚠️ No audio stream found in video")
                background_tasks.add_task(
                    cleanup_files, [temp_video_path, temp_audio_path, txt_path])
                return JSONResponse({
                    "success": True,
                    "transcription_file": "",
                    "message": "No audio found in this video",
                    "has_audio": False
                })
            raise

        update_strapi_progress(STRAPI_URL, STRAPI_TOKEN, upload_record_id, 35, status="Audio extracted")

        # Check if audio file was created and has content
        if not os.path.exists(temp_audio_path) or os.path.getsize(temp_audio_path) == 0:
            print(f"{log_prefix} ⚠️ No audio extracted from video")
            background_tasks.add_task(
                cleanup_files, [temp_video_path, temp_audio_path, txt_path])
            return JSONResponse({
                "success": True,
                "transcription_file": "",
                "message": "No audio found in this video",
                "has_audio": False
            })

        # 3️⃣ Transcribe audio with progress tracking
        print(f"{log_prefix} Transcribing audio...")
        update_strapi_progress(STRAPI_URL, STRAPI_TOKEN, upload_record_id, 40, status="Loading transcription model")

        # Use the transcribe_audio function but we need to modify it for progress
        # For now, let's use the Whisper model directly for better control
        segments, info = whisper_model.transcribe(temp_audio_path, beam_size=5)
        
        update_strapi_progress(STRAPI_URL, STRAPI_TOKEN, upload_record_id, 50, status="Transcribing audio")

        # Collect segments with progress tracking
        segments_list = []
        segment_count = 0
        last_progress = 50
        
        for segment in segments:
            segments_list.append(segment)
            segment_count += 1
            
            # Update progress every 10 segments (50% to 85%)
            if segment_count % 10 == 0:
                progress = min(92, 50 + int((segment_count ** 0.7)))
                if progress > last_progress:
                    update_strapi_progress(
                        STRAPI_URL, 
                        STRAPI_TOKEN, 
                        upload_record_id, 
                        progress, 
                        status="Transcribing audio"
                    )
                    last_progress = progress

        update_strapi_progress(STRAPI_URL, STRAPI_TOKEN, upload_record_id, 92, status="Processing transcription")

        # Combine transcription text
        transcription_text = " ".join([seg.text.strip() for seg in segments_list if seg.text.strip()])

        # Check if transcription returned no speech
        if not transcription_text or transcription_text == "No speech detected.":
            print(f"{log_prefix} ⚠️ No speech detected in video")
            background_tasks.add_task(
                cleanup_files, [temp_video_path, temp_audio_path, txt_path])
            return JSONResponse({
                "success": True,
                "transcription_file": "",
                "message": "No voice detected in this video",
                "has_audio": True,
                "has_speech": False
            })

        update_strapi_progress(STRAPI_URL, STRAPI_TOKEN, upload_record_id, 95, status="Uploading transcription")

        # 4️⃣ Update module with transcription
        update_payload = {
            "transcription_file": transcription_text,
            "content_type": "video"
        }

        update_url = f"{STRAPI_URL}/api/moduleContent/unifiedTranscription"

        update_res = requests.put(update_url, json=update_payload)

        if update_res.status_code not in (200, 201):
            raise Exception(
                f"Transcription failed: {update_res.status_code} -> {update_res.text}")

        update_strapi_progress(STRAPI_URL, STRAPI_TOKEN, upload_record_id, 98, status="Finalizing")
        print(f"{log_prefix} ✅ Transcription exported successfully")

        # 5️⃣ Cleanup
        background_tasks.add_task(
            cleanup_files, [temp_video_path, temp_audio_path, txt_path])

        update_strapi_progress(STRAPI_URL, STRAPI_TOKEN, upload_record_id, 100, status="completed")

        return JSONResponse(
            {
                "success": True,
                "transcription_file": transcription_text,
                "message": "Transcription completed successfully",
                "has_audio": True,
                "has_speech": True
            }
        )

    except Exception as e:
        print(f"{log_prefix} ❌ Error: {e}")
        for file_path in [temp_video_path, temp_audio_path, txt_path]:
            if os.path.exists(file_path):
                try:
                    os.remove(file_path)
                except:
                    pass
        raise HTTPException(
            status_code=500, detail=f"Video transcription failed: {e}")
@app.post("/extract-slides-text")
async def extract_slides_text(
    request: Request,
    background_tasks: BackgroundTasks,
    slide_urls: list[str] = Body(..., embed=True),
    upload_record_id: Optional[str] = Body(None)
):
    log_prefix = "[🖼️ Slides Text Extraction]"
    
    print(f"{log_prefix} Started with {len(slide_urls)} slides")
    
    origin = request.headers.get("origin")
    config = get_strapi_config(origin)

    if not config:
        raise HTTPException(
            status_code=400, detail=f"Invalid or unsupported origin: {origin}")

    STRAPI_URL = config["STRAPI_URL"]
    STRAPI_TOKEN = config["STRAPI_TOKEN"]

    temp_dir = "temp_slides"
    os.makedirs(temp_dir, exist_ok=True)
    file_id = str(uuid.uuid4())
    downloaded_files = []

    try:
        # Local debounce state to avoid sending too many progress requests in tight loops
        local_last_send = 0.0
        local_last_percent = -1
        local_min_interval = 3.0  # seconds between non-critical sends for slides
        local_delta_percent = 8    # percent jump required to force a send sooner

        def send_progress_local(percent: int, status: Optional[str] = None, force: bool = False):
            nonlocal local_last_send, local_last_percent
            now = time.time()
            percent = int(max(0, min(100, percent)))

            # Always force critical or explicitly forced updates
            critical = bool(status and status.lower() in {"failed", "completed", "error"})
            if force or critical:
                update_strapi_progress(STRAPI_URL, STRAPI_TOKEN, upload_record_id, percent, status=status, force=True)
                local_last_send = now
                local_last_percent = percent
                return

            # If percent jump is large enough, send immediately
            if percent - local_last_percent >= local_delta_percent:
                update_strapi_progress(STRAPI_URL, STRAPI_TOKEN, upload_record_id, percent, status=status)
                local_last_send = now
                local_last_percent = percent
                return

            # Otherwise, only send if min interval elapsed
            if (now - local_last_send) >= local_min_interval:
                update_strapi_progress(STRAPI_URL, STRAPI_TOKEN, upload_record_id, percent, status=status)
                local_last_send = now
                local_last_percent = percent

        # Initial progress
        send_progress_local(5, status="Starting slide extraction")

        # Process each slide URL and extract text
        all_extracted_text = []
        total_slides = len(slide_urls)

        # Phase 1: Download all slides (5% to 50%)
        for i, image_url in enumerate(slide_urls):
            try:
                if not image_url:
                    print(f"{log_prefix} ⚠️ Empty URL for slide {i+1}, skipping")
                    continue

                print(f"{log_prefix} Downloading slide {i+1}/{total_slides}: {image_url}")

                # Calculate download progress: 5% to 50%
                download_progress = 5 + int(((i + 1) / total_slides) * 45)
                send_progress_local(download_progress, status=f"Downloading slide {i+1}/{total_slides}")

                # Download image
                image_filename = f"slide_{i+1}_{file_id}_{os.path.basename(image_url)}"
                temp_image_path = os.path.join(temp_dir, image_filename)

                image_res = requests.get(image_url, stream=True, timeout=30)
                if image_res.status_code != 200:
                    print(f"{log_prefix} ⚠️ Failed to download slide {i+1}, skipping")
                    continue

                with open(temp_image_path, "wb") as f:
                    for chunk in image_res.iter_content(chunk_size=8192):
                        f.write(chunk)

                downloaded_files.append(temp_image_path)

            except Exception as e:
                print(f"{log_prefix} ❌ Error downloading slide {i+1}: {e}")
                continue

        # Update after all downloads complete
        send_progress_local(50, status="All slides downloaded")

        # Phase 2: Extract text from all downloaded slides (50% to 90%)
        for i, temp_image_path in enumerate(downloaded_files):
            try:
                # Calculate extraction progress: 50% to 90%
                extraction_progress = 50 + int(((i + 1) / len(downloaded_files)) * 40)
                send_progress_local(extraction_progress, status=f"Extracting text from slide {i+1}/{len(downloaded_files)}")

                # Extract text from image using existing helper function
                extracted_text = extract_text_from_image(temp_image_path)

                # Clean up the extracted text
                cleaned_text = extracted_text.strip()
                if cleaned_text and cleaned_text != "No text found in image.":
                    all_extracted_text.append({
                        "slide_number": i + 1,
                        "extracted_text": cleaned_text
                    })
                    print(f"{log_prefix} ✅ Extracted text from slide {i+1}")
                else:
                    print(f"{log_prefix} ⚠️ No text found in slide {i+1}")

            except Exception as e:
                print(f"{log_prefix} ❌ Error extracting text from slide {i+1}: {e}")
                continue

        # Combine all extracted text
        update_strapi_progress(STRAPI_URL, STRAPI_TOKEN, upload_record_id, 92, status="Combining extracted text")
        
        if not all_extracted_text:
            update_strapi_progress(STRAPI_URL, STRAPI_TOKEN, upload_record_id, 0, status="failed")
            raise HTTPException(
                status_code=404, detail="No text could be extracted from any slides")

        combined_text = "\n\n".join([
            f"Slide {item['slide_number']}:\n{item['extracted_text']}"
            for item in all_extracted_text
        ])

        update_strapi_progress(STRAPI_URL, STRAPI_TOKEN, upload_record_id, 95, status="Finalizing")

        # Cleanup temporary files in background
        background_tasks.add_task(cleanup_files, downloaded_files)

        update_strapi_progress(STRAPI_URL, STRAPI_TOKEN, upload_record_id, 100, status="completed")

        return JSONResponse({
            "success": True,
            "slides_processed": len(all_extracted_text),
            "total_slides": len(slide_urls),
            "transcription": combined_text,
            "extracted_text_summary": {
                "total_slides_with_text": len(all_extracted_text),
                "slides": all_extracted_text
            }
        })

    except HTTPException:
        raise
    except Exception as e:
        print(f"{log_prefix} ❌ Error: {e}")
        # Cleanup on error
        cleanup_files(downloaded_files)
        if upload_record_id:
            update_strapi_progress(STRAPI_URL, STRAPI_TOKEN, upload_record_id, 0, status="failed")
        raise HTTPException(
            status_code=500, detail=f"Slides text extraction failed: {e}")  
@app.post("/transcribe-course-audio")
async def transcribe_course_audio_api(
    request: Request,
    background_tasks: BackgroundTasks,
    audio_url: str = Body(...),
    audio_id: str = Body(...),
    upload_record_id: Optional[str] = Body(None),
    model_size: str = Body("base"),
    language: Optional[str] = Body(None),
    task: str = Body("transcribe")
):
    log_prefix = "[🎧Course Audio Subtitle Generator]"
    origin = request.headers.get("origin")
    config = get_strapi_config(origin)

    print(f"{log_prefix} Started for audio_id: {audio_id} | Origin: {origin} | upload_record_id: {upload_record_id}")

    if not config:
        raise HTTPException(status_code=400, detail=f"Invalid or unsupported origin: {origin}")

    STRAPI_URL = config["STRAPI_URL"]
    STRAPI_TOKEN = config["STRAPI_TOKEN"]

    temp_dir = "temp_audio"
    os.makedirs(temp_dir, exist_ok=True)

    file_id = str(uuid.uuid4())
    temp_audio_path = os.path.join(temp_dir, f"{file_id}.mp3")
    vtt_path = os.path.join(temp_dir, f"{file_id}.vtt")

    try:
        # 1️⃣ Download audio file
        print(f"{log_prefix} Downloading from {audio_url}")
        update_strapi_progress(STRAPI_URL, STRAPI_TOKEN, upload_record_id, 10, status="downloading file")
        audio_res = requests.get(audio_url, stream=True, timeout=60)
        if audio_res.status_code != 200:
            update_strapi_progress(STRAPI_URL, STRAPI_TOKEN, upload_record_id, 0, status="failed")
            raise HTTPException(status_code=400, detail=f"Failed to download audio: {audio_res.status_code}")

        with open(temp_audio_path, "wb") as f:
            for chunk in audio_res.iter_content(chunk_size=8192):
                f.write(chunk)

        # Check if audio file has content
        if not os.path.exists(temp_audio_path) or os.path.getsize(temp_audio_path) == 0:
            print(f"{log_prefix} ⚠️ Downloaded audio file is empty")
            background_tasks.add_task(cleanup_files, [temp_audio_path, vtt_path])
            if upload_record_id:
                update_strapi_progress(STRAPI_URL, STRAPI_TOKEN, upload_record_id, 0, status="failed")
            return JSONResponse({
                "success": True,
                "transcription_file": "",
                "message": "No audio content found in this file",
                "has_audio": False,
                "has_speech": False
            })

        # 2️⃣ Load Whisper model
        print(f"{log_prefix} Loading model: {model_size}")
        update_strapi_progress(STRAPI_URL, STRAPI_TOKEN, upload_record_id, 18, status="loading model")
        transcribe_model = get_model(model_size)

        # 3️⃣ Transcribe audio (get segments + info)
        print(f"{log_prefix} Transcribing audio...")
        update_strapi_progress(STRAPI_URL, STRAPI_TOKEN, upload_record_id, 28, status="transcribing audio")
        segments, info = transcribe_model.transcribe(
            temp_audio_path,
            language=language,
            task=task
        )

        # Track progress during segment processing
        segments_list = []
        segment_count = 0
        last_reported_progress = 28
        
        for segment in segments:
            segments_list.append(segment)
            segment_count += 1
            
            # Update progress every 5 segments
            if segment_count % 5 == 0:
                # Gradually increase from 43% to 60% during transcription
                # Using a smooth increment that slows down as we approach 60%
                progress = min(87, 28 + int((segment_count ** 0.7)))
                
                if progress > last_reported_progress:
                    update_strapi_progress(
                        STRAPI_URL, 
                        STRAPI_TOKEN, 
                        upload_record_id, 
                        progress, 
                        status="transcribing audio"
                    )
                    last_reported_progress = progress
        
        # Ensure we're at 60% when transcription is done
        update_strapi_progress(STRAPI_URL, STRAPI_TOKEN, upload_record_id, 87, status="transcription complete")
        
        total_duration = getattr(info, "duration", None) or (max((seg.end for seg in segments_list), default=None) if segments_list else None)
    
        # Check if no segments were detected
        if not segments_list:
            print(f"{log_prefix} ⚠️ No speech detected in audio")
            background_tasks.add_task(cleanup_files, [temp_audio_path, vtt_path])
            if upload_record_id:
                update_strapi_progress(STRAPI_URL, STRAPI_TOKEN, upload_record_id, 0, status="failed")
            return JSONResponse({
                "success": True,
                "transcription_file": "",
                "message": "No voice detected in this audio",
                "has_audio": True,
                "has_speech": False
            })

        # 4️⃣ Generate VTT file and collect full text while reporting progress
        print(f"{log_prefix} Generating VTT and collecting text...")
        update_strapi_progress(STRAPI_URL, STRAPI_TOKEN, upload_record_id, 91, status="Generating VTT")

        full_text = ""
        last_percent_sent = -1
        with open(vtt_path, "w", encoding="utf-8") as vtt_file:
            vtt_file.write("WEBVTT\n\n")
            for i, segment in enumerate(segments_list, start=1):
                start_time = format_time_vtt(segment.start)
                end_time = format_time_vtt(segment.end)
                text = segment.text.strip()
                vtt_file.write(f"{i}\n{start_time} --> {end_time}\n{text}\n\n")
                full_text += text + " "

                if total_duration:
                    percent = math.floor((segment.end / total_duration) * 100)
                else:
                    percent = math.floor((i / max(1, len(segments_list))) * 100)

                if upload_record_id and percent != last_percent_sent:
                    last_percent_sent = percent

        full_text = full_text.strip()
        update_strapi_progress(STRAPI_URL, STRAPI_TOKEN, upload_record_id, 94, status="processing")

        # Check if transcription is empty or too short
        if not full_text or len(full_text) < 3:
            print(f"{log_prefix} ⚠️ No meaningful speech detected in audio")
            background_tasks.add_task(cleanup_files, [temp_audio_path, vtt_path])
            if upload_record_id:
                update_strapi_progress(STRAPI_URL, STRAPI_TOKEN, upload_record_id, 0, status="failed")
            return JSONResponse({
                "success": True,
                "transcription_file": "",
                "message": "No voice detected in this audio",
                "has_audio": True,
                "has_speech": False
            })

        print(f"{log_prefix} full_text length: {len(full_text)} characters")

        # 5️⃣ Upload VTT to Strapi
        print(f"{log_prefix} Uploading VTT to Strapi...")
        update_strapi_progress(STRAPI_URL, STRAPI_TOKEN, upload_record_id, 96, status="Uploading VTT")

        with open(vtt_path, "rb") as file_data:
            files = {"files": (os.path.basename(vtt_path), file_data, "text/vtt")}
            headers = {"Authorization": f"Bearer {STRAPI_TOKEN}"}
            upload_res = requests.post(f"{STRAPI_URL}/api/upload", headers=headers, files=files)

        if upload_res.status_code not in (200, 201):
            raise Exception(f"Upload failed: {upload_res.status_code} -> {upload_res.text}")

        # ✅ Handle Strapi upload response
        uploaded_json = upload_res.json()
        if isinstance(uploaded_json, list) and len(uploaded_json) > 0:
            uploaded_vtt = uploaded_json[0]
        elif isinstance(uploaded_json, dict) and "id" in uploaded_json:
            uploaded_vtt = uploaded_json
        else:
            raise Exception(f"Unexpected upload response format: {uploaded_json}")

        subtitle_file_id = uploaded_vtt["id"]
        print(f"{log_prefix} ✅ Uploaded VTT file (id={subtitle_file_id})")

        # 6️⃣ Create subtitle entry in Strapi
        print(f"{log_prefix} Creating subtitle entry in Strapi...")
        subtitle_payload = {"data": {"subtitle": subtitle_file_id, "audio": audio_id}}
        headers = {"Authorization": f"Bearer {STRAPI_TOKEN}", "Content-Type": "application/json"}
        create_res = requests.post(f"{STRAPI_URL}/api/subtitles", headers=headers, json=subtitle_payload)
        if create_res.status_code not in (200, 201):
            raise Exception(f"Subtitle entry creation failed: {create_res.status_code} -> {create_res.text}")
        print(f"{log_prefix} ✅ Subtitle entry created successfully")

        # 7️⃣ Update module with transcription text (directly send text content)
        print(f"{log_prefix} Updating module with transcription text...")
        update_strapi_progress(STRAPI_URL, STRAPI_TOKEN, upload_record_id, 98, status="Updating module with transcription text")

        update_payload = {
            "transcription_file": full_text,
            "content_type": "courseAudio"
        }
        update_url = f"{STRAPI_URL}/api/moduleContent/unifiedTranscription"
        update_res = requests.put(update_url, headers=headers, json=update_payload)
        if update_res.status_code not in (200, 201):
            print(f"{log_prefix} ⚠️ Module update failed: {update_res.status_code} -> {update_res.text}")

        # 8️⃣ Mark upload-record completed (if provided)
        if upload_record_id:
            update_strapi_progress(STRAPI_URL, STRAPI_TOKEN, upload_record_id, 100, status="completed")

        # 9️⃣ Cleanup after success
        background_tasks.add_task(cleanup_files, [temp_audio_path, vtt_path])

        # 10️⃣ Return success JSON
        return JSONResponse({
            "success": True,
            "subtitle_file": uploaded_vtt,
            "subtitle_entry": create_res.json(),
            "transcription_file": full_text,
            "transcription_text_length": len(full_text),
            "message": "Transcription completed successfully",
            "has_audio": True,
            "has_speech": True
        })

    except Exception as e:
        print(f"{log_prefix} ❌ Error: {e}")
        cleanup_files([temp_audio_path, vtt_path])
        if upload_record_id:
            try:
                update_strapi_progress(STRAPI_URL, STRAPI_TOKEN, upload_record_id, 0, status="failed")
            except:
                pass
        raise HTTPException(status_code=500, detail=f"Transcription failed: {e}")
