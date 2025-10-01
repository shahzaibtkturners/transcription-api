import os
import re
import logging
import torch
import shutil
import subprocess
from faster_whisper import WhisperModel
from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.responses import JSONResponse

import pdfplumber
import pytesseract
from PIL import Image
import docx
import openpyxl
import csv

# Configure logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
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
    whisper_model = WhisperModel(MODEL_NAME, device=DEVICE, compute_type="int8")
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
        transcription = " ".join([seg.text.strip() for seg in segments if seg.text.strip()])
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
        raise HTTPException(status_code=500, detail="PDF text extraction failed")

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
        raise HTTPException(status_code=500, detail="DOCX text extraction failed")

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
                row_text = " ".join([str(cell) for cell in row if cell is not None])
                if row_text.strip():
                    text.append(row_text)
        return "\n".join(text) if text else "No text found in Excel file."
    except Exception as e:
        logger.error(f"XLSX extraction failed: {e}")
        raise HTTPException(status_code=500, detail="XLSX text extraction failed")

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
        raise HTTPException(status_code=500, detail="CSV text extraction failed")

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
                ["ffmpeg", "-i", file_path, "-vn", "-acodec", "pcm_s16le", "-ar", "16000", audio_path],
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
