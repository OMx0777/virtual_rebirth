import os
import shutil
from fastapi import FastAPI, File, UploadFile, BackgroundTasks
from datetime import datetime

app = FastAPI(title="Virtual Rebirth Core API")

# Setup directories for the pipeline
TEMP_VIDEO_DIR = "temp_videos"
TEXT_MEMORY_DIR = "memory_logs"

os.makedirs(TEMP_VIDEO_DIR, exist_ok=True)
os.makedirs(TEXT_MEMORY_DIR, exist_ok=True)

def process_and_purge(file_path: str, filename: str):
    """
    Background task to extract data and delete the raw video.
    """
    print(f"[*] Starting extraction pipeline for {filename}...")
    
    # ---------------------------------------------------------
    # TODO: Phase 1.1 - Audio Extraction (faster-whisper)
    # TODO: Phase 1.2 - Vision Extraction (moondream2)
    # ---------------------------------------------------------
    
    # Simulated extraction for the skeleton pipeline
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    memory_file = os.path.join(TEXT_MEMORY_DIR, f"log_{timestamp}.txt")
    
    with open(memory_file, "w") as f:
        f.write(f"--- Raw Memory Log for {filename} ---\n")
        f.write("Audio Transcript: [Whisper extraction will go here]\n")
        f.write("Visual Context: [Moondream2 extraction will go here]\n")
    
    print(f"[+] Memory extracted to {memory_file}")

    # The most crucial step: Purge the raw video to save storage
    if os.path.exists(file_path):
        os.remove(file_path)
        print(f"[-] Raw video {filename} securely purged from disk.")

@app.post("/upload_chunk/")
async def upload_video_chunk(background_tasks: BackgroundTasks, file: UploadFile = File(...)):
    """
    Receives the video from the mobile app, saves it temporarily, 
    and offloads processing to a background task.
    """
    file_path = os.path.join(TEMP_VIDEO_DIR, file.filename)
    
    # Save the incoming video chunk
    with open(file_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)
        
    # Queue the extraction and deletion process
    background_tasks.add_task(process_and_purge, file_path, file.filename)
    
    # Instantly return success so the mobile app can delete its copy
    return {"status": "success", "message": "Chunk received. Processing in background.", "filename": file.filename}