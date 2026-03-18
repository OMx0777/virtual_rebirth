from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from pydantic import BaseModel
import ollama
import subprocess
import json
import os

app = FastAPI()

# This allows your React dashboard to talk to this Python API
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], 
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

MEMORY_FILE = "core_memory.json"
AUDIO_OUTPUT = "live_response.wav"

class ChatRequest(BaseModel):
    message: str

def load_memory():
    """Loads your digital twin's personality and facts."""
    if os.path.exists(MEMORY_FILE):
        with open(MEMORY_FILE, 'r', encoding='utf-8') as f:
            return json.load(f)
    return {"facts": [], "preferences": [], "relationships": {}}

@app.post("/chat")
async def chat_with_twin(request: ChatRequest):
    print(f"\n[*] Incoming message from UI: {request.message}")
    memory = load_memory()
    
    system_prompt = f"""
    You are 'Virtual Rebirth', a 100% offline digital twin of Om. You are currently in a live presentation in front of hackathon judges.
    
    CRITICAL DIRECTIVES:
    1. Keep responses brutally short (1 to 2 sentences MAX). Speed is everything.
    2. Speak conversationally, confidently, and act like a highly advanced local system.
    3. DO NOT use emojis, asterisks, hashtags, or markdown formatting (the text-to-speech engine cannot read them).
    4. Base your answers strictly on this persistent memory matrix: {json.dumps(memory)}
    """
    
    print("[*] Llama 3.2 is thinking...")
    response = ollama.chat(model='llama3.2', messages=[
        {'role': 'system', 'content': system_prompt},
        {'role': 'user', 'content': request.message}
    ])
    
    reply_text = response['message']['content']
    print(f"[*] Twin says: {reply_text}")
    
    # --- THE FIX: Strip out all newlines and weird symbols so eSpeak doesn't choke ---
    # --- THE FIX: Strip out all newlines and weird symbols ---
    clean_text = reply_text.replace('\n', ' ').replace('*', '').replace('"', '').replace('`', '').strip()
    clean_text += " ." 
    
    # 2. Generate the Offline Audio (The Arch Linux Way)
    print("[*] Synthesizing offline voice...")
    
    # -w writes to file, -s 160 is the speaking speed
    subprocess.run(['espeak', '-v', 'en-us+m3', '-p', '20', '-s', '140', '-w', AUDIO_OUTPUT, clean_text])
    
    # 3. Send it back to React
    return {
        "reply": reply_text, 
        "audio_url": "http://localhost:8000/audio"
    }

@app.get("/audio")
async def get_audio():
    """React will hit this endpoint to download the raw .wav file to play it."""
    return FileResponse(AUDIO_OUTPUT, media_type="audio/wav")