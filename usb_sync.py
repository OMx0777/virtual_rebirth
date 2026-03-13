import os
import subprocess
import cv2
import torch
import torchaudio
import huggingface_hub

# --- BLEEDING EDGE FIX FOR PYTORCH 2.9+ ---
if not hasattr(torchaudio, "list_audio_backends"):
    torchaudio.list_audio_backends = lambda: ["soundfile"]

# --- THE ULTIMATE 404 INTERCEPTOR ---
_orig_hf_download = huggingface_hub.hf_hub_download
def patched_hf_download(*args, **kwargs):
    # Fix the deprecated token argument
    if 'use_auth_token' in kwargs:
        kwargs['token'] = kwargs.pop('use_auth_token')
        
    # Extract the filename being requested
    filename = kwargs.get('filename')
    if not filename and len(args) >= 2:
        filename = args[1]
        
    # INTERCEPT custom.py BEFORE it goes to the internet!
    if filename == 'custom.py':
        fake_dir = "pretrained_models/spkrec-ecapa-voxceleb"
        os.makedirs(fake_dir, exist_ok=True)
        fake_path = os.path.abspath(os.path.join(fake_dir, "custom.py"))
        with open(fake_path, "w") as f:
            f.write("# Fake custom.py to bypass 404 crash\n")
        return fake_path
        
    # Normal download for everything else
    return _orig_hf_download(*args, **kwargs)

huggingface_hub.hf_hub_download = patched_hf_download

_orig_snapshot = huggingface_hub.snapshot_download
def patched_snapshot(*args, **kwargs):
    if 'use_auth_token' in kwargs:
        kwargs['token'] = kwargs.pop('use_auth_token')
    return _orig_snapshot(*args, **kwargs)
huggingface_hub.snapshot_download = patched_snapshot
# ------------------------------------------

from datetime import datetime
from PIL import Image
from faster_whisper import WhisperModel
from transformers import AutoProcessor, AutoModelForImageTextToText
from pydub import AudioSegment
from speechbrain.inference.speaker import SpeakerRecognition

# ==========================================
# CONFIGURATION
# ==========================================
PHONE_VIDEO_DIR = "/storage/emulated/0/Movies/VirtualRebirth/" 
PC_TEMP_DIR = "temp_videos"
TEXT_MEMORY_DIR = "memory_logs"
USER_VOICE_FILE = "om_voice.wav" 

CONFIDENCE_THRESHOLD = 0.15 

os.makedirs(PC_TEMP_DIR, exist_ok=True)
os.makedirs(TEXT_MEMORY_DIR, exist_ok=True)

# ==========================================
# INITIALIZE AI MODELS
# ==========================================
print("[*] Loading Whisper AI (CPU) for audio extraction...")
whisper_model = WhisperModel("base", device="cpu", compute_type="int8")

print("[*] Loading Biometric Voice Engine (CPU, Token-Free)...")
try:
    voice_verifier = SpeakerRecognition.from_hparams(
        source="speechbrain/spkrec-ecapa-voxceleb",
        savedir="pretrained_models/spkrec-ecapa-voxceleb"
    )
except Exception as e:
    print(f"[!] Warning: Biometric engine failed to load: {e}")
    voice_verifier = None

print("[*] Loading SmolVLM (GPU, 256M Native) for visual extraction...")
model_id = "HuggingFaceTB/SmolVLM-256M-Instruct"
processor = AutoProcessor.from_pretrained(model_id)
vlm_model = AutoModelForImageTextToText.from_pretrained(
    model_id, 
    torch_dtype=torch.float16,
).to("cuda")

# ==========================================
# HELPER FUNCTIONS
# ==========================================
def run_adb_command(command: list):
    result = subprocess.run(["adb"] + command, capture_output=True, text=True)
    return result.stdout.strip(), result.returncode

def get_video_files_from_phone():
    stdout, code = run_adb_command(["shell", "ls", PHONE_VIDEO_DIR])
    if code != 0 or "No such file or directory" in stdout:
        return []
    return [f.strip() for f in stdout.split('\n') if f.strip().endswith(('.mp4', '.mkv', '.avi'))]

def process_and_purge(local_file_path: str, filename: str):
    print(f"\n[*] Starting extraction pipeline for {filename}...")
    
    print("    -> Listening to audio and biometrically tagging speakers...")
    transcript_text = ""
    
    try:
        full_audio = AudioSegment.from_file(local_file_path)
    except Exception as e:
        print(f"    [!] Could not load audio for biometric slicing: {e}")
        full_audio = None

    has_voiceprint = os.path.exists(USER_VOICE_FILE)

    try:
        segments, _ = whisper_model.transcribe(local_file_path, beam_size=5)
        for segment in segments:
            speaker_tag = "[Guest]"
            confidence_log = ""
            
            if full_audio and has_voiceprint and voice_verifier:
                start_ms = int(segment.start * 1000)
                end_ms = int(segment.end * 1000)
                duration_ms = end_ms - start_ms
                
                if duration_ms > 500: 
                    temp_chunk_path = "temp_chunk.wav"
                    chunk = full_audio[start_ms:end_ms]
                    chunk.export(temp_chunk_path, format="wav", parameters=["-ac", "1", "-ar", "16000"])
                    
                    try:
                        score, _ = voice_verifier.verify_files(USER_VOICE_FILE, temp_chunk_path)
                        similarity = score.item()
                        
                        if similarity >= CONFIDENCE_THRESHOLD:
                            speaker_tag = "[Om]"
                            
                        confidence_log = f" (Score: {similarity:.2f})"
                    except Exception as e:
                        confidence_log = f" (Bio Error: {e})"

            line = f"[{segment.start:.2f}s -> {segment.end:.2f}s] {speaker_tag}{confidence_log}: {segment.text}"
            print(f"       {line}")
            transcript_text += line + "\n"
            
        if os.path.exists("temp_chunk.wav"):
            os.remove("temp_chunk.wav")
            
    except Exception as e:
        print(f"    [!] Audio extraction failed: {e}")
        transcript_text = "No audio detected."

    print("    -> Watching video and extracting visual context...")
    visual_context_text = ""
    try:
        vidcap = cv2.VideoCapture(local_file_path)
        fps = vidcap.get(cv2.CAP_PROP_FPS)
        if fps == 0: fps = 30.0 
            
        frame_interval = int(fps * 5)
        success, image = vidcap.read()
        count = 0
        
        while success:
            if count % frame_interval == 0:
                rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                pil_image = Image.fromarray(rgb_image)
                messages = [{"role": "user", "content": [{"type": "image"}, {"type": "text", "text": "Describe this image in detail. What is the user seeing or doing?"}]}]
                prompt = processor.apply_chat_template(messages, add_generation_prompt=True)
                inputs = processor(text=prompt, images=[pil_image], return_tensors="pt").to("cuda")
                generated_ids = vlm_model.generate(**inputs, max_new_tokens=50)
                generated_ids_trimmed = [out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)]
                description = processor.batch_decode(generated_ids_trimmed, skip_special_tokens=True)[0].strip()
                
                timestamp_sec = count / fps
                log_line = f"[Frame @ {timestamp_sec:.2f}s]: {description}"
                print(f"       {log_line}")
                visual_context_text += log_line + "\n"
            
            success, image = vidcap.read()
            count += 1
            
        vidcap.release()
    except Exception as e:
        print(f"    [!] Visual extraction failed: {e}")
        visual_context_text = "Visual extraction failed."

    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    memory_file = os.path.join(TEXT_MEMORY_DIR, f"log_{filename}_{timestamp}.txt")
    with open(memory_file, "w", encoding="utf-8") as f:
        f.write(f"--- Raw Memory Log for {filename} ---\n")
        f.write("=== Audio Transcript ===\n")
        f.write(transcript_text + "\n")
        f.write("=== Visual Context ===\n")
        f.write(visual_context_text + "\n")
    print(f"[+] Memory securely saved to: {memory_file}")
    if os.path.exists(local_file_path):
        os.remove(local_file_path)
        print(f"[-] Local raw video {filename} securely purged from PC.")

def main():
    print("\n[*] Checking for connected devices...")
    stdout, _ = run_adb_command(["devices"])
    if not any("device" in line and "unauthorized" not in line for line in stdout.splitlines()[1:]):
        print("[!] No device connected or USB debugging not authorized.")
        return
    print("[*] Device found! Scanning phone for new life logs...")
    files = get_video_files_from_phone()
    if not files:
        print(f"[*] No new memories found in {PHONE_VIDEO_DIR}. Exiting.")
        return
    for filename in files:
        phone_file_path = f"{PHONE_VIDEO_DIR}{filename}"
        local_file_path = os.path.join(PC_TEMP_DIR, filename)
        print(f"\n[*] Pulling {filename} from phone...")
        _, code = run_adb_command(["pull", phone_file_path, local_file_path])
        if code == 0:
            print(f"[-] Deleting {filename} from phone...")
            run_adb_command(["shell", "rm", phone_file_path])
            process_and_purge(local_file_path, filename)
    print("\n[*] All tasks complete. Pipeline shutting down.")

if __name__ == "__main__":
    main()