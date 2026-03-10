import os
import subprocess
import cv2
import torch
from datetime import datetime
from PIL import Image
from faster_whisper import WhisperModel
# Notice the new class name here!
from transformers import AutoProcessor, AutoModelForImageTextToText

# ==========================================
# CONFIGURATION
# ==========================================
PHONE_VIDEO_DIR = "/storage/emulated/0/Movies/VirtualRebirth/" 
PC_TEMP_DIR = "temp_videos"
TEXT_MEMORY_DIR = "memory_logs"

os.makedirs(PC_TEMP_DIR, exist_ok=True)
os.makedirs(TEXT_MEMORY_DIR, exist_ok=True)

# ==========================================
# INITIALIZE AI MODELS
# ==========================================
print("[*] Loading Whisper AI (CPU) for audio extraction...")
whisper_model = WhisperModel("base", device="cpu", compute_type="int8")

print("[*] Loading SmolVLM (GPU, 256M Native) for visual extraction...")
# We switched to SmolVLM! It fits natively in your 4GB VRAM without ANY compression.
model_id = "HuggingFaceTB/SmolVLM-256M-Instruct"
processor = AutoProcessor.from_pretrained(model_id)

# Using the brand new Transformers 5.0 class name
vlm_model = AutoModelForImageTextToText.from_pretrained(
    model_id, 
    torch_dtype=torch.float16,
).to("cuda")

# ==========================================
# HELPER FUNCTIONS
# ==========================================
def run_adb_command(command: list):
    """Executes an ADB command and returns the output."""
    result = subprocess.run(["adb"] + command, capture_output=True, text=True)
    return result.stdout.strip(), result.returncode

def get_video_files_from_phone():
    """Gets a list of video files from the phone's directory."""
    stdout, code = run_adb_command(["shell", "ls", PHONE_VIDEO_DIR])
    if code != 0 or "No such file or directory" in stdout:
        return []
    return [f.strip() for f in stdout.split('\n') if f.strip().endswith(('.mp4', '.mkv', '.avi'))]

def process_and_purge(local_file_path: str, filename: str):
    """Extracts data and deletes the raw video from the PC."""
    print(f"\n[*] Starting extraction pipeline for {filename}...")
    
    # --- 1. Audio Extraction ---
    print("    -> Listening to audio and transcribing...")
    transcript_text = ""
    try:
        segments, _ = whisper_model.transcribe(local_file_path, beam_size=5)
        for segment in segments:
            line = f"[{segment.start:.2f}s -> {segment.end:.2f}s] {segment.text}"
            print(f"       {line}")
            transcript_text += line + "\n"
    except Exception as e:
        print(f"    [!] Audio extraction failed: {e}")
        transcript_text = "No audio detected."

    # --- 2. Vision Extraction ---
    print("    -> Watching video and extracting visual context...")
    visual_context_text = ""
    try:
        vidcap = cv2.VideoCapture(local_file_path)
        fps = vidcap.get(cv2.CAP_PROP_FPS)
        
        if fps == 0:
            fps = 30.0 
            
        frame_interval = int(fps * 5) # Extract 1 frame every 5 seconds
        success, image = vidcap.read()
        count = 0
        
        while success:
            if count % frame_interval == 0:
                # Convert OpenCV's BGR format to standard RGB
                rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                pil_image = Image.fromarray(rgb_image)
                
                # Format the prompt exactly how SmolVLM expects it
                messages = [
                    {
                        "role": "user",
                        "content": [
                            {"type": "image"},
                            {"type": "text", "text": "Describe this image in detail. What is the user seeing or doing?"}
                        ]
                    }
                ]
                
                prompt = processor.apply_chat_template(messages, add_generation_prompt=True)
                inputs = processor(text=prompt, images=[pil_image], return_tensors="pt").to("cuda")
                
                generated_ids = vlm_model.generate(**inputs, max_new_tokens=50)
                
                # Trim the prompt tokens out so we only log the new description
                generated_ids_trimmed = [
                    out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
                ]
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

    # --- 3. Save Memory Log ---
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

# ==========================================
# MAIN EXECUTION
# ==========================================
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
