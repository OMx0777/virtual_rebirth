import os
from huggingface_hub import snapshot_download

print("========================================")
print("  VIRTUAL REBIRTH: MODEL PRE-FETCHER")
print("========================================\n")

# 1. Whisper Base (Audio)
print("[1/3] Downloading Whisper (Base)...")
try:
    snapshot_download(repo_id="Systran/faster-whisper-base", local_files_only=False)
    print("      [+] Whisper downloaded successfully.\n")
except Exception as e:
    print(f"      [!] Whisper download failed: {e}\n")

# 2. SpeechBrain (Biometrics)
print("[2/3] Downloading SpeechBrain Voice Engine...")
try:
    snapshot_download(repo_id="speechbrain/spkrec-ecapa-voxceleb", local_files_only=False)
    print("      [+] SpeechBrain downloaded successfully.\n")
except Exception as e:
    print(f"      [!] SpeechBrain download failed: {e}\n")

# 3. SmolVLM (Vision)
print("[3/3] Downloading SmolVLM Vision Model...")
try:
    snapshot_download(repo_id="HuggingFaceTB/SmolVLM-256M-Instruct", local_files_only=False)
    print("      [+] SmolVLM downloaded successfully.\n")
except Exception as e:
    print(f"      [!] SmolVLM download failed: {e}\n")

print("========================================")
print(" All downloads attempted. You are good to go!")