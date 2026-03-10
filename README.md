Step 1: Create the file

Run this command in your terminal to create the file and open it in nano (or use your favorite code editor like VS Code):
Bash

nano README.md

Step 2: Paste this documentation

Copy and paste this entire block into the file:
Markdown

# Virtual Rebirth - Backend Extraction Pipeline

## Project Overview
Virtual Rebirth is a local, privacy-first multimodal AI pipeline. The ultimate goal of this project is to create a "digital twin" by capturing daily life logs (video and audio) and processing them into dense, structured behavioral text memories. 

This repository contains Phase 1: **The Multimodal Extraction Engine**.

## Architecture & Hardware Optimization
This pipeline was engineered to run locally on an Arch Linux environment with strict VRAM constraints (4GB RTX 3050 Mobile) by splitting workloads between the CPU and GPU.

* **Mobile Sync (ADB):** Automatically detects the Android device, pulls raw daily video logs (`.mp4`) from the `/Movies/VirtualRebirth/` directory, and securely purges them from the device.
* **Audio Extraction (Whisper):** Uses `faster-whisper` (Base model). To preserve GPU memory, audio transcription is explicitly offloaded to the CPU (int8 compute type).
* **Visual Extraction (SmolVLM):** Uses Hugging Face's `SmolVLM-256M-Instruct`. A highly optimized 256-parameter Vision-Language Model running natively in `float16` on the GPU. It samples 1 frame every 5 seconds to generate spatial and contextual awareness.

## Pipeline Flow
1. **Pull:** Raw video is pulled via USB debugging.
2. **Listen:** Whisper generates a timestamped audio transcript.
3. **Watch:** SmolVLM generates timestamped visual descriptions.
4. **Log:** Data is compiled into a single `.txt` file inside `/memory_logs/`.
5. **Purge:** The raw `.mp4` is permanently deleted from the local PC to save storage.

## Installation & Setup

**Prerequisites:** * Arch Linux (or any Linux distro)
* Android device with USB Debugging enabled
* `uv` package manager (for lightning-fast Python environments)

**1. Clone the repository**
```bash
git clone [https://github.com/YOUR_USERNAME/virtual_rebirth_backend.git](https://github.com/YOUR_USERNAME/virtual_rebirth_backend.git)
cd virtual_rebirth_backend

2. Setup the Environment (Python 3.11 recommended)
Bash

uv venv --python 3.11 venv
source venv/bin/activate

3. Install Dependencies
Bash

uv pip install -r requirements.txt

4. Run the Pipeline
Bash

python usb_sync.py

Next Steps (Phase 2)

    Integrate a local LLM via Ollama to analyze the memory_logs/ daily.

    Generate behavioral profiling and automated interview questions based on the day's events to further train the digital twin.


### Step 3: Save and Commit
If you used `nano`, press `Ctrl + O`, hit `Enter` to save, and then `Ctrl + X` to exit.

Since you already initialized Git, you just need to add this new file to your staging area before you do your final push:

```bash
git add README.md
git commit -m "docs: added project architecture and setup documentation"