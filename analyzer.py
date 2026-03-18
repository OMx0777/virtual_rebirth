import os
import glob
import json
import re
import ollama
from datetime import datetime

# ==========================================
# CONFIGURATION
# ==========================================
MEMORY_DIR = "memory_logs"
INSIGHTS_DIR = "insights"
CORE_MEMORY_FILE = "core_memory.json"

os.makedirs(INSIGHTS_DIR, exist_ok=True)

# Ensure core_memory.json exists so we don't crash on the first run
if not os.path.exists(CORE_MEMORY_FILE):
    with open(CORE_MEMORY_FILE, 'w', encoding='utf-8') as f:
        json.dump({"facts": [], "preferences": [], "relationships": {}}, f, indent=4)

# ==========================================
# HELPER FUNCTIONS
# ==========================================
def get_all_unprocessed_logs():
    """Grabs EVERY raw text file in the memory directory."""
    return glob.glob(f"{MEMORY_DIR}/*.txt")

def update_core_memory(extracted_data):
    """Safely merges the newly extracted AI data into the permanent core_memory.json."""
    with open(CORE_MEMORY_FILE, 'r', encoding='utf-8') as f:
        core_mem = json.load(f)

    added_count = 0

    # 1. Update Facts (avoiding duplicates)
    for fact in extracted_data.get("new_facts", []):
        if fact not in core_mem["facts"]:
            core_mem["facts"].append(fact)
            added_count += 1

    # 2. Update Preferences
    for pref in extracted_data.get("new_preferences", []):
        if pref not in core_mem["preferences"]:
            core_mem["preferences"].append(pref)
            added_count += 1

    # 3. Update Relationships
    for name, desc in extracted_data.get("new_relationships", {}).items():
        if name not in core_mem["relationships"] or core_mem["relationships"][name] != desc:
            core_mem["relationships"][name] = desc
            added_count += 1

    # Save it back to the master drive
    with open(CORE_MEMORY_FILE, 'w', encoding='utf-8') as f:
        json.dump(core_mem, f, indent=4)
        
    print(f"    [+] Successfully merged {added_count} new data points into core memory.")

def analyze_memory(log_path):
    """Feeds a single memory log to Llama 3.2 and updates the JSON memory."""
    with open(log_path, 'r', encoding='utf-8') as f:
        memory_content = f.read()

    filename = os.path.basename(log_path)
    print(f"\n[*] Initializing Cognitive Engine (Llama 3.2) for: {filename}...")

    # The prompt FORCES Llama to reply in pure JSON based on the biometric tags
    prompt = f"""
    You are the core analytical engine of "Virtual Rebirth", an AI system designed to create a digital twin of the user, whose name is Om.
    Analyze the following raw multimodal log.

    RAW LOG:
    {memory_content}

    CRITICAL INSTRUCTION ON DIALOGUE: 
    - Lines tagged with `[Om]` are spoken by the primary user, Om. 
    - Lines tagged with `[Guest]` are spoken by other people.
    Do NOT profile the guests. Focus entirely on extracting data about Om.

    OUTPUT FORMAT:
    You MUST output your analysis as a valid JSON object matching this exact structure. Do not include any text outside of the JSON block.
    {{
        "summary": "A concise paragraph summarizing the context of this log.",
        "new_facts": ["Om did X", "Om was wearing Y"],
        "new_preferences": ["Om expressed a dislike for Z", "Om wants to achieve A"],
        "new_relationships": {{
            "GuestName": "Description of who they are to Om based on this log"
        }},
        "interview_questions": [
            "A specific question to ask Om later about this event",
            "Another specific question"
        ]
    }}
    """
    print("    -> Analyzing multimodal context and extracting JSON traits...")
    
    try:
        response = ollama.chat(model='llama3.2', messages=[
            {
                'role': 'user',
                'content': prompt
            }
        ])
        
        raw_output = response['message']['content']
        
        # Robust regex to strip out Markdown formatting if Llama tries to be helpful
        clean_json_str = re.sub(r"```json\n|\n```|```", "", raw_output).strip()
        
        # Parse the JSON string into a Python dictionary
        extracted_data = json.loads(clean_json_str)
        
        # Save the daily insight to a file for the Chat Engine to use later
        timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        insight_file = os.path.join(INSIGHTS_DIR, f"insight_{timestamp}.json")
        
        with open(insight_file, "w", encoding="utf-8") as f:
            json.dump(extracted_data, f, indent=4)
            
        print(f"    [+] Neural mapping saved to: {insight_file}")
        
        # Merge the extracted data directly into the master memory file
        update_core_memory(extracted_data)
        
    except json.JSONDecodeError as e:
        print(f"    [!] AI failed to return valid JSON. Error: {e}")
        print("    Raw AI Output was:\n", raw_output)
    except Exception as e:
        print(f"    [!] Analysis failed. Is Ollama running? Error: {e}")

# ==========================================
# MAIN EXECUTION (The Batch Processor)
# ==========================================
if __name__ == "__main__":
    all_logs = get_all_unprocessed_logs()
    
    if not all_logs:
        print("[*] No raw memory logs found in the directory. Run usb_sync.py first!")
    else:
        print(f"[*] Found {len(all_logs)} unprocessed memory logs. Starting batch extraction...\n")
        
        for log_path in all_logs:
            # 1. Feed it to Llama 3.2 to extract JSON traits
            analyze_memory(log_path)
            
            # 2. Burn after reading! Delete the raw text file so we don't process it again.
            try:
                os.remove(log_path)
                print(f"[-] Purged raw text log: {os.path.basename(log_path)}\n")
            except Exception as e:
                print(f"[!] Failed to delete {log_path}: {e}")
                
        print("[+] Batch processing complete! All memories successfully integrated into the Core Drive.")