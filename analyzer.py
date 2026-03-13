import os
import glob
import ollama
from datetime import datetime

# ==========================================
# CONFIGURATION
# ==========================================
MEMORY_DIR = "memory_logs"
INSIGHTS_DIR = "insights"

os.makedirs(INSIGHTS_DIR, exist_ok=True)

# ==========================================
# HELPER FUNCTIONS
# ==========================================
def get_latest_log():
    """Finds the most recently created text file in the memory directory."""
    list_of_files = glob.glob(f"{MEMORY_DIR}/*.txt")
    if not list_of_files:
        return None
    return max(list_of_files, key=os.path.getctime)

def analyze_memory(log_path):
    """Feeds the memory log to Llama 3.2 and saves the behavioral insights."""
    with open(log_path, 'r', encoding='utf-8') as f:
        memory_content = f.read()

    filename = os.path.basename(log_path)
    print(f"\n[*] Found latest memory: {filename}")
    print("[*] Initializing Virtual Rebirth Cognitive Engine (Llama 3.2)...")

    # This is the master prompt that turns raw data into human insight
    # This is the master prompt that turns raw data into human insight
    prompt = f"""
    You are the core analytical engine of "Virtual Rebirth", an AI system designed to create a digital twin of the user, whose name is Om.
    Analyze the following raw multimodal log (containing timestamped audio transcripts and visual frame descriptions) of Om's recent activity.

    RAW LOG:
    {memory_content}

    CRITICAL INSTRUCTION ON DIALOGUE: 
    The audio transcript uses biometric tagging. 
    - Lines tagged with `[Om]` are spoken by the primary user, Om. 
    - Lines tagged with `[Guest]` are spoken by other people (friends, guests, etc.).
    Your entire analysis MUST focus on building the digital twin of OM. You can note Om's interactions with guests, but do NOT profile the guests.

    Based strictly on the log provided, generate a structured profile using the following format:

    ### 1. BEHAVIORAL & CONTEXT SUMMARY
    (Write a concise paragraph summarizing exactly what Om was doing, where Om was, who Om was interacting with, and Om's dynamic/mood).

    ### 2. OM'S TRAIT & INTEREST EXTRACTION
    (List any personality traits, relationships, or technical interests demonstrated specifically BY OM in this log).

    ### 3. DIGITAL TWIN INTERVIEW QUESTIONS
    (Generate 3 highly specific, conversational questions to ask OM later. Address him directly as "you". These questions should prompt Om to elaborate on the events in the log so you can learn more about how his mind works).
    """
    print("    -> Analyzing multimodal context and extracting behavioral traits...")
    
    try:
        # Pinging your local Ollama server
        response = ollama.chat(model='llama3.2', messages=[
            {
                'role': 'user',
                'content': prompt
            }
        ])
        
        insight_text = response['message']['content']
        
        # Save the insights to a new file
        timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        insight_file = os.path.join(INSIGHTS_DIR, f"insight_{timestamp}.txt")
        
        with open(insight_file, "w", encoding="utf-8") as f:
            f.write(f"--- Cognitive Insights for {filename} ---\n\n")
            f.write(insight_text)
            
        print(f"[+] Analysis complete! Neural mapping saved to: {insight_file}\n")
        
        # Print it to the terminal so we can see the magic instantly
        print("================ VIRTUAL REBIRTH INSIGHTS ================")
        print(insight_text)
        print("==========================================================")

    except Exception as e:
        print(f"[!] Analysis failed. Is Ollama running? Error: {e}")

# ==========================================
# MAIN EXECUTION
# ==========================================
if __name__ == "__main__":
    latest_log = get_latest_log()
    if latest_log:
        analyze_memory(latest_log)
    else:
        print("[!] No memory logs found in the directory. Run usb_sync.py to extract some life logs first!")