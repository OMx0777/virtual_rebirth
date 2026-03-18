import json
import os

class MemoryManager:
    def __init__(self, filepath="core_memory.json"):
        self.filepath = filepath
        self.memory = self._load_memory()

    def _load_memory(self):
        """Loads the JSON file or initializes a default structure."""
        if os.path.exists(self.filepath):
            with open(self.filepath, 'r') as file:
                return json.load(file)
        return {"facts": [], "preferences": [], "relationships": {}}

    def _save_memory(self):
        """Saves the current state back to the JSON file."""
        with open(self.filepath, 'w') as file:
            json.dump(self.memory, file, indent=4)

    def add_fact(self, fact: str):
        """Adds a fact if it doesn't already exist."""
        if fact not in self.memory["facts"]:
            self.memory["facts"].append(fact)
            self._save_memory()

    def add_relationship(self, name: str, description: str):
        """Adds or updates a relationship."""
        self.memory["relationships"][name] = description
        self._save_memory()

    def generate_system_prompt(self) -> str:
        """Translates the JSON memory into a context prompt for an LLM."""
        prompt_lines = ["You are interacting with Om. Here is what you know about him:\n"]
        
        if self.memory.get("facts"):
            prompt_lines.append("**Core Facts:**")
            for fact in self.memory["facts"]:
                prompt_lines.append(f"- {fact}")
                
        if self.memory.get("preferences"):
            prompt_lines.append("\n**Preferences & Goals:**")
            for pref in self.memory["preferences"]:
                prompt_lines.append(f"- {pref}")
                
        if self.memory.get("relationships"):
            prompt_lines.append("\n**Known Relationships:**")
            for name, desc in self.memory["relationships"].items():
                prompt_lines.append(f"- {name}: {desc}")

        return "\n".join(prompt_lines)

# --- Example Usage ---
if __name__ == "__main__":
    # 1. Initialize the manager
    om_memory = MemoryManager("core_memory.json")
    
    # 2. Add a new fact dynamically
    om_memory.add_fact("Om prefers working in a virtual environment.")
    
    # 3. Generate the context prompt to feed to your LLM
    system_context = om_memory.generate_system_prompt()
    
    print("--- Generated LLM Context ---")
    print(system_context)