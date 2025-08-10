import os
from openai import OpenAI
from dotenv import load_dotenv
import time # Import time for potential rate limiting

# --- 1. SETUP ---
load_dotenv()
try:
    client = OpenAI()
except Exception as e:
    print("Error initializing OpenAI client. Is your OPENAI_API_KEY set in the .env file?")
    print(e)
    exit()

MODEL_NAME = "gpt-5"
TEMPERATURE = 0
REASONING = "minimal"
max_completion_tokens = 4000

# --- 2. LOAD ASSETS FROM FILES ---
def load_text(filepath):
    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            return f.read()
    except FileNotFoundError:
        print(f"Error: File not found at {filepath}")
        return ""

# Load the new, structured prompt components
system_prompt = load_text('data/system_prompt.md')
user_prompt_template = load_text('data/user_prompt.md')
case_scenario = load_text('data/Il caso.md')

# --- 3. DEFINE THE EXPERIMENTAL CONDITIONS ---
def get_rag_context(condition_name):
    if condition_name == "no_rag":
        return "Nessun documento legale fornito."
    elif condition_name == "ideal_rag":
        return load_text('data/Paragrafo c-sexies Articolo 67.md')
    elif condition_name == "noisy_rag":
        return load_text('data/Testo unico del 22.12.1986 n.917 Articolo 67.md')
    elif condition_name == "complex_rag":
        doc1 = load_text('data/Paragrafo c-sexies Articolo 67.md')
        doc2 = load_text('data/Estratte dalla Circolare N30.md')
        doc3 = load_text('data/Faq del 30 aprile 2025. Tassazione sostitutiva delle.md')
        doc4 = load_text('data/Estratte della Legge di Bilancio 2025.md')
        return f"DOCUMENTO 1:\n{doc1}\n\nDOCUMENTO 2:\n{doc2}\n\nDOCUMENTO 3:\n{doc3}\n\nDOCUMENTO 4:\n{doc4}"
    else:
        return ""

# --- 4. MAIN EXPERIMENT EXECUTION ---
def run_condition(condition_name):
    print(f"--- Running condition: {condition_name} ---")
    
    rag_context = get_rag_context(condition_name)
    
    # Use the template to construct the final user prompt
    user_prompt = user_prompt_template.format(
        case_scenario=case_scenario,
        rag_context=rag_context
    )
    
    try:
        # Call the OpenAI API using the structured messages format
        response = client.chat.completions.create(
            model=MODEL_NAME,
            ## reasoning_effort=REASONING,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ],
            temperature=TEMPERATURE,
            reasoning_effort=REASONING,
            max_completion_tokens=max_completion_tokens
        )
        
        generated_opinion = response.choices[0].message.content or ""
        
        output_path = f'outputs/output_{condition_name}.txt'
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write(generated_opinion)
            
        print(f"Successfully generated and saved output to {output_path}\n")
        
    except Exception as e:
        print(f"An error occurred during the API call for condition '{condition_name}': {e}")

if __name__ == "__main__":
    conditions = ["no_rag", "ideal_rag", "noisy_rag", "complex_rag"]
    os.makedirs("outputs", exist_ok=True)
    
    for condition in conditions:
        run_condition(condition)
        # Optional: Add a small delay to avoid hitting API rate limits
        time.sleep(5) 
        
    print("--- Experiment complete. All outputs are in the 'outputs' folder. ---")
