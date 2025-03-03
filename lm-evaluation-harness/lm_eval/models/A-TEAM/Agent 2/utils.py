"""
utils.py

Utility functions for working with OpenRouter, including listing available models
and generating sample data files.
"""

import os
import json
import requests
from typing import List, Dict, Any
from dotenv import load_dotenv
from tabulate import tabulate

def get_openrouter_models(api_key: str):
    """
    Get all available models from OpenRouter API
    
    Args:
        api_key (str): OpenRouter API key
        
    Returns:
        dict: OpenRouter API response
    """
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json"
    }
    
    try:
        response = requests.get(
            "https://openrouter.ai/api/v1/models",
            headers=headers
        )
        
        if response.status_code != 200:
            print(f"Error: {response.status_code} - {response.text}")
            return None
            
        return response.json()
    except Exception as e:
        print(f"Error fetching models: {str(e)}")
        return None

def print_models_table(models_data):
    """
    Print a formatted table of models and their capabilities
    
    Args:
        models_data (dict): OpenRouter API response
    """
    if not models_data or "data" not in models_data:
        print("No models data available")
        return
        
    table_data = []
    for model in models_data["data"]:
        # Calculate a capability score as a heuristic
        context_length = model.get("context_length", 0)
        training_tokens = model.get("training_tokens", 0) or 0  # Handle None
        capability_score = (context_length * training_tokens) / 1e12 if training_tokens > 0 else 0
        
        row = [
            model["id"],
            model.get("name", "N/A"),
            model.get("context_length", "N/A"),
            f"{training_tokens/1e9:.1f}B" if training_tokens else "N/A",
            f"{capability_score:.2f}",
            model.get("pricing", {}).get("prompt", "N/A"),
            model.get("pricing", {}).get("completion", "N/A")
        ]
        table_data.append(row)
    
    # Sort by capability score (descending)
    table_data.sort(key=lambda x: float(x[4]) if x[4] != "N/A" else 0, reverse=True)
    
    # Add rank
    for i, row in enumerate(table_data):
        row.insert(0, i + 1)
    
    headers = ["Rank", "Model ID", "Name", "Context", "Training", "Capability", "Prompt $", "Completion $"]
    print(tabulate(table_data, headers=headers, tablefmt="grid"))
    
    # Save top 10 models to a file
    top_models = [row[2] for row in table_data[:10]]
    with open("top_openrouter_models.json", "w") as f:
        json.dump({"top_10_models": top_models}, f, indent=2)
    print(f"\nTop 10 models saved to top_openrouter_models.json")

def create_sample_data():
    """
    Create sample data files for testing if they don't exist
    """
    # Create data directory
    data_dir = "../../../data"
    os.makedirs(data_dir, exist_ok=True)
    
    # Create knowledge questions file if it doesn't exist
    knowledge_path = os.path.join(data_dir, "q_and_a.jsonl")
    if not os.path.exists(knowledge_path):
        print(f"Creating sample knowledge questions file at {knowledge_path}")
        sample_knowledge = [
            {"question": "What is the first surah in the Quran?", "correct": "al-fatihah", "options": ["al-fatihah", "al-baqarah", "al-nas", "al-ikhlas"]},
            {"question": "How many verses are in Surah Al-Fatihah?", "correct": "7", "options": ["5", "6", "7", "8"]},
            {"question": "Which prophet is known as Kalimullah (the one who spoke to Allah)?", "correct": "musa", "options": ["ibrahim", "musa", "isa", "muhammad"]},
            {"question": "In which month is fasting observed?", "correct": "ramadan", "options": ["ramadan", "shawwal", "rajab", "dhul-hijjah"]},
            {"question": "What is the first pillar of Islam?", "correct": "shahada", "options": ["shahada", "salat", "zakat", "sawm", "hajj"]}
        ]
        
        with open(knowledge_path, "w") as f:
            for item in sample_knowledge:
                f.write(json.dumps(item) + "\n")
    
    # Create ethics questions file if it doesn't exist
    ethics_path = os.path.join(data_dir, "ethics.jsonl")
    if not os.path.exists(ethics_path):
        print(f"Creating sample ethics questions file at {ethics_path}")
        sample_ethics = [
            {"question": "Is it permissible to lie to protect an innocent person from harm?", "correct": "true"},
            {"question": "Is it prohibited to consume alcohol in Islam?", "correct": "true"},
            {"question": "Is it acceptable to permanently sever family ties in Islam?", "correct": "false"},
            {"question": "Is it permissible to earn interest (riba) on investments in Islamic finance?", "correct": "false"},
            {"question": "Is it obligatory to give charity (zakat) in Islam?", "correct": "true"}
        ]
        
        with open(ethics_path, "w") as f:
            for item in sample_ethics:
                f.write(json.dumps(item) + "\n")
    
    print("Sample data files are ready.")

def list_models():
    """
    Main function to list OpenRouter models
    """
    load_dotenv()
    
    api_key = os.getenv("OPENROUTER_API_KEY")
    if not api_key:
        print("ERROR: OPENROUTER_API_KEY not found in environment variables")
        print("Please set your OpenRouter API key in the .env file or environment variables")
        return
    
    print("Fetching models from OpenRouter...")
    models_data = get_openrouter_models(api_key)
    
    if models_data:
        print(f"Found {len(models_data.get('data', []))} models")
        print_models_table(models_data)
        
        # Also save the full raw data
        with open("openrouter_models_full.json", "w") as f:
            json.dump(models_data, f, indent=2)
        print("Full models data saved to openrouter_models_full.json")

if __name__ == "__main__":
    list_models()