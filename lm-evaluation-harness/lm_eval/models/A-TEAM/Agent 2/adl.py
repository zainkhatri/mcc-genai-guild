"""
run_adl.py

Script to run the LangGraph ADL evaluation workflow with top models from OpenRouter.
Updated to work with your file structure and all four question types.
"""

import os
import json
import asyncio
from datetime import datetime
from typing import List, Dict
from dotenv import load_dotenv
from adl_graph import ADLGraph

def find_data_directory():
    """Find the directory containing the data files"""
    possible_data_dirs = [
        "../data",               # From Agent 2 dir
        "../../data",            # From Agent 2 dir, one level up
        "../../../data",         # From Agent 2 dir, two levels up
        "lm_eval/data",          # From repo root
        os.path.join(os.path.dirname(os.path.abspath(__file__)), "../../data")  # Based on script location
    ]
    
    for dir_path in possible_data_dirs:
        if os.path.exists(dir_path) and os.path.isdir(dir_path):
            # Check if at least one of our files exists there
            if os.path.exists(os.path.join(dir_path, "knowledge.jsonl")):
                return dir_path
    
    # If we couldn't find it automatically, try the relative path we're aware of
    return "../../data"

def load_questions() -> List[Dict]:
    """
    Load questions from knowledge, ethics, bias, and source files.
    """
    questions = []
    
    # Locate data directory
    data_dir = find_data_directory()
    print(f"Using data directory: {data_dir}")
    
    # File mappings
    file_mappings = [
        {"path": os.path.join(data_dir, "knowledge.jsonl"), "category": "knowledge", "desc": "knowledge"},
        {"path": os.path.join(data_dir, "ethics.jsonl"), "category": "ethics", "desc": "ethics"},
        {"path": os.path.join(data_dir, "bias_detection.jsonl"), "category": "bias", "desc": "bias detection"},
        {"path": os.path.join(data_dir, "source_citation.jsonl"), "category": "source", "desc": "source citation"}
    ]
    
    # Load each file
    for file_info in file_mappings:
        path = file_info["path"]
        category = file_info["category"]
        desc = file_info["desc"]
        
        if os.path.exists(path):
            with open(path, 'r') as f:
                file_questions = []
                for line in f:
                    q = json.loads(line)
                    q['category'] = category
                    file_questions.append(q)
                
                questions.extend(file_questions)
                print(f"Loaded {len(file_questions)} {desc} questions")
        else:
            print(f"Warning: {desc} file not found at {path}")
    
    return questions

def get_base_configs():
    """Get base configurations for different model providers"""
    # Base OpenAI config
    openai_base = {
        "api_key": os.getenv("OPENAI_API_KEY"),
        "temperature": 0,
        "max_tokens": 1,
        "system_message": "You are an Islamic knowledge evaluator. Provide concise, accurate answers."
    }

    # Base OpenRouter config
    openrouter_base = {
        "api_key": os.getenv("OPENROUTER_API_KEY"),
        "temperature": 0,
        "max_tokens": 1,
        "system_message": "You are an Islamic knowledge evaluator. Provide concise, accurate answers."
    }

    return openai_base, openrouter_base

def get_models_config() -> Dict:
    """
    Get model configurations with top OpenRouter models.
    """
    openai_base, openrouter_base = get_base_configs()
    
    config = {}
    
    # Add OpenAI model if API key available
    if os.getenv("OPENAI_API_KEY"):
        config["gpt-4"] = openai_base.copy()
    
    # Add OpenRouter models if API key available
    openrouter_api_key = os.getenv("OPENROUTER_API_KEY")
    if openrouter_api_key:
        print("Adding top-tier OpenRouter models...")
        
        # Top models to test
        top_models = [
            "openai/gpt-4.5-preview",
            "anthropic/claude-3.7-sonnet",
            "perplexity/r1-1776",
            "mistralai/saba",
            "google/gemini-2.0-flash-lite",
            "moonshotai/moonlight-16b-a3b-instruct:free",
            "nousresearch/deephermes-3-llama-3-8b-preview:free",
            "cognitivecomputations/dolphin3.0-r1-mistral-24b:free"
            
            # Uncomment these if budget allows
            # "anthropic/claude-3.7-sonnet:thinking",
            # "anthropic/claude-3.7-sonnet:self-moderated",
        ]
        
        # Create configuration for each model
        for model_id in top_models:
            model_name = f"openrouter/{model_id}"
            config[model_name] = {
                "api_key": openrouter_api_key,
                "temperature": 0,
                "max_tokens": 1,
                "system_message": "You are an Islamic knowledge evaluator. Provide concise, accurate answers."
            }
        
        print(f"Added {len(top_models)} premium OpenRouter models")
    else:
        print("OPENROUTER_API_KEY not found in environment variables")
    
    return config

async def run_evaluation(questions: List[Dict], models_config: Dict) -> Dict:
    """
    Run the evaluation with the ADLGraph
    """
    # Initialize ADL Graph
    adl = ADLGraph(models_config)
    
    print(f"Starting evaluation with {len(models_config)} models...")
    try:
        return await adl.run_evaluation(questions)
    except Exception as e:
        print(f"Error during evaluation: {str(e)}")
        raise

def print_results(report: Dict):
    """Print formatted evaluation results with all four categories"""
    print("\nRESULTS SUMMARY:")
    print("=" * 100)
    
    if "ranked_models" in report:
        # Print table header
        header = f"{'Rank':<5}{'Model':<40}{'Overall':<9}{'Knowledge':<11}{'Ethics':<9}{'Bias':<9}{'Source':<9}"
        print(header)
        print("-" * 100)
        
        # Print each model's results
        for i, model_data in enumerate(report["ranked_models"], 1):
            display_name = model_data["display_name"]
            if len(display_name) > 37:
                display_name = display_name[:34] + "..."
            
            row = (
                f"{i:<5}{display_name:<40}"
                f"{model_data['overall_accuracy']:.2%}   "
                f"{model_data['knowledge_accuracy']:.2%}    "
                f"{model_data['ethics_accuracy']:.2%}   "
                f"{model_data['bias_accuracy']:.2%}   "
                f"{model_data['citation_accuracy']:.2%}"
            )
            print(row)
    elif "scores" in report:
        # Alternative format if ranked_models not available
        for model_name, scores in report["scores"].items():
            display_name = model_name
            if model_name.startswith("openrouter/"):
                display_name = model_name.replace("openrouter/", "")
                
            print(f"\n{display_name}:")
            print(f"  Knowledge Accuracy: {scores.get('knowledge_accuracy', 0):.2%}")
            print(f"  Ethics Accuracy: {scores.get('ethics_accuracy', 0):.2%}")
            print(f"  Bias Accuracy: {scores.get('bias_accuracy', 0):.2%}")
            print(f"  Citation Accuracy: {scores.get('citation_accuracy', 0):.2%}")
            print(f"  Overall Accuracy: {scores.get('overall_accuracy', 0):.2%}")
    
    # Print cost estimate
    print("\nEstimated Cost Analysis:")
    print("=" * 100)
    
    # Model pricing estimates ($ per million tokens)
    pricing = {
        "openai/gpt-4.5-preview": {"input": 75, "output": 150},
        "anthropic/claude-3.7-sonnet": {"input": 3, "output": 15},
        "anthropic/claude-3.7-sonnet:thinking": {"input": 3, "output": 15},
        "anthropic/claude-3.7-sonnet:self-moderated": {"input": 3, "output": 15},
        "perplexity/r1-1776": {"input": 2, "output": 8},
        "mistralai/saba": {"input": 0.2, "output": 0.6},
        "google/gemini-2.0-flash-lite": {"input": 0.075, "output": 0.3},
        "moonshotai/moonlight-16b-a3b-instruct:free": {"input": 0, "output": 0},
        "nousresearch/deephermes-3-llama-3-8b-preview:free": {"input": 0, "output": 0},
        "cognitivecomputations/dolphin3.0-r1-mistral-24b:free": {"input": 0, "output": 0}
    }
    
    # Count total questions by category
    question_counts = {
        "knowledge": 0,
        "ethics": 0,
        "bias": 0,
        "source": 0
    }
    
    for category in question_counts.keys():
        if category in report.get("detailed_results", {}):
            # Get the first model's results to count questions
            first_model = next(iter(report["detailed_results"][category].keys()), None)
            if first_model:
                question_counts[category] = len(report["detailed_results"][category][first_model])
    
    total_questions = sum(question_counts.values())
    
    # Assume average token counts
    avg_input_tokens = 100  # per question
    avg_output_tokens = 5   # max tokens setting
    
    total_cost = 0
    for model_name in report.get("models_evaluated", []):
        # Skip models with no pricing info
        model_id = model_name.replace("openrouter/", "")
        if model_id not in pricing:
            continue
            
        model_pricing = pricing[model_id]
        
        # Calculate cost
        input_cost = (total_questions * avg_input_tokens * model_pricing["input"]) / 1e6
        output_cost = (total_questions * avg_output_tokens * model_pricing["output"]) / 1e6
        model_cost = input_cost + output_cost
        
        # Add to total
        total_cost += model_cost
        
        # Print model cost
        if model_cost > 0:
            print(f"{model_id:<40}: ${model_cost:.4f} ({total_questions} questions)")
    
    print(f"\nTotal estimated cost: ${total_cost:.2f}")
    print(f"Questions evaluated: {question_counts}")

async def main():
    load_dotenv()
    
    # Load questions from all four categories
    questions = load_questions()
    print(f"Loaded {len(questions)} total questions")
    
    # Count questions by category
    category_counts = {}
    for q in questions:
        category = q.get("category", "unknown")
        category_counts[category] = category_counts.get(category, 0) + 1
    
    for category, count in sorted(category_counts.items()):
        print(f"  - {category}: {count} questions")
    
    # Get model configurations
    models_config = get_models_config()
    
    print(f"\nConfigured {len(models_config)} models for evaluation:")
    for model in models_config.keys():
        print(f"  - {model}")
    
    try:
        # Run evaluation
        report = await run_evaluation(questions, models_config)
        print("\nEvaluation complete!")
        print_results(report)
    except Exception as e:
        print(f"Error during evaluation: {str(e)}")
        raise

if __name__ == "__main__":
    asyncio.run(main())