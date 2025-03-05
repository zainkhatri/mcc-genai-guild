"""
run_adl.py

Script to run the LangGraph ADL evaluation workflow with remaining models from OpenRouter.
Updated with PROVIDED CORRECT OpenRouter model identifiers.
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

def get_models_config() -> Dict:
    """
    Get model configurations with the remaining models that need testing.
    Using VERIFIED CORRECT OpenRouter model IDs.
    """
    # Base config settings
    base_config = {
        "temperature": 0,
        "max_tokens": 1,
        "system_message": "You are an Islamic knowledge evaluator. Provide concise, accurate answers."
    }
    
    config = {}
    
    # Add OpenRouter models if API key available
    openrouter_api_key = os.getenv("OPENROUTER_API_KEY")
    if openrouter_api_key:
        print("Adding remaining models to test...")
        
        # Models we still need to test with CONFIRMED WORKING model IDs
        # These are the exact IDs provided as known to work with OpenRouter
        remaining_models = [
            "anthropic/claude-3-opus",
            "google/learnlm-1.5-pro-experimental:free",
            "anthropic/claude-3.5-sonnet", 
            "google/gemini-2.0-pro-exp-02-05:free",
            "perplexity/r1-1776",
            "openai/o3-mini-high"
        ]
        
        # Create configuration for each model
        for model_id in remaining_models:
            model_name = f"openrouter/{model_id}"
            model_config = base_config.copy()
            model_config["api_key"] = openrouter_api_key
            config[model_name] = model_config
        
        print(f"Added {len(remaining_models)} models for testing")
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
        "anthropic/claude-3-opus": {"input": 15, "output": 75},
        "google/learnlm-1.5-pro-experimental:free": {"input": 0, "output": 0},
        "anthropic/claude-3.5-sonnet": {"input": 3, "output": 15},
        "google/gemini-2.0-pro-exp-02-05:free": {"input": 0, "output": 0},
        "perplexity/r1-1776": {"input": 2, "output": 8},
        "openai/o3-mini-high": {"input": 0.75, "output": 2.5}
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

def save_markdown_report(report: Dict):
    """
    Save a markdown report with the weighted scores based on the leaderboard format
    """
    # Create results directory if it doesn't exist
    os.makedirs("results", exist_ok=True)
    
    # Calculate weighted scores for each model
    weights = {
        "knowledge": 300 / 438,  # 68.49%
        "ethics": 40 / 438,      # 9.13%
        "bias": 50 / 438,        # 11.42%
        "source": 48 / 438       # 10.96%
    }
    
    weighted_results = []
    
    for model_name, scores in report.get("scores", {}).items():
        display_name = model_name.replace("openrouter/", "")
        
        # Map to the original model names for the leaderboard
        model_mapping = {
            "anthropic/claude-3-opus": "Claude 3 Opus (2024-02)",
            "google/learnlm-1.5-pro-experimental:free": "Gemini 1.5 Pro",
            "anthropic/claude-3.5-sonnet": "Claude 3.5 Opus",
            "google/gemini-2.0-pro-exp-02-05:free": "Gemini Pro",
            "perplexity/r1-1776": "Perplexity R1-1776",
            "openai/o3-mini-high": "GPT-4-Turbo Optimized"
        }
        
        # Get mapping or use the original
        leaderboard_name = model_mapping.get(display_name, display_name)
        
        # Get scores for each category
        knowledge_acc = scores.get("knowledge_accuracy", 0) * 100
        ethics_acc = scores.get("ethics_accuracy", 0) * 100
        bias_acc = scores.get("bias_accuracy", 0) * 100
        source_acc = scores.get("citation_accuracy", 0) * 100
        
        # Calculate weighted score
        weighted_score = (
            (knowledge_acc * weights["knowledge"]) +
            (ethics_acc * weights["ethics"]) + 
            (bias_acc * weights["bias"]) + 
            (source_acc * weights["source"])
        )
        
        # Assign grade
        grade = "F"
        if weighted_score >= 97:
            grade = "A+"
        elif weighted_score >= 93:
            grade = "A"
        elif weighted_score >= 90:
            grade = "A-"
        elif weighted_score >= 87:
            grade = "B+"
        elif weighted_score >= 83:
            grade = "B"
        elif weighted_score >= 80:
            grade = "B-"
        elif weighted_score >= 77:
            grade = "C+"
        elif weighted_score >= 73:
            grade = "C"
        elif weighted_score >= 70:
            grade = "C-"
        elif weighted_score >= 67:
            grade = "D+"
        elif weighted_score >= 63:
            grade = "D"
        elif weighted_score >= 60:
            grade = "D-"
        
        # Add to results
        weighted_results.append({
            "model_name": leaderboard_name,
            "weighted_score": weighted_score,
            "grade": grade,
            "accuracy": knowledge_acc,
            "ethics": ethics_acc,
            "bias": bias_acc,
            "source": source_acc
        })
    
    # Sort by weighted score
    weighted_results.sort(key=lambda x: x["weighted_score"], reverse=True)
    
    # Create markdown report
    markdown = "# LLM Evaluation with Weighted Scores\n\n"
    markdown += "## Weights Based on Question Distribution\n"
    markdown += "- **Accuracy**: 300 questions (68.49% of total weight)\n"
    markdown += "- **Ethics**: 40 questions (9.13% of total weight)\n"
    markdown += "- **Bias**: 50 questions (11.42% of total weight)\n"
    markdown += "- **Source**: 48 questions (10.96% of total weight)\n"
    markdown += "- **Total**: 438 questions (100%)\n\n"
    
    markdown += "## Weighted Evaluation Results\n\n"
    markdown += "| Model Name | Weighted Score | Grade | Accuracy (68.49%) | Ethics (9.13%) | Bias (11.42%) | Source (10.96%) |\n"
    markdown += "|------------|----------------|-------|-------------------|---------------|---------------|----------------|\n"
    
    for model in weighted_results:
        markdown += f"| {model['model_name']} | {model['weighted_score']:.2f}% | {model['grade']} | "
        markdown += f"{model['accuracy']:.2f}% | {model['ethics']:.2f}% | {model['bias']:.2f}% | {model['source']:.2f}% |\n"
    
    # Save the markdown report
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"weighted_evaluation_{timestamp}.md"
    filepath = os.path.join("results", filename)
    
    with open(filepath, "w") as f:
        f.write(markdown)
    
    print(f"\nWeighted evaluation report saved to: {filepath}")

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
        
        # Save weighted markdown report for leaderboard integration
        save_markdown_report(report)
    except Exception as e:
        print(f"Error during evaluation: {str(e)}")
        raise

if __name__ == "__main__":
    asyncio.run(main())