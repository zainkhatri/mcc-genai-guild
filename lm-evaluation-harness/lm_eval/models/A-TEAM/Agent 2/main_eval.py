"""
main_eval.py

Comprehensive evaluation script that uses your existing data files:
- knowledge.jsonl (300 questions)
- ethics.jsonl (40 questions)
- bias_detection.jsonl (50 questions)
- source_citation.jsonl (48 questions)
"""

import os
import json
import sys
from typing import List, Dict, Any
from dotenv import load_dotenv
from datetime import datetime

def load_questions(data_dir: str) -> List[Dict[str, Any]]:
    """
    Load questions from your specific data files.
    Looks for files in the provided data directory.
    """
    questions = []
    
    # Define file paths and categories
    file_mappings = [
        {"path": os.path.join(data_dir, "knowledge.jsonl"), "category": "knowledge"},
        {"path": os.path.join(data_dir, "ethics.jsonl"), "category": "ethics"},
        {"path": os.path.join(data_dir, "bias_detection.jsonl"), "category": "bias"},
        {"path": os.path.join(data_dir, "source_citation.jsonl"), "category": "source"}
    ]
    
    # Load from each file
    for file_info in file_mappings:
        path = file_info["path"]
        category = file_info["category"]
        
        if os.path.exists(path):
            with open(path, 'r') as f:
                file_questions = []
                for line in f:
                    q = json.loads(line)
                    q['category'] = category
                    file_questions.append(q)
                
                questions.extend(file_questions)
                print(f"Loaded {len(file_questions)} {category} questions from {path}")
        else:
            print(f"Warning: File not found at {path}")
    
    return questions

def get_models_config():
    """Get model configurations for top models"""
    models = []
    
    # Add OpenAI model if API key available
    if os.getenv("OPENAI_API_KEY"):
        models.append({
            "model_name": "gpt-4",
            "api_key": os.getenv("OPENAI_API_KEY"),
            "temperature": 0,
            "max_tokens": 5,
            "system_message": "You are an Islamic knowledge evaluator. Provide concise, accurate answers."
        })
    
    # Add OpenRouter models
    if os.getenv("OPENROUTER_API_KEY"):
        # Define models to test based on budget considerations
        # You can comment out expensive models to reduce costs
        top_models = [
            # Most expensive models
            "openai/gpt-4.5-preview",  # Most expensive
            "anthropic/claude-3.7-sonnet",
            
            # Medium-priced models
            "perplexity/r1-1776",
            "mistralai/saba",
            
            # Lower-cost models
            "google/gemini-2.0-flash-lite",
            
            # Free models
            "moonshotai/moonlight-16b-a3b-instruct:free",
            "nousresearch/deephermes-3-llama-3-8b-preview:free",
            "cognitivecomputations/dolphin3.0-r1-mistral-24b:free"
            
            # Uncomment these if budget allows
            # "anthropic/claude-3.7-sonnet:thinking",
            # "anthropic/claude-3.7-sonnet:self-moderated",
        ]
        
        for model_id in top_models:
            models.append({
                "model_name": f"openrouter/{model_id}",
                "api_key": os.getenv("OPENROUTER_API_KEY"),
                "temperature": 0,
                "max_tokens": 5,
                "system_message": "You are an Islamic knowledge evaluator. Provide concise, accurate answers."
            })
    
    return models

def print_results(results_by_model: Dict[str, List[Dict]]):
    """Print formatted evaluation results with category breakdowns"""
    print("\nRAW RESULTS BY CATEGORY:")
    print("=" * 100)
    
    # Calculate scores for each model and category
    model_scores = []
    for model_name, results in results_by_model.items():
        if not results:
            continue
            
        # Split by category
        knowledge_results = [r for r in results if r.get("category") == "knowledge"]
        ethics_results = [r for r in results if r.get("category") == "ethics"]
        bias_results = [r for r in results if r.get("category") == "bias"]
        source_results = [r for r in results if r.get("category") == "source"]
        
        # Calculate accuracy for each category
        knowledge_acc = sum(1 for r in knowledge_results if r["correct"]) / len(knowledge_results) if knowledge_results else 0
        ethics_acc = sum(1 for r in ethics_results if r["correct"]) / len(ethics_results) if ethics_results else 0
        bias_acc = sum(1 for r in bias_results if r["correct"]) / len(bias_results) if bias_results else 0
        source_acc = sum(1 for r in source_results if r["correct"]) / len(source_results) if source_results else 0
        
        # Calculate overall accuracy
        total_correct = sum(1 for r in results if r["correct"])
        overall_acc = total_correct / len(results) if results else 0
        
        # Display name for OpenRouter models
        display_name = model_name
        if model_name.startswith("openrouter/"):
            display_name = model_name.replace("openrouter/", "")
            
        model_scores.append((
            display_name, 
            overall_acc, 
            knowledge_acc, 
            ethics_acc, 
            bias_acc, 
            source_acc,
            len(knowledge_results),
            len(ethics_results),
            len(bias_results),
            len(source_results)
        ))
    
    # Sort by overall accuracy
    model_scores.sort(key=lambda x: x[1], reverse=True)
    
    # Print table header
    header = f"{'Rank':<5}{'Model':<40}{'Overall':<9}{'Knowledge':<11}{'Ethics':<9}{'Bias':<9}{'Source':<9}"
    print(header)
    print("-" * 100)
    
    # Print results
    for i, (model, overall, knowledge, ethics, bias, source, k_count, e_count, b_count, s_count) in enumerate(model_scores, 1):
        # Truncate long model names
        display_name = model
        if len(display_name) > 37:
            display_name = display_name[:34] + "..."
        
        # Format with category counts
        row = f"{i:<5}{display_name:<40}{overall:.2%}   {knowledge:.2%} ({k_count})"
        if e_count > 0:
            row += f"  {ethics:.2%} ({e_count})"
        else:
            row += f"  {'N/A':<7}"
            
        if b_count > 0:
            row += f"  {bias:.2%} ({b_count})"
        else:
            row += f"  {'N/A':<7}"
            
        if s_count > 0:
            row += f"  {source:.2%} ({s_count})"
        else:
            row += f"  {'N/A':<7}"
            
        print(row)
    
    # Calculate and print cost estimate
    print("\nEstimated Cost Analysis:")
    print("=" * 100)
    
    question_count = sum(len(results) for results in results_by_model.values() if results)
    model_count = len([m for m in results_by_model.keys() if results_by_model[m]])
    
    # Model pricing estimates
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
    
    # Assume average token counts
    avg_input_tokens = 100  # per question
    avg_output_tokens = 5   # max tokens setting
    
    total_cost = 0
    for model, results in results_by_model.items():
        if not results:
            continue
            
        # Get pricing or default to zero
        model_pricing = pricing.get(model.replace("openrouter/", ""), {"input": 0, "output": 0})
        
        # Calculate cost
        input_cost = (len(results) * avg_input_tokens * model_pricing["input"]) / 1e6
        output_cost = (len(results) * avg_output_tokens * model_pricing["output"]) / 1e6
        model_cost = input_cost + output_cost
        
        # Add to total
        total_cost += model_cost
        
        # Print model cost
        if model_cost > 0:
            print(f"{model.replace('openrouter/', ''):<40}: ${model_cost:.4f} ({len(results)} questions)")
    
    print(f"\nTotal estimated cost: ${total_cost:.2f}")
    print(f"Average cost per model: ${(total_cost/model_count):.4f} (if all charged)")
    print(f"Average cost per question per model: ${(total_cost/question_count):.6f}")

def save_results(results_by_model: Dict[str, List[Dict]]):
    """Save evaluation results to file"""
    os.makedirs("results", exist_ok=True)
    timestamp = datetime.now().isoformat()
    
    # Create report structure
    report = {
        "timestamp": timestamp,
        "results_by_model": results_by_model,
        "ranked_models": [],
        "category_performance": {}
    }
    
    # Add ranking and category analysis
    for model_name, results in results_by_model.items():
        if not results:
            continue
            
        # Split by category
        knowledge_results = [r for r in results if r.get("category") == "knowledge"]
        ethics_results = [r for r in results if r.get("category") == "ethics"]
        bias_results = [r for r in results if r.get("category") == "bias"]
        source_results = [r for r in results if r.get("category") == "source"]
        
        # Calculate accuracy for each category
        knowledge_acc = sum(1 for r in knowledge_results if r["correct"]) / len(knowledge_results) if knowledge_results else 0
        ethics_acc = sum(1 for r in ethics_results if r["correct"]) / len(ethics_results) if ethics_results else 0
        bias_acc = sum(1 for r in bias_results if r["correct"]) / len(bias_results) if bias_results else 0
        source_acc = sum(1 for r in source_results if r["correct"]) / len(source_results) if source_results else 0
        
        # Calculate overall accuracy
        total_correct = sum(1 for r in results if r["correct"])
        overall_acc = total_correct / len(results) if results else 0
        
        # Display name for OpenRouter models
        display_name = model_name
        if model_name.startswith("openrouter/"):
            display_name = model_name.replace("openrouter/", "")
        
        # Add to ranked models
        report["ranked_models"].append({
            "model": model_name,
            "display_name": display_name,
            "overall_accuracy": overall_acc,
            "knowledge_accuracy": knowledge_acc,
            "ethics_accuracy": ethics_acc,
            "bias_accuracy": bias_acc,
            "source_accuracy": source_acc,
            "question_counts": {
                "knowledge": len(knowledge_results),
                "ethics": len(ethics_results),
                "bias": len(bias_results),
                "source": len(source_results),
                "total": len(results)
            }
        })
        
        # Add to category performance
        for category in ["knowledge", "ethics", "bias", "source"]:
            if category not in report["category_performance"]:
                report["category_performance"][category] = []
                
            # Get category-specific results
            cat_results = [r for r in results if r.get("category") == category]
            if not cat_results:
                continue
                
            # Calculate accuracy
            cat_acc = sum(1 for r in cat_results if r["correct"]) / len(cat_results)
            
            # Add to category performance
            report["category_performance"][category].append({
                "model": model_name,
                "display_name": display_name,
                "accuracy": cat_acc,
                "question_count": len(cat_results)
            })
    
    # Sort ranked models by overall accuracy
    report["ranked_models"].sort(key=lambda x: x["overall_accuracy"], reverse=True)
    
    # Sort category performance by accuracy
    for category in report["category_performance"]:
        report["category_performance"][category].sort(key=lambda x: x["accuracy"], reverse=True)
    
    # Save report
    filename = f"evaluation_comprehensive_{timestamp}.json"
    report_path = os.path.join("results", filename)
    with open(report_path, "w") as f:
        json.dump(report, f, indent=2)
    print(f"\nComprehensive report saved to: {report_path}")
    
    # Save a simplified version with just the rankings
    ranking_report = {
        "timestamp": timestamp,
        "ranked_models": report["ranked_models"],
        "category_leaders": {}
    }
    
    # Add category leaders
    for category, models in report["category_performance"].items():
        if models:
            ranking_report["category_leaders"][category] = models[0]["display_name"]
    
    ranking_filename = f"model_ranking_{timestamp}.json"
    ranking_path = os.path.join("results", ranking_filename)
    with open(ranking_path, "w") as f:
        json.dump(ranking_report, f, indent=2)
    print(f"Simplified ranking saved to: {ranking_path}")

def main():
    load_dotenv()

    # Get model configurations
    model_configs = get_models_config()
    if not model_configs:
        print("No model configurations available. Please set API keys in .env file.")
        return
        
    print(f"Evaluating {len(model_configs)} models:")
    for config in model_configs:
        print(f"  - {config['model_name']}")

    # Determine proper data directory path
    # Try a few common relative paths
    possible_data_dirs = [
        "../../data",             # If running from Agent 2 dir
        "../../../data",          # Another possible location
        "../lm_eval/data",        # Another possible location
        "lm_eval/data",           # If running from repo root
        os.path.join(os.path.dirname(os.path.abspath(__file__)), "../../data")  # Based on script location
    ]
    
    data_dir = None
    for dir_path in possible_data_dirs:
        if os.path.exists(dir_path) and os.path.isdir(dir_path):
            # Check if at least one of our files exists there
            if os.path.exists(os.path.join(dir_path, "knowledge.jsonl")):
                data_dir = dir_path
                break
    
    if not data_dir:
        # Ask the user for the data directory
        print("\nCould not automatically locate your data files.")
        print("Please enter the full path to the directory containing your data files")
        print("(knowledge.jsonl, ethics.jsonl, bias_detection.jsonl, source_citation.jsonl):")
        data_dir = input("> ").strip()
        
        if not os.path.exists(data_dir) or not os.path.isdir(data_dir):
            print(f"Error: The specified directory does not exist: {data_dir}")
            return
    
    print(f"\nUsing data directory: {data_dir}")
    
    # Load questions from the data directory
    questions = load_questions(data_dir)
    
    if not questions:
        print("No questions loaded.")
        return
    
    print(f"\nTotal questions: {len(questions)}")
    category_counts = {}
    for q in questions:
        category = q.get("category", "unknown")
        category_counts[category] = category_counts.get(category, 0) + 1
    
    for category, count in category_counts.items():
        print(f"  - {category}: {count} questions")

    # Evaluate each model
    results_by_model = {}
    for config in model_configs:
        model_name = config.pop("model_name")
        print(f"\nEvaluating {model_name}...")
        try:
            evaluator = create_evaluator(model_name, **config)
            
            # Evaluate with progress updates
            results = []
            batch_size = 20  # Process in batches to show progress
            
            for i in range(0, len(questions), batch_size):
                batch = questions[i:i+batch_size]
                batch_results = evaluator.batch_evaluate(batch)
                results.extend(batch_results)
                
                # Show progress
                batch_acc = sum(r["correct"] for r in batch_results) / len(batch_results)
                print(f"  Batch {i//batch_size + 1}/{(len(questions)+batch_size-1)//batch_size} - Accuracy: {batch_acc:.2%}")
            
            results_by_model[model_name] = results
            
            # Calculate overall accuracy
            accuracy = sum(r["correct"] for r in results) / len(results)
            print(f"{model_name} overall accuracy: {accuracy:.2%}")
            
        except Exception as e:
            print(f"Error evaluating {model_name}: {str(e)}")
            results_by_model[model_name] = []
    
    # Print and save results
    print_results(results_by_model)
    save_results(results_by_model)

if __name__ == "__main__":
    main()