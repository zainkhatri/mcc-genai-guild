#!/usr/bin/env python3
"""
Updated processor for Agent 2 results to generate Mizan rankings with markdown export.
This script handles the custom JSON structure and generates a nicely formatted markdown table.
"""

import os
import sys
import json
import traceback
from datetime import datetime

# Import required modules
try:
    from mizan_ranker import MizanRanker
    from markdown_formatter import generate_markdown_leaderboard
except ImportError:
    print("Error: Cannot import required modules. Make sure mizan_ranker.py and markdown_formatter.py are in the current directory.")
    sys.exit(1)

def process_results_file(file_path, output_dir="mizan_results"):
    """
    Process a specific results.json file with custom structure and generate Mizan rankings.
    
    Args:
        file_path: Path to the results.json file
        output_dir: Directory to save the output
    """
    print(f"Processing file: {file_path}")
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    try:
        # Load the JSON file
        with open(file_path, 'r') as f:
            try:
                data = json.load(f)
                print(f"Successfully loaded JSON data from {file_path}")
            except json.JSONDecodeError:
                print(f"Error: Could not parse JSON in {file_path}")
                traceback.print_exc()
                return
    except Exception as e:
        print(f"Error loading file {file_path}: {e}")
        traceback.print_exc()
        return
    
    # Print the structure of the data
    print("\nExamining data structure:")
    print(f"Keys in data: {list(data.keys())}")
    
    # Identify model groups
    model_groups = []
    for key in data.keys():
        if key.endswith('_models'):
            model_groups.append(key)
            print(f"Found model group: {key} with {len(data[key])} models")
    
    if not model_groups:
        print("Error: No model groups found in the data")
        return
    
    # Combine models from all groups
    combined_scores = {}
    for group in model_groups:
        models = data[group]
        print(f"\nExamining models in {group}:")
        
        # See if this is a list or dict
        if isinstance(models, list):
            for model in models:
                if 'name' in model:
                    model_name = model['name']
                    print(f"  Model: {model_name}")
                    print(f"  Keys: {list(model.keys())}")
                    
                    # Extract scores
                    combined_scores[model_name] = extract_scores_from_model(model)
        elif isinstance(models, dict):
            for model_name, model_data in models.items():
                print(f"  Model: {model_name}")
                print(f"  Keys: {list(model_data.keys())}")
                
                # Extract scores
                combined_scores[model_name] = extract_scores_from_model(model_data)
    
    # Check if we got any scores
    if not combined_scores:
        print("Error: No model scores could be extracted")
        return
    
    # Create a data structure for MizanRanker
    data_for_mizan = {"scores": combined_scores}
    
    # Transform data for MizanRanker
    mizan_input = transform_data_for_mizan(data_for_mizan)
    
    if not mizan_input.get("model_name"):
        print("Error: Could not extract valid model data")
        return
    
    # Initialize MizanRanker
    print("\nInitializing MizanRanker...")
    ranker = MizanRanker()
    
    # Process data
    print("Processing data with MizanRanker...")
    report = ranker.process_results(mizan_input)
    
    # Export report
    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    
    # JSON report
    json_report = ranker.export(report, fmt='json')
    json_path = os.path.join(output_dir, f"leaderboard_{timestamp}.json")
    with open(json_path, 'w') as f:
        f.write(json_report)
    
    # CSV report
    csv_report = ranker.export(report, fmt='csv')
    csv_path = os.path.join(output_dir, f"leaderboard_{timestamp}.csv")
    with open(csv_path, 'w') as f:
        f.write(csv_report)
    
    # Generate markdown leaderboard
    markdown = generate_markdown_leaderboard(report, include_legacy_models=True)
    markdown_path = os.path.join(output_dir, f"leaderboard_{timestamp}.md")
    with open(markdown_path, 'w') as f:
        f.write(markdown)
    
    # Print summary
    print("\nResults Summary:")
    print("=" * 50)
    
    if "status" in report and report["status"] == "error":
        print(f"Error: {report.get('error', 'Unknown error')}")
        if "errors" in report:
            for error in report["errors"]:
                print(f"- {error}")
        return
    
    if "leaderboard" in report:
        print("Leaderboard:")
        for entry in report["leaderboard"]:
            print(f"  {entry['model_name']}: {entry.get('total_score', 0):.4f}")
    
    if "ethical_ranking" in report:
        print("\nEthical Ranking:")
        for entry in report["ethical_ranking"]:
            print(f"  {entry['model_name']}: {entry.get('ethical_compliance', 0):.4f}")
    
    if "areas_for_improvement" in report:
        print("\nAreas for Improvement:")
        for area in report["areas_for_improvement"]:
            print(f"  - {area}")
    
    print(f"\nReports saved to:")
    print(f"- JSON: {json_path}")
    print(f"- CSV: {csv_path}")
    print(f"- Markdown: {markdown_path}")
    
    # Print markdown preview
    print("\nMarkdown Leaderboard Preview:")
    print("-" * 50)
    print(markdown)

def extract_scores_from_model(model_data):
    """
    Extract relevant scores from a model data structure.
    
    Args:
        model_data: Dict with model evaluation data
        
    Returns:
        Dict with standardized scores
    """
    scores = {}
    
    # Map of possible keys to standardized score names
    key_mappings = {
        # Accuracy related
        'knowledge_accuracy': 'accuracy',
        'factual_accuracy': 'accuracy', 
        'accuracy': 'accuracy',
        'knowledge_score': 'accuracy',
        'factual_score': 'accuracy',
        'correctness': 'accuracy',
        
        # Ethical alignment related
        'ethics_accuracy': 'ethical_alignment',
        'ethical_alignment': 'ethical_alignment',
        'islamic_alignment': 'ethical_alignment',
        'ethical_score': 'ethical_alignment',
        'ethics_score': 'ethical_alignment',
        'ethical_correctness': 'ethical_alignment',
        
        # Bias related
        'bias': 'bias',
        'bias_score': 'bias',
        'model_bias': 'bias'
    }
    
    # Check if the model data has a scores or results subfield
    if 'scores' in model_data:
        data_to_check = model_data['scores']
    elif 'results' in model_data:
        data_to_check = model_data['results']
    else:
        data_to_check = model_data
    
    # Look through all possible keys
    found_keys = []
    for source_key, target_key in key_mappings.items():
        if source_key in data_to_check:
            scores[target_key] = data_to_check[source_key]
            found_keys.append(f"{source_key} -> {target_key}")
    
    # Print what we found
    if found_keys:
        print(f"    Found scores: {', '.join(found_keys)}")
    else:
        print(f"    No recognizable scores found. Available keys: {list(data_to_check.keys())}")
        
        # If we still didn't find any scores, try looking for numeric values
        for key, value in data_to_check.items():
            if isinstance(value, (int, float)) and 0 <= value <= 1:
                print(f"    Found numeric value for '{key}': {value}")
                
                # Make a guess based on the key name
                if 'ethics' in key.lower() or 'ethical' in key.lower() or 'islamic' in key.lower():
                    scores['ethical_alignment'] = value
                    print(f"    Interpreting '{key}' as ethical_alignment")
                elif 'accuracy' in key.lower() or 'knowledge' in key.lower() or 'factual' in key.lower():
                    scores['accuracy'] = value
                    print(f"    Interpreting '{key}' as accuracy")
    
    # If we still don't have essential scores, check for overall scores
    if 'accuracy' not in scores and 'ethical_alignment' not in scores:
        if 'overall_score' in data_to_check:
            overall = data_to_check['overall_score']
            scores['accuracy'] = overall
            scores['ethical_alignment'] = overall
            print(f"    Using overall_score {overall} for both accuracy and ethical_alignment")
    
    return scores

def transform_data_for_mizan(data):
    """
    Transform evaluation data for MizanRanker.
    
    Args:
        data: Input data with scores dict
        
    Returns:
        Transformed data for MizanRanker
    """
    model_names = []
    accuracy_scores = []
    ethical_scores = []
    bias_scores = []
    
    # Extract scores
    scores_data = data.get("scores", {})
    print(f"Transforming data for {len(scores_data)} models")
    
    for model_name, scores in scores_data.items():
        # Skip models with no scores
        if not scores:
            print(f"Skipping {model_name}: No scores found")
            continue
            
        # Make sure we have the essential scores
        if 'accuracy' not in scores and 'ethical_alignment' not in scores:
            print(f"Skipping {model_name}: Missing both accuracy and ethical_alignment")
            continue
            
        model_names.append(model_name)
        
        # Add scores with defaults if missing
        accuracy_scores.append(scores.get('accuracy', 0.5))
        ethical_scores.append(scores.get('ethical_alignment', 0.5))
        
        if 'bias' in scores:
            bias_scores.append(scores['bias'])
    
    # Create input for MizanRanker
    mizan_input = {
        'model_name': model_names,
        'accuracy': accuracy_scores,
        'ethical_alignment': ethical_scores
    }
    
    # Add bias if available for all models
    if bias_scores and len(bias_scores) == len(model_names):
        mizan_input['bias'] = bias_scores
    
    print(f"Transformed input with {len(model_names)} models")
    for i, model in enumerate(model_names):
        print(f"  {model}: accuracy={accuracy_scores[i]}, ethical_alignment={ethical_scores[i]}")
    
    return mizan_input

def main():
    """Main function to find and process results.json"""
    # Check current directory
    current_dir = os.getcwd()
    print(f"Current directory: {current_dir}")
    
    # Define possible paths to results.json
    possible_paths = [
        os.path.join(current_dir, "results.json"),
        os.path.join(current_dir, "..", "Agent 2", "results", "results.json"),
        os.path.join(os.path.dirname(current_dir), "Agent 2", "results", "results.json")
    ]
    
    # Check if we can find results.json
    results_path = None
    for path in possible_paths:
        if os.path.exists(path):
            results_path = path
            print(f"Found results.json at: {path}")
            break
    
    if results_path:
        # Process the file
        process_results_file(results_path)
    else:
        # Ask for manual path
        print("Could not automatically find results.json.")
        user_path = input("Please enter the full path to results.json: ")
        if os.path.exists(user_path):
            process_results_file(user_path)
        else:
            print(f"Error: File not found at {user_path}")

if __name__ == "__main__":
    main()