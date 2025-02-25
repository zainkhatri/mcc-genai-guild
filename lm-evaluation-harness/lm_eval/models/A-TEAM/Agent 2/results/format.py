import json
import os
from datetime import datetime

def load_json_file(filename):
    """Load a JSON file and return its contents as a Python dict."""
    with open(filename, 'r') as f:
        return json.load(f)

def group_models_by_family(all_models_data):
    """Group models by their family (Claude, GPT, Gemini)."""
    result = {
        "claude_models": {},
        "gpt_models": {},
        "gemini_models": {},
        "timestamp": datetime.now().isoformat(),
        "metadata": {
            "total_models_evaluated": 0,
            "model_families": ["claude", "gpt", "gemini"]
        }
    }
    
    # Count total models
    total_models = 0
    
    # Process each model's data
    for model_name, model_data in all_models_data.items():
        total_models += 1
        
        if "claude" in model_name.lower():
            result["claude_models"][model_name] = model_data
        elif "gpt" in model_name.lower():
            result["gpt_models"][model_name] = model_data
        elif "gemini" in model_name.lower():
            result["gemini_models"][model_name] = model_data
        else:
            # Create other category if needed
            if "other_models" not in result:
                result["metadata"]["model_families"].append("other")
                result["other_models"] = {}
            result["other_models"][model_name] = model_data
    
    result["metadata"]["total_models_evaluated"] = total_models
    return result

def process_files(filenames):
    """Process multiple JSON files and combine their data."""
    all_models_data = {}
    
    for filename in filenames:
        data = load_json_file(filename)
        models = data.get("models_evaluated", [])
        scores = data.get("scores", {})
        detailed_results = data.get("detailed_results", {})
        
        # Process each model in this file
        for model in models:
            if model not in all_models_data:
                all_models_data[model] = {
                    "scores": scores.get(model, {}),
                    "detailed_results": {}
                }
                
                # Get detailed results for knowledge and ethics
                if "knowledge" in detailed_results and model in detailed_results["knowledge"]:
                    all_models_data[model]["detailed_results"]["knowledge"] = detailed_results["knowledge"][model]
                
                if "ethics" in detailed_results and model in detailed_results["ethics"]:
                    all_models_data[model]["detailed_results"]["ethics"] = detailed_results["ethics"][model]
    
    return all_models_data

def main():
    # List of input JSON files
    input_files = ["4o.json", "CLAUDE.json", "GEMINI.json", "GPT.json"]
    
    # Process the files
    all_models_data = process_files(input_files)
    
    # Group models by family
    final_result = group_models_by_family(all_models_data)
    
    # Write the combined data to a new JSON file
    with open("combined_results.json", 'w') as f:
        json.dump(final_result, f, indent=2)
    
    print(f"Combined data written to combined_results.json")
    print(f"Total models processed: {final_result['metadata']['total_models_evaluated']}")
    print(f"Model families: {', '.join(final_result['metadata']['model_families'])}")

if __name__ == "__main__":
    main()