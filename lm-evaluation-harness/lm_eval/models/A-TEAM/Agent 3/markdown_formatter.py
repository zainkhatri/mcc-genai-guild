import pandas as pd
import json
import os
from typing import Dict, List, Any, Optional, Union

def calculate_grade(score: float) -> str:
    """
    Calculate letter grade based on score percentage.
    
    Args:
        score: Score as a float between 0 and 1
        
    Returns:
        Letter grade (A+, A, B+, etc.)
    """
    if score >= 0.97:
        return "A+"
    elif score >= 0.93:
        return "A"
    elif score >= 0.90:
        return "A-"
    elif score >= 0.87:
        return "B+"
    elif score >= 0.83:
        return "B"
    elif score >= 0.80:
        return "B-"
    elif score >= 0.77:
        return "C+"
    elif score >= 0.73:
        return "C"
    elif score >= 0.70:
        return "C-"
    elif score >= 0.67:
        return "D+"
    elif score >= 0.63:
        return "D"
    elif score >= 0.60:
        return "D-"
    else:
        return "F"

def clean_model_name(name: str) -> str:
    """
    Clean and format model names for better readability.
    
    Args:
        name: Raw model name
        
    Returns:
        Cleaned and formatted model name
    """
    # Replace hyphens with spaces for readability
    name = name.replace('-', ' ').strip()
    
    # Handle specific models with custom formatting
    name_mapping = {
        # GPT models
        "gpt 4o 2024 11 20": "GPT-4o (2024-11-20)",
        "gpt 4 0125 preview": "GPT-4-0125 Optimized",
        "gpt 4 turbo preview": "GPT-4-Turbo Optimized",
        "gpt 4": "GPT-4",
        
        # Claude models
        "claude 3 opus 20240229": "Claude 3 Opus (2024-02)",
        "claude 3 sonnet 20240229": "Claude 3 Sonnet (2024-02)",
        "claude 2.1": "Claude 2.1",
        "claude 3.5 opus": "Claude 3.5 Opus",
        "claude 3.5 sonnet": "Claude 3.5 Sonnet",
        
        # Gemini models
        "gemini 2.0 flash": "Gemini 2.0 Flash",
        "gemini pro": "Gemini Pro",
        "gemini 1.5 pro": "Gemini 1.5 Pro", 
        "gemini 1.5 flash": "Gemini 1.5 Flash",
        
        # Open source models
        "zephyr 7b beta": "Zephyr-7B Beta (7B)",
        "microsoft phi 2": "microsoft/phi-2 (2.7B)",
        "stablelm 2 zephyr": "StableLM-2 Zephyr (1.6B)"
    }
    
    # Check for exact matches after normalization
    normalized = name.lower()
    if normalized in name_mapping:
        return name_mapping[normalized]
    
    # Check for partial matches
    for key, value in name_mapping.items():
        if key in normalized:
            return value
    
    # Default: capitalize words for better formatting
    words = name.split()
    return ' '.join(word.capitalize() for word in words)

def format_as_percentage(value: float) -> str:
    """
    Format float as percentage with 2 decimal places.
    
    Args:
        value: Float value between 0 and 1
        
    Returns:
        Formatted percentage string
    """
    if value is None:
        return "N/A"
    return f"{value*100:.2f}%"

def generate_markdown_leaderboard(
    report_data: Union[Dict[str, Any], str, pd.DataFrame],
    include_legacy_models: bool = True,
    legacy_models: Optional[List[Dict[str, Any]]] = None
) -> str:
    """
    Generate a markdown leaderboard table from report data.
    
    Args:
        report_data: Report data from MizanRanker (dict), path to JSON file, or DataFrame
        include_legacy_models: Whether to include legacy models from previous evaluations
        legacy_models: List of legacy model data to include
        
    Returns:
        Markdown formatted leaderboard
    """
    # Process input based on type
    if isinstance(report_data, str):
        # Load from file
        if os.path.exists(report_data):
            with open(report_data, 'r') as f:
                if report_data.endswith('.json'):
                    data = json.load(f)
                elif report_data.endswith('.csv'):
                    df = pd.read_csv(report_data)
                    data = {"complete_rankings": df.to_dict(orient='records')}
                else:
                    raise ValueError(f"Unsupported file type: {report_data}")
        else:
            raise FileNotFoundError(f"File not found: {report_data}")
    elif isinstance(report_data, dict):
        # Use directly as dict
        data = report_data
    elif isinstance(report_data, pd.DataFrame):
        # Convert DataFrame to dict
        data = {"complete_rankings": report_data.to_dict(orient='records')}
    else:
        raise ValueError(f"Unsupported data type: {type(report_data)}")
    
    # Get rankings data
    if "complete_rankings" in data and data["complete_rankings"]:
        rankings = data["complete_rankings"]
    elif "leaderboard" in data and data["leaderboard"]:
        # Use leaderboard, ethical_ranking, and accuracy_ranking to build complete data
        model_data = {}
        
        # Process leaderboard for total scores
        for entry in data.get("leaderboard", []):
            model_name = entry.get("model_name")
            if model_name:
                if model_name not in model_data:
                    model_data[model_name] = {}
                model_data[model_name]["total_score"] = entry.get("total_score")
        
        # Add ethical compliance scores
        for entry in data.get("ethical_ranking", []):
            model_name = entry.get("model_name")
            if model_name and model_name in model_data:
                model_data[model_name]["ethical_compliance"] = entry.get("ethical_compliance")
                # Use as ethical_alignment if not present
                if "ethical_alignment" not in model_data[model_name]:
                    model_data[model_name]["ethical_alignment"] = entry.get("ethical_compliance")
        
        # Add accuracy scores
        for entry in data.get("accuracy_ranking", []):
            model_name = entry.get("model_name")
            if model_name and model_name in model_data:
                model_data[model_name]["accuracy"] = entry.get("accuracy")
        
        # Convert to list format
        rankings = []
        for model_name, scores in model_data.items():
            entry = {"model_name": model_name}
            entry.update(scores)
            rankings.append(entry)
    else:
        raise ValueError("No ranking data found in the input")
    
    # Include legacy models if requested
    if include_legacy_models and legacy_models:
        for legacy_model in legacy_models:
            # Check if model is already in the rankings
            if legacy_model.get("model_name") not in [r.get("model_name") for r in rankings]:
                rankings.append(legacy_model)
    
    # Convert to DataFrame for easier manipulation
    df = pd.DataFrame(rankings)
    
    # Make sure we have required columns, add placeholders if missing
    if "total_score" not in df.columns:
        if "accuracy" in df.columns and "ethical_alignment" in df.columns:
            df["total_score"] = (df["accuracy"] + df["ethical_alignment"]) / 2
        else:
            df["total_score"] = 0.5
    
    # Sort by total score descending
    df = df.sort_values("total_score", ascending=False)
    
    # Add grades
    df["grade"] = df["total_score"].apply(calculate_grade)
    
    # Clean model names
    df["formatted_name"] = df["model_name"].apply(clean_model_name)
    
    # Format as markdown table
    markdown = "##### **Overall Model Rankings**\n"
    markdown += "| Model Name | Total Score | Grade | Accuracy | Ethics |\n"
    markdown += "|------------|-------------|-------|----------|--------|\n"
    
    # Add rows
    for _, row in df.iterrows():
        ethics_value = row.get("ethical_alignment", row.get("ethical_compliance", None))
        accuracy = format_as_percentage(row.get("accuracy", None))
        ethics = format_as_percentage(ethics_value)
        
        markdown += f"| **{row['formatted_name']}** | {format_as_percentage(row['total_score'])} | "
        markdown += f"{row['grade']} | {accuracy} | {ethics} |\n"
    
    return markdown

def format_leaderboard_from_file(file_path: str, output_file: Optional[str] = None) -> str:
    """
    Generate markdown leaderboard from a file and optionally save to output file.
    
    Args:
        file_path: Path to JSON or CSV report file
        output_file: Optional path to save markdown output
        
    Returns:
        Markdown formatted leaderboard
    """
    # Legacy models to include (from previous evaluations)
    legacy_models = [
        {
            "model_name": "gemini_1.5_pro",
            "total_score": 0.955,
            "accuracy": 0.96,
            "ethical_alignment": 0.95
        },
        {
            "model_name": "claude_3.5_opus",
            "total_score": 0.95,
            "accuracy": 0.92,
            "ethical_alignment": 0.98
        },
        {
            "model_name": "gemini_1.5_flash",
            "total_score": 0.875,
            "accuracy": 0.84,
            "ethical_alignment": 0.91
        },
        {
            "model_name": "claude_3.5_sonnet",
            "total_score": 0.86,
            "accuracy": 0.76,
            "ethical_alignment": 0.96
        },
        {
            "model_name": "zephyr_7b_beta",
            "total_score": 0.437,
            "accuracy": 0.437,
            "ethical_alignment": None
        },
        {
            "model_name": "microsoft_phi_2",
            "total_score": 0.3733,
            "accuracy": 0.3733,
            "ethical_alignment": None
        },
        {
            "model_name": "stablelm_2_zephyr",
            "total_score": 0.2433,
            "accuracy": 0.2433,
            "ethical_alignment": None
        }
    ]
    
    # Generate markdown
    markdown = generate_markdown_leaderboard(
        file_path, 
        include_legacy_models=True,
        legacy_models=legacy_models
    )
    
    # Save to file if requested
    if output_file:
        with open(output_file, 'w') as f:
            f.write(markdown)
        print(f"Leaderboard saved to {output_file}")
    
    return markdown

if __name__ == "__main__":
    import sys
    
    if len(sys.argv) > 1:
        file_path = sys.argv[1]
        output_file = sys.argv[2] if len(sys.argv) > 2 else "leaderboard.md"
        
        try:
            markdown = format_leaderboard_from_file(file_path, output_file)
            print("\nLeaderboard Preview:\n")
            print(markdown)
        except Exception as e:
            print(f"Error: {e}")
    else:
        print("Usage: python markdown_formatter.py input_file.json [output_file.md]")