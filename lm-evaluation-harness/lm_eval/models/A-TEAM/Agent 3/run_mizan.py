#!/usr/bin/env python3
import os
import glob
import pandas as pd
from mizan_ranker import MizanRanker

def collect_csv_data(results_dir: str) -> dict:
    """
    1. Reads all CSV files in the given results_dir.
    2. Concatenates them into one DataFrame.
    3. Filters out rows with total_score < 10.
    4. Drops duplicate models, keeping only the row with the highest total_score.
    5. Returns a dictionary suitable for MizanRanker:
       {
         'model_name': [...],
         'accuracy': [...],
         'ethical_alignment': [...],
         'bias': [...],
         'source_reliability': [...]
       }
    """
    csv_files = glob.glob(os.path.join(results_dir, "*.csv"))
    if not csv_files:
        print(f"No CSV files found in {results_dir}.")
        return {}
    
    all_dfs = []
    for csv_path in csv_files:
        print(f"Reading CSV: {csv_path}")
        # Each CSV file is expected to have the following columns (in order):
        # model_name, total_score, grade, accuracy, ethical_alignment, bias, source_reliability
        df = pd.read_csv(
            csv_path,
            header=None,
            names=["model_name", "total_score", "grade", "accuracy", 
                   "ethical_alignment", "bias", "source_reliability"]
        )
        all_dfs.append(df)
    
    combined = pd.concat(all_dfs, ignore_index=True)
    
    # Convert numeric columns to floats
    combined["total_score"] = combined["total_score"].astype(float)
    combined["accuracy"] = combined["accuracy"].astype(float)
    combined["ethical_alignment"] = combined["ethical_alignment"].astype(float)
    combined["bias"] = combined["bias"].astype(float)
    
    # Filter out models with total_score less than 10
    combined = combined[combined["total_score"] >= 10]
    
    # Sort by total_score descending and drop duplicate model entries (keeping the best score)
    combined.sort_values("total_score", ascending=False, inplace=True)
    combined.drop_duplicates(subset=["model_name"], keep="first", inplace=True)
    combined.sort_values("total_score", ascending=False, inplace=True)
    
    # Build the dictionary for MizanRanker
    data_dict = {
        "model_name": combined["model_name"].tolist(),
        "accuracy": combined["accuracy"].tolist(),
        "ethical_alignment": combined["ethical_alignment"].tolist(),
        "bias": combined["bias"].tolist(),
        "source_reliability": combined["source_reliability"].tolist(),
    }
    return data_dict

def generate_markdown_leaderboard(report: dict) -> str:
    """
    Converts the MizanRanker 'report' into a Markdown table with numbers rounded
    to two decimal places.
    """
    if report.get("status") == "error":
        return "Error in report. No leaderboard available."
    
    rows = report.get("complete_rankings", [])
    if not rows:
        return "No complete_rankings in report."
    
    md = []
    md.append("| Model Name | Accuracy | Ethical Alignment | Bias | Total Score |")
    md.append("|------------|----------|-------------------|------|------------|")
    
    for r in rows:
        def fmt2(x):
            return f"{x:.2f}" if isinstance(x, (int, float)) else str(x)
        
        model  = str(r.get("model_name", ""))
        acc    = fmt2(r.get("accuracy", 0))
        eth    = fmt2(r.get("ethical_alignment", 0))
        bias   = fmt2(r.get("bias", 0))
        total  = fmt2(r.get("total_score", 0))
        
        md.append(f"| {model} | {acc} | {eth} | {bias} | {total} |")
    
    return "\n".join(md)

def main():
    # Define the directory where Agent 2 CSV files are located
    agent2_results_dir = os.path.join(os.path.dirname(__file__), "..", "Agent 2", "results")
    
    # Collect data from CSV files
    data_dict = collect_csv_data(agent2_results_dir)
    if not data_dict:
        print("No data to process.")
        return
    
    # Initialize MizanRanker (LangGraph-based ranking)
    ranker = MizanRanker(debug=True)
    
    # Process the results and generate the report
    report = ranker.process_results(data_dict)
    
    # Generate the Markdown leaderboard from the report
    leaderboard_md = generate_markdown_leaderboard(report)
    
    # Save the Markdown leaderboard to a file in the Agent 3 folder
    output_md_path = os.path.join(os.path.dirname(__file__), "leaderboard.md")
    with open(output_md_path, "w") as f:
        f.write(leaderboard_md)
    
    print("\n===== Leaderboard =====")
    print(leaderboard_md)
    print(f"\nSaved Markdown to: {output_md_path}")

if __name__ == "__main__":
    main()