import os
import sys
import asyncio
import json
import pandas as pd
from datetime import datetime
import traceback
from typing import Dict, List, Any, Optional, Tuple

# Add proper paths for module imports
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.append(current_dir)  # Add current directory to path

# Import MizanRanker with error handling
try:
    from mizan_ranker import MizanRanker
except ImportError as e:
    print(f"Error importing MizanRanker: {e}")
    print("Make sure mizan_ranker.py is in the current directory")
    sys.exit(1)

class MizanConnector:
    """
    Enhanced connector between Agent 2 (ADL Evaluator) and Agent 3 (Mizan Ranker)
    with improved error handling, validation, and reporting capabilities.
    """
    
    def __init__(self, output_dir: str = "results"):
        """
        Initialize the connector with the MizanRanker instance.
        
        Args:
            output_dir: Directory to save results
        """
        self.ranker = MizanRanker()
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)
        
        # Initialize logging
        self.log_file = os.path.join(output_dir, f"mizan_log_{datetime.now().strftime('%Y%m%d-%H%M%S')}.txt")
        with open(self.log_file, 'w') as f:
            f.write(f"MizanConnector initialized at {datetime.now().isoformat()}\n")
    
    def log(self, message: str):
        """Log a message to both console and log file"""
        print(message)
        with open(self.log_file, 'a') as f:
            f.write(f"{datetime.now().isoformat()}: {message}\n")
    
    def transform_adl_data(self, adl_report: Dict[str, Any]) -> Tuple[Dict[str, Any], List[str]]:
        """
        Transform ADL Evaluator output into MizanRanker input format with validation.
        
        Args:
            adl_report: The raw output from Agent 2 (ADL Evaluator)
            
        Returns:
            Tuple containing:
                - Transformed data for MizanRanker
                - List of warnings during transformation
        """
        model_names = []
        accuracy_scores = []
        ethical_scores = []
        bias_scores = []
        warnings = []
        
        # Validate input
        if not adl_report:
            warnings.append("Empty ADL report received")
            return {"model_name": [], "accuracy": [], "ethical_alignment": []}, warnings
            
        if "scores" not in adl_report:
            warnings.append("No 'scores' key found in ADL report")
            self.log(f"ADL report structure: {json.dumps(adl_report, indent=2)[:500]}...")
            return {"model_name": [], "accuracy": [], "ethical_alignment": []}, warnings
        
        # Extract scores with flexible key mapping
        scores_data = adl_report.get("scores", {})
        self.log(f"Processing scores for {len(scores_data)} models")
        
        for model_name, scores in scores_data.items():
            model_names.append(model_name)
            
            # Handle different possible key names flexibly
            # For accuracy metrics
            accuracy_keys = ["knowledge_accuracy", "accuracy", "factual_accuracy", "knowledge_score"]
            accuracy_value = 0
            for key in accuracy_keys:
                if key in scores:
                    accuracy_value = scores[key]
                    break
            
            # For ethical alignment metrics  
            ethical_keys = ["ethics_accuracy", "ethical_alignment", "islamic_alignment", "ethical_score"]
            ethical_value = 0
            for key in ethical_keys:
                if key in scores:
                    ethical_value = scores[key]
                    break
            
            # For bias metrics (optional)
            bias_keys = ["bias", "bias_score", "model_bias"]
            bias_value = None
            for key in bias_keys:
                if key in scores:
                    bias_value = scores[key]
                    break
            
            # Add extracted values
            accuracy_scores.append(accuracy_value)
            ethical_scores.append(ethical_value)
            if bias_value is not None:
                bias_scores.append(bias_value)
            
            # Log what was found
            self.log(f"Model: {model_name}, Accuracy: {accuracy_value}, Ethics: {ethical_value}" + 
                    (f", Bias: {bias_value}" if bias_value is not None else ""))
            
            # Check for potential issues
            if accuracy_value == 0:
                warnings.append(f"Zero accuracy score for model {model_name}")
            if ethical_value == 0:
                warnings.append(f"Zero ethical alignment score for model {model_name}")
        
        # Create input for MizanRanker
        mizan_input = {
            'model_name': model_names,
            'accuracy': accuracy_scores,
            'ethical_alignment': ethical_scores
        }
        
        # Add bias if available for all models
        if bias_scores and len(bias_scores) == len(model_names):
            mizan_input['bias'] = bias_scores
        
        return mizan_input, warnings
    
    async def process_file(self, file_path: str) -> Optional[Dict[str, Any]]:
        """
        Process a single ADL Evaluator results file.
        
        Args:
            file_path: Path to the ADL results JSON file
            
        Returns:
            The Mizan report or None if processing failed
        """
        try:
            filename = os.path.basename(file_path)
            self.log(f"Processing file: {filename}")
            
            # Load ADL results
            with open(file_path, 'r') as f:
                adl_report = json.load(f)
            
            # Transform data
            mizan_input, warnings = self.transform_adl_data(adl_report)
            
            # Log any warnings
            for warning in warnings:
                self.log(f"WARNING: {warning}")
            
            # Check if we have valid data
            if not mizan_input.get('model_name'):
                self.log(f"ERROR: No valid models found in {filename}")
                return None
            
            # Process with MizanRanker
            self.log("Running MizanRanker workflow...")
            report = self.ranker.process_results(mizan_input)
            
            # Check report results
            if not report or "error" in report:
                self.log(f"ERROR: MizanRanker processing failed: {report.get('error', 'Unknown error')}")
                return None
                
            # Save reports
            base_name = os.path.splitext(filename)[0]
            timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
            
            # JSON report
            json_report = self.ranker.export(report, fmt='json')
            json_path = os.path.join(self.output_dir, f"{base_name}_ranked_{timestamp}.json")
            with open(json_path, 'w') as f:
                f.write(json_report)
            
            # CSV leaderboard
            csv_report = self.ranker.export(report, fmt='csv')
            csv_path = os.path.join(self.output_dir, f"{base_name}_leaderboard_{timestamp}.csv")
            with open(csv_path, 'w') as f:
                f.write(csv_report)
            
            # Print summary
            self.log(f"\nResults for {filename}:")
            self.log("=" * 50)
            
            if "leaderboard" in report:
                self.log("Leaderboard:")
                for entry in report["leaderboard"]:
                    self.log(f"  {entry['model_name']}: {entry.get('total_score', 0):.4f}")
            
            if "ethical_ranking" in report:
                self.log("\nEthical Ranking:")
                for entry in report["ethical_ranking"]:
                    self.log(f"  {entry['model_name']}: {entry.get('ethical_compliance', 0):.4f}")
            
            if "areas_for_improvement" in report:
                self.log("\nAreas for Improvement:")
                for area in report["areas_for_improvement"]:
                    self.log(f"  - {area}")
            
            self.log(f"\nSaved to: {json_path}")
            self.log(f"Leaderboard saved to: {csv_path}")
            
            return report
            
        except Exception as e:
            self.log(f"ERROR processing {file_path}: {str(e)}")
            self.log(traceback.format_exc())
            return None
    
    async def process_directory(self, dir_path: str) -> List[Dict[str, Any]]:
        """
        Process all JSON files in a directory.
        
        Args:
            dir_path: Path to directory with ADL result files
            
        Returns:
            List of successfully processed reports
        """
        if not os.path.exists(dir_path):
            self.log(f"Directory not found: {dir_path}")
            return []
            
        self.log(f"Scanning directory: {dir_path}")
        reports = []
        found_files = False
        
        for filename in os.listdir(dir_path):
            if filename.endswith('.json'):
                found_files = True
                file_path = os.path.join(dir_path, filename)
                report = await self.process_file(file_path)
                if report:
                    reports.append(report)
        
        if not found_files:
            self.log(f"No JSON files found in {dir_path}")
        
        return reports
    
    async def generate_meta_report(self, reports: List[Dict[str, Any]]) -> None:
        """
        Generate a meta-report combining results from multiple files.
        
        Args:
            reports: List of Mizan reports to combine
        """
        if not reports:
            self.log("No reports to combine for meta-report")
            return
            
        self.log(f"Generating meta-report from {len(reports)} reports...")
        
        # Combine leaderboards
        all_models = {}
        
        for report in reports:
            for entry in report.get("leaderboard", []):
                model_name = entry.get("model_name")
                if model_name:
                    if model_name not in all_models:
                        all_models[model_name] = {
                            "total_scores": [],
                            "ethical_scores": []
                        }
                    
                    all_models[model_name]["total_scores"].append(entry.get("total_score", 0))
            
            # Add ethical scores
            for entry in report.get("ethical_ranking", []):
                model_name = entry.get("model_name")
                if model_name and model_name in all_models:
                    all_models[model_name]["ethical_scores"].append(
                        entry.get("ethical_compliance", 0))
        
        # Calculate averages
        meta_leaderboard = []
        for model_name, scores in all_models.items():
            avg_total = sum(scores["total_scores"]) / len(scores["total_scores"]) if scores["total_scores"] else 0
            avg_ethical = sum(scores["ethical_scores"]) / len(scores["ethical_scores"]) if scores["ethical_scores"] else 0
            
            meta_leaderboard.append({
                "model_name": model_name,
                "avg_total_score": avg_total,
                "avg_ethical_compliance": avg_ethical,
                "reports_count": len(scores["total_scores"])
            })
        
        # Sort by average total score
        meta_leaderboard.sort(key=lambda x: x["avg_total_score"], reverse=True)
        
        # Collect improvement areas
        all_improvement_areas = {}
        for report in reports:
            for area in report.get("areas_for_improvement", []):
                if area not in all_improvement_areas:
                    all_improvement_areas[area] = 0
                all_improvement_areas[area] += 1
        
        # Sort improvement areas by frequency
        sorted_areas = sorted(
            [(area, count) for area, count in all_improvement_areas.items()],
            key=lambda x: x[1],
            reverse=True
        )
        
        # Create meta-report
        meta_report = {
            "meta_leaderboard": meta_leaderboard,
            "common_improvement_areas": [
                {"area": area, "frequency": count} 
                for area, count in sorted_areas
            ],
            "reports_analyzed": len(reports),
            "generated_at": datetime.now().isoformat(),
            "version": "1.0"
        }
        
        # Save meta-report
        timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
        meta_path = os.path.join(self.output_dir, f"meta_report_{timestamp}.json")
        
        with open(meta_path, 'w') as f:
            json.dump(meta_report, f, indent=2)
            
        # Also save as CSV
        df = pd.DataFrame(meta_report["meta_leaderboard"])
        csv_path = os.path.join(self.output_dir, f"meta_leaderboard_{timestamp}.csv")
        df.to_csv(csv_path, index=False)
        
        # Print summary
        self.log("\nMeta-Report Summary:")
        self.log("=" * 50)
        self.log("Overall Model Performance:")
        for entry in meta_leaderboard:
            self.log(f"  {entry['model_name']}: {entry['avg_total_score']:.4f} (from {entry['reports_count']} reports)")
        
        self.log("\nCommon Areas for Improvement:")
        for area_data in meta_report["common_improvement_areas"]:
            self.log(f"  - {area_data['area']} (mentioned in {area_data['frequency']} reports)")
        
        self.log(f"\nMeta-report saved to: {meta_path}")
        self.log(f"Meta-leaderboard saved to: {csv_path}")

async def main():
    """Main function to run the enhanced Mizan Connector"""
    # Setup output directory
    output_dir = os.path.join(os.getcwd(), "mizan_results")
    
    # Initialize connector
    connector = MizanConnector(output_dir=output_dir)
    connector.log(f"MizanConnector initialized. Output directory: {output_dir}")
    
    # Check for test data or use sample data
    test_data_path = os.path.join(os.getcwd(), "test_data.json")
    agent2_results_dir = os.path.join(os.path.dirname(os.getcwd()), "Agent 2", "results")
    
    # Process test data if available
    if os.path.exists(test_data_path):
        connector.log(f"Processing test data from {test_data_path}")
        await connector.process_file(test_data_path)
    
    # Try to process Agent 2 results if available
    reports = []
    
    if os.path.exists(agent2_results_dir):
        connector.log(f"Found Agent 2 results directory: {agent2_results_dir}")
        
        # Process different result directories
        for subdir in ["full data", "short results", "single results"]:
            subdir_path = os.path.join(agent2_results_dir, subdir)
            if os.path.exists(subdir_path):
                connector.log(f"Processing {subdir} directory...")
                subdir_reports = await connector.process_directory(subdir_path)
                reports.extend(subdir_reports)
    else:
        connector.log(f"Agent 2 results directory not found at: {agent2_results_dir}")
        connector.log("Looking for .json files in current directory...")
        
        # Fall back to current directory
        current_reports = await connector.process_directory(os.getcwd())
        reports.extend(current_reports)
        
        # If no files found, create and process sample data
        if not reports:
            connector.log("No JSON files found. Creating sample data for testing...")
            
            # Create sample data
            sample_data = {
                "scores": {
                    "gpt-4": {"knowledge_accuracy": 0.85, "ethics_accuracy": 0.90},
                    "claude-3-opus": {"knowledge_accuracy": 0.82, "ethics_accuracy": 0.88},
                    "gemini-2.0-flash": {"knowledge_accuracy": 0.78, "ethics_accuracy": 0.95},
                    "mistral-large": {"knowledge_accuracy": 0.80, "ethics_accuracy": 0.92}
                }
            }
            
            # Save sample data
            sample_path = os.path.join(output_dir, "sample_data.json")
            with open(sample_path, 'w') as f:
                json.dump(sample_data, f, indent=2)
                
            connector.log(f"Created sample data at {sample_path}")
            
            # Process sample data
            sample_report = await connector.process_file(sample_path)
            if sample_report:
                reports.append(sample_report)
    
    # Generate meta-report if multiple reports were processed
    if len(reports) > 1:
        await connector.generate_meta_report(reports)
    
    connector.log("Processing complete!")

if __name__ == "__main__":
    asyncio.run(main())