import pandas as pd
from langgraph.graph import StateGraph
from typing import Dict, List, Any, TypedDict, Optional, Union
import json
from datetime import datetime
import traceback
import os

class MizanState(TypedDict):
    """State schema for the Mizan workflow"""
    evaluation_data: Dict[str, List[Any]]
    scores_df: Optional[pd.DataFrame]
    metrics: Dict[str, Any]
    summary: Dict[str, Any]
    report: Dict[str, Any]
    errors: List[str]

class MizanRanker:
    """
    Enhanced MizanRanker for evaluating and ranking AI models based on
    accuracy, ethical alignment with Islamic values, and other metrics.
    
    The ranker uses a LangGraph workflow for processing evaluation data.
    """
    
    def __init__(self, debug: bool = True):
        """
        Initialize the MizanRanker.
        
        Args:
            debug: Whether to print debug information
        """
        self.debug = debug
        self.graph = self._setup_workflow()
        
        # Create a log directory
        self.log_dir = "mizan_logs"
        os.makedirs(self.log_dir, exist_ok=True)
        self.log_file = os.path.join(self.log_dir, f"mizan_log_{datetime.now().strftime('%Y%m%d-%H%M%S')}.txt")
    
    def log(self, message: str) -> None:
        """Log a message if debug mode is enabled"""
        if self.debug:
            print(message)
            with open(self.log_file, 'a') as f:
                f.write(f"{datetime.now().isoformat()}: {message}\n")

    def _setup_workflow(self) -> StateGraph:
        """
        Set up the LangGraph workflow for the ranking process.
        
        Returns:
            A compiled StateGraph for the workflow
        """
        # Create StateGraph with state schema
        workflow = StateGraph(MizanState)

        # Add nodes
        workflow.add_node("validate_input", self._validate_input)
        workflow.add_node("score_aggregation", self._aggregate_scores)
        workflow.add_node("compute_metrics", self._compute_islamic_metrics)
        workflow.add_node("generate_summary", self._generate_summary)
        workflow.add_node("create_report", self._build_report)

        # Add conditional routing based on validation results
        workflow.add_conditional_edges(
            "validate_input",
            lambda state: "score_aggregation" if not state.get("errors", []) else "create_report"
        )
        
        # Add remaining edges
        workflow.add_edge("score_aggregation", "compute_metrics")
        workflow.add_edge("compute_metrics", "generate_summary")
        workflow.add_edge("generate_summary", "create_report")
        
        # Set entry point
        workflow.set_entry_point("validate_input")

        return workflow.compile()

    def _validate_input(self, state: MizanState) -> MizanState:
        """
        Validate input data before processing.
        
        Args:
            state: Current workflow state
            
        Returns:
            Updated state with validation results
        """
        errors = []
        
        try:
            # Check if evaluation data exists
            data = state.get("evaluation_data", {})
            if not data:
                errors.append("No evaluation data provided")
                return {
                    "evaluation_data": {},
                    "errors": errors
                }
            
            # Check required keys
            required_keys = ['model_name', 'accuracy', 'ethical_alignment']
            missing_keys = [key for key in required_keys if key not in data]
            
            if missing_keys:
                errors.append(f"Missing required keys: {', '.join(missing_keys)}")
            
            # Check model names and scores lists have same length
            if 'model_name' in data and 'accuracy' in data:
                if len(data['model_name']) != len(data['accuracy']):
                    errors.append("Length mismatch: 'model_name' and 'accuracy' lists have different lengths")
            
            if 'model_name' in data and 'ethical_alignment' in data:
                if len(data['model_name']) != len(data['ethical_alignment']):
                    errors.append("Length mismatch: 'model_name' and 'ethical_alignment' lists have different lengths")
            
            # Check for empty model list
            if 'model_name' in data and not data['model_name']:
                errors.append("Model list is empty")
            
            # Log validation results
            if errors:
                self.log(f"Validation failed with {len(errors)} errors: {errors}")
            else:
                self.log("Input validation successful")
                
            return {
                "evaluation_data": state.get("evaluation_data", {}),
                "errors": errors
            }
            
        except Exception as e:
            self.log(f"Error in input validation: {e}")
            errors.append(f"Validation error: {str(e)}")
            return {
                "evaluation_data": state.get("evaluation_data", {}),
                "errors": errors
            }

    def _aggregate_scores(self, state: MizanState) -> MizanState:
        """
        Aggregate evaluation scores and create a DataFrame.
        
        Args:
            state: Current workflow state
            
        Returns:
            Updated state with aggregated scores
        """
        try:
            # Get evaluation data from state
            data = state.get("evaluation_data", {})
            
            # Create a dictionary to build DataFrame
            df_dict = {}
            for key in data:
                df_dict[key] = data[key]
            
            # Create DataFrame
            df = pd.DataFrame(df_dict)
            
            self.log(f"Processing data with columns: {df.columns.tolist()}")
            
            # Simple average of accuracy and ethical_alignment for total score
            # No arbitrary weights - just using the raw scores directly
            if 'accuracy' in df.columns and 'ethical_alignment' in df.columns:
                df['total_score'] = (df['accuracy'] + df['ethical_alignment']) / 2
                self.log("Calculated total_score as simple average of accuracy and ethical_alignment")
            else:
                # Fallback if columns are missing
                if 'accuracy' in df.columns:
                    df['total_score'] = df['accuracy']
                    self.log("Using only accuracy for total_score (ethical_alignment missing)")
                elif 'ethical_alignment' in df.columns:
                    df['total_score'] = df['ethical_alignment']
                    self.log("Using only ethical_alignment for total_score (accuracy missing)")
                else:
                    df['total_score'] = 0.0
                    self.log("Warning: Neither accuracy nor ethical_alignment found - using zero score")
            
            # Print calculated scores for verification
            if self.debug and 'model_name' in df.columns:
                for idx, row in df.iterrows():
                    self.log(f"Model: {row['model_name']}, Total Score: {row['total_score']:.4f}")
            
            # Create new state with updated scores_df
            return {
                "evaluation_data": state.get("evaluation_data", {}),
                "scores_df": df,
                "errors": state.get("errors", [])
            }

        except Exception as e:
            self.log(f"Error in _aggregate_scores: {e}")
            self.log(traceback.format_exc())
            errors = state.get("errors", [])
            errors.append(f"Score aggregation error: {str(e)}")
            return {
                "evaluation_data": state.get("evaluation_data", {}),
                "scores_df": pd.DataFrame(),
                "errors": errors
            }

    def _compute_islamic_metrics(self, state: MizanState) -> MizanState:
        """
        Compute Islamic-aligned metrics based on evaluation scores.
        
        Args:
            state: Current workflow state
            
        Returns:
            Updated state with Islamic metrics
        """
        try:
            df = state.get("scores_df", pd.DataFrame())
            
            # Simple calculation of ethical compliance based on ethical_alignment
            # Include bias consideration if available
            if 'ethical_alignment' in df.columns:
                if 'bias' in df.columns:
                    # Lower bias is better, so we subtract from 1
                    df['ethical_compliance'] = df['ethical_alignment'] * (1 - df['bias'])
                    self.log("Calculated ethical_compliance with bias consideration")
                else:
                    # If bias isn't provided, use ethical_alignment directly
                    df['ethical_compliance'] = df['ethical_alignment']
                    self.log("Using ethical_alignment directly as ethical_compliance (no bias data)")
            else:
                # Fallback
                df['ethical_compliance'] = 0.5
                self.log("Warning: No ethical_alignment found - using default ethical_compliance value")
            
            # Create metrics summary
            metrics = {
                'avg_compliance': df['ethical_compliance'].mean() if 'ethical_compliance' in df.columns else 0,
                'avg_accuracy': df['accuracy'].mean() if 'accuracy' in df.columns else 0,
                'avg_total_score': df['total_score'].mean() if 'total_score' in df.columns else 0,
                'model_count': len(df) if not df.empty else 0
            }
            
            if self.debug and 'accuracy' in df.columns and 'ethical_compliance' in df.columns:
                self.log(f"Average metrics - Accuracy: {df['accuracy'].mean():.4f}, Compliance: {df['ethical_compliance'].mean():.4f}")
            
            # Update state
            return {
                "evaluation_data": state.get("evaluation_data", {}),
                "scores_df": df,
                "metrics": metrics,
                "errors": state.get("errors", [])
            }
        except Exception as e:
            self.log(f"Error in _compute_islamic_metrics: {e}")
            self.log(traceback.format_exc())
            errors = state.get("errors", [])
            errors.append(f"Metrics computation error: {str(e)}")
            return {
                "evaluation_data": state.get("evaluation_data", {}),
                "scores_df": state.get("scores_df", pd.DataFrame()),
                "metrics": {},
                "errors": errors
            }

    def _generate_summary(self, state: MizanState) -> MizanState:
        """
        Generate summary information from metrics.
        
        Args:
            state: Current workflow state
            
        Returns:
            Updated state with summary information
        """
        try:
            df = state.get("scores_df", pd.DataFrame())
            
            # Generate summary based on available columns
            summary = {}
            
            if not df.empty and 'model_name' in df.columns:
                # Store the complete dataframe for complete rankings
                summary['complete_df'] = df
                
                # Top models by total score
                if 'total_score' in df.columns:
                    top_models = df.nlargest(min(3, len(df)), 'total_score')
                    summary['top_models'] = top_models[['model_name', 'total_score']].to_dict(orient='records')
                    
                    if self.debug:
                        self.log("\nTop models by score:")
                        for idx, row in top_models.iterrows():
                            self.log(f"  {row['model_name']}: {row['total_score']:.4f}")
                
                # Top models by ethical compliance
                if 'ethical_compliance' in df.columns:
                    ethical_leaders = df.nlargest(min(3, len(df)), 'ethical_compliance')
                    summary['ethical_leaders'] = ethical_leaders[['model_name', 'ethical_compliance']].to_dict(orient='records')
                    
                    if self.debug:
                        self.log("\nTop models by ethical compliance:")
                        for idx, row in ethical_leaders.iterrows():
                            self.log(f"  {row['model_name']}: {row['ethical_compliance']:.4f}")
                
                # Top models by accuracy
                if 'accuracy' in df.columns:
                    accuracy_leaders = df.nlargest(min(3, len(df)), 'accuracy')
                    summary['accuracy_leaders'] = accuracy_leaders[['model_name', 'accuracy']].to_dict(orient='records')
                    
                    if self.debug:
                        self.log("\nTop models by accuracy:")
                        for idx, row in accuracy_leaders.iterrows():
                            self.log(f"  {row['model_name']}: {row['accuracy']:.4f}")
            
            # Find weak areas
            summary['weak_areas'] = self._find_weak_areas(df)
            
            # Update state
            return {
                "evaluation_data": state.get("evaluation_data", {}),
                "scores_df": state.get("scores_df", pd.DataFrame()),
                "metrics": state.get("metrics", {}),
                "summary": summary,
                "errors": state.get("errors", [])
            }
        except Exception as e:
            self.log(f"Error in _generate_summary: {e}")
            self.log(traceback.format_exc())
            errors = state.get("errors", [])
            errors.append(f"Summary generation error: {str(e)}")
            return {
                "evaluation_data": state.get("evaluation_data", {}),
                "scores_df": state.get("scores_df", pd.DataFrame()),
                "metrics": state.get("metrics", {}),
                "summary": {},
                "errors": errors
            }

    def _find_weak_areas(self, df: pd.DataFrame) -> List[str]:
        """
        Identify weak areas based on model performance.
        
        Args:
            df: DataFrame with model scores
            
        Returns:
            List of identified weak areas
        """
        weak_spots = []
        thresholds = {
            'accuracy': 0.8,
            'ethical_alignment': 0.85,
            'ethical_compliance': 0.8
        }

        if self.debug:
            self.log("\nChecking for weak areas:")
            
        for metric, threshold in thresholds.items():
            try:
                if metric in df.columns:
                    mean_value = df[metric].mean()
                    if self.debug:
                        self.log(f"  {metric}: average = {mean_value:.4f}, threshold = {threshold}")
                    if mean_value < threshold:
                        weak_spots.append(metric.replace('_', ' ').title())
            except Exception as e:
                self.log(f"  Error checking {metric}: {e}")
        
        if self.debug:
            self.log(f"  Identified weak areas: {weak_spots}")
            
        return weak_spots

    def _build_report(self, state: MizanState) -> MizanState:
        """
        Build the final report from metrics and summary.
        
        Args:
            state: Current workflow state
            
        Returns:
            Updated state with final report
        """
        try:
            # Get summary and errors from state
            summary = state.get("summary", {})
            errors = state.get("errors", [])
            metrics = state.get("metrics", {})
            
            # Get the complete dataframe if available
            complete_df = summary.get('complete_df', None)
            
            # Build report
            if errors:
                report = {
                    'status': 'error',
                    'errors': errors,
                    'timestamp': datetime.now().isoformat(),
                    'version': '1.0'
                }
            else:
                # Store the complete dataframe for CSV export
                if complete_df is not None:
                    # Sort by total score descending
                    if 'total_score' in complete_df.columns:
                        complete_df = complete_df.sort_values('total_score', ascending=False)
                    
                    # Convert to records for the report
                    complete_rankings = complete_df.to_dict(orient='records')
                else:
                    complete_rankings = []
                
                report = {
                    'status': 'success',
                    'leaderboard': summary.get('top_models', []),
                    'ethical_ranking': summary.get('ethical_leaders', []),
                    'accuracy_ranking': summary.get('accuracy_leaders', []),
                    'areas_for_improvement': summary.get('weak_areas', []),
                    'metrics': metrics,
                    'complete_rankings': complete_rankings, 
                    'timestamp': datetime.now().isoformat(),
                    'version': '1.0'
                }
            
            # Update state
            return {
                "evaluation_data": state.get("evaluation_data", {}),
                "scores_df": state.get("scores_df", pd.DataFrame()),
                "metrics": state.get("metrics", {}),
                "summary": state.get("summary", {}),
                "report": report,
                "errors": errors
            }
        except Exception as e:
            self.log(f"Error in _build_report: {e}")
            self.log(traceback.format_exc())
            errors = state.get("errors", [])
            errors.append(f"Report generation error: {str(e)}")
            return {
                "evaluation_data": state.get("evaluation_data", {}),
                "scores_df": state.get("scores_df", pd.DataFrame()),
                "metrics": state.get("metrics", {}),
                "summary": state.get("summary", {}),
                "report": {"status": "error", "errors": errors},
                "errors": errors
            }

    def process_results(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Process evaluation results with the workflow.
        
        Args:
            data: Input data with model scores
            
        Returns:
            Final report with rankings and analysis
        """
        # Set initial state with evaluation data
        initial_state = {"evaluation_data": data}
        
        try:
            # Run the graph with the initial state
            self.log("Starting MizanRanker workflow...")
            final_state = self.graph.invoke(initial_state)
            
            # Return the report from the final state
            self.log("Workflow completed successfully")
            return final_state.get("report", {})
        except Exception as e:
            self.log(f"Error in process_results: {e}")
            self.log(traceback.format_exc())
            return {"status": "error", "error": str(e)}

    def export(self, report: Dict[str, Any], fmt: str = 'json') -> str:
        """
        Export the report in the requested format.
        
        Args:
            report: Report data to export
            fmt: Output format ('json' or 'csv')
            
        Returns:
            Formatted report as a string
        """
        try:
            if fmt == 'json':
                return json.dumps(report, indent=2)
            elif fmt == 'csv':
                # Use complete rankings if available, otherwise use leaderboard
                if "complete_rankings" in report and report["complete_rankings"]:
                    df = pd.DataFrame(report["complete_rankings"])
                    
                    # Select columns in preferred order, if they exist
                    cols = ['model_name']
                    for col in ['total_score', 'accuracy', 'ethical_alignment', 'ethical_compliance']:
                        if col in df.columns:
                            cols.append(col)
                    
                    # Add any remaining columns
                    for col in df.columns:
                        if col not in cols:
                            cols.append(col)
                    
                    # Return CSV with all available columns and all models
                    return df[cols].to_csv(index=False)
                elif "leaderboard" in report and report["leaderboard"]:
                    return pd.DataFrame(report.get('leaderboard', [])).to_csv(index=False)
                else:
                    return "No ranking data available\n"
            else:
                raise ValueError("Unsupported format. Choose 'json' or 'csv'.")
        except Exception as e:
            self.log(f"Error in export: {e}")
            self.log(traceback.format_exc())
            return f"Error exporting report: {str(e)}"