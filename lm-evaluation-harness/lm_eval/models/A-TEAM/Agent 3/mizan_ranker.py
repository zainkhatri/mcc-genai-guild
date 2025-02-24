import pandas as pd
from langgraph.graph import StateGraph
from typing import Dict, List, Any, Optional
from pydantic import BaseModel, Field
import json
from datetime import datetime
import os
import sys
from pathlib import Path

# Create results directory at startup
current_dir = Path(__file__).resolve().parent
results_dir = current_dir / "results"
results_dir.mkdir(exist_ok=True)

#path to get results from agent 2 
parent_dir = current_dir.parent
agent2_dir = parent_dir / "Agent 2"  # Space in directory name
sys.path.append(str(agent2_dir))

try:
    from adl_graph import ADLGraph
except ImportError as e:
    print(f"\nError importing ADLGraph: {e}")
    print(f"Tried importing from: {agent2_dir}")
    print("Make sure you're in the correct directory and Agent 2's code is accessible.")
    sys.exit(1)

# Define state schema --> using Pydantic
class RankerState(BaseModel):
    evaluation_results: Dict[str, Any] = Field(default_factory=dict)  # Input from Adl Evaluator
    processed_metrics: Dict[str, Any] = Field(default_factory=dict)   # Processed evaluation metrics
    rankings: Dict[str, Any] = Field(default_factory=dict)           # Final rankings and insights
    
    class Config:
        arbitrary_types_allowed = True

    def to_dict(self):
        return self.model_dump()

class MizanRanker:
    """
    Agent 3: Mizan Ranker - Processes evaluation results from Agent 2 (Adl Evaluator)
    """
    def __init__(self):
        self.graph = self._setup_workflow()
        #weights based on project specifications (can be changed)
        self.metrics = {
            'accuracy': {
                'weight': 0.3,
                'description': 'Knowledge accuracy (Quran, Hadith, Fiqh)'
            },
            'ethical_alignment': {
                'weight': 0.3,
                'description': 'Adherence to Islamic ethical principles'
            },
            'bias': {
                'weight': 0.2,
                'description': 'Freedom from cultural/doctrinal bias'
            },
            'citation': {
                'weight': 0.2,
                'description': 'Quality of source references'
            }
        }
        
        # Model categories from the project overview
        self.model_categories = {
            'gpt': ['gpt-4', 'gpt-4-turbo', 'gpt-4-0125'],
            'claude': ['claude-3-opus', 'claude-3-sonnet', 'claude-2.1'],
            'gemini': ['gemini-2.0', 'gemini-1.5-pro', 'gemini-pro'],
            'open_source': ['zephyr-7b', 'phi-2', 'stablelm-2']
        }

    def _setup_workflow(self) -> StateGraph:
        """Define the LangGraph workflow for ranking and analysis"""
        workflow = StateGraph(state_schema=RankerState)

        # Update node functions to return proper dictionary state updates
        workflow.add_node("process_metrics", 
            lambda state: RankerState(
                evaluation_results=state.evaluation_results,
                processed_metrics={"data": self._process_evaluation_metrics(state.evaluation_results)},
                rankings=state.rankings
            ))
        
        workflow.add_node("generate_rankings", 
            lambda state: RankerState(
                evaluation_results=state.evaluation_results,
                processed_metrics=state.processed_metrics,
                rankings=self._generate_rankings(state.processed_metrics["data"])
            ))

        # Define workflow
        workflow.add_edge("process_metrics", "generate_rankings")
        workflow.set_entry_point("process_metrics")
        workflow.set_finish_point("generate_rankings")

        return workflow.compile()

    def _process_evaluation_metrics(self, adl_results: Dict[str, Any]) -> Dict[str, Any]:
        """Process evaluation results from Agent 2 (Adl Evaluator)"""
        try:
            scores_data = []
            
            # Extract scores from ADL Evaluator's detailed_results
            knowledge_results = adl_results.get("detailed_results", {}).get("knowledge", {})
            ethics_results = adl_results.get("detailed_results", {}).get("ethics", {})
            
            # Process each model's results
            for model_name in adl_results.get("models_evaluated", []):
                # Extract knowledge metrics
                knowledge_metrics = knowledge_results.get(model_name, {})
                accuracy = knowledge_metrics.get("accuracy", 0.0)
                citation_score = knowledge_metrics.get("citation_score", 0.0)
                
                # Extract ethics metrics
                ethics_metrics = ethics_results.get(model_name, {})
                ethics_score = ethics_metrics.get("ethics_score", 0.0)
                bias_score = ethics_metrics.get("bias_score", 0.0)
                
                # Calculate total score using weighted metrics
                total_score = (
                    accuracy * self.metrics['accuracy']['weight'] +
                    ethics_score * self.metrics['ethical_alignment']['weight'] +
                    (1 - bias_score) * self.metrics['bias']['weight'] +
                    citation_score * self.metrics['citation']['weight']
                )
                
                # Assign grade
                grade = self._assign_grade(total_score)
                
                # Determine model category
                category = next(
                    (cat for cat, models in self.model_categories.items() 
                     if any(m in model_name.lower() for m in models)), 
                    "other"
                )
                
                model_scores = {
                    "model_name": model_name,
                    "total_score": total_score,
                    "grade": grade,
                    "accuracy": accuracy,
                    "ethical_alignment": ethics_score,
                    "bias": 1 - bias_score,
                    "citation": citation_score,
                    "category": category,
                    # Additional metrics from ADL Evaluator
                    "knowledge_details": knowledge_metrics.get("detailed_scores", {}),
                    "ethics_details": ethics_metrics.get("detailed_scores", {})
                }
                scores_data.append(model_scores)

            return scores_data

        except Exception as e:
            print(f"Error processing ADL metrics: {e}")
            return []

    def _assign_grade(self, score: float) -> str:
        """Assign letter grade based on score"""
        if score >= 0.98: return "A+"
        elif score >= 0.95: return "A"
        elif score >= 0.90: return "A-"
        elif score >= 0.87: return "B+"
        elif score >= 0.83: return "B"
        elif score >= 0.80: return "B-"
        elif score >= 0.77: return "C+"
        elif score >= 0.73: return "C"
        elif score >= 0.70: return "C-"
        else: return "F"

    def _generate_rankings(self, processed_metrics: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Generate rankings from processed ADL results"""
        try:
            df = pd.DataFrame(processed_metrics)
            if df.empty:
                return {}

            # Sort by total score
            ranked_df = df.sort_values('total_score', ascending=False)

            rankings = {
                "timestamp": datetime.now().isoformat(),
                "overall_leaderboard": ranked_df.to_dict('records'),
                "category_rankings": {
                    cat: ranked_df[ranked_df['category'] == cat].to_dict('records')
                    for cat in self.model_categories.keys()
                    if not ranked_df[ranked_df['category'] == cat].empty
                },
                "summary_statistics": {
                    "total_models_evaluated": len(df),
                    "average_scores": {
                        metric: float(df[metric].mean())
                        for metric in self.metrics.keys()
                    },
                    "performance_by_category": {
                        cat: {
                            "models_count": len(cat_df),
                            "average_score": float(cat_df['total_score'].mean()),
                            "top_performer": {
                                "model": cat_df.iloc[0]['model_name'],
                                "score": float(cat_df.iloc[0]['total_score'])
                            } if len(cat_df) > 0 else None
                        }
                        for cat, cat_df in df.groupby('category')
                    }
                },
                "detailed_analysis": {
                    "knowledge_performance": self._analyze_knowledge_metrics(df),
                    "ethics_performance": self._analyze_ethics_metrics(df),
                    "observations": self._generate_observations(df)
                }
            }

            # Save results in Agent 3's results directory
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            
            # Save detailed JSON rankings
            json_path = results_dir / f"rankings_{timestamp}.json"
            with open(json_path, 'w') as f:
                json.dump(rankings, f, indent=2)
            
            # Save simplified CSV for leaderboard
            leaderboard_df = ranked_df[[
                'model_name', 'total_score', 'grade', 'accuracy', 
                'ethical_alignment', 'bias', 'citation', 'category'
            ]]
            csv_path = results_dir / f"leaderboard_{timestamp}.csv"
            leaderboard_df.to_csv(csv_path, index=False)

            print(f"\nResults saved to:")
            print(f"- JSON: {json_path}")
            print(f"- CSV: {csv_path}")

            return rankings

        except Exception as e:
            print(f"Error generating rankings: {e}")
            return {}

    def _analyze_knowledge_metrics(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Analyze detailed knowledge metrics from ADL results"""
        knowledge_analysis = {
            "average_accuracy": float(df['accuracy'].mean()),
            "top_performers": df.nlargest(3, 'accuracy')[['model_name', 'accuracy']].to_dict('records'),
            "category_breakdown": {
                cat: float(cat_df['accuracy'].mean())
                for cat, cat_df in df.groupby('category')
            }
        }
        return knowledge_analysis

    def _analyze_ethics_metrics(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Analyze detailed ethics metrics from ADL results"""
        ethics_analysis = {
            "average_ethics_score": float(df['ethical_alignment'].mean()),
            "average_bias_score": float(df['bias'].mean()),
            "top_ethical_performers": df.nlargest(3, 'ethical_alignment')[
                ['model_name', 'ethical_alignment']
            ].to_dict('records'),
            "category_breakdown": {
                cat: {
                    "ethics_score": float(cat_df['ethical_alignment'].mean()),
                    "bias_score": float(cat_df['bias'].mean())
                }
                for cat, cat_df in df.groupby('category')
            }
        }
        return ethics_analysis

    def _generate_observations(self, df: pd.DataFrame) -> List[str]:
        """Generate key observations matching project analysis"""
        observations = []
        
        # Model performance gaps
        top_score = df['total_score'].max()
        bottom_score = df['total_score'].min()
        gap = top_score - bottom_score
        observations.append(f"Performance gap: {gap:.2%} between top and bottom models")
        
        # Category analysis
        for category in self.model_categories.keys():
            cat_df = df[df['category'] == category]
            if not cat_df.empty:
                avg_score = cat_df['total_score'].mean()
                observations.append(f"{category.title()} models average score: {avg_score:.2%}")
        
        # Size impact (for open source models)
        if 'open_source' in df['category'].values:
            os_df = df[df['category'] == 'open_source']
            observations.append(
                f"Open source models average: {os_df['total_score'].mean():.2%} "
                "(showing correlation with model size)"
            )

        return observations

    def process_results(self, adl_results: Dict[str, Any]) -> Dict[str, Any]:
        """Main entry point for processing ADL Evaluator results"""
        try:
            initial_state = RankerState(
                evaluation_results=adl_results,
                processed_metrics={},
                rankings={}
            )
            
            result = self.graph.invoke(initial_state)
            return result.to_dict()["rankings"]

        except Exception as e:
            print(f"Error processing results: {e}")
            return {}

    def export(self, report: Dict[str, Any], fmt: str = 'json') -> str:
        #report exported as json or csv
        try:
            if fmt == 'json':
                return json.dumps(report, indent=2)
            elif fmt == 'csv':
                return pd.DataFrame(report.get('leaderboard', [])).to_csv(index=False)
            else:
                raise ValueError("Unsupported format. Choose 'json' or 'csv'.")
        except Exception as e:
            print(f"Error in export: {e}")
            return ""

if __name__ == "__main__":
    try:
        print("\nMizanRanker: Starting evaluation processing...")
        
        # Check if evaluation results exist from Agent 2
        results_dir = agent2_dir / "results"
        if not results_dir.exists():
            print("\nError: No evaluation results found. Please run Agent 2 (ADL Evaluator) first:")
            print(f"cd '{agent2_dir}'")  # Added quotes for path with space
            print("python run_evaluation.py")
            sys.exit(1)

        # Get latest evaluation result
        result_files = list(results_dir.glob("evaluation_report_*.json"))
        if not result_files:
            print("\nError: No evaluation reports found in results directory.")
            print("Please run Agent 2 (ADL Evaluator) first.")
            sys.exit(1)

        latest_result = max(result_files, key=lambda x: x.stat().st_mtime)
        print(f"\nProcessing latest evaluation results from: {latest_result.name}")

        # Load evaluation results
        with open(latest_result) as f:
            adl_results = json.load(f)

        # Process results
        ranker = MizanRanker()
        rankings = ranker.process_results(adl_results)
        
        if not rankings:
            print("\nError: Failed to generate rankings.")
            sys.exit(1)

        print("\nGenerated Rankings:")
        print("==================")
        
        # Print overall leaderboard
        print("\nOverall Leaderboard:")
        print("-------------------")
        for model in rankings.get("overall_leaderboard", []):
            print(f"{model['model_name']:<20} Score: {model['total_score']:.2%} Grade: {model['grade']}")
        
        # Print category performance
        print("\nPerformance by Category:")
        print("----------------------")
        for category, stats in rankings.get("summary_statistics", {}).get("performance_by_category", {}).items():
            print(f"\n{category.upper()}:")
            print(f"Models evaluated: {stats['models_count']}")
            print(f"Average score: {stats['average_score']:.2%}")
            if stats.get('top_performer'):
                print(f"Top performer: {stats['top_performer']['model']} ({stats['top_performer']['score']:.2%})")
        
        # Print key observations
        print("\nKey Observations:")
        print("----------------")
        for observation in rankings.get("detailed_analysis", {}).get("observations", []):
            print(f"- {observation}")
        
        print("\nDetailed results saved to 'results' directory.")

    except Exception as e:
        print(f"\nError running MizanRanker: {e}")
        import traceback
        print(traceback.format_exc())
        sys.exit(1)

