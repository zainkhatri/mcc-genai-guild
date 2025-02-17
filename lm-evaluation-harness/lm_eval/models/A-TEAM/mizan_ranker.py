import pandas as pd
from langgraph.graph import StateGraph
from typing import Dict, List, Any
import json

class MizanRanker:
    def __init__(self):
        """Initialize the Mizan Ranker agent"""
        self.graph = self._create_workflow()
        
    def _create_workflow(self) -> StateGraph:
        """Create the LangGraph workflow for ranking and reporting"""
        workflow = StateGraph()
        
        workflow.add_node("aggregate_scores", self._aggregate_scores)
        workflow.add_node("calculate_islamic_metrics", self._calculate_islamic_metrics)
        workflow.add_node("generate_insights", self._generate_insights)
        workflow.add_node("report", self._prepare_report)
        
        workflow.add_edge("aggregate_scores", "calculate_islamic_metrics")
        workflow.add_edge("calculate_islamic_metrics", "generate_insights")
        workflow.add_edge("generate_insights", "report")
        
        return workflow

    def _aggregate_scores(self, evaluation_data: Dict[str, Any]) -> pd.DataFrame:
        
        df = pd.DataFrame(evaluation_data)
        
        # Define weights (this can be adjusted not sure what is best).
        weights = {
            'accuracy': 0.3,
            'ethical_alignment': 0.3,
            'bias': 0.2,
            'citation': 0.2
        }
        
        
        for metric in weights.keys():
            if metric not in df.columns:
                df[metric] = 0.0
        
        # calculator for weighted total score.
        df['total_score'] = sum(df[metric] * weight for metric, weight in weights.items())
        return df

    def _calculate_islamic_metrics(self, df: pd.DataFrame) -> pd.DataFrame:
        
        for metric in ['citation', 'accuracy', 'ethical_alignment', 'bias']:
            if metric not in df.columns:
                df[metric] = 0.0
        
        df['source_authenticity'] = df['citation'] * df['accuracy']
        df['ethical_compliance'] = df['ethical_alignment'] * (1 - df['bias'])
        return df

    def _generate_insights(self, df: pd.DataFrame) -> Dict[str, Any]:
        
        insights = {
            'top_performers': df.nlargest(3, 'total_score')[['model_name', 'total_score']].to_dict(orient='records'),
            'ethical_leaders': df.nlargest(3, 'ethical_compliance')[['model_name', 'ethical_compliance']].to_dict(orient='records'),
            'areas_for_improvement': self._identify_improvement_areas(df)
        }
        return insights

    def _identify_improvement_areas(self, df: pd.DataFrame) -> List[str]:
        
        areas = []
        thresholds = {
            'accuracy': 0.8,
            'ethical_alignment': 0.85,
            'bias': 0.3,  # For bias, lower values indicate better performance.
            'citation': 0.75
        }
        
        for metric, threshold in thresholds.items():
            if metric in df.columns and df[metric].mean() < threshold:
                areas.append(metric.replace('_', ' ').title())
        return areas

    def _prepare_report(self, insights: Dict[str, Any]) -> Dict[str, Any]:
        
        report = {
            'leaderboard': insights['top_performers'],
            'ethical_performance': insights['ethical_leaders'],
            'improvement_areas': insights['areas_for_improvement'],
            'timestamp': pd.Timestamp.now().isoformat(),
            'format_version': '1.0'
        }
        return report

    def process_evaluation_results(self, evaluation_data: Dict[str, Any]) -> Dict[str, Any]:
        
        # initialize evaluation data in the workflow state.
        state = {"evaluation_data": evaluation_data}
        # Run the workflow --> node output is stored under 'report'.
        final_state = self.graph.run(state)
        return final_state['report']

    def export_results(self, report: Dict[str, Any], format: str = 'json') -> str:
        
        if format == 'json':
            return json.dumps(report, indent=2)
        elif format == 'csv':
            return pd.DataFrame(report['leaderboard']).to_csv()
        else:
            raise ValueError(f"Unsupported format: {format}")

# testing individual models if needed
if __name__ == "__main__":
    
    data = {
        'model_name': ['Model A', 'Model B', 'Model C'],
        'accuracy': [0.85, 0.90, 0.78],
        'ethical_alignment': [0.88, 0.87, 0.80],
        'bias': [0.2, 0.1, 0.25],
        'citation': [0.80, 0.75, 0.70]
    }
    
    ranker = MizanRanker()
    report = ranker.process_evaluation_results(data)
    print(ranker.export_results(report)) 