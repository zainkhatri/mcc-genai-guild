"""
adl_graph.py

Implements ADLGraph (Agent 2) as a LangGraph workflow with multiple evaluation nodes:
- Knowledge
- Ethics
- Bias
- Citation
- Score Calculation
- Report Generation

Outputs JSON-formatted results.
"""

import os
import json
from datetime import datetime
from typing import Dict, Any, List, Optional
from typing_extensions import TypedDict
from pydantic import Field
from langgraph.graph import StateGraph, END
from functools import reduce

class GraphState(TypedDict):
    """
    Graph state for storing intermediate results.
    """
    questions: List[Dict[str, Any]]
    knowledge_results: Optional[Dict[str, List[Dict]]]
    ethics_results: Optional[Dict[str, List[Dict]]]
    bias_results: Optional[Dict[str, List[Dict]]]
    citation_results: Optional[Dict[str, List[Dict]]]
    scores: Optional[Dict[str, Dict[str, float]]]
    report: Optional[Dict[str, Any]]

class ADLGraph:
    """
    LangGraph implementation for ADL Evaluator (Agent 2).
    Evaluates:
      - Accuracy (knowledge)
      - Ethical alignment
      - Bias detection (placeholder)
      - Source citation (placeholder)
    Produces JSON-formatted evaluation results.
    """
    
    def __init__(self, models_config: Dict[str, Any]):
        """
        Initialize ADLGraph with model configurations.
        Each model should have: api_key, temperature, max_tokens, system_message, etc.
        """
        self.models = models_config
        self.graph = self._build_graph()

    async def evaluate_knowledge(self, state: GraphState) -> GraphState:
        """
        Evaluate knowledge questions for all models.
        """
        from adl import create_evaluator  # local import
        try:
            knowledge_results = {}
            # Filter knowledge questions
            knowledge_qs = [q for q in state["questions"] if q.get("category") == "knowledge"]
            if not knowledge_qs:
                print("No knowledge questions found.")
                state["knowledge_results"] = {}
                return state

            for model_name, config in self.models.items():
                print(f"Evaluating KNOWLEDGE questions for {model_name}...")
                evaluator = create_evaluator(model_name=model_name, **config)
                results = evaluator.batch_evaluate(knowledge_qs)
                knowledge_results[model_name] = results

                # Compute accuracy
                if results:
                    accuracy = sum(r["correct"] for r in results) / len(results)
                    print(f"{model_name} knowledge accuracy: {accuracy:.2%}")

            state["knowledge_results"] = knowledge_results
            return state
        except Exception as e:
            print(f"Error in evaluate_knowledge: {str(e)}")
            raise

    async def evaluate_ethics(self, state: GraphState) -> GraphState:
        """
        Evaluate ethics questions for all models.
        """
        from adl import create_evaluator  # local import
        try:
            ethics_results = {}
            # Filter ethics questions
            ethics_qs = [q for q in state["questions"] if q.get("category") == "ethics"]
            if not ethics_qs:
                print("No ethics questions found.")
                state["ethics_results"] = {}
                return state

            for model_name, config in self.models.items():
                print(f"Evaluating ETHICS questions for {model_name}...")
                evaluator = create_evaluator(model_name=model_name, **config)
                results = evaluator.batch_evaluate(ethics_qs)
                ethics_results[model_name] = results

                # Compute accuracy or alignment
                if results:
                    alignment_score = sum(r["correct"] for r in results) / len(results)
                    print(f"{model_name} ethics alignment: {alignment_score:.2%}")

            state["ethics_results"] = ethics_results
            return state
        except Exception as e:
            print(f"Error in evaluate_ethics: {str(e)}")
            raise

    async def evaluate_bias(self, state: GraphState) -> GraphState:
        """
        Placeholder for bias detection.
        For now, we just store an empty structure or a dummy score.
        """

        bias_results = {}
        for model_name in self.models:
            # Placeholder: no actual questions, so no real "correct" check
            bias_results[model_name] = [{
                "bias_score": 0.0,
                "notes": "No bias questions defined yet."
            }]
        state["bias_results"] = bias_results
        return state

    async def evaluate_citation(self, state: GraphState) -> GraphState:
        """
        Placeholder for source citation checking.
        For now, we just store a dummy result.
        """
        citation_results = {}
        for model_name in self.models:
            citation_results[model_name] = [{
                "citation_score": 0.0,
                "notes": "No citation checks defined yet."
            }]
        state["citation_results"] = citation_results
        return state

    async def calculate_scores(self, state: GraphState) -> GraphState:
        """
        Calculate comprehensive scores for each model.
        """
        try:
            scores = {}
            model_names = list(self.models.keys())

            for model_name in model_names:
                # knowledge
                knowledge_data = state["knowledge_results"].get(model_name, [])
                knowledge_acc = (sum(r["correct"] for r in knowledge_data) / len(knowledge_data)) if knowledge_data else 0

                # ethics
                ethics_data = state["ethics_results"].get(model_name, [])
                ethics_acc = (sum(r["correct"] for r in ethics_data) / len(ethics_data)) if ethics_data else 0

                scores[model_name] = {
                    "knowledge_accuracy": knowledge_acc,
                    "ethics_accuracy": ethics_acc,
                    "timestamp": datetime.now().isoformat()
                }

                print(f"\nScores for {model_name}:")
                print(f"Knowledge Accuracy: {knowledge_acc:.2%}")
                print(f"Ethics Accuracy: {ethics_acc:.2%}")

            state["scores"] = scores
            return state
        except Exception as e:
            print(f"Error in calculate_scores: {str(e)}")
            raise

    async def generate_report(self, state: GraphState) -> GraphState:
        """
        Generate final evaluation report (JSON).
        """
        try:
            report = {
                "timestamp": datetime.now().isoformat(),
                "models_evaluated": list(self.models.keys()),
                "scores": state["scores"],
                "detailed_results": {
                    "knowledge": state["knowledge_results"],
                    "ethics": state["ethics_results"]
                }
            }
            # Ensure results directory
            os.makedirs("results", exist_ok=True)
            # Save
            filename = f"evaluation_report_{report['timestamp']}.json"
            report_path = os.path.join("results", filename)
            with open(report_path, "w") as f:
                json.dump(report, f, indent=2)

            print(f"\nReport saved to: {report_path}")
            state["report"] = report
            return state
        except Exception as e:
            print(f"Error in generate_report: {str(e)}")
            raise

    def _build_graph(self) -> StateGraph:
        """
        Build the evaluation workflow graph.
        """
        from langgraph.graph import StateGraph, END

        workflow = StateGraph(state_schema=GraphState)

        # Add nodes
        workflow.add_node("evaluate_knowledge", self.evaluate_knowledge)
        workflow.add_node("evaluate_ethics", self.evaluate_ethics)
        workflow.add_node("calculate_scores", self.calculate_scores)
        workflow.add_node("generate_report", self.generate_report)

        # Set the entry point
        workflow.set_entry_point("evaluate_knowledge")

        # Link them in order
        workflow.add_edge("evaluate_knowledge", "evaluate_ethics")
        workflow.add_edge("evaluate_ethics", "calculate_scores")
        workflow.add_edge("calculate_scores", "generate_report")
        workflow.add_edge("generate_report", END)

        return workflow.compile()

    async def run_evaluation(self, questions: List[Dict]) -> Dict:
        """
        Run the complete evaluation workflow.
        """
        try:
            initial_state: GraphState = {
                "questions": questions,
                "knowledge_results": None,
                "ethics_results": None,
                "scores": None,
                "report": None
            }
            
            print("Starting evaluation workflow...")
            final_state = await self.graph.ainvoke(initial_state)
            print("Evaluation workflow completed.")

            if "report" not in final_state:
                raise ValueError("Evaluation completed but no report was generated.")
                
            return final_state["report"]
        except Exception as e:
            print(f"Error in run_evaluation: {str(e)}")
            raise