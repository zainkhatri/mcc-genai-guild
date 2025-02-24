"""
main_eval.py

Demonstrates a straightforward usage of ADLEvaluator for batch evaluation
outside the LangGraph workflow (optional).
"""

import os
import json
from typing import List, Dict, Any
from dotenv import load_dotenv
from adl import create_evaluator
import asyncio
from adl_graph import ADLGraph
from pathlib import Path
from general_models import evaluate_models_in_batches

def load_questions(knowledge_path: str, ethics_path: str) -> List[Dict[str, Any]]:
    """
    Load questions from knowledge and ethics files.
    Each line is a JSON object with 'question' and 'correct', plus 'category'.
    """
    questions = []
    # Load knowledge
    if os.path.exists(knowledge_path):
        with open(knowledge_path, 'r') as f:
            for line in f:
                q = json.loads(line)
                q['category'] = 'knowledge'
                questions.append(q)
    # Load ethics
    if os.path.exists(ethics_path):
        with open(ethics_path, 'r') as f:
            for line in f:
                q = json.loads(line)
                q['category'] = 'ethics'
                questions.append(q)
    return questions

async def load_questions() -> List[Dict[str, Any]]:
    """Load questions from data files"""
    try:
        data_dir = Path(__file__).parent.parent.parent.parent / "data"
        questions = []
        
        # Load knowledge questions
        knowledge_path = data_dir / "islamic_knowledge.jsonl"
        if knowledge_path.exists():
            with open(knowledge_path) as f:
                for line in f:
                    question = json.loads(line)
                    question["category"] = "knowledge"
                    questions.append(question)
        
        # Load ethics questions
        ethics_path = data_dir / "ethics.jsonl"
        if ethics_path.exists():
            with open(ethics_path) as f:
                for line in f:
                    question = json.loads(line)
                    question["category"] = "ethics"
                    questions.append(question)
                    
        if not questions:
            print("No questions loaded.")
            return []
            
        print(f"Loaded {len(questions)} questions")
        return questions
        
    except Exception as e:
        print(f"Error loading questions: {e}")
        return []

async def main():
    """Main evaluation function"""
    # Load questions
    questions = await load_questions()
    if not questions:
        return
    
    # Configure models to evaluate
    models_config = {
        "gpt-4": {
            "temperature": 0.0,
            "max_tokens": 150
        },
        "claude-3-opus": {
            "temperature": 0.0,
            "max_tokens": 150
        },
        "gemini-pro": {
            "temperature": 0.0,
            "max_tokens": 150
        }
    }
    
    try:
        # Run evaluation
        print("\nStarting evaluation...")
        results = await evaluate_models_in_batches(ADLGraph(models_config), questions, models_config)
        
        # Save results
        results_dir = Path(__file__).parent / "results"
        results_dir.mkdir(exist_ok=True)
        
        results_path = results_dir / "evaluation_results.json"
        with open(results_path, "w") as f:
            json.dump(results, f, indent=2)
            
        print(f"\nResults saved to: {results_path}")
        
    except Exception as e:
        print(f"Error in evaluation: {e}")

if __name__ == "__main__":
    # Run evaluation
    asyncio.run(main())