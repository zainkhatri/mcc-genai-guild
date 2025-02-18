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

def main():
    load_dotenv()

    # Example model configs
    model_configs = [
        {
            "model_name": "gpt-4",
            "api_key": os.getenv("OPENAI_API_KEY"),
            "temperature": 0,
            "max_tokens": 1,
            "system_message": "You are an Islamic knowledge evaluator. Provide concise, accurate answers."
        },
        {
            "model_name": "claude-3-sonnet-20240229",
            "api_key": os.getenv("ANTHROPIC_API_KEY"),
            "temperature": 0,
            "max_tokens": 1,
            "system_message": "You are an Islamic knowledge evaluator. Provide concise, accurate answers."
        }
    ]

    # Load questions
    questions = load_questions("data/q_and_a.jsonl", "data/ethics.jsonl")
    if not questions:
        print("No questions loaded.")
        return

    # Evaluate each model
    for config in model_configs:
        model_name = config.pop("model_name")
        print(f"\nEvaluating {model_name} with direct batch_evaluate (no graph)...")
        evaluator = create_evaluator(model_name, **config)

        # For demonstration, limit to first 5 questions
        limited_questions = questions[:5]

        results = evaluator.batch_evaluate(limited_questions)
        accuracy = sum(r["correct"] for r in results) / len(results)
        print(f"{model_name} accuracy on first 5: {accuracy:.2%}")

if name == "main":
    main()