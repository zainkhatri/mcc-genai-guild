import os
import json
import asyncio
from dotenv import load_dotenv
from typing import List, Dict
from adl_graph import ADLGraph

def load_questions(knowledge_path: str, ethics_path: str) -> List[Dict]:
    questions = []
    # Load all knowledge questions (or slice to 300 if there are more)
    if os.path.exists(knowledge_path):
        with open(knowledge_path, 'r') as f:
            all_knowledge = [json.loads(line) for line in f]
            for q in all_knowledge[:5]:  # limit to 300
                q['category'] = 'knowledge'
                questions.append(q)
    # Load all ethics questions (or slice to 40)
    if os.path.exists(ethics_path):
        with open(ethics_path, 'r') as f:
            all_ethics = [json.loads(line) for line in f]
            for q in all_ethics[:5]:  # limit to 40
                q['category'] = 'ethics'
                questions.append(q)
    return questions

async def main():
    load_dotenv()

    models_config = {
        # Anthropic models:
        "claude-3-opus-20240229": {
            "api_key": os.getenv("ANTHROPIC_API_KEY"),
            "temperature": 0,
            "max_tokens": 1,  
            "system_message": "You are an Islamic knowledge evaluator. Provide concise, accurate answers."
        },
        "claude-3-sonnet-20240229": {
            "api_key": os.getenv("ANTHROPIC_API_KEY"),
            "temperature": 0,
            "max_tokens": 1,
            "system_message": "You are an Islamic knowledge evaluator. Provide concise, accurate answers."
        },
        "claude-2.1": {
            "api_key": os.getenv("ANTHROPIC_API_KEY"),
            "temperature": 0,
            "max_tokens": 1,
            "system_message": "You are an Islamic knowledge evaluator. Provide concise, accurate answers."
        },
        # Google models:
        "gemini-1.5-pro": {
            "api_key": os.getenv("GOOGLE_API_KEY"),
            "temperature": 0,
            "max_tokens": 1,
            "system_message": "You are an Islamic knowledge evaluator. Provide concise, accurate answers."
        },
        "gemini-1.5-flash": {
            "api_key": os.getenv("GOOGLE_API_KEY"),
            "temperature": 0,
            "max_tokens": 1,
            "system_message": "You are an Islamic knowledge evaluator. Provide concise, accurate answers."
        },
        "gemini-2.0-flash-exp": {
            "api_key": os.getenv("GOOGLE_API_KEY"),
            "temperature": 0,
            "max_tokens": 1,
            "system_message": "You are an Islamic knowledge evaluator. Provide concise, accurate answers."
        },
        # OpenAI models:
        "gpt-4-0125-preview": {
            "api_key": os.getenv("OPENAI_API_KEY"),
            "temperature": 0,
            "max_tokens": 1,
            "system_message": "You are an Islamic knowledge evaluator. Provide concise, accurate answers."
        },
        "gpt-4-turbo-preview": {
            "api_key": os.getenv("OPENAI_API_KEY"),
            "temperature": 0,
            "max_tokens": 1,
            "system_message": "You are an Islamic knowledge evaluator. Provide concise, accurate answers."
        },
        "gpt-4": {
            "api_key": os.getenv("OPENAI_API_KEY"),
            "temperature": 0,
            "max_tokens": 1,
            "system_message": "You are an Islamic knowledge evaluator. Provide concise, accurate answers."
        }
    }

    # Use correct relative paths: from tasks/islamic_knowledge_task go up two levels to data folder.
    questions = load_questions("../../data/q_and_a.jsonl", "../../data/ethics.jsonl")
    print(f"Loaded {len(questions)} questions")

    # Initialize the ADL Graph
    adl = ADLGraph(models_config)

    print("Starting evaluation...")
    try:
        report = await adl.run_evaluation(questions)
        print("\nEvaluation complete!")
        print("\nResults Summary:")
        print("=" * 50)

        for model_name, scores in report["scores"].items():
            print(f"\n{model_name}:")
            print(f"Knowledge Accuracy: {scores['knowledge_accuracy']:.2%}")
            print(f"Ethics Accuracy: {scores['ethics_accuracy']:.2%}")
            print(f"Bias Score: {scores['bias_score']:.2f}")
            print(f"Citation Score: {scores['citation_score']:.2f}")
            print(f"Timestamp: {scores['timestamp']}")

    except Exception as e:
        print(f"Error during evaluation: {str(e)}")

if __name__ == "__main__":
    asyncio.run(main())