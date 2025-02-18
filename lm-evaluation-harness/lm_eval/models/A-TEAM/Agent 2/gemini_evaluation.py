import os
import json
import asyncio
from datetime import datetime
from typing import List, Dict
from dotenv import load_dotenv
from adl_graph import ADLGraph

def load_questions(knowledge_path: str, ethics_path: str) -> List[Dict]:
    questions = []
    # Load knowledge questions
    if os.path.exists(knowledge_path):
        with open(knowledge_path, 'r') as f:
            all_knowledge = [json.loads(line) for line in f]
            for q in all_knowledge[:300]:  # limit to 5 for testing
                q['category'] = 'knowledge'
                questions.append(q)
    # Load ethics questions
    if os.path.exists(ethics_path):
        with open(ethics_path, 'r') as f:
            all_ethics = [json.loads(line) for line in f]
            for q in all_ethics[:40]:  # limit to 5 for testing
                q['category'] = 'ethics'
                questions.append(q)
    return questions

async def main():
    load_dotenv()

    # Base configuration for all models
    base_config = {
        "api_key": os.getenv("GOOGLE_API_KEY"),
        "temperature": 0,
        "generation_config": {
            "temperature": 0,
            "top_p": 0.95,
            "top_k": 40,
            "max_output_tokens": 1024,
        },
        "system_message": "You are an Islamic knowledge evaluator. Provide concise, accurate answers."
    }

    # Enhanced configuration for Ultra model
    ultra_config = {
        **base_config,
        "generation_config": {
            **base_config["generation_config"],
            "max_output_tokens": 2048,  # Increased token limit for Ultra
        }
    }

    models_config = {
        "gemini-pro": base_config.copy(),
        "gemini-2.0-flash": base_config.copy(),
    }

    # Load questions
    questions = load_questions("../../../data/q_and_a.jsonl", "../../../data/ethics.jsonl")
    print(f"Loaded {len(questions)} questions")

    # Initialize ADL Graph with all Gemini models
    adl = ADLGraph(models_config)

    print("\nStarting Gemini models evaluation...")
    try:
        report = await adl.run_evaluation(questions)
        
        print("\nEvaluation complete!")
        print("\nResults Summary:")
        print("=" * 50)

        for model_name, scores in report["scores"].items():
            print(f"\n{model_name}:")
            print(f"Knowledge Accuracy: {scores['knowledge_accuracy']:.2%}")
            print(f"Ethics Accuracy: {scores['ethics_accuracy']:.2%}")
            if 'bias_score' in scores:
                print(f"Bias Score: {scores['bias_score']:.2f}")
            if 'citation_score' in scores:
                print(f"Citation Score: {scores['citation_score']:.2f}")
            print(f"Timestamp: {scores['timestamp']}")

        # Save detailed report
        os.makedirs("results", exist_ok=True)
        filename = f"gemini_evaluation_{datetime.now().isoformat()}.json"
        report_path = os.path.join("results", filename)
        with open(report_path, "w") as f:
            json.dump(report, f, indent=2)
        print(f"\nDetailed report saved to: {report_path}")

        # Save summary report
        summary_filename = f"evaluation_summary_{datetime.now().isoformat()}.json"
        summary_path = os.path.join("results", summary_filename)
        summary_report = {
            "timestamp": datetime.now().isoformat(),
            "models_evaluated": list(report["scores"].keys()),
            "summary_metrics": report["scores"]
        }
        with open(summary_path, "w") as f:
            json.dump(summary_report, f, indent=2)
        print(f"Summary report saved to: {summary_path}")

    except Exception as e:
        print(f"Error during evaluation: {str(e)}")
        raise

if __name__ == "__main__":
    asyncio.run(main())