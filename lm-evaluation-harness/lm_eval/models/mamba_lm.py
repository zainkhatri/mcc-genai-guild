import os
import json
from datetime import datetime
from dotenv import load_dotenv
from lm_eval import evaluator
import lm_eval.tasks
from lm_eval.tasks.islamic_knowledge_task.islamic_knowledge_task import IslamicKnowledgeTask

class IslamicKnowledgeTaskModified(IslamicKnowledgeTask):
    DATASET_PATH = "data/q_and_a.jsonl"

def evaluate_new_models(limit=50, num_fewshot=2):
    load_dotenv()
    
    # Register task
    lm_eval.tasks.TASK_REGISTRY["islamic_knowledge"] = IslamicKnowledgeTaskModified
    
    all_results = []
    system_message = ("You are taking an Islamic knowledge test. "
                     "For each question, respond with only a single letter (A, B, C, or D).")

    # Common evaluation parameters
    eval_params = {
        "tasks": ["islamic_knowledge"],
        "num_fewshot": num_fewshot,
        "limit": limit,
        "apply_chat_template": True
    }

    # Start with just one small model to test
    hf_models = [
        ("microsoft/phi-2", "Phi-2")
    ]

    for model_id, model_name in hf_models:
        print(f"\nEvaluating {model_name}...")
        try:
            results = evaluator.simple_evaluate(
                model="hf-auto",  # Changed from hf to hf-auto
                model_args={
                    "pretrained": model_id,
                    "trust_remote_code": True,
                    "use_fast": True,
                    "device": "cuda",
                    "batch_size": 1
                },
                **eval_params
            )
            all_results.append({
                "model": model_name,
                "provider": "Hugging Face",
                "results": results
            })
        except Exception as e:
            print(f"Error evaluating {model_name}: {str(e)}")

    # Print results in a table format
    print("\nEvaluation Results")
    print("=" * 60)
    print(f"{'Model Name':<30} {'Accuracy':<10} {'Grade':<5}")
    print("-" * 60)
    
    def get_grade(accuracy):
        if accuracy >= 0.90: return "A"
        elif accuracy >= 0.80: return "B"
        elif accuracy >= 0.70: return "C"
        elif accuracy >= 0.60: return "D"
        else: return "F"
    
    # Sort results by accuracy
    sorted_results = []
    for result in all_results:
        try:
            metrics = result['results']['results']['islamic_knowledge'] if 'results' in result['results'] else result['results']['islamic_knowledge']
            accuracy = metrics.get('accuracy,none', 0.0)
            sorted_results.append((result['model'], accuracy))
        except KeyError as e:
            print(f"Could not extract accuracy for {result['model']}: {str(e)}")
    
    sorted_results.sort(key=lambda x: x[1], reverse=True)
    
    for model_name, accuracy in sorted_results:
        grade = get_grade(accuracy)
        print(f"{model_name:<30} {accuracy:>7.2%}  {grade:>4}")

    # Save results with timestamp
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f'new_models_comparison_{timestamp}.json'
    with open(filename, 'w') as f:
        json.dump(all_results, f, indent=2)
    print(f"\nDetailed results saved to {filename}")

if __name__ == "__main__":
    evaluate_new_models()