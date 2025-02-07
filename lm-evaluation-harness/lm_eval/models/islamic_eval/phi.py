import json
import os
from datetime import datetime
from lm_eval import evaluator
import lm_eval.tasks
from lm_eval.tasks.islamic_knowledge_task.islamic_knowledge_task import IslamicKnowledgeTask

class IslamicKnowledgeTaskModified(IslamicKnowledgeTask):
    DATASET_PATH = "data/q_and_a.jsonl"

def main():
    # Register our modified task
    lm_eval.tasks.TASK_REGISTRY["islamic_knowledge"] = IslamicKnowledgeTaskModified
    
    all_results = []

    # Define Phi models to test
    phi_models = [
        ("microsoft/phi-1", "Phi-1"),
        ("microsoft/phi-1_5", "Phi-1.5"),
        ("microsoft/phi-1-small", "Phi-1-small"),
        ("microsoft/phi-1_5-small", "Phi-1.5-small"),
        ("microsoft/phi-4", "Phi-4")
    ]
    
    for model_id, model_name in phi_models:
        print(f"\nEvaluating {model_name}...")
        results = evaluator.simple_evaluate(
            model="hf",  # Use the Hugging Face model interface
            model_args={
                "pretrained": model_id,
                "device": "cuda",  # Use GPU if available
                "dtype": "bfloat16",  # Use bfloat16 for Phi-4 compatibility
                "trust_remote_code": True,
                "max_length": 2048,
                "use_fast_tokenizer": True,
                "system_message": (
                    "You are taking an Islamic knowledge test. "
                    "For each question, respond with only a single letter (A, B, C, or D)."
                ),
            },
            tasks=["islamic_knowledge"],
            num_fewshot=2,
            limit=50,
            batch_size=1
        )
        
        all_results.append({
            "model": model_name,
            "provider": "Microsoft",
            "results": results
        })

    print("\nComparative Results:")
    print("=" * 50)
    
    # Print results
    print("\nMicrosoft Models:")
    print("-" * 30)
    for result in all_results:
        try:
            metrics = result['results']['results']['islamic_knowledge'] if 'results' in result['results'] else result['results']['islamic_knowledge']
            accuracy = metrics.get('accuracy,none', 0.0)
            print(f"{result['model']}: {accuracy:.2%}")
        except KeyError as e:
            print(f"Could not extract accuracy for {result['model']}: {str(e)}")

    # Save results with timestamp
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f'phi_model_comparison_{timestamp}.json'
    with open(filename, 'w') as f:
        json.dump(all_results, f, indent=2)
    print(f"\nResults saved to {filename}")

if __name__ == "__main__":
    main()