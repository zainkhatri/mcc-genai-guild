import json
import os
from dotenv import load_dotenv
from lm_eval import evaluator
import lm_eval.tasks
from lm_eval.tasks.islamic_knowledge_task.islamic_knowledge_task import IslamicKnowledgeTask
from datetime import datetime

class IslamicKnowledgeTaskModified(IslamicKnowledgeTask):
    DATASET_PATH = "data/q_and_a.jsonl"

def evaluate_model(model_id, model_name, api_key):
    print(f"\nEvaluating {model_name}...")
    try:
        results = evaluator.simple_evaluate(
            model="anthropic-chat",
            model_args={
                "model": model_id,
                "api_key": api_key,
                "temperature": 0,
                "max_tokens": 1,
                "system_message": (
                    "You are taking an Islamic knowledge test. "
                    "For each question, respond with only a single letter (A, B, C, or D)."
                ),
                "anthropic_api_version": "2023-06-01"
            },
            tasks=["islamic_knowledge"],
            num_fewshot=2,
            limit=50,
            apply_chat_template=True,
            batch_size=1
        )
        
        # Immediately print the results for this model
        try:
            metrics = results['results']['islamic_knowledge'] if 'results' in results else results['islamic_knowledge']
            accuracy = metrics.get('accuracy,none', 0.0)
            print(f"\nResults for {model_name}:")
            print(f"Accuracy: {accuracy:.2%}")
        except KeyError as e:
            print(f"Could not extract accuracy for {model_name}: {str(e)}")
            
        return {
            "model": model_name,
            "provider": "Anthropic",
            "results": results
        }
    except Exception as e:
        print(f"\nError evaluating {model_name}: {str(e)}")
        return None

def main():
    load_dotenv()
    
    anthropic_key = os.getenv("ANTHROPIC_API_KEY")
    all_results = []

    if anthropic_key:
        # Test each model individually
        models = [
            ("claude-3-opus-20240229", "Claude 3.5 Opus"),
            ("claude-3-sonnet-20240229", "Claude 3.5 Sonnet")
            # Note: Claude 3 Ultra removed as it's not available yet
        ]
        
        for model_id, model_name in models:
            result = evaluate_model(model_id, model_name, anthropic_key)
            if result:
                all_results.append(result)

    # Save final results
    if all_results:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f'model_comparison_{timestamp}.json'
        with open(filename, 'w') as f:
            json.dump(all_results, f, indent=2)
        print(f"\nFull results saved to {filename}")
    else:
        print("\nNo results were successfully generated to save.")

if __name__ == "__main__":
    main()