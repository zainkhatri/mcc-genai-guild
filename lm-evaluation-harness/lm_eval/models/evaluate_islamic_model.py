from lm_eval import evaluator
import json
import os
from dotenv import load_dotenv
import lm_eval.tasks
from lm_eval.tasks.islamic_knowledge_task.islamic_knowledge_task import IslamicKnowledgeTask

class IslamicKnowledgeTaskModified(IslamicKnowledgeTask):
    DATASET_PATH = "data/q_and_a.jsonl"

def main():
    # Load environment variables
    load_dotenv()

    # Get API key
    openai_key = os.getenv('OPENAI_API_KEY')

    # Register task
    lm_eval.tasks.TASK_REGISTRY["islamic_knowledge"] = IslamicKnowledgeTaskModified

    # Store results
    all_results = []

    try:
        if openai_key:
            # Test GPT-4
            print("Evaluating GPT-4...")
            gpt4_results = evaluator.simple_evaluate(
                model="openai-chat-completions",
                model_args={
                    "model": "gpt-4",
                    "api_key": openai_key,
                    "temperature": 0,
                    "max_tokens": 1,
                    "eos_token": "\n",
                    "chat_format": "chatml"
                },
                tasks=["islamic_knowledge"],
                num_fewshot=2,
                limit=50,
                apply_chat_template=True,
                batch_size=1,
            )
            all_results.append({"model": "GPT-4", "results": gpt4_results})

            # Test GPT-3.5
            print("Evaluating GPT-3.5...")
            gpt35_results = evaluator.simple_evaluate(
                model="openai-chat-completions",
                model_args={
                    "model": "gpt-3.5-turbo",
                    "api_key": openai_key,
                    "temperature": 0,
                    "max_tokens": 1,
                    "eos_token": "\n",
                    "chat_format": "chatml"
                },
                tasks=["islamic_knowledge"],
                num_fewshot=2,
                limit=50,
                apply_chat_template=True,
                batch_size=1,
            )
            all_results.append({"model": "GPT-3.5", "results": gpt35_results})

    except Exception as e:
        print(f"Error during evaluation: {str(e)}")
        import traceback
        traceback.print_exc()

    # Print results
    print("\nComparative Results:")
    print("=" * 50)
    for result in all_results:
        print(f"\nModel: {result['model']}")
        try:
            if 'results' in result['results']:
                metrics = result['results']['results']['islamic_knowledge']
            else:
                metrics = result['results']['islamic_knowledge']
            accuracy = metrics.get('accuracy,none', 0.0)
            print(f"Accuracy: {accuracy:.2%}")
        except KeyError as e:
            print(f"Could not extract accuracy from results: {str(e)}")
            print("Available keys:", result['results'].keys())

    # Save results
    with open('evaluation_results.json', 'w') as f:
        json.dump(all_results, f, indent=2)

if __name__ == "__main__":
    main()