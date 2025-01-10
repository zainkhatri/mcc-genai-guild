import json
import os
from dotenv import load_dotenv
from lm_eval import evaluator
import lm_eval.tasks
from lm_eval.tasks.islamic_knowledge_task.islamic_knowledge_task import IslamicKnowledgeTask

class IslamicKnowledgeTaskModified(IslamicKnowledgeTask):
    DATASET_PATH = "data/q_and_a.jsonl"

def main():
    # Load environment variables
    load_dotenv()

    # Get Anthropic API key
    anthropic_key = os.getenv("ANTHROPIC_API_KEY")

    # Register task
    lm_eval.tasks.TASK_REGISTRY["islamic_knowledge"] = IslamicKnowledgeTaskModified

    # Store results
    all_results = []

    try:
        if anthropic_key:
            # Test Claude 2.1
            print("Evaluating Claude 2.1...")
            claude_2_results = evaluator.simple_evaluate(
                model="anthropic-chat",
                model_args={
                    "model": "claude-2.1",
                    "api_key": anthropic_key,
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
            all_results.append({"model": "Claude 2.1", "results": claude_2_results})

            # Test Claude 3
            print("\nEvaluating Claude 3...")
            claude_3_results = evaluator.simple_evaluate(
                model="anthropic-chat",
                model_args={
                    "model": "claude-3-opus-20240229",
                    "api_key": anthropic_key,
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
            all_results.append({"model": "Claude 3", "results": claude_3_results})
        else:
            print("No ANTHROPIC_API_KEY found in environment variables")

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