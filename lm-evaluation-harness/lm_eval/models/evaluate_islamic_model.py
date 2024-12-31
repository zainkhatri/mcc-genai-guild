from lm_eval import evaluator
from lm_eval.api.model import LM
from lm_eval.tasks.islamic_knowledge_task import IslamicKnowledgeTask
import json
import os
from dotenv import load_dotenv

def main():
    # Load environment variables from .env file (you guys cant see it bc its in gitignore)
    load_dotenv()

    # Get API keys from environment variables
    openai_key = os.getenv('OPENAI_API_KEY')
    anthropic_key = os.getenv('ANTHROPIC_API_KEY')
    google_key = os.getenv('GOOGLE_API_KEY')
    ansari_key = os.getenv('ANSARI_API_KEY')

    # Register the task
    task_dict = {
        "islamic_knowledge": IslamicKnowledgeTask
    }

    # List to store results for each model
    all_results = []

    try:
        # Test GPT-4
        print("Evaluating GPT-4...")
        gpt4_results = evaluator.simple_evaluate(
            model="openai",
            model_args=f"model=gpt-4,api_key={openai_key}",
            tasks=["islamic_knowledge"],
            num_fewshot=0,
            task_dict=task_dict
        )
        all_results.append({"model": "GPT-4", "results": gpt4_results})

        # Test GPT-3.5
        print("Evaluating GPT-3.5...")
        gpt35_results = evaluator.simple_evaluate(
            model="openai",
            model_args=f"model=gpt-3.5-turbo,api_key={openai_key}",
            tasks=["islamic_knowledge"],
            num_fewshot=0,
            task_dict=task_dict
        )
        all_results.append({"model": "GPT-3.5", "results": gpt35_results})

        # Test Claude
        print("Evaluating Claude...")
        claude_results = evaluator.simple_evaluate(
            model="anthropic",
            model_args=f"model=claude-2,api_key={anthropic_key}",
            tasks=["islamic_knowledge"],
            num_fewshot=0,
            task_dict=task_dict
        )
        all_results.append({"model": "Claude", "results": claude_results})

        # Test Gemini
        print("Evaluating Gemini...")
        gemini_results = evaluator.simple_evaluate(
            model="google",
            model_args=f"model=gemini-pro,api_key={google_key}",
            tasks=["islamic_knowledge"],
            num_fewshot=0,
            task_dict=task_dict
        )
        all_results.append({"model": "Gemini", "results": gemini_results})

        # Test Ansari.ai
        print("Evaluating Ansari.ai...")
        ansari_results = evaluator.simple_evaluate(
            model="ansari",
            model_args=f"model=ansari-1,api_key={ansari_key}",  # Update with correct model idk how ansaris key works
            tasks=["islamic_knowledge"],
            num_fewshot=0,
            task_dict=task_dict
        )
        all_results.append({"model": "Ansari.ai", "results": ansari_results})

    except Exception as e:
        print(f"Error during evaluation: {str(e)}")

    # Print results
    print("\nComparative Results:")
    print("=" * 50)
    for result in all_results:
        print(f"\nModel: {result['model']}")
        print(f"Accuracy: {result['results']['islamic_knowledge']['accuracy']:.2%}")
        
        # Print category wise performance if its there
        if 'category_accuracy' in result['results']['islamic_knowledge']:
            print("\nCategory-wise Performance:")
            for category, acc in result['results']['islamic_knowledge']['category_accuracy'].items():
                print(f"{category}: {acc:.2%}")

    # Save results to file
    with open('evaluation_results.json', 'w') as f:
        json.dump(all_results, f, indent=2)

if __name__ == "__main__":
    main()