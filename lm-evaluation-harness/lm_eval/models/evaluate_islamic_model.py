import json
import os
from dotenv import load_dotenv
from lm_eval import evaluator
import lm_eval.tasks
from lm_eval.tasks.islamic_knowledge_task.islamic_knowledge_task import IslamicKnowledgeTask

# Import the google palm implementation
from google_palm import GooglePalmLM

class IslamicKnowledgeTaskModified(IslamicKnowledgeTask):
    DATASET_PATH = "data/q_and_a.jsonl"

def main():
    load_dotenv()
    
    # Load API keys
    anthropic_key = os.getenv("ANTHROPIC_API_KEY")
    google_key = os.getenv("GOOGLE_API_KEY")
    openai_key = os.getenv("OPENAI_API_KEY")
    
    lm_eval.tasks.TASK_REGISTRY["islamic_knowledge"] = IslamicKnowledgeTaskModified
    
    all_results = []

    # Test Anthropic Models
    if anthropic_key:
        anthropic_models = [
            ("claude-3-opus-20240229", "Claude 3 Opus"),
            ("claude-3-sonnet-20240229", "Claude 3 Sonnet"),
            ("claude-2.1", "Claude 2.1")
        ]
        
        for model_id, model_name in anthropic_models:
            print(f"\nEvaluating {model_name}...")
            results = evaluator.simple_evaluate(
                model="anthropic-chat",
                model_args={
                    "model": model_id,
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
            all_results.append({
                "model": model_name,
                "provider": "Anthropic",
                "results": results
            })

    # Test Google Models
    if google_key:
        google_models = [
            ("gemini-1.5-pro", "Gemini 1.5 Pro"),
            ("gemini-1.5-flash", "Gemini 1.5 Flash"),
            ("gemini-2.0-flash-exp", "Gemini 2.0 Flash (Beta)")
        ]
        
        for model_id, model_name in google_models:
            print(f"\nEvaluating {model_name}...")
            results = evaluator.simple_evaluate(
                model="google-palm",
                model_args={
                    "model": model_id,
                    "api_key": google_key,
                    "temperature": 0,
                    "max_tokens": 1,
                    "system_message": (
                        "You are taking an Islamic knowledge test. "
                        "For each question, respond with only a single letter (A, B, C, or D)."
                    )
                },
                tasks=["islamic_knowledge"],
                num_fewshot=2,
                limit=50,
                apply_chat_template=True,
                batch_size=1
            )
            all_results.append({
                "model": model_name,
                "provider": "Google",
                "results": results
            })

    # Test OpenAI Models
    if openai_key:
        openai_models = [
            ("gpt-4-0125-preview", "GPT-4 O1"),
            ("gpt-4-turbo-preview", "GPT-4 Turbo"),
            ("gpt-4", "GPT-4")
        ]
        
        for model_id, model_name in openai_models:
            print(f"\nEvaluating {model_name}...")
            results = evaluator.simple_evaluate(
                model="openai-chat-completions",
                model_args={
                    "model": model_id,
                    "api_key": openai_key,
                    "temperature": 0,
                    "max_tokens": 1,
                    "system_message": (
                        "You are taking an Islamic knowledge test. "
                        "For each question, respond with only a single letter (A, B, C, or D)."
                    )
                },
                tasks=["islamic_knowledge"],
                num_fewshot=2,
                limit=50,
                apply_chat_template=True,
                batch_size=1
            )
            all_results.append({
                "model": model_name,
                "provider": "OpenAI",
                "results": results
            })

    print("\nComparative Results:")
    print("=" * 50)
    
    # Print results by provider
    for provider in ["Anthropic", "Google", "OpenAI"]:
        provider_results = [r for r in all_results if r["provider"] == provider]
        if provider_results:
            print(f"\n{provider} Models:")
            print("-" * 30)
            for result in provider_results:
                try:
                    metrics = result['results']['results']['islamic_knowledge'] if 'results' in result['results'] else result['results']['islamic_knowledge']
                    accuracy = metrics.get('accuracy,none', 0.0)
                    print(f"{result['model']}: {accuracy:.2%}")
                except KeyError as e:
                    print(f"Could not extract accuracy for {result['model']}: {str(e)}")

    # Save results with timestamp
    from datetime import datetime
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f'model_comparison_{timestamp}.json'
    with open(filename, 'w') as f:
        json.dump(all_results, f, indent=2)
    print(f"\nResults saved to {filename}")

if __name__ == "__main__":
    main()