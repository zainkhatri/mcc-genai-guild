import os
from lm_eval import evaluator
import lm_eval.tasks
from lm_eval.tasks.islamic_knowledge_task.islamic_knowledge_task import IslamicKnowledgeTask

def main():
    # Register the Islamic knowledge task
    lm_eval.tasks.TASK_REGISTRY["islamic_knowledge"] = IslamicKnowledgeTask

    # List of DeepSeek models to evaluate
    deepseek_models = [
        ("deepseek-ai/deepseek-llm-7b-base", "DeepSeek 7B Base"),
        ("deepseek-ai/deepseek-llm-7b-chat", "DeepSeek 7B Chat"),
        ("deepseek-ai/deepseek-coder-7b-base", "DeepSeek Coder 7B Base"),
        ("deepseek-ai/deepseek-coder-7b-instruct", "DeepSeek Coder 7B Instruct"),
        ("deepseek-ai/deepseek-math-7b-base", "DeepSeek Math 7B Base"),
        ("deepseek-ai/deepseek-math-7b-instruct", "DeepSeek Math 7B Instruct")
    ]

    all_results = []

    # Evaluate each model
    for model_id, model_name in deepseek_models:
        print(f"\nEvaluating {model_name}...")
        try:
            results = evaluator.simple_evaluate(
                model="hf",
                model_args={
                    "pretrained": model_id,
                    "device": "cuda",
                    "dtype": "bfloat16",
                    "trust_remote_code": True,
                    "max_length": 2048,
                    "use_fast_tokenizer": True
                },
                tasks=["islamic_knowledge"],
                num_fewshot=2,
                limit=50,
                batch_size=1
            )
            
            all_results.append({
                "model": model_name,
                "results": results
            })
        except Exception as e:
            print(f"Error evaluating {model_name}: {str(e)}")
            continue

    # Print results
    print("\nComparative Results:")
    print("=" * 50)
    
    for result in all_results:
        try:
            metrics = result['results']['results']['islamic_knowledge'] if 'results' in result['results'] else result['results']['islamic_knowledge']
            accuracy = metrics.get('accuracy,none', 0.0)
            print(f"{result['model']}: {accuracy:.2%}")
        except KeyError as e:
            print(f"Could not extract accuracy for {result['model']}: {str(e)}")

    # Save results with timestamp
    from datetime import datetime
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f'deepseek_comparison_{timestamp}.json'
    
    import json
    with open(filename, 'w') as f:
        json.dump(all_results, f, indent=2)
    print(f"\nResults saved to {filename}")

if __name__ == "__main__":
    main()