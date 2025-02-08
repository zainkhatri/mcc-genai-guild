import json
import os
from datetime import datetime
from lm_eval import evaluator, utils
import lm_eval.tasks
from lm_eval.tasks.islamic_knowledge_task.islamic_knowledge_task import IslamicKnowledgeTask
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer

def main():
    print("\nEvaluating Phi models...")
    
    all_results = []

    # Define Phi models to test
    phi_models = [
        ("microsoft/phi-1", "Phi-1"),
        ("microsoft/phi-1_5", "Phi-1.5"),
        ("microsoft/phi-1-small", "Phi-1-small"),
        ("microsoft/phi-1_5-small", "Phi-1.5-small")
    ]
    
    for model_id, model_name in phi_models:
        print(f"\nEvaluating {model_name}...")
        
        results = evaluator.simple_evaluate(
            model="hf",
            model_args={
                "pretrained": model_id,
                "device": "cuda",
                "dtype": "bfloat16",
                "trust_remote_code": True,
                "use_fast_tokenizer": True,
                "tokenizer_kwargs": {"padding_side": "left"},
                "config_kwargs": {"pad_token_id": 0}
            },
            tasks=["islamic_knowledge"],
            num_fewshot=2,
            limit=50,
            batch_size=1,
            no_cache=True,
            generation_kwargs={
                "max_new_tokens": 10,
                "do_sample": False,
                "num_beams": 1,
                "pad_token_id": 0,
                "eos_token_id": None
            }
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