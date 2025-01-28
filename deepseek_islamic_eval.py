#The following file is dedicated

import json
import os
from datetime import datetime
from dotenv import load_dotenv
from transformers import AutoTokenizer, AutoModelForCausalLM
from lm_eval import evaluator
import lm_eval.tasks
from lm_eval.tasks.islamic_knowledge_task.islamic_knowledge_task import IslamicKnowledgeTask
import torch

# Define a modified IslamicKnowledgeTask class
class IslamicKnowledgeTaskModified(IslamicKnowledgeTask):
    DATASET_PATH = "data/q_and_a.jsonl"

def main():
    load_dotenv()
    
    lm_eval.tasks.TASK_REGISTRY["islamic_knowledge"] = IslamicKnowledgeTaskModified
    all_results = []

    # Evaluate Hugging Face Model: DeepSeek R1 Distill Qwen 32B
    hf_model_id = "deepseek-ai/DeepSeek-R1-Distill-Qwen-32B"
    hf_model_name = "DeepSeek R1 Distill Qwen 32B"

    print(f"\nEvaluating {hf_model_name}...")
    results = evaluator.simple_evaluate(
        model="hf-auto",
        model_args={
            "pretrained": hf_model_id,
            "tokenizer": hf_model_id,
            "device": "cuda" if torch.cuda.is_available() else "cpu",
            "max_length": 2048,
            "temperature": 0,
        },
        tasks=["islamic_knowledge"],
        num_fewshot=2,
        limit=50,                # Evaluate on first 50 questions
        apply_chat_template=False,  # Disable chat template
        batch_size=1
    )

    all_results.append({
        "model": hf_model_name,
        "provider": "Hugging Face",
        "results": results
    })

    print("\nEvaluation Results:")
    print("=" * 50)
    try:
        metrics = results['results']['islamic_knowledge']
        accuracy = metrics.get('accuracy,none', 0.0)
        print(f"{hf_model_name}: {accuracy:.2%}")
    except KeyError as e:
        print(f"Could not extract accuracy for {hf_model_name}: {str(e)}")
    
    # Save results with timestamp
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f'model_evaluation_{timestamp}.json'
    with open(filename, 'w') as f:
        json.dump(all_results, f, indent=2)
    print(f"\nResults saved to {filename}")

if __name__ == "__main__":
    main()

