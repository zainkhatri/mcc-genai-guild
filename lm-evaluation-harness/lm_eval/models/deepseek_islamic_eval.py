import json
import os
import logging
from datetime import datetime
from typing import List, Dict, Any

import torch
from dotenv import load_dotenv
from transformers import AutoTokenizer, AutoModelForCausalLM

from lm_eval import evaluator
import lm_eval.tasks
from lm_eval.tasks.islamic_knowledge_task.islamic_knowledge_task import IslamicKnowledgeTask
from lm_eval.api.instance import Instance

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class IslamicKnowledgeTaskModified(IslamicKnowledgeTask):
    """Modified version of IslamicKnowledgeTask with custom dataset path."""
    DATASET_PATH = "data/q_and_a.jsonl"
    
    def construct_requests(self, doc, ctx, **kwargs):
        """Override to handle generation parameters at task level."""
        return Instance(
            request_type="generate_until",
            doc=doc,
            arguments=(ctx, {
                'until': ['\n', '.', ' '],  # Stop at newline, period, or space
                'max_new_tokens': 1,
                'temperature': 0.0,
                'do_sample': False
            }),
            idx=0,
            metadata=kwargs.get('metadata')
        )

def evaluate_model(model_id: str, model_name: str, num_questions: int = 50) -> Dict[str, Any]:
    """
    Evaluate a specific model on the Islamic Knowledge task.
    """
    try:
        logger.info(f"Evaluating {model_name}...")

        device = "cuda" if torch.cuda.is_available() else "cpu"
        logger.info(f"Using device: {device}")

        # Basic model configuration
        model_args = {
            "pretrained": model_id,
            "tokenizer": model_id,
            "device": device,
            "trust_remote_code": True
        }

        # Initialize evaluator with modified settings
        results = evaluator.simple_evaluate(
            model="hf-auto",  # Use standard HuggingFace auto model
            model_args=model_args,
            tasks=["islamic_knowledge"],
            num_fewshot=2,
            limit=num_questions,
            batch_size=1
        )

        return {
            "model": model_name,
            "provider": "Hugging Face",
            "results": results
        }

    except Exception as e:
        logger.error(f"Error evaluating {model_name}: {str(e)}")
        return {
            "model": model_name,
            "provider": "Hugging Face",
            "error": str(e)
        }

def save_results(results: List[Dict[str, Any]]) -> str:
    """Save evaluation results to a JSON file with timestamp."""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"model_evaluation_{timestamp}.json"

    try:
        with open(filename, 'w') as f:
            json.dump(results, f, indent=2)
        logger.info(f"Results saved to {filename}")
        return filename
    except Exception as e:
        logger.error(f"Error saving results: {str(e)}")
        raise

def main():
    """Main execution function."""
    try:
        load_dotenv()

        # Register the modified task
        lm_eval.tasks.TASK_REGISTRY["islamic_knowledge"] = IslamicKnowledgeTaskModified

        # Model configuration
        model_config = {
            "id": "deepseek-ai/DeepSeek-R1-Distill-Qwen-32B",
            "name": "DeepSeek R1 Distill Qwen 32B"
        }

        # Evaluate model
        results = evaluate_model(model_config["id"], model_config["name"])
        all_results = [results]

        # Print results
        print("\nEvaluation Results:")
        print("=" * 50)

        try:
            metrics = results.get("results", {}).get("islamic_knowledge", {})
            accuracy = metrics.get("accuracy", 0.0)
            print(f"{model_config['name']}: {accuracy:.2%}")
        except (KeyError, AttributeError) as e:
            print(f"Could not extract accuracy for {model_config['name']}: {str(e)}")

        # Save results
        save_results(all_results)

    except Exception as e:
        logger.error(f"Fatal error in main execution: {str(e)}")
        raise

if __name__ == "__main__":
    main()