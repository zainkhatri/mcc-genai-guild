#evaluate_islamic_model.py:

from lm_eval import evaluator
from lm_eval.api.model import LM
from lm_eval.tasks.islamic_knowledge_task import IslamicKnowledgeTask

def main():
    # Register the task
    task_dict = {
        "islamic_knowledge": IslamicKnowledgeTask
    }

    # Configure models
    model_configs = {
        "gpt-4": {
            "model": "gpt-4",
            "model_args": "openai_key=YOUR_API_KEY"
        }
        # Add other models as needed
    }

    # Run evaluation
    results = evaluator.simple_evaluate(
        model="gpt-4",
        tasks=["islamic_knowledge"],
        num_fewshot=0,
        task_dict=task_dict
    )

    print(results)

if __name__ == "__main__":
    main()