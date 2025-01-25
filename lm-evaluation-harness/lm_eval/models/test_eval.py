from lm_eval import evaluator
import os
from dotenv import load_dotenv
import logging
import lm_eval.tasks
from lm_eval.tasks.islamic_knowledge_task import IslamicKnowledgeTask

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def test_single_model():
    # Load environment variables
    load_dotenv()
    
    # Get API key
    openai_key = os.getenv('OPENAI_API_KEY')
    
    if not openai_key:
        logger.error("Error: OPENAI_API_KEY not found in .env file")
        return
    
    try:
        logger.info("\nInitializing evaluation...")

        # Register task
        lm_eval.tasks.TASK_REGISTRY["islamic_knowledge"] = IslamicKnowledgeTask
        
        # Run evaluation - using a model that supports loglikelihoods
        results = evaluator.simple_evaluate(
            model="openai-completions",
            model_args=f"model=davinci-002,api_key={openai_key}",  # Changed to davinci-002
            tasks=["islamic_knowledge"],
            num_fewshot=0,
            limit=5,
            verbosity="INFO"
        )
        
        # Print results
        logger.info("\nTest Results:")
        logger.info("=" * 50)
        if results:
            for task_name, task_results in results['results'].items():
                logger.info(f"\nTask: {task_name}")
                for metric, value in task_results.items():
                    if metric != "samples":  # Skip printing sample count
                        logger.info(f"{metric}: {value}")
        else:
            logger.info("No results returned from evaluation")
            
    except Exception as e:
        logger.error(f"Error during evaluation: {str(e)}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    test_single_model()