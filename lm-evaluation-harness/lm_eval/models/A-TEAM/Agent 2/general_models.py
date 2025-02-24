import os
import json
import asyncio
from datetime import datetime
from typing import List, Dict
from dotenv import load_dotenv
from adl_graph import ADLGraph

def load_questions(knowledge_path: str, ethics_path: str) -> List[Dict]:
    questions = []
    # Load knowledge questions
    if os.path.exists(knowledge_path):
        with open(knowledge_path, 'r') as f:
            all_knowledge = [json.loads(line) for line in f]
            for q in all_knowledge[:300]:  # limit to 5 for testing
                q['category'] = 'knowledge'
                questions.append(q)
    # Load ethics questions
    if os.path.exists(ethics_path):
        with open(ethics_path, 'r') as f:
            all_ethics = [json.loads(line) for line in f]
            for q in all_ethics[:40]:  # limit to 5 for testing
                q['category'] = 'ethics'
                questions.append(q)
    return questions

def get_base_configs():
    """Get base configurations for different model providers"""
    # Base Gemini config
    gemini_base = {
        "api_key": os.getenv("GOOGLE_API_KEY"),
        "temperature": 0,
        "generation_config": {
            "temperature": 0,
            "top_p": 0.95,
            "top_k": 40,
            "max_output_tokens": 1024,
        },
        "system_message": "You are an Islamic knowledge evaluator. Provide concise, accurate answers."
    }

    # Base OpenAI config
    openai_base = {
        "api_key": os.getenv("OPENAI_API_KEY"),
        "temperature": 0,
        "max_tokens": 1,
        "system_message": "You are an Islamic knowledge evaluator. Provide concise, accurate answers."
    }

    # Base Anthropic config
    anthropic_base = {
        "api_key": os.getenv("ANTHROPIC_API_KEY"),
        "temperature": 0,
        "max_tokens": 1,
        "system_message": "You are an Islamic knowledge evaluator. Provide concise, accurate answers."
    }

    return gemini_base, openai_base, anthropic_base

def get_models_config(eval_mode: str = "all") -> Dict:
    """
    Get model configurations based on evaluation mode.
    
    Args:
        eval_mode (str): Either "all" for all providers or "gemini" for Gemini-only
    """
    gemini_base, openai_base, anthropic_base = get_base_configs()
    
    if eval_mode == "gemini":
        return {
            "gemini-pro": gemini_base.copy(),
            "gemini-2.0-flash": gemini_base.copy(),
        }
    
    # Full configuration for all providers
    return {
        
        # OpenAI models
        "gpt-4-0125-preview": openai_base.copy(),
        "gpt-4-turbo-preview": openai_base.copy(),
        "gpt-4": openai_base.copy(),
        "gpt-4o-2024-11-20": openai_base.copy()
    }

async def evaluate_models_in_batches(adl: ADLGraph, questions: List[Dict], models_config: Dict) -> Dict:
    """Evaluates models in separate batches by provider"""
    all_results = {
        "models_evaluated": [],
        "scores": {},
        "detailed_results": {
            "knowledge": {},
            "ethics": {}
        },
        "timestamp": datetime.now().isoformat()
    }

    # Group models by provider
    provider_groups = {
        "anthropic": {},
        "openai": {},
        "google": {}
    }

    for model_name, config in models_config.items():
        if model_name.startswith("claude"):
            provider_groups["anthropic"][model_name] = config
        elif model_name.startswith("gpt"):
            provider_groups["openai"][model_name] = config
        elif model_name.startswith("gemini"):
            provider_groups["google"][model_name] = config

    # Evaluate each provider group
    for provider, models in provider_groups.items():
        if not models:
            continue
            
        print(f"\nEvaluating {provider.upper()} models...")
        try:
            provider_adl = ADLGraph(models)
            results = await provider_adl.run_evaluation(questions)
            
            # Merge results
            all_results["models_evaluated"].extend(results.get("models_evaluated", []))
            all_results["scores"].update(results.get("scores", {}))
            for result_type in ["knowledge", "ethics"]:
                all_results["detailed_results"][result_type].update(
                    results.get("detailed_results", {}).get(result_type, {})
                )
                
        except Exception as e:
            print(f"Error evaluating {provider} models: {str(e)}")
            all_results["errors"] = all_results.get("errors", {})
            all_results["errors"][provider] = str(e)

    return all_results

def save_reports(report: Dict, prefix: str = "evaluation"):
    """Save both detailed and summary reports"""
    os.makedirs("results", exist_ok=True)
    timestamp = datetime.now().isoformat()
    
    # Save detailed report
    filename = f"{prefix}_{timestamp}.json"
    report_path = os.path.join("results", filename)
    with open(report_path, "w") as f:
        json.dump(report, f, indent=2)
    print(f"\nDetailed report saved to: {report_path}")
    
    # Save summary report
    summary_filename = f"{prefix}_summary_{timestamp}.json"
    summary_path = os.path.join("results", summary_filename)
    summary_report = {
        "timestamp": timestamp,
        "models_evaluated": list(report["scores"].keys()),
        "summary_metrics": report["scores"]
    }
    with open(summary_path, "w") as f:
        json.dump(summary_report, f, indent=2)
    print(f"Summary report saved to: {summary_path}")

def print_results(report: Dict):
    """Print formatted evaluation results"""
    print("\nResults Summary:")
    print("=" * 50)

    if report["scores"]:
        for model_name, scores in report["scores"].items():
            print(f"\n{model_name}:")
            print(f"Knowledge Accuracy: {scores['knowledge_accuracy']:.2%}")
            print(f"Ethics Accuracy: {scores['ethics_accuracy']:.2%}")
            print(f"Timestamp: {scores['timestamp']}")
    
    if "errors" in report:
        print("\nErrors encountered:")
        print("=" * 50)
        for provider, error in report["errors"].items():
            print(f"{provider}: {error}")

async def main():
    load_dotenv()
    
    # Get evaluation mode from environment or default to "all"
    eval_mode = os.getenv("EVAL_MODE", "all").lower()
    
    # Get model configurations
    models_config = get_models_config(eval_mode)
    
    # Load questions
    questions = load_questions("../../../data/q_and_a.jsonl", "../../../data/ethics.jsonl")
    print(f"Loaded {len(questions)} questions")

    # Initialize ADL Graph
    adl = ADLGraph(models_config)

    print(f"\nStarting evaluation in {eval_mode.upper()} mode...")
    try:
        if eval_mode == "gemini":
            report = await adl.run_evaluation(questions)
        else:
            report = await evaluate_models_in_batches(adl, questions, models_config)
        
        print("\nEvaluation complete!")
        print_results(report)
        save_reports(report, prefix=f"{eval_mode}_evaluation")

    except Exception as e:
        print(f"Error during evaluation: {str(e)}")
        raise

if __name__ == "__main__":
    asyncio.run(main())