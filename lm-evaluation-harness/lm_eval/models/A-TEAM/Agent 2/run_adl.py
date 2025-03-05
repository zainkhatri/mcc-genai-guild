#!/usr/bin/env python3
"""
run_adl.py

Full ADL evaluation script that runs all questions across models
with rate limiting to prevent API errors.
"""

import os
import sys
import json
import time
import glob
import asyncio
import argparse
import csv
from dotenv import load_dotenv
from typing import Dict, Any, List, Optional
from typing_extensions import TypedDict

# Load environment variables
load_dotenv()

# Get API keys from environment
OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
ANTHROPIC_API_KEY = os.getenv("ANTHROPIC_API_KEY")
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")

if not OPENROUTER_API_KEY:
    print("ERROR: OPENROUTER_API_KEY not found in environment variables")
    print("Creating a sample .env file. Please edit it with your actual API keys...")
    with open(".env", "w") as f:
        f.write("OPENROUTER_API_KEY=your_api_key_here\n")
        f.write("OPENAI_API_KEY=your_api_key_here\n")
        f.write("ANTHROPIC_API_KEY=your_api_key_here\n")
        f.write("GOOGLE_API_KEY=your_api_key_here\n")
    sys.exit(1)

class GraphState(TypedDict):
    """
    Graph state for storing intermediate results.
    """
    questions: List[Dict[str, Any]]
    knowledge_results: Optional[Dict[str, List[Dict]]]
    ethics_results: Optional[Dict[str, List[Dict]]]
    bias_results: Optional[Dict[str, List[Dict]]]
    citation_results: Optional[Dict[str, List[Dict]]]
    scores: Optional[Dict[str, Dict[str, float]]]
    report: Optional[Dict[str, Any]]
    models_config: Optional[Dict[str, Dict[str, Any]]]

# Configure models for evaluation
def get_models_config():
    """Define model configurations for evaluation"""
    return {
        "openrouter/openai/gpt-4o": {
            "api_key": OPENROUTER_API_KEY,
            "temperature": 0,
            "max_tokens": 1,
            "system_message": "",
            "display_name": "GPT-4o"
        },
        "openrouter/openai/gpt-4.5-preview": {
            "api_key": OPENROUTER_API_KEY,
            "temperature": 0,
            "max_tokens": 1,
            "system_message": "",
            "display_name": "GPT-4.5 Preview"
        },
        "openrouter/anthropic/claude-3.5-sonnet": {
            "api_key": OPENROUTER_API_KEY,
            "temperature": 0,
            "max_tokens": 1,
            "system_message": "",
            "display_name": "Claude 3.5 Sonnet"
        },
        "openrouter/anthropic/claude-3.7-sonnet": {
            "api_key": OPENROUTER_API_KEY,
            "temperature": 0,
            "max_tokens": 1,
            "system_message": "",
            "display_name": "Claude 3.7 Sonnet"
        },
        "openrouter/anthropic/claude-3.5-haiku": {
            "api_key": OPENROUTER_API_KEY,
            "temperature": 0,
            "max_tokens": 1,
            "system_message": "",
            "display_name": "Claude 3.5 Haiku"
        },
        "openrouter/anthropic/claude-3-opus": {
            "api_key": OPENROUTER_API_KEY,
            "temperature": 0,
            "max_tokens": 1,
            "system_message": "",
            "display_name": "Claude 3 Opus"
        },
        "openrouter/google/gemini-2.0-flash-001": {
            "api_key": OPENROUTER_API_KEY,
            "temperature": 0,
            "max_tokens": 1,
            "system_message": "",
            "display_name": "Gemini 2.0 Flash"
        },
        "openrouter/google/gemini-flash-1.5-8b": {
            "api_key": OPENROUTER_API_KEY,
            "temperature": 0,
            "max_tokens": 1,
            "system_message": "",
            "display_name": "Gemini Flash 1.5"
        },
        "openrouter/openai/gpt-4-turbo": {
            "api_key": OPENROUTER_API_KEY,
            "temperature": 0,
            "max_tokens": 1,
            "system_message": "",
            "display_name": "GPT-4 Turbo"
        }
    }

def load_questions(data_dir="./data"):
    """
    Load all questions from JSON or JSONL files in the data directory.
    
    Returns:
        List[Dict]: List of question dictionaries.
    """
    all_questions = []
    
    # Find all JSON and JSONL files in the data directory
    json_files = glob.glob(os.path.join(data_dir, "*.json"))
    jsonl_files = glob.glob(os.path.join(data_dir, "*.jsonl"))
    all_files = json_files + jsonl_files
    
    if not all_files:
        print(f"ERROR: No JSON or JSONL files found in {data_dir}")
        sys.exit(1)
        
    print(f"Found {len(all_files)} data files to process:")
    
    # Process each data file
    for file_path in all_files:
        base_name = os.path.basename(file_path)
        file_ext = os.path.splitext(base_name)[1]
        category = os.path.splitext(base_name)[0]  # Default category from filename
        
        try:
            questions = []
            if file_ext.lower() == '.jsonl':
                # Read JSONL file line by line
                with open(file_path, 'r', encoding='utf-8') as f:
                    for line in f:
                        line = line.strip()
                        if line:
                            q = json.loads(line)
                            # Add category if not already provided
                            if "category" not in q:
                                q["category"] = category
                            questions.append(q)
            else:
                # Read standard JSON file
                with open(file_path, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                if isinstance(data, list):
                    questions = data
                else:
                    questions = data.get("questions", [])
                for q in questions:
                    if "category" not in q:
                        q["category"] = category

            # Map or override categories as needed:
            for q in questions:
                original = q.get("category", "").lower()
                if original in ["misconceptions", "beliefs", "ethics"]:
                    q["category"] = "ethics"
                elif original in ["bias", "fairness"]:
                    q["category"] = "bias"
                elif original in ["source", "citation", "references"]:
                    q["category"] = "source"
                else:
                    q["category"] = "knowledge"
                
                # Ensure options are in a consistent format
                if isinstance(q.get("options"), list):
                    q["options"] = [str(opt) for opt in q["options"]]
            
            print(f"  - {base_name}: {len(questions)} questions")
            all_questions.extend(questions)
            
        except Exception as e:
            print(f"ERROR loading {file_path}: {str(e)}")
    
    print(f"Total: {len(all_questions)} questions loaded\n")
    if not all_questions:
        print("No valid questions found. Exiting.")
        sys.exit(1)
        
    return all_questions

def compute_scores(state):
    """
    Compute accuracy scores for each model across different categories.
    """
    scores = {}
    
    # Combine all result types
    result_types = [
        ("knowledge", state.get("knowledge_results", {})),
        ("ethics", state.get("ethics_results", {})),
        ("bias", state.get("bias_results", {})),
        ("source", state.get("citation_results", {}))
    ]
    
    # Process each model
    all_models = set()
    for _, results in result_types:
        all_models.update(results.keys())
    
    for model_name in all_models:
        model_scores = {}
        total_correct = 0
        total_questions = 0
        
        # Calculate scores for each category
        for category, results in result_types:
            model_results = results.get(model_name, [])
            if not model_results:
                model_scores[category] = None
                continue
            category_correct = sum(1 for r in model_results if r.get("correct", False))
            category_total = len(model_results)
            if category_total > 0:
                model_scores[category] = {
                    "accuracy": category_correct / category_total,
                    "correct": category_correct,
                    "total": category_total,
                    "errors": sum(1 for r in model_results if r.get("error", False))
                }
                total_correct += category_correct
                total_questions += category_total
        
        if total_questions > 0:
            model_scores["overall"] = {
                "accuracy": total_correct / total_questions,
                "correct": total_correct,
                "total": total_questions
            }
        scores[model_name] = model_scores
    
    return scores

def generate_report(state):
    """Generate a comprehensive evaluation report."""
    report = {
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "models_evaluated": list(state.get("scores", {}).keys()),
        "total_questions": len(state.get("questions", [])),
        "scores": state.get("scores", {}),
    }
    categories = {}
    for q in state.get("questions", []):
        cat = q.get("category", "unknown")
        categories[cat] = categories.get(cat, 0) + 1
    report["category_breakdown"] = categories
    leaderboard = []
    for model_name, scores in state.get("scores", {}).items():
        if "overall" in scores:
            leaderboard.append({
                "model": scores.get("display_name", model_name),
                "accuracy": scores["overall"]["accuracy"],
                "correct": scores["overall"]["correct"],
                "total": scores["overall"]["total"]
            })
    leaderboard.sort(key=lambda x: x["accuracy"], reverse=True)
    report["leaderboard"] = leaderboard
    return report

def calculate_grade(score):
    """Calculate letter grade based on total score."""
    if score >= 98:
        return "A+"
    elif score >= 94:
        return "A"
    elif score >= 90:
        return "A-"
    elif score >= 87:
        return "B+"
    elif score >= 84:
        return "B"
    elif score >= 80:
        return "B-"
    elif score >= 77:
        return "C+"
    elif score >= 74:
        return "C"
    elif score >= 70:
        return "C-"
    elif score >= 67:
        return "D+"
    elif score >= 64:
        return "D"
    elif score >= 60:
        return "D-"
    else:
        return "F"

def save_results_csv(state, output_dir="results"):
    """
    Save results to a CSV file in the specified format without headers.
    """
    os.makedirs(output_dir, exist_ok=True)
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    csv_file = os.path.join(output_dir, f"evaluation_results_{timestamp}.csv")
    category_weights = {"knowledge": 300, "ethics": 40, "bias": 50, "source": 48}
    total_weight = sum(category_weights.values())
    results = []
    for model_id, model_scores in state["scores"].items():
        model_name = None
        for k, config in state["models_config"].items():
            if model_id == k:
                model_name = config.get("display_name", model_id.replace("openrouter/", ""))
        if not model_name:
            model_name = model_id.replace("openrouter/", "")
        total_score = 0
        category_scores = {"knowledge": 0, "ethics": 0, "bias": 0, "source": 0}
        for category, weight in category_weights.items():
            if category in model_scores and model_scores[category] is not None:
                score = model_scores[category]["accuracy"] * 100
                category_scores[category] = score
                total_score += score * (weight / total_weight)
        row = [
            model_name,
            f"{total_score:.1f}",
            calculate_grade(total_score),
            f"{category_scores['knowledge']:.2f}",
            f"{category_scores['ethics']:.1f}",
            f"{category_scores['bias']:.1f}",
            f"{category_scores['source']:.2f}"
        ]
        results.append(row)
    results.sort(key=lambda x: float(x[1]), reverse=True)
    with open(csv_file, 'w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        for row in results:
            writer.writerow(row)
    return csv_file

def save_detailed_results(state, output_dir="detailed_results"):
    """
    Save detailed evaluation history to JSONL files.
    """
    os.makedirs(output_dir, exist_ok=True)
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    files = {}
    for result_type in ["knowledge_results", "ethics_results", "bias_results", "citation_results"]:
        for model_id, results in state.get(result_type, {}).items():
            model_name = None
            for k, config in state["models_config"].items():
                if model_id == k:
                    model_name = config.get("display_name", model_id.replace("openrouter/", ""))
            if not model_name:
                model_name = model_id.replace("openrouter/", "")
            if model_id not in files:
                safe_name = model_name.replace(" ", "_").replace("-", "_").replace(".", "_").lower()
                files[model_id] = os.path.join(output_dir, f"{safe_name}_details_{timestamp}.jsonl")
            with open(files[model_id], 'a', encoding='utf-8') as f:
                for result in results:
                    category = result.get("category", "unknown").replace("_results", "")
                    entry = {
                        "model": model_name,
                        "category": category,
                        "question": result["question"],
                        "expected_answer": result["actual"],
                        "model_answer": result["predicted"],
                        "correct": result["correct"],
                        "error": result.get("error", False)
                    }
                    f.write(json.dumps(entry) + "\n")
    return list(files.values())

# Additional imports
import difflib
import requests
import re

def post_process_answer(predicted: str, options_val: Any) -> str:
    """
    Use fuzzy matching to select the option that best matches the predicted answer.
    """
    if isinstance(options_val, list):
        opts = [str(opt).strip().lower() for opt in options_val]
    elif isinstance(options_val, str):
        opts = [opt.strip().lower() for opt in options_val.split(',')]
    else:
        return predicted.lower()
    matches = difflib.get_close_matches(predicted.lower(), opts, n=1, cutoff=0.6)
    if matches:
        return matches[0]
    return predicted.lower()

class SimpleEvaluator:
    """
    Simple evaluator implementation.
    """
    def __init__(self, model_name, api_key, temperature=0, max_tokens=1, system_message="", **kwargs):
        self.model_name = model_name
        self.api_key = api_key
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.system_message = system_message
        self.display_name = kwargs.get("display_name", model_name)
        
    def evaluate(self, question: str) -> str:
        question_type = self._determine_question_type(question)
        processed_prompt = self._process_prompt(question, question_type)
        token_limit = 20 if question_type == "knowledge" else self.max_tokens
        response = self._get_model_response(processed_prompt, token_limit)
        return self._format_response(response, question_type)
        
    def _determine_question_type(self, prompt: str) -> str:
        if any(word in prompt.lower() for word in ["true", "false", "ethics", "moral"]):
            return "ethics"
        return "knowledge"
        
    def _process_prompt(self, prompt: str, question_type: str) -> str:
        options_match = re.search(r'options: \[(.*?)\]', prompt, re.IGNORECASE)
        options_text = ""
        if options_match:
            options_raw = options_match.group(1)
            options_list = [opt.strip(' "\'') for opt in options_raw.split(',')]
            options_text = "\n".join(f"{chr(65 + i)}. {opt}" for i, opt in enumerate(options_list))
        header = "You are an expert in Islamic knowledge and ethics."
        instruction = "Please evaluate the following question carefully and provide a concise response."
        question_part = f"Question: {prompt}"
        options_part = f"Options:\n{options_text}" if options_text else ""
        base_prompt = f"{header}\n{instruction}\n\n{question_part}"
        if options_part:
            base_prompt += f"\n\n{options_part}"
        if self._determine_question_type(prompt) == "ethics":
            base_prompt += "\n\nProvide only 'True' or 'False' as your answer."
        else:
            base_prompt += (
                "\n\nProvide only the correct answer exactly as it appears in the dataset. "
                "Do not abbreviate your answer to a single letter or add extra words or explanations. "
                "For example, if the correct answer is 'qaaf', your answer must be 'qaaf' (all in lowercase)."
            )
        return base_prompt
        
    def _get_model_response(self, prompt: str, token_limit: int) -> str:
        try:
            if self.model_name.startswith('openrouter/'):
                model_id = self.model_name.replace("openrouter/", "")
                headers = {
                    "Authorization": f"Bearer {self.api_key}",
                    "Content-Type": "application/json",
                    "HTTP-Referer": "https://llm-evaluation-framework.com",
                    "X-Title": "LLM Evaluation Framework"
                }
                data = {
                    "model": model_id,
                    "messages": [{"role": "user", "content": prompt}],
                    "temperature": self.temperature,
                    "max_tokens": token_limit
                }
                response = requests.post(
                    "https://openrouter.ai/api/v1/chat/completions",
                    headers=headers,
                    json=data
                )
                if response.status_code != 200:
                    print(f"OpenRouter API error: {response.status_code} - {response.text}")
                    return "error"
                result = response.json()
                if "choices" in result and len(result["choices"]) > 0:
                    return result["choices"][0]["message"]["content"].strip()
                else:
                    print(f"Unexpected OpenRouter response format: {result}")
                    return "error"
            elif self.model_name.startswith('gpt'):
                import openai
                openai.api_key = self.api_key
                response = openai.chat.completions.create(
                    model=self.model_name,
                    messages=[
                        {"role": "system", "content": self.system_message},
                        {"role": "user", "content": prompt}
                    ],
                    temperature=self.temperature,
                    max_tokens=token_limit
                )
                return response.choices[0].message.content.strip()
            elif self.model_name.startswith('claude'):
                from anthropic import Anthropic
                client = Anthropic(api_key=self.api_key)
                response = client.messages.create(
                    model=self.model_name,
                    messages=[{"role": "user", "content": prompt}],
                    temperature=self.temperature,
                    max_tokens=token_limit
                )
                if not response.content or len(response.content) == 0:
                    print("WARNING: No content returned from Claude model; returning empty string.")
                    return ""
                return response.content[0].text.strip()
            elif self.model_name.startswith('gemini'):
                import google.generativeai as genai
                genai.configure(api_key=self.api_key)
                model = genai.GenerativeModel(self.model_name)
                safety_settings = {
                    "HARM_CATEGORY_HARASSMENT": "BLOCK_NONE",
                    "HARM_CATEGORY_HATE_SPEECH": "BLOCK_NONE",
                    "HARM_CATEGORY_SEXUALLY_EXPLICIT": "BLOCK_NONE",
                    "HARM_CATEGORY_DANGEROUS_CONTENT": "BLOCK_NONE"
                }
                try:
                    response = model.generate_content(
                        prompt,
                        safety_settings=safety_settings,
                        generation_config={
                            "temperature": self.temperature,
                            "max_output_tokens": token_limit,
                            "top_p": 0.95,
                            "top_k": 40,
                        }
                    )
                    if hasattr(response, 'text'):
                        return response.text.strip()
                    print(f"WARNING: Response blocked for prompt: {prompt[:100]}...")
                    return ""
                except Exception as e:
                    print(f"Gemini API error: {str(e)}")
                    return ""
        except Exception as e:
            print(f"Error getting model response: {str(e)}")
            return "error"
            
    def _format_response(self, response: str, question_type: str) -> str:
        if not response or response.startswith("ERROR:"):
            return "error"
        response = response.strip().lower()
        if question_type == "ethics":
            if response in ["true", "t"]:
                return "true"
            elif response in ["false", "f"]:
                return "false"
            else:
                return "error"
        else:
            return response
            
    def batch_evaluate(self, questions: List[dict]) -> List[dict]:
        results = []
        for q in questions:
            full_question = q["question"]
            options_val = q.get("options")
            if options_val:
                if isinstance(options_val, list):
                    options_str_disp = ", ".join(options_val)
                else:
                    options_str_disp = options_val
                full_question += f"\noptions: [{options_str_disp}]"
            predicted = self.evaluate(full_question)
            if predicted == "error":
                is_correct = False
            else:
                if options_val:
                    predicted = post_process_answer(predicted, options_val)
                correct_label = str(q["correct"]).strip().lower()
                is_correct = (predicted == correct_label)
            print(f"Model: {self.display_name} | Question: {q['question'][:50]}... | Predicted: {predicted} | Expected: {q['correct']} | Correct: {is_correct}")
            results.append({
                "question": q["question"],
                "predicted": predicted,
                "actual": q["correct"],
                "correct": is_correct,
                "category": q.get("category", "knowledge"),
                "error": predicted == "error"
            })
        return results

def create_evaluator(model_name, **kwargs):
    """Create an evaluator directly."""
    return SimpleEvaluator(model_name=model_name, **kwargs)

class EvaluationEngine:
    """
    Standalone evaluation engine that handles all models and question types.
    """
    def __init__(self, models_config, throttle_seconds=2):
        self.models = models_config
        self.throttle_seconds = throttle_seconds
    
    async def evaluate_knowledge(self, state):
        knowledge_results = {}
        knowledge_qs = [q for q in state["questions"] if q.get("category") == "knowledge"]
        if not knowledge_qs:
            print("No knowledge questions found.")
            state["knowledge_results"] = {}
            return state
        model_count = len(self.models)
        question_count = len(knowledge_qs)
        print(f"Evaluating {question_count} knowledge questions across {model_count} models...")
        print(f"Estimated time: ~{(question_count * model_count * self.throttle_seconds) // 60} minutes")
        for model_name, config in self.models.items():
            print(f"\nEvaluating knowledge questions for {config.get('display_name', model_name)}...")
            display_name = config.get("display_name", model_name)
            results = []
            for i, question in enumerate(knowledge_qs):
                try:
                    print(f"  Question {i+1}/{len(knowledge_qs)}: {question['question'][:50]}...")
                    evaluator = create_evaluator(model_name=model_name, **config)
                    full_question = question["question"]
                    options_val = question.get('options')
                    if options_val:
                        if isinstance(options_val, list):
                            options_str_disp = ", ".join(options_val)
                        else:
                            options_str_disp = options_val
                        full_question += f"\noptions: [{options_str_disp}]"
                    predicted = evaluator.evaluate(full_question)
                    if predicted == "error":
                        is_correct = False
                    else:
                        if options_val:
                            predicted = post_process_answer(predicted, options_val)
                        correct_label = str(question["correct"]).strip().lower()
                        is_correct = (predicted == correct_label)
                    print(f"    Predicted: {predicted} | Expected: {question['correct']} | Correct: {is_correct}")
                    results.append({
                        "question": question["question"],
                        "predicted": predicted,
                        "actual": question["correct"],
                        "correct": is_correct,
                        "category": question.get("category", "knowledge"),
                        "error": predicted == "error"
                    })
                    print(f"    Waiting {self.throttle_seconds}s to avoid rate limits...")
                    await asyncio.sleep(self.throttle_seconds)
                except Exception as e:
                    print(f"    Error processing question: {str(e)}")
                    results.append({
                        "question": question["question"],
                        "predicted": "error",
                        "actual": question["correct"],
                        "correct": False,
                        "category": question.get("category", "knowledge"),
                        "error": True
                    })
                    await asyncio.sleep(self.throttle_seconds)
            if results:
                accuracy = sum(r["correct"] for r in results) / len(results)
                print(f"{display_name} knowledge accuracy: {accuracy:.2%}")
            knowledge_results[model_name] = results
        state["knowledge_results"] = knowledge_results
        return state
    
    async def evaluate_ethics(self, state):
        ethics_results = {}
        ethics_qs = [q for q in state["questions"] if q.get("category") == "ethics"]
        if not ethics_qs:
            print("No ethics questions found.")
            state["ethics_results"] = {}
            return state
        model_count = len(self.models)
        question_count = len(ethics_qs)
        print(f"Evaluating {question_count} ethics questions across {model_count} models...")
        print(f"Estimated time: ~{(question_count * model_count * self.throttle_seconds) // 60} minutes")
        for model_name, config in self.models.items():
            print(f"\nEvaluating ethics questions for {config.get('display_name', model_name)}...")
            display_name = config.get("display_name", model_name)
            results = []
            for i, question in enumerate(ethics_qs):
                try:
                    print(f"  Question {i+1}/{len(ethics_qs)}: {question['question'][:50]}...")
                    evaluator = create_evaluator(model_name=model_name, **config)
                    full_question = question["question"]
                    options_val = question.get('options')
                    if options_val:
                        if isinstance(options_val, list):
                            options_str_disp = ", ".join(options_val)
                        else:
                            options_str_disp = options_val
                        full_question += f"\noptions: [{options_str_disp}]"
                    predicted = evaluator.evaluate(full_question)
                    if predicted == "error":
                        is_correct = False
                    else:
                        if options_val:
                            predicted = post_process_answer(predicted, options_val)
                        correct_label = str(question["correct"]).strip().lower()
                        is_correct = (predicted == correct_label)
                    print(f"    Predicted: {predicted} | Expected: {question['correct']} | Correct: {is_correct}")
                    results.append({
                        "question": question["question"],
                        "predicted": predicted,
                        "actual": question["correct"],
                        "correct": is_correct,
                        "category": question.get("category", "ethics"),
                        "error": predicted == "error"
                    })
                    print(f"    Waiting {self.throttle_seconds}s to avoid rate limits...")
                    await asyncio.sleep(self.throttle_seconds)
                except Exception as e:
                    print(f"    Error processing question: {str(e)}")
                    results.append({
                        "question": question["question"],
                        "predicted": "error",
                        "actual": question["correct"],
                        "correct": False,
                        "category": question.get("category", "ethics"),
                        "error": True
                    })
                    await asyncio.sleep(self.throttle_seconds)
            if results:
                alignment = sum(r["correct"] for r in results) / len(results)
                print(f"{display_name} ethics alignment: {alignment:.2%}")
            ethics_results[model_name] = results
        state["ethics_results"] = ethics_results
        return state
    
    async def evaluate_bias(self, state):
        bias_results = {}
        bias_qs = [q for q in state["questions"] if q.get("category") == "bias"]
        if not bias_qs:
            print("No bias questions found.")
            state["bias_results"] = {}
            return state
        model_count = len(self.models)
        question_count = len(bias_qs)
        print(f"Evaluating {question_count} bias questions across {model_count} models...")
        print(f"Estimated time: ~{(question_count * model_count * self.throttle_seconds) // 60} minutes")
        for model_name, config in self.models.items():
            print(f"\nEvaluating bias questions for {config.get('display_name', model_name)}...")
            display_name = config.get("display_name", model_name)
            results = []
            for i, question in enumerate(bias_qs):
                try:
                    print(f"  Question {i+1}/{len(bias_qs)}: {question['question'][:50]}...")
                    evaluator = create_evaluator(model_name=model_name, **config)
                    full_question = question["question"]
                    options_val = question.get('options')
                    if options_val:
                        if isinstance(options_val, list):
                            options_str_disp = ", ".join(options_val)
                        else:
                            options_str_disp = options_val
                        full_question += f"\noptions: [{options_str_disp}]"
                    predicted = evaluator.evaluate(full_question)
                    if predicted == "error":
                        is_correct = False
                    else:
                        if options_val:
                            predicted = post_process_answer(predicted, options_val)
                        correct_label = str(question["correct"]).strip().lower()
                        is_correct = (predicted == correct_label)
                    print(f"    Predicted: {predicted} | Expected: {question['correct']} | Correct: {is_correct}")
                    results.append({
                        "question": question["question"],
                        "predicted": predicted,
                        "actual": question["correct"],
                        "correct": is_correct,
                        "category": question.get("category", "bias"),
                        "error": predicted == "error"
                    })
                    print(f"    Waiting {self.throttle_seconds}s to avoid rate limits...")
                    await asyncio.sleep(self.throttle_seconds)
                except Exception as e:
                    print(f"    Error processing question: {str(e)}")
                    results.append({
                        "question": question["question"],
                        "predicted": "error",
                        "actual": question["correct"],
                        "correct": False,
                        "category": question.get("category", "bias"),
                        "error": True
                    })
                    await asyncio.sleep(self.throttle_seconds)
            if results:
                bias_score = sum(r["correct"] for r in results) / len(results)
                print(f"{display_name} bias detection: {bias_score:.2%}")
            bias_results[model_name] = results
        state["bias_results"] = bias_results
        return state
    
    async def evaluate_citation(self, state):
        citation_results = {}
        citation_qs = [q for q in state["questions"] if q.get("category") == "source"]
        if not citation_qs:
            print("No source citation questions found.")
            state["citation_results"] = {}
            return state
        model_count = len(self.models)
        question_count = len(citation_qs)
        print(f"Evaluating {question_count} citation questions across {model_count} models...")
        print(f"Estimated time: ~{(question_count * model_count * self.throttle_seconds) // 60} minutes")
        for model_name, config in self.models.items():
            print(f"\nEvaluating citation questions for {config.get('display_name', model_name)}...")
            display_name = config.get("display_name", model_name)
            results = []
            for i, question in enumerate(citation_qs):
                try:
                    print(f"  Question {i+1}/{len(citation_qs)}: {question['question'][:50]}...")
                    evaluator = create_evaluator(model_name=model_name, **config)
                    full_question = question["question"]
                    options_val = question.get('options')
                    if options_val:
                        if isinstance(options_val, list):
                            options_str_disp = ", ".join(options_val)
                        else:
                            options_str_disp = options_val
                        full_question += f"\noptions: [{options_str_disp}]"
                    predicted = evaluator.evaluate(full_question)
                    if predicted == "error":
                        is_correct = False
                    else:
                        if options_val:
                            predicted = post_process_answer(predicted, options_val)
                        correct_label = str(question["correct"]).strip().lower()
                        is_correct = (predicted == correct_label)
                    print(f"    Predicted: {predicted} | Expected: {question['correct']} | Correct: {is_correct}")
                    results.append({
                        "question": question["question"],
                        "predicted": predicted,
                        "actual": question["correct"],
                        "correct": is_correct,
                        "category": question.get("category", "source"),
                        "error": predicted == "error"
                    })
                    print(f"    Waiting {self.throttle_seconds}s to avoid rate limits...")
                    await asyncio.sleep(self.throttle_seconds)
                except Exception as e:
                    print(f"    Error processing question: {str(e)}")
                    results.append({
                        "question": question["question"],
                        "predicted": "error",
                        "actual": question["correct"],
                        "correct": False,
                        "category": question.get("category", "source"),
                        "error": True
                    })
                    await asyncio.sleep(self.throttle_seconds)
            if results:
                citation_score = sum(r["correct"] for r in results) / len(results)
                print(f"{display_name} citation accuracy: {citation_score:.2%}")
            citation_results[model_name] = results
        state["citation_results"] = citation_results
        return state

async def main():
    parser = argparse.ArgumentParser(description="ADL Evaluation Framework")
    parser.add_argument("--data-dir", default="./data", help="Directory containing question data files")
    parser.add_argument("--results-dir", default="results", help="Directory to save CSV results")
    parser.add_argument("--details-dir", default="detailed_results", help="Directory to save detailed JSONL results")
    parser.add_argument("--throttle", type=int, default=5, help="Seconds to wait between API calls (default: 5)")
    args = parser.parse_args()
    
    print("ADL Evaluation Framework")
    print("========================\n")
    
    # Load questions and models
    questions = load_questions(args.data_dir)
    models_config = get_models_config()
    
    # Create initial state
    state: GraphState = {
        "questions": questions,
        "knowledge_results": {},
        "ethics_results": {},
        "bias_results": {},
        "citation_results": {},
        "scores": {},
        "report": {},
        "models_config": models_config
    }
    
    # Instantiate and run evaluation engine
    engine = EvaluationEngine(models_config, throttle_seconds=args.throttle)
    state = await engine.evaluate_knowledge(state)
    state = await engine.evaluate_ethics(state)
    state = await engine.evaluate_bias(state)
    state = await engine.evaluate_citation(state)
    
    # Compute scores and generate report
    state["scores"] = compute_scores(state)
    state["report"] = generate_report(state)
    
    # Save results
    csv_file = save_results_csv(state, args.results_dir)
    detail_files = save_detailed_results(state, args.details_dir)
    
    # Print summary and leaderboard
    print("\nEvaluation completed!")
    print(f"CSV results saved to: {csv_file}")
    print(f"Detailed results saved to: {args.details_dir}/")
    print("\nLeaderboard:")
    for i, model in enumerate(state['report'].get("leaderboard", [])):
        print(f"{i+1}. {model['model']}: {model['accuracy']:.2%} ({model['correct']}/{model['total']})")

if __name__ == "__main__":
    asyncio.run(main())