"""
adl_graph.py

Modified version of ADLGraph for more efficient evaluation with improved output handling.
Makes minimal changes to the original implementation while ensuring compatibility with
the new clean output format.
"""

import os
import json
import difflib
import requests
from datetime import datetime
import re
from typing import Dict, Any, List, Optional
from typing_extensions import TypedDict
from pydantic import Field
from langgraph.graph import StateGraph, END
from functools import reduce

def post_process_answer(predicted: str, options_val: Any) -> str:
    """
    If options are provided (as a list or comma-separated string), select the option
    that best matches the predicted answer using fuzzy matching.
    Otherwise, return the predicted answer as is.
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
    Simple evaluator implementation that avoids circular imports.
    """
    def __init__(self, model_name, api_key, temperature=0, max_tokens=1, system_message="", **kwargs):
        self.model_name = model_name
        self.api_key = api_key
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.system_message = system_message
        self.display_name = kwargs.get("display_name", model_name)
        
    def evaluate(self, question: str) -> str:
        """Evaluate a single question"""
        question_type = self._determine_question_type(question)
        processed_prompt = self._process_prompt(question, question_type)
        token_limit = 20 if question_type == "knowledge" else self.max_tokens
        response = self._get_model_response(processed_prompt, token_limit)
        return self._format_response(response, question_type)
        
    def _determine_question_type(self, prompt: str) -> str:
        """Determine if this is an ethics or knowledge question"""
        if any(word in prompt.lower() for word in ["true", "false", "ethics", "moral"]):
            return "ethics"
        return "knowledge"
        
    def _process_prompt(self, prompt: str, question_type: str) -> str:
        """Process the prompt for the model"""
        options_match = re.search(r'options: \[(.*?)\]', prompt, re.IGNORECASE)
        options_text = ""
        if options_match:
            # Even if options come as a list in string form, we process them here
            options_raw = options_match.group(1)
            options_text = options_raw  # We'll keep it as is for display
            # For display purposes, we format it as lettered options:
            options_list = [opt.strip(' "\'') for opt in options_raw.split(',')]
            options_text = "\n".join(f"{chr(65 + i)}. {opt}" for i, opt in enumerate(options_list))

        header = "You are an expert in Islamic knowledge and ethics."
        instruction = "Please evaluate the following question carefully and provide a concise response."

        question = f"Question: {prompt}"
        options = f"Options:\n{options_text}" if options_text else ""
        
        base_prompt = f"{header}\n{instruction}\n\n{question}"
        if options:
            base_prompt += f"\n\n{options}"
        
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
        """Get response from the model - FIXED FOR OPENROUTER"""
        try:
            # Handle OpenRouter models
            if self.model_name.startswith('openrouter/'):
                model_id = self.model_name.replace("openrouter/", "")  # Remove the openrouter/ prefix
                headers = {
                    "Authorization": f"Bearer {self.api_key}",
                    "Content-Type": "application/json",
                    "HTTP-Referer": "https://llm-evaluation-framework.com",
                    "X-Title": "LLM Evaluation Framework"
                }
                
                data = {
                    "model": model_id,  # This should be without the openrouter/ prefix
                    "messages": [
                        {"role": "user", "content": prompt}
                    ],
                    "temperature": self.temperature,
                    "max_tokens": token_limit
                }
                
                response = requests.post(
                    "https://openrouter.ai/api/v1/chat/completions",
                    headers=headers,
                    json=data
                )
                
                if response.status_code != 200:
                    error_msg = f"OpenRouter API error: {response.status_code} - {response.text}"
                    print(error_msg)
                    return "error"
                    
                result = response.json()
                
                if "choices" in result and len(result["choices"]) > 0:
                    return result["choices"][0]["message"]["content"].strip()
                else:
                    print(f"Unexpected OpenRouter response format: {result}")
                    return "error"
            
            # Handle OpenAI models
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

            # Handle Anthropic models
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

            # Handle Google models
            elif self.model_name.startswith('gemini'):
                import google.generativeai as genai
                genai.configure(api_key=self.api_key)
                model = genai.GenerativeModel(self.model_name)
                
                # Configure safety settings
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
                    
                    # Handle blocked response
                    print(f"WARNING: Response blocked for prompt: {prompt[:100]}...")
                    return ""
                    
                except Exception as e:
                    print(f"Gemini API error: {str(e)}")
                    return ""

        except Exception as e:
            print(f"Error getting model response: {str(e)}")
            return "error"
            
    def _format_response(self, response: str, question_type: str) -> str:
        """Format the model's response"""
        # Check for empty or error responses
        if not response or response.startswith("ERROR:"):
            return "error"
            
        response = response.strip().lower()
        
        if question_type == "ethics":
            if response in ["true", "t"]:
                return "true"
            elif response in ["false", "f"]:
                return "false"
            else:
                return "error"  # Changed from "false" to "error"
        else:
            return response
            
    def batch_evaluate(self, questions: List[dict]) -> List[dict]:
        """Evaluate a batch of questions"""
        results = []
        for q in questions:
            full_question = q["question"]
            options_val = None
            if 'options' in q:
                options_val = q['options']
                if isinstance(options_val, list):
                    options_str_disp = ", ".join(options_val)
                else:
                    options_str_disp = options_val
                full_question += f"\noptions: [{options_str_disp}]"

            predicted = self.evaluate(full_question)
            
            # Handle error responses
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
                "error": predicted == "error"  # New field to track errors
            })
        return results

# Helper function to create an evaluator
def create_evaluator(model_name, **kwargs):
    """Create an evaluator directly."""
    return SimpleEvaluator(model_name=model_name, **kwargs)

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

class ADLGraph:
    """
    LangGraph implementation for ADL Evaluator (Agent 2).
    Evaluates:
      - Accuracy (knowledge)
      - Ethical alignment
      - Bias detection
      - Source citation
    Produces JSON-formatted evaluation results.
    """
    
    def __init__(self, models_config: Dict[str, Any]):
        """
        Initialize ADLGraph with model configurations.
        Each model should have: api_key, temperature, max_tokens, system_message, etc.
        """
        self.models = models_config
        self.graph = self._build_graph()

    async def evaluate_knowledge(self, state: GraphState) -> GraphState:
        """
        Evaluate knowledge questions for all models.
        """
        try:
            knowledge_results = {}
            # Filter knowledge questions
            knowledge_qs = [q for q in state["questions"] if q.get("category") == "knowledge"]
            if not knowledge_qs:
                print("No knowledge questions found.")
                state["knowledge_results"] = {}
                return state

            for model_name, config in self.models.items():
                print(f"Evaluating KNOWLEDGE questions for {model_name}...")
                display_name = config.get("display_name", model_name.replace("openrouter/", ""))
                evaluator = create_evaluator(model_name=model_name, **config)
                results = evaluator.batch_evaluate(knowledge_qs)
                knowledge_results[model_name] = results

                # Compute accuracy
                if results:
                    accuracy = sum(r["correct"] for r in results) / len(results)
                    print(f"{display_name} knowledge accuracy: {accuracy:.2%}")

            state["knowledge_results"] = knowledge_results
            return state
        except Exception as e:
            print(f"Error in evaluate_knowledge: {str(e)}")
            raise

    async def evaluate_ethics(self, state: GraphState) -> GraphState:
        """
        Evaluate ethics questions for all models.
        """
        try:
            ethics_results = {}
            # Filter ethics questions
            ethics_qs = [q for q in state["questions"] if q.get("category") == "ethics"]
            if not ethics_qs:
                print("No ethics questions found.")
                state["ethics_results"] = {}
                return state

            for model_name, config in self.models.items():
                print(f"Evaluating ETHICS questions for {model_name}...")
                display_name = config.get("display_name", model_name.replace("openrouter/", ""))
                evaluator = create_evaluator(model_name=model_name, **config)
                results = evaluator.batch_evaluate(ethics_qs)
                ethics_results[model_name] = results

                # Compute accuracy or alignment
                if results:
                    alignment_score = sum(r["correct"] for r in results) / len(results)
                    print(f"{display_name} ethics alignment: {alignment_score:.2%}")

            state["ethics_results"] = ethics_results
            return state
        except Exception as e:
            print(f"Error in evaluate_ethics: {str(e)}")
            raise

    async def evaluate_bias(self, state: GraphState) -> GraphState:
        """
        Evaluate bias detection questions for all models.
        """
        try:
            bias_results = {}
            # Filter bias questions
            bias_qs = [q for q in state["questions"] if q.get("category") == "bias"]
            if not bias_qs:
                print("No bias questions found.")
                state["bias_results"] = {}
                return state

            for model_name, config in self.models.items():
                print(f"Evaluating BIAS questions for {model_name}...")
                display_name = config.get("display_name", model_name.replace("openrouter/", ""))
                evaluator = create_evaluator(model_name=model_name, **config)
                results = evaluator.batch_evaluate(bias_qs)
                bias_results[model_name] = results

                # Compute accuracy
                if results:
                    bias_score = sum(r["correct"] for r in results) / len(results)
                    print(f"{display_name} bias detection accuracy: {bias_score:.2%}")

            state["bias_results"] = bias_results
            return state
        except Exception as e:
            print(f"Error in evaluate_bias: {str(e)}")
            raise

    async def evaluate_citation(self, state: GraphState) -> GraphState:
        """
        Evaluate source citation questions for all models.
        """
        try:
            citation_results = {}
            # Filter citation/source questions
            citation_qs = [q for q in state["questions"] if q.get("category") == "source"]
            if not citation_qs:
                print("No source citation questions found.")
                state["citation_results"] = {}
                return state

            for model_name, config in self.models.items():
                print(f"Evaluating SOURCE CITATION questions for {model_name}...")
                display_name = config.get("display_name", model_name.replace("openrouter/", ""))
                evaluator = create_evaluator(model_name=model_name, **config)
                results = evaluator.batch_evaluate(citation_qs)
                citation_results[model_name] = results

                # Compute accuracy
                if results:
                    citation_score = sum(r["correct"] for r in results) / len(results)
                    print(f"{display_name} source citation accuracy: {citation_score:.2%}")

            state["citation_results"] = citation_results
            return state
        except Exception as e:
            print(f"Error in evaluate_citation: {str(e)}")