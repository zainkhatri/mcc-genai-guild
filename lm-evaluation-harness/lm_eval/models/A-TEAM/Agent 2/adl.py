"""
adl.py

Defines the ADLEvaluator (LLM wrapper) and the IslamicEvalChain for multi-LLM evaluation.
"""

import difflib
from typing import Any, List, Mapping, Optional
from langchain.callbacks.manager import CallbackManagerForLLMRun
from langchain.llms.base import LLM
from pydantic import Field, model_validator
from langchain.utils import get_from_dict_or_env
import json
import os
import re

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

class ADLEvaluator(LLM):
    """
    ADL (Adl Dynamic Learning) evaluator for Islamic knowledge and ethics testing.
    """

    client: Optional[Any] = None  # Optional model client
    model_name: str = Field(default="gpt-4")
    api_key: Optional[str] = None
    temperature: float = 0.0
    # Default token limit; we'll override for knowledge questions in _call.
    max_tokens: int = 5  
    model_kwargs: dict = Field(default_factory=dict)
    
    @model_validator(mode="before")
    def validate_environment(cls, values: dict) -> dict:
        """
        Validate that the API key is set based on model name.
        """
        model_name = values.get("model_name", "")
        if (model_name.startswith("gpt") or 
            model_name.endswith("-preview")):
            api_key = get_from_dict_or_env(values, "api_key", "OPENAI_API_KEY")
        elif model_name.startswith("claude"):
            api_key = get_from_dict_or_env(values, "api_key", "ANTHROPIC_API_KEY")
        elif model_name.startswith("gemini"):
            api_key = get_from_dict_or_env(values, "api_key", "GOOGLE_API_KEY")
        else:
            raise ValueError(f"Unsupported model: {model_name}")
        values["api_key"] = api_key
        return values

    @property
    def _llm_type(self) -> str:
        return "adl_evaluator"

    def _call(
        self,
        prompt: str,
        stop: Optional[List[str]] = None,
        run_manager: Optional[CallbackManagerForLLMRun] = None,
        **kwargs: Any,
    ) -> str:
        """
        Internal call method that builds and processes the prompt.
        """
        question_type = self._determine_question_type(prompt)
        processed_prompt = self._process_prompt(prompt, question_type)
        # For knowledge questions, allow more tokens (e.g., 20 tokens)
        token_limit = 20 if question_type == "knowledge" else self.max_tokens
        response = self._get_model_response(processed_prompt, stop, token_limit=token_limit)
        formatted_response = self._format_response(response, question_type)
        return formatted_response

    def _determine_question_type(self, prompt: str) -> str:
        """
        Determine if the question is for ethics or knowledge evaluation.
        """
        if any(word in prompt.lower() for word in ["true", "false", "ethics", "moral"]):
            return "ethics"
        return "knowledge"

    def _process_prompt(self, prompt: str, question_type: str) -> str:
        """
        Process and format the prompt based on the question type.
        """
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

    def _get_model_response(self, prompt: str, stop: Optional[List[str]], token_limit: int) -> str:
        """
        Get response from the underlying model, using the specified token limit.
        """
        try:
            if self.model_name.startswith('gpt'):
                import openai
                openai.api_key = self.api_key
                response = openai.chat.completions.create(
                    model=self.model_name,
                    messages=[
                        {"role": "system", "content": self.model_kwargs.get("system_message", "")},
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
            raise ValueError(f"Error getting model response: {str(e)}")

    def _format_response(self, response: str, question_type: str) -> str:
        """
        Format and validate the model's response.
        - For ethics: expect 'true' or 'false'.
        - For knowledge: return the raw response (trimmed and lower-cased) for comparison.
        - For any failed response: return 'error'
        """
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
        """
        Evaluate a batch of questions and return detailed results.
        Now handles error responses properly.
        """
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
                
            print(f"DEBUG:\nQuestion: {q['question']}\nPredicted: {predicted}\nExpected: {q['correct']}\n")

            results.append({
                "question": q["question"],
                "predicted": predicted,
                "actual": q["correct"],
                "correct": is_correct,
                "category": q.get("category", "knowledge"),
                "error": predicted == "error"  # New field to track errors
            })
        return results

    def get_num_tokens(self, text: str) -> int:
        """
        Return the approximate number of tokens in the text.
        """
        return len(text.split())

    @property
    def _identifying_params(self) -> Mapping[str, Any]:
        """
        Return the identifying parameters for the model.
        """
        return {
            "model_name": self.model_name,
            "temperature": self.temperature,
            "max_tokens": self.max_tokens,
            **self.model_kwargs
        }

class IslamicEvalChain:
    """
    Chain for evaluating Islamic knowledge and ethics questions.
    """
    
    def __init__(self, llm: ADLEvaluator):
        self.llm = llm

    def evaluate(self, question: str) -> str:
        """
        Evaluate a single question using the LLM's invoke method.
        """
        return self.llm.invoke(question)

    def batch_evaluate(self, questions: List[dict]) -> List[dict]:
        """
        Evaluate a batch of questions and return detailed results.
        Debug logging is added to print each question, its predicted answer, and the expected answer.
        Additionally, if options are provided, post-process the answer to match one of the options.
        """
        results = []
        for q in questions:
            full_question = q["question"]
            options_val = None
            if 'options' in q:
                options_val = q['options']
                # If options is a list, join it into a string for display
                if isinstance(options_val, list):
                    options_str_disp = ", ".join(options_val)
                else:
                    options_str_disp = options_val
                full_question += f"\noptions: [{options_str_disp}]"

            predicted = self.evaluate(full_question)
            if options_val:
                predicted = post_process_answer(predicted, options_val)
            print(f"Question: {q['question']}\nPredicted: {predicted}\nCorrect Answer: {q['correct']}\n")

            correct_label = str(q["correct"]).strip().lower()
            is_correct = (predicted == correct_label)
            results.append({
                "question": q["question"],
                "predicted": predicted,
                "actual": q["correct"],
                "correct": is_correct,
                "category": q.get("category", "knowledge")
            })
        return results

def create_evaluator(model_name: str = "gpt-4", **kwargs) -> IslamicEvalChain:
    """
    Create an Islamic knowledge evaluator chain.
    """
    llm = ADLEvaluator(model_name=model_name, **kwargs)
    return IslamicEvalChain(llm=llm)