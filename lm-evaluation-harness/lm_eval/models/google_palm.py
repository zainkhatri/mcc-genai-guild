import os
from typing import Any, Dict, List, Tuple
from tqdm import tqdm
import google.generativeai as genai
from lm_eval import utils
from lm_eval.api.model import LM
from lm_eval.api.registry import register_model

eval_logger = utils.eval_logger

@register_model("google-palm")
class GooglePalmLM(LM):
    def __init__(
        self,
        model="gemini-pro",
        api_key=None,
        temperature=0.0,
        max_tokens=1,
        debug_output=False,
        **kwargs,
    ):
        super().__init__()
        if api_key is None:
            api_key = os.getenv("GOOGLE_API_KEY")
        if api_key is None:
            raise ValueError("GOOGLE_API_KEY environment variable not set")
        
        genai.configure(api_key=api_key)
        self.model = genai.GenerativeModel(model)
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.debug_output = debug_output
        self.kwargs = kwargs

    @property
    def tokenizer_name(self) -> str:
        """Required property for chat template support"""
        return "google/gemini-pro"
    
    def apply_chat_template(self, chat_history: List[Dict[str, str]]) -> str:
        """Convert chat history into Gemini format."""
        prompt_parts = []
        
        # Add system message if present
        if chat_history and chat_history[0]["role"] == "system":
            system_msg = chat_history[0]["content"]
            chat_history = chat_history[1:]
        else:
            system_msg = None

        if system_msg:
            prompt_parts.append(f"Instructions: {system_msg}\n")
        
        # Format rest of the conversation
        for msg in chat_history:
            role = msg["role"].capitalize()
            content = msg["content"]
            prompt_parts.append(f"{role}: {content}")

        final_prompt = "\n".join(prompt_parts)
        if self.debug_output:
            print(f"\nFinal prompt:\n{final_prompt}\n")
        return final_prompt

    def generate_until(self, requests, disable_tqdm=False) -> List[str]:
        if not requests:
            return []
        
        res = []
        raw_responses = []
        
        for request in tqdm(requests, disable=disable_tqdm):
            prompt = request.args[0]
            try:
                generation_config = genai.types.GenerationConfig(
                    temperature=self.temperature,
                    max_output_tokens=self.max_tokens,
                    candidate_count=1,
                    top_p=1.0,
                    top_k=1,
                )
                
                if self.debug_output:
                    print(f"\nPrompt: {prompt[:200]}...")
                    print(f"Generation config: {generation_config}")
                
                response = self.model.generate_content(
                    prompt,
                    generation_config=generation_config
                )
                
                # Extract just the letter response
                text = response.text.strip()
                raw_responses.append(text)
                
                # Get first letter that matches A, B, C, or D
                import re
                match = re.search(r'[ABCD]', text.upper())
                result = match.group(0) if match else ""
                
                if self.debug_output:
                    print(f"Raw response: {text}")
                    print(f"Extracted letter: {result}")
                
                res.append(result)
            except Exception as e:
                eval_logger.warning(f"Error in generate_until: {e}")
                print(f"Exception details: {str(e)}")
                res.append("")
                raw_responses.append(f"ERROR: {str(e)}")

        # Store raw responses for debugging
        if hasattr(self, 'raw_responses'):
            self.raw_responses.extend(raw_responses)
        else:
            self.raw_responses = raw_responses

        return res

    def loglikelihood(self, requests, disable_tqdm=False):
        raise NotImplementedError("Loglikelihood not implemented for Google Palm")

    def loglikelihood_rolling(self, requests, disable_tqdm=False):
        raise NotImplementedError("Loglikelihood rolling not implemented for Google Palm")