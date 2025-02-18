"""Islamic knowledge evaluation task implementation."""
from lm_eval.api.task import Task
from lm_eval.api.instance import Instance
import os
import jsonlines
import re

class IslamicKnowledgeTask(Task):
    """Task for evaluating models on Islamic knowledge."""
    VERSION = 0
    OUTPUT_TYPE = "generate_until"
    DATASET_PATH = None
    
    def __init__(self, data_dir=None, cache_dir=None, download_mode=None, config=None):
        super().__init__(data_dir, cache_dir, download_mode)
        self._data = None
        self._download_data()
        self.task_name = "islamic_knowledge"
    
    def _download_data(self):
        """Load data from local JSONL file."""
        try:
            base_path = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
            data_path = os.path.join(base_path, "data", "q_and_a.jsonl")
            
            if not os.path.exists(data_path):
                raise FileNotFoundError(f"Data file not found at {data_path}")
                
            data = []
            with jsonlines.open(data_path) as reader:
                for obj in reader:
                    if isinstance(obj['options'], str):
                        obj['options'] = [opt.strip() for opt in obj['options'].split(',')]
                    data.append(obj)
            self._data = data
            print(f"Successfully loaded {len(data)} questions")
        except Exception as e:
            print(f"Error loading data: {str(e)}")
            raise

    def doc_to_target(self, doc, **kwargs):
        """Get the correct answer letter.
        
        The correct answer can be stored in two ways:
        1. As the actual answer text that needs to be matched with options
        2. As the index of the correct option
        """
        try:
            # First try to find the index of the exact answer in options
            correct_idx = doc['options'].index(doc['correct'])
        except (ValueError, KeyError):
            try:
                # If that fails, try to use the answer as a direct index
                correct_idx = int(doc['correct'])
            except (ValueError, TypeError):
                # If both fail, try to find partial match
                correct = str(doc['correct']).strip().lower()
                for idx, option in enumerate(doc['options']):
                    if str(option).strip().lower() == correct:
                        correct_idx = idx
                        break
                else:
                    # If no match found, log the error and default to first option
                    print(f"Warning: Could not find correct answer '{doc['correct']}' in options {doc['options']}")
                    correct_idx = 0
        
        # Convert index to letter (0 -> A, 1 -> B, etc.)
        return chr(65 + correct_idx)

    def doc_to_text(self, doc, **kwargs):
        """Format question with clear instruction for a single-letter answer."""
        # Format options with letters, handling any type of value
        options_text = '\n'.join(
            f"{chr(65 + i)}. {str(opt).strip()}" 
            for i, opt in enumerate(doc['options'])
        )
        
        return (
            f"Question: {doc['question']}\n\n"
            f"Choices:\n{options_text}\n\n"
            f"Provide only a single letter (A, B, C, or D) as your answer:"
        )

    def process_results(self, doc, results, **kwargs):
        """Process and score response."""
        if not results or not results[0]:
            return {"accuracy": 0.0}

        try:
            # Get first letter of response
            response = results[0].strip().upper()
            if not response:
                return {"accuracy": 0.0}
                
            # Find first letter A, B, C, or D
            match = re.search(r'[ABCD]', response)
            if not match:
                return {"accuracy": 0.0}
                
            generated = match.group(0)
            true = self.doc_to_target(doc)
            
            # Compare normalized letters
            correct = (generated.upper() == true.upper())
            return {"accuracy": float(correct)}
            
        except Exception as e:
            print(f"Error processing result '{results[0]}': {str(e)}")
            return {"accuracy": 0.0}
    

    def download(self, data_dir=None, cache_dir=None, download_mode=None):
        pass

    def has_training_docs(self):
        return False

    def has_validation_docs(self):
        return True

    def has_test_docs(self):
        return False

    def training_docs(self):
        return []

    def validation_docs(self):
        return self._data if self._data else []

    def test_docs(self):
        return []

    def fewshot_context(
        self,
        doc,
        num_fewshot,
        system_instruction=None,
        apply_chat_template=False,
        fewshot_as_multiturn=False,
        chat_template=None,
        **kwargs
    ):
        """Generate context with examples."""
        if not system_instruction:
            system_instruction = ("You are a knowledgeable assistant for Islamic topics. "
                                "You will be given multiple choice questions. "
                                "Respond with ONLY a single letter (A, B, C, or D) as your answer. "
                                "Do not include any other text, explanation, or punctuation.")

        if apply_chat_template:
            messages = []
            messages.append({"role": "system", "content": system_instruction})

            if num_fewshot > 0:
                if self.has_training_docs():
                    fewshot_examples = self.fewshot_examples(num_fewshot, self.fewshot_rnd) 
                else:
                    if self._fewshot_docs is None:
                        self._fewshot_docs = list(self.validation_docs())
                    fewshot_examples = self._fewshot_docs[:num_fewshot]
                
                for ex in fewshot_examples:
                    messages.append({
                        "role": "user",
                        "content": self.doc_to_text(ex)
                    })
                    messages.append({
                        "role": "assistant",
                        "content": self.doc_to_target(ex)
                    })

            messages.append({
                "role": "user",
                "content": self.doc_to_text(doc)
            })

            if chat_template:
                return chat_template(messages)
            return messages

        else:
            if num_fewshot == 0:
                return self.doc_to_text(doc)

            examples = []
            if system_instruction:
                examples.append(system_instruction)
            
            if self.has_training_docs():
                fewshot_examples = self.fewshot_examples(num_fewshot, self.fewshot_rnd)
            else:
                if self._fewshot_docs is None:
                    self._fewshot_docs = list(self.validation_docs())
                fewshot_examples = self._fewshot_docs[:num_fewshot]

            for ex in fewshot_examples:
                ex_text = self.doc_to_text(ex)
                ex_target = self.doc_to_target(ex)
                examples.append(f"{ex_text}\n{ex_target}")

            examples.append(self.doc_to_text(doc))
            return "\n\n".join(examples)

    def construct_requests(self, doc, ctx, apply_chat_template=False, metadata=None, **kwargs):
        """Construct a single token generation request."""
        return Instance(
            request_type="generate_until",
            doc=doc,
            arguments=(ctx, {
                'until': ['\n', '.', ' '],  # Stop at newline, period, or space
                'max_tokens': 1
            }),
            idx=0,
            metadata=metadata
        )
    
    def aggregation(self):
        return {
            "accuracy": lambda list_of_scores: sum(list_of_scores) / len(list_of_scores) if list_of_scores else 0
        }

    def higher_is_better(self):
        return {
            "accuracy": True
        }