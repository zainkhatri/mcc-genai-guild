"""Islamic knowledge evaluation task implementation with ethics evaluation."""
from lm_eval.api.task import Task
from lm_eval.api.instance import Instance
import os
import jsonlines
import re

class IslamicKnowledgeTask(Task):
    """Task for evaluating models on Islamic knowledge and ethics."""
    VERSION = 1
    OUTPUT_TYPE = "generate_until"
    DATASET_PATH = None  # Not used directly; we load from multiple files
    
    def __init__(self, data_dir=None, cache_dir=None, download_mode=None, config=None):
        super().__init__(data_dir, cache_dir, download_mode)
        self._knowledge_data = []   # List for Q&A knowledge documents
        self._ethics_data = []      # List for ethics documents
        self._data = []             # Combined data
        self._download_data()
        self.task_name = "islamic_knowledge_and_ethics"
    
    def _download_data(self):
        """Load data from local JSONL files for knowledge and ethics."""
        try:
            # Assume the data directory is three levels up in a folder called "data"
            base_path = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
            knowledge_path = os.path.join(base_path, "data", "q_and_a.jsonl")
            ethics_path = os.path.join(base_path, "data", "ethics.jsonl")
            
            # Load knowledge questions
            if not os.path.exists(knowledge_path):
                raise FileNotFoundError(f"Knowledge data file not found at {knowledge_path}")
            knowledge = []
            with jsonlines.open(knowledge_path) as reader:
                for obj in reader:
                    # If options are provided as a comma‐separated string, convert to list
                    if isinstance(obj.get('options'), str):
                        obj['options'] = [opt.strip() for opt in obj['options'].split(',')]
                    obj['task_type'] = "knowledge"
                    knowledge.append(obj)
            self._knowledge_data = knowledge
            print(f"Successfully loaded {len(knowledge)} knowledge questions")
            
            # Load ethics questions
            if not os.path.exists(ethics_path):
                raise FileNotFoundError(f"Ethics data file not found at {ethics_path}")
            ethics = []
            with jsonlines.open(ethics_path) as reader:
                for obj in reader:
                    # Ensure options are a list; if not present, default to ["True", "False"]
                    if 'options' not in obj or not obj['options']:
                        obj['options'] = ["True", "False"]
                    else:
                        if isinstance(obj['options'], str):
                            obj['options'] = [opt.strip() for opt in obj['options'].split(',')]
                    obj['task_type'] = "ethics"
                    ethics.append(obj)
            self._ethics_data = ethics
            print(f"Successfully loaded {len(ethics)} ethics questions")
            
            # Combine both sets into a single list
            self._data = self._knowledge_data + self._ethics_data
            print(f"Total questions for evaluation: {len(self._data)}")
        except Exception as e:
            print(f"Error loading data: {str(e)}")
            raise

    def doc_to_target(self, doc, **kwargs):
        """Return the target answer for a document.
        
        For 'knowledge' documents, returns a single letter (A, B, C, or D) corresponding to the correct option.
        For 'ethics' documents, returns the normalized answer ('True' or 'False').
        """
        if doc.get("task_type") == "knowledge":
            try:
                # Attempt to find the index of the correct answer in the options list
                correct_idx = doc['options'].index(doc['correct'])
            except (ValueError, KeyError):
                try:
                    # Fallback: try to interpret 'correct' as an index
                    correct_idx = int(doc['correct'])
                except (ValueError, TypeError):
                    # Try a case-insensitive match
                    correct = str(doc['correct']).strip().lower()
                    for idx, option in enumerate(doc['options']):
                        if str(option).strip().lower() == correct:
                            correct_idx = idx
                            break
                    else:
                        print(f"Warning: Could not match correct answer '{doc['correct']}' in options {doc['options']}")
                        correct_idx = 0
            # Convert the index to a letter (0 -> A, 1 -> B, etc.)
            return chr(65 + correct_idx)
        
        elif doc.get("task_type") == "ethics":
            # For ethics, simply normalize the answer to title case (e.g., "True" or "False")
            return str(doc['correct']).strip().title()
        
        else:
            return ""

    def doc_to_text(self, doc, **kwargs):
        """Format the prompt text for a document.
        
        For 'knowledge' questions, includes the multiple-choice options labeled A–D.
        For 'ethics' questions, shows the question along with "True, False" as the available options.
        """
        if doc.get("task_type") == "knowledge":
            options_text = '\n'.join(
                f"{chr(65 + i)}. {str(opt).strip()}" 
                for i, opt in enumerate(doc['options'])
            )
            return (
                f"Question: {doc['question']}\n\n"
                f"Choices:\n{options_text}\n\n"
                f"Provide only a single letter (A, B, C, or D) as your answer:"
            )
        elif doc.get("task_type") == "ethics":
            # For ethics questions, the answer should be either True or False.
            options_text = "True, False"
            return (
                f"Question: {doc['question']}\n\n"
                f"Choices: {options_text}\n\n"
                f"Answer with either 'True' or 'False' (no extra text):"
            )
        else:
            return doc.get("question", "")
    
    def process_results(self, doc, results, **kwargs):
        """Process the model's response and compute an accuracy score.
        
        For 'knowledge' documents, looks for a single letter (A–D).
        For 'ethics' documents, compares the response to the expected answer (True/False).
        """
        if not results or not results[0]:
            return {"accuracy": 0.0}
        try:
            response = results[0].strip()
            if doc.get("task_type") == "knowledge":
                # Look for a single letter A–D in the response (case-insensitive)
                match = re.search(r'[ABCD]', response, re.IGNORECASE)
                if not match:
                    return {"accuracy": 0.0}
                generated = match.group(0).upper()
                target = self.doc_to_target(doc).upper()
                correct = (generated == target)
                return {"accuracy": float(correct)}
            elif doc.get("task_type") == "ethics":
                # For ethics, normalize to title case and compare
                generated = response.title()
                target = self.doc_to_target(doc)
                correct = (generated == target)
                return {"accuracy": float(correct)}
            else:
                return {"accuracy": 0.0}
        except Exception as e:
            print(f"Error processing result '{results[0]}': {str(e)}")
            return {"accuracy": 0.0}
    
    def download(self, data_dir=None, cache_dir=None, download_mode=None):
        # No additional download logic needed since _download_data() handles file loading.
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

    def fewshot_context(self, doc, num_fewshot, system_instruction=None, 
                        apply_chat_template=False, fewshot_as_multiturn=False, 
                        chat_template=None, **kwargs):
        """Generate a few-shot prompt context with examples."""
        if not system_instruction:
            system_instruction = (
                "You are a knowledgeable assistant for Islamic topics. "
                "You will be given a multiple choice question. "
                "Respond with only the correct answer as specified (a single letter for knowledge or True/False for ethics)."
            )
        if apply_chat_template:
            messages = [{"role": "system", "content": system_instruction}]
            if num_fewshot > 0:
                # Use training docs if available; otherwise, use validation docs
                if self.has_training_docs():
                    fewshot_examples = self.fewshot_examples(num_fewshot, self.fewshot_rnd)
                else:
                    if not hasattr(self, '_fewshot_docs'):
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
            examples = [system_instruction]
            if num_fewshot > 0:
                if self.has_training_docs():
                    fewshot_examples = self.fewshot_examples(num_fewshot, self.fewshot_rnd)
                else:
                    if not hasattr(self, '_fewshot_docs'):
                        self._fewshot_docs = list(self.validation_docs())
                    fewshot_examples = self._fewshot_docs[:num_fewshot]
                for ex in fewshot_examples:
                    ex_text = self.doc_to_text(ex)
                    ex_target = self.doc_to_target(ex)
                    examples.append(f"{ex_text}\nAnswer: {ex_target}")
            examples.append(self.doc_to_text(doc))
            return "\n\n".join(examples)
    
    def construct_requests(self, doc, ctx, apply_chat_template=False, metadata=None, **kwargs):
        """Construct a single token generation request."""
        return Instance(
            request_type="generate_until",
            doc=doc,
            arguments=(ctx, {'until': ['\n', '.', ' '], 'max_tokens': 1}),
            idx=0,
            metadata=metadata
        )
    
    def aggregation(self):
        return {
            "accuracy": lambda scores: sum(scores) / len(scores) if scores else 0.0
        }
    
    def higher_is_better(self):
        return {
            "accuracy": True
        }