from lm_eval.base import Task
from lm_eval.metrics import mean, accuracy
import json
import jsonlines

class IslamicKnowledgeTask(Task):
    VERSION = 0
    DATASET_PATH = "data/islamic_knowledge.jsonl"
    DATASET_NAME = None

    def __init__(self):
        super().__init__()
        self.data = self._load_data()

    def _load_data(self):
        # Read JSONL file
        data = []
        with jsonlines.open(self.DATASET_PATH) as reader:
            for obj in reader:
                # Process options into a list
                obj['options'] = [opt.strip() for opt in obj['options'].split(',')]
                data.append(obj)
                
        print(f"Loaded {len(data)} questions from JSONL file")
        return data

    def has_training_docs(self):
        return False

    def has_validation_docs(self):
        return True

    def has_test_docs(self):
        return False

    def training_docs(self):
        return []

    def validation_docs(self):
        return self.data

    def test_docs(self):
        return []

    def doc_to_text(self, doc):
        # Format the question with multiple choice options
        options_text = '\n'.join(f"{chr(65 + i)}. {opt}" for i, opt in enumerate(doc['options']))
        return f"{doc['question']}\n\n{options_text}"

    def doc_to_target(self, doc):
        return doc['correct']

    def construct_requests(self, doc, ctx):
        return [{"request_type": "generate_until", "request_args": {"stop": ["\n"]}}]

    def process_results(self, doc, results):
        pred = results[0].strip()
        true = self.doc_to_target(doc)
        return {"accuracy": float(pred == true)}

    def aggregation(self):
        return {
            "accuracy": mean
        }

    def higher_is_better(self):
        return {
            "accuracy": True
        }