# lm_eval/tasks/islamic_knowledge_task.py

from lm_eval.base import Task
from lm_eval.metrics import mean, accuracy
import json

class IslamicKnowledgeTask(Task):
    VERSION = 0
    DATASET_PATH = "data/q_and_a.jsonl"
    DATASET_NAME = None

    def __init__(self):
        super().__init__()
        self.data = self._load_data()

    def _load_data(self):
        with open(self.DATASET_PATH, 'r') as f:
            return [json.loads(line) for line in f]

    def has_training_docs(self):
        return False

    def has_validation_docs(self):
        return True

    def has_test_docs(self):
        return False

    def training_docs(self):
        return []  # No training docs

    def validation_docs(self):
        return self.data

    def test_docs(self):
        return []  # No test docs

    def doc_to_text(self, doc):
        # Convert document to input text format
        return doc["question"]

    def doc_to_target(self, doc):
        # Return expected answer
        return doc["answer"]

    def construct_requests(self, doc, ctx):
        # Construct model request from document
        return [{"request_type": "generate_until", "request_args": {"stop": ["\n"]}}]

    def process_results(self, doc, results):
        # Process model outputs and compare with targets
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
