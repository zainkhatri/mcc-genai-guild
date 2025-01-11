"""Base abstractions for defining a language model wrapper and task."""

class Task:
    """A task represents an entire benchmark including its dataset, problems, answers, and evaluation methods."""

    VERSION = None

    def __init__(self):
        self.download()
        self._training_docs = None
        self._validation_docs = None
        self._test_docs = None

    def download(self):
        """Downloads the task dataset if necessary."""
        pass

    def has_training_docs(self):
        """Whether the task has a training set."""
        return False

    def has_validation_docs(self):
        """Whether the task has a validation set."""
        return False

    def has_test_docs(self):
        """Whether the task has a test set."""
        return False

    def training_docs(self):
        """Training documents."""
        return []

    def validation_docs(self):
        """Validation documents."""
        return []

    def test_docs(self):
        """Test documents."""
        return []

    def fewshot_examples(self, k, rng):
        """Returns k fewshot examples."""
        if k == 0:
            return []
        training_docs = self.training_docs()
        return rng.sample(training_docs, k)

    def doc_to_text(self, doc):
        """Convert a document to text for processing."""
        raise NotImplementedError

    def doc_to_target(self, doc):
        """Convert a document to target for evaluation."""
        raise NotImplementedError

    def construct_requests(self, doc, ctx):
        """Construct requests for the language model."""
        raise NotImplementedError

    def process_results(self, doc, results):
        """Process results from the language model."""
        raise NotImplementedError

    def aggregation(self):
        """Specify how to aggregate results."""
        raise NotImplementedError

    def higher_is_better(self):
        """Whether a higher value is better for the metrics."""
        raise NotImplementedError