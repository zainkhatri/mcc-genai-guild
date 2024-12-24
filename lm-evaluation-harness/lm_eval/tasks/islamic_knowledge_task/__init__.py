"""Islamic Knowledge evaluation."""
from . islamic_knowledge_task import IslamicKnowledgeTask

TASK_REGISTRY = {
    "islamic_knowledge": IslamicKnowledgeTask,
}

ALL_TASKS = sorted(list(TASK_REGISTRY.keys()))

def get_task(task_name):
    try:
        return TASK_REGISTRY[task_name]
    except KeyError:
        print(f"Available tasks: {ALL_TASKS}")
        raise KeyError(f"Missing task {task_name}")