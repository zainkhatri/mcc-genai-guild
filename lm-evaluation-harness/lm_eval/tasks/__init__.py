import collections
from .task_manager import TaskManager

"""Task registry."""

from . import islamic_knowledge_task

TASK_REGISTRY = collections.defaultdict(dict)

# Update the registry with tasks from each module 
for task_name, task_class in islamic_knowledge_task.TASK_REGISTRY.items():
    TASK_REGISTRY[task_name] = task_class

ALL_TASKS = sorted(list(TASK_REGISTRY.keys()))

def get_task(task_name):
    try:
        return TASK_REGISTRY[task_name]
    except KeyError:
        print(f"Available tasks: {ALL_TASKS}")
        raise KeyError(f"Missing task {task_name}")

def list_tasks():
    return ALL_TASKS

def get_task_dict(task_list, task_manager=None):
    """Return a dictionary of tasks with specified names."""
    return {name: TASK_REGISTRY[name]() for name in task_list if name in TASK_REGISTRY}
