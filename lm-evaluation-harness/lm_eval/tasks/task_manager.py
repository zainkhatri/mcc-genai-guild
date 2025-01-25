class TaskManager:
    def __init__(self, verbosity="INFO"):
        self.verbosity = verbosity
        self.tasks = {}

    def register_task(self, task_name, task_class):
        self.tasks[task_name] = task_class

    def get_task(self, task_name):
        return self.tasks.get(task_name)

    def list_tasks(self):
        return list(self.tasks.keys())