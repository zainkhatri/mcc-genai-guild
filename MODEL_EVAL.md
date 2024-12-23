# Model Evaluations Build Log

## We're Using a Virtual Environment (venv)
We're using a virtual environment to keep things clean and organized. It has its own copy of Python and its own set of tools. This way:
- All the packages we install won't mess with anyones computer's main Python setup
- Everyone on the team can have the exact same setup
- It's easier to track what packages we're using
- If something goes wrong, we can just delete the venv and start fresh

## Getting Into the Environment
1. First, go to the right folder:
```bash
cd lm-evaluation-harness
```

2. Activate the virtual environment:
```bash
source venv/bin/activate
```
You'll know it worked when you see `(venv)` at the start of your command line.

## Check Your Packages

1. Install lm-evaluation-harness and its requirements:
```bash
pip install lm-eval
```

2. Install OpenAI package (for GPT-4 access):
```bash
pip install openai
```

3. Verify installations:
```bash
pip list
```

You should see these core packages (versions may vary):
- lm_eval
- openai
- transformers
- torch
- numpy
- pandas
- tqdm
- datasets
- evaluate
- jsonlines
- scikit-learn

If lm-eval installation shows any missing dependencies, it will tell you and you can install them.

## Our Project Structure
```
mcc-genai-guild/
└── lm-evaluation-harness/
    ├── lm_eval/
    │   ├── models/
    │   │   └── evaluate_islamic_model.py    # Our main testing script
    │   ├── tasks/
    │   │   └── islamic_knowledge_task/      # Our task folder
    │   │       ├── __init__.py             # Makes Python treat this as a package
    │   │       └── islamic_knowledge_task.py # Our actual task code
    │   └── data/
    │       └── q_and_a.jsonl               # A sample of our question-answer pairs (needs improvement)
    └── venv/                               # Our virtual environment
```

## What Each File Does

### 1. evaluate_islamic_model.py
This is our main script that runs everything. It:
- Loads up the AI models (like GPT-4)
- Gets our Islamic knowledge questions ready
- Tests how well the models answer these questions
- Shows you the results

### 2. islamic_knowledge_task.py
This is where we define how to test the AI models. It:
- Loads our questions and answers
- Tells the system how to score the AI's answers
- Handles all the testing logic

### 3. __init__.py files
These are like signposts that tell Python "hey, there's important code in this folder" 

### 4. q_and_a.jsonl
This is our actual test data. It has pairs of:
- Questions about Islamic topics
- Their correct answers
The AI models will try to answer these questions, and we'll see how well they do...

## Next Steps

1. **We need to set up our API Key**
   ```bash
   export OPENAI_API_KEY='insert-key-here'
   ```

2. **Check the Test Data**
   - Update it to have proper question-answer pairs (its a jsonl file at the moment)

3. **Run the Tests**
   ```bash
   python lm_eval/models/evaluate_islamic_model.py
   ```