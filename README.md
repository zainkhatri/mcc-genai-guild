Below is a detailed version of your README that explains every part of the project, including its structure, components, evaluation pipeline, and additional benchmarks. You can save the following content as your README.md:

# Islamic LLM Evaluation Project

## Overview

The **Islamic LLM Evaluation Project** is a comprehensive framework to benchmark and evaluate AI language models on Islamic knowledge and ethics. The project utilizes the open-source `lm-evaluation-harness` framework to execute standardized evaluations across multiple benchmarks such as factual accuracy, ethical alignment, and bias detection. It not only tests models using structured Islamic Q&A and ethics datasets but also aggregates the results into an interactive leaderboard deployed on Hugging Face Spaces.

With advanced evaluation agents and support for multiple languages and benchmarks, this project serves both the research community and developers interested in assessing language model performance in culturally relevant contexts.

---

## Repository Structure

The project repository is organized as follows:

```
mcc-genai-guild/
├── .env                    # Environment file with API keys (excluded from Git)
├── .gitignore              # Files and folders to be ignored by Git (e.g., .env, venv)
├── README.md               # This documentation
└── lm-evaluation-harness/
    ├── venv/              # Virtual environment for Python dependencies
    ├── lm_eval/
    │   ├── data/
    │   │   ├── islamic_knowledge.jsonl   # Islamic Q&A dataset (e.g., questions on Quran, Hadith, Fiqh)
    │   │   └── ethics.jsonl              # Ethics evaluation dataset (assessing ethical alignment)
    │   ├── models/
    │   │   ├── A-TEAM/
    │   │   │   ├── Agent 2/             # ADL Evaluator: Asynchronous evaluation pipeline that orchestrates multi-LLM testing
    │   │   │   └── Agent 3/             # MizanRanker: Aggregates and ranks model performance using weighted metrics
    │   │   ├── openai_completions.py     # Wrapper for interfacing with OpenAI completions API
    │   │   ├── google_palm.py            # Integration for testing Google PaLM models via the Cloud AI Platform
    │   │   ├── huggingface.py            # Support for evaluating HuggingFace models
    │   │   └── evaluate_islamic_model.py  # Core script for running Islamic knowledge tests across different models
    │   └── tasks/
    │       ├── islamic_knowledge_task/
    │       │   ├── __init__.py
    │       │   ├── islamic_knowledge_task.py  # Task definition for loading, prompting, and scoring Islamic knowledge 
```

---

## Understanding the Evaluation Framework

### The Role of `lm-evaluation-harness`

At its core, the project uses the `lm-evaluation-harness` framework, which divides evaluation into two main components:

1. **Tasks:**  
   - Each task defines what aspect of the model is being tested. For example, the `islamic_knowledge_task.py` is responsible for loading Q&A and ethics datasets, formatting data (e.g., converting comma-separated answer options into lists), and establishing the scoring procedure.
   - Other tasks (e.g., TurkishMMLU, arabic_leaderboard*, hendrycks_ethics) evaluate additional dimensions such as language comprehension, cultural context, and ethical reasoning.

2. **Models:**  
   - Modules in the `lm_eval/models/` directory wrap API calls to different providers. They standardize prompts and parameters—such as instructing the model to return a single letter (A, B, C, or D) for multiple-choice questions.
   - For example, `openai_completions.py` handles interactions with OpenAI's API, while other modules (e.g., `google_palm.py`) do the same for Google's models.
   - Advanced evaluation is carried out by the agents in the A-TEAM folder:
     - **Agent 2 (Adl Evaluator):** Uses an asynchronous pipeline (see `run_evaluation.py`) to evaluate models across multiple dimensions (knowledge accuracy, ethics, bias).
     - **Agent 3 (MizanRanker):** Aggregates scores using weighted metrics — typically 30% for accuracy, 30% for ethical alignment, 20% for bias detection, and 20% for citation quality.

---

## The Evaluation Pipeline

### 1. Setting Up

- **Virtual Environment:**  
  Create and activate a virtual environment to keep dependencies isolated.
  ```bash
  python3 -m venv venv
  source venv/bin/activate
  ```

- **Installing Dependencies:**  
  Install the required packages including the custom pluggable evaluation harness and cloud libraries.
  ```bash
  pip install -r requirements.txt
  pip install lm-eval[api] anthropic google-cloud-aiplatform generativeai openai python-dotenv
  ```

### 2. Organizing the Data

- **Datasets:**  
  The evaluations use two primary datasets:
  - `islamic_knowledge.jsonl` – Contains structured multiple-choice questions on topics like Quranic knowledge, Hadith, and Islamic jurisprudence.
  - `ethics.jsonl` – Contains ethically oriented questions that assess model alignment with Islamic values and ethical reasoning.
  
- **Data Loading:**  
  The `IslamicKnowledgeTask` class (in `islamic_knowledge_task.py`) is designed to read both datasets, perform necessary preprocessing (such as converting option strings into lists), and combine them for evaluation.

### 3. Running Model Evaluations

- **Core Evaluation Script:**  
  The script `evaluate_islamic_model.py` ties tasks and model wrappers together. It:
  - Loads questions from the designated dataset files.
  - Iterates over a list of models (for example, various GPT-4 versions) and invokes their respective evaluation functions.
  - Applies a standardized prompt such as:  
    "You are taking an Islamic knowledge test. For each question, respond with only a single letter (A, B, C, or D)."
  - Computes key metrics including Accuracy, F1 Score, and Exact Match.

- **Advanced Agents:**  
  - **Adl Evaluator (Agent 2):**  
    Utilizes asynchronous execution in `run_evaluation.py` to call multiple models concurrently, aggregating evaluations that include not only knowledge-based accuracy but also ethical assessments and bias scores.
  
  - **MizanRanker (Agent 3):**  
    Processes model performance using a weighted scoring system where each performance metric contributes a percentage to the overall rank. The final report is outputted in JSON/CSV formats, which can then be visualized on the leaderboard.

### 4. Leaderboard Generation

- **Result Aggregation:**  
  After evaluations are complete, the results are stored in a JSON file with detailed statistics for each model.
  
- **Deployment:**  
  The aggregated results are used to generate an interactive leaderboard. This leaderboard is designed for deployment on Hugging Face Spaces using Gradio or Streamlit, offering a public-facing interface to compare model scores.

---

## Using the Framework

### Basic Evaluation

To run a basic evaluation using the provided Islamic knowledge task:
```bash
cd lm-evaluation-harness/lm_eval/models
python evaluate_islamic_model.py
```
This script evaluates a range of models (e.g., multiple GPT-4 variants) on a predefined set of 50 questions and prints a comparative report.

### Asynchronous Evaluation with ADL Graph

For an advanced evaluation that leverages asynchronous pipelines:
```bash
cd lm-evaluation-harness/lm_eval/models/A-TEAM/Agent\ 2
python run_evaluation.py
```
This execution will load questions from the data folder, run evaluations concurrently across different provider models, and output detailed results including knowledge accuracy, ethics accuracy, bias score, and more.

### Direct Batch Evaluation Without Graph

A demonstration of direct batch evaluation (without using the asynchronous graph) is available in the `main_eval.py` script:
```bash
cd lm-evaluation-harness/lm_eval/models/A-TEAM
python main_eval.py
```
This script evaluates a small subset (e.g., 5 questions) of the test set and prints the accuracy for each model.

#### Comparative Model Performance

##### Overall Model Rankings
| Model Name                 | Accuracy | Grade |
|----------------------------|----------|-------|
| **Gemini 1.5 Pro**         | 96.00%   | A     |
| **GPT-4 O1**               | 94.00%   | A     |
| **GPT-4o**                 | 94.00%   | A     |
| **GPT-4 Turbo**            | 92.00%   | A-    |
| **GPT-4**                  | 92.00%   | A-    |
| **Claude 3 Opus**          | 92.00%   | A-    |
| **Claude 3.5 Opus**        | 92.00%   | A-    |
| **Gemini 1.5 Flash**       | 84.00%   | B     |
| **Claude 3 Sonnet**        | 76.00%   | C     |
| **Claude 3.5 Sonnet**      | 76.00%   | C     |
| **Claude 2.1**             | 72.00%   | C-    |
| **Zephyr-7B Beta (7B)**    | 43.70%   | F     |
| **microsoft/phi-2 (2.7B)** | 37.33%   | F     |
| **Gemini 2.0 Flash (Beta)**| 28.00%   | F     |
| **StableLM-2 Zephyr (1.6B)**| 24.33%  | F     |

Key Observations:
1. **Model Performance**:
   - Gemini 1.5 Pro leads
   - GPT-4 variants show strong consistent performance
   - Significant gap between large and small models
   - Open-source models currently underperform significantly


3. **Model Size Impact**:
   - Clear correlation between model size and performance
   - Smaller models (<10B parameters) struggle significantly
   - Latest model versions generally outperform older versions

---

## Contributing

We welcome contributions from the community. To contribute:

1. Fork the repository and create your feature or bug-fix branch.
2. Follow the project's coding style and add appropriate tests.
3. Update documentation if your changes affect usage or configuration.
4. Submit a pull request outlining your changes and improvements.

Additionally, if you add a new evaluation task (or benchmark) please:
- Include a clear README in the task directory detailing its purpose.
- Reference any original papers or implementations.
- Document how the task integrates with the main evaluation harness.

---

## Troubleshooting

- **Configuration:**  
  Ensure the `.env` file is correctly populated with your API keys for providers like OpenAI and Anthropic.

- **Dataset Paths:**  
  Verify that the dataset files (e.g., `data/q_and_a.jsonl` and `data/ethics.jsonl`) are present and paths are correctly specified in the tasks.

- **Error Handling:**  
  Review logged errors (e.g., KeyError during metric extraction) and check that API responses conform to expected formats.

---