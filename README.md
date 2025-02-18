# Islamic LLM Evaluation Project

## Overview

The **Islamic LLM Evaluation Project** benchmarks AI language models on Islamic knowledge and ethics using structured datasets and evaluation frameworks. We use **`lm-evaluation-harness`** to test models across various benchmarks, measuring **accuracy, ethical alignment, and bias detection**. This project automates the ranking of models and deploys an interactive **leaderboard on Hugging Face Spaces**.

---

## Repository Structure  

```plaintext
mcc-genai-guild/
├── .env                   # API keys (excluded from Git)
├── .gitignore             # Lists .env, venv, etc.
├── README.md              # This documentation
└── lm-evaluation-harness/
   ├── venv/                  # Virtual environment
   ├── lm_eval/
   │   ├── data/
   │   │   ├── islamic_knowledge.jsonl   # Islamic Q&A dataset
   │   │   ├── ethics.jsonl              # Ethical evaluation dataset
   │   ├── models/
   │   │   ├── A-TEAM/Agent 2/  # Adl Evaluator code
   │   │   ├── A-TEAM/Agent 3/  # MizanRanker code
   │   │   ├── openai_completions.py
   │   │   ├── google_palm.py
   │   │   ├── huggingface.py
   │   │   ├── evaluate_islamic_model.py  # Core script running evaluations
   │   ├── tasks/
   │   │   ├── islamic_knowledge_task/
   │   │   │   ├── init.py
   │   │   │   ├── islamic_knowledge_task.py  # Custom task for LLM evaluation
   │   │   │   └── old/
   │   │   └── (other tasks)
   │   └── (other framework code)
```

---

## Understanding `lm-evaluation-harness`

The `lm-evaluation-harness` is a framework used for **standardized AI model evaluation**. It defines **tasks** (datasets and scoring logic) and connects them to **models** (GPT-4, Claude, Gemini, etc.).

- **Tasks**: Define **what** is being tested (e.g., “Islamic Knowledge Q&A”).
- **Models**: Define **who** is being evaluated (e.g., GPT-4, Claude, Gemini).

### How it Works
1. **The `islamic_knowledge_task.py`** defines how questions are loaded, prompted, and scored.
2. **`evaluate_islamic_model.py`** connects to multiple models and runs structured evaluations.
3. The output is stored in JSON/CSV and **ranked for leaderboard visualization**.

---

## Evaluation Pipeline

### Step 1: Setting Up
- Virtual environment created with **`venv`**.
- Install necessary dependencies:

```bash
pip install lm-eval openai google-cloud-aiplatform python-dotenv
```

### Step 2: Organizing Data
- **`islamic_knowledge.jsonl`** contains structured multiple-choice questions on Quran, Hadith, and Fiqh.
- **`ethics.jsonl`** contains questions assessing model ethical alignment.

### Step 3: Running Model Evaluations
- `evaluate_islamic_model.py` fetches models from OpenAI, Anthropic, and Google.
- Each model answers the same 50-question Islamic Knowledge Q&A.
- **Metrics measured**:
  - **Accuracy**: % of correct answers.
  - **F1 Score**: Measures how well the model balances precision and recall.
  - **Exact Match**: Checks if the response matches the expected output.

### Step 4: Generating the Leaderboard
- Results are stored in structured JSON format.
- Aggregated scores are ranked.
- Deployed to Hugging Face Spaces via Gradio/Streamlit.

---

## Comparative Model Performance

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

---

## Agent 2: Adl Evaluator

The Adl Evaluator automates multi-LLM evaluation. It scores models on:

- **Accuracy (Islamic Knowledge Q&A)**
- **Ethical Alignment (Ethics dataset)**
- **Bias Detection (Future feature)**
- **Source Citation (Future feature)**

### Pipeline
1. **adl.py** – Wraps LLM APIs (OpenAI, Anthropic, Google).
2. **adl_graph.py** – LangGraph-based workflow:
   - `evaluate_knowledge` → tests knowledge Q&A.
   - `evaluate_ethics` → evaluates model bias.
   - `calculate_scores` → computes aggregate performance.
3. **run_evaluation.py** – Runs the entire pipeline asynchronously and outputs results.

---

## Agent 3: MizanRanker

The MizanRanker evaluates and ranks models based on a weighted scoring system.

| Metric             | Weight |
|--------------------|--------|
| Accuracy           | 30%    |
| Ethical Alignment  | 30%    |
| Bias Detection     | 20%    |
| Citation Quality   | 20%    |

### Workflow
1. `_aggregate_scores` - Computes total ranking.
2. `_compute_islamic_metrics` - Adjusts ranking based on Islamic ethics.
3. `_generate_summary` - Identifies top-performing models.
4. `_build_report` - Outputs JSON/CSV for leaderboard.

---

## How to Get Started

1. **Clone the Repository**
   ```bash
   git clone https://github.com/zainkhatri/mcc-genai-guild
   cd mcc-genai-guild
   ```

2. **Enter the lm-evaluation-harness Directory**
   ```bash
   cd lm-evaluation-harness
   ```

3. **Create/Activate Virtual Environment**
   ```bash
   python3 -m venv venv
   source venv/bin/activate
   ```

4. **Install Dependencies**
   ```bash
   pip install -r requirements.txt
   pip install lm-eval[api] anthropic google generativeai openai
   ```

5. **Run an Evaluation**
   ```bash
   cd lm_eval/models
   python evaluate_islamic_model.py
   ```

---

## Conclusion

This project provides a transparent, automated evaluation framework for AI models on Islamic Knowledge & Ethics. By maintaining structured datasets, rigorous benchmarks, and an interactive leaderboard, we ensure responsible AI deployment in Muslim communities.