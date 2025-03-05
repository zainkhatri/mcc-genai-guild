# Islamic LLM Evaluation Project

## Overview

The **Islamic LLM Evaluation Project** benchmarks AI language models on **Islamic knowledge, ethics, and bias detection**. Using `lm-evaluation-harness`, we test models across structured Islamic Q&A, ethical reasoning, and source reliability. Results are compiled into a **leaderboard** deployed on Hugging Face Spaces.

---

## Evaluation Criteria

Each model is assessed on four key metrics:

- **Islamic Knowledge Accuracy** – Evaluates factual correctness on Quran, Hadith, and Fiqh.
- **Ethical Understanding** – Measures alignment with Islamic social norms and values.
- **Bias Against Islam** – Detects potential biases or misrepresentations.
- **Source Reliability** – Assesses citations from Quran, Hadith, or scholarly texts.

### Weighted Scoring Formula:
```plaintext
Final Score = 
  (Accuracy × 0.6849) + 
  (Ethics × 0.0913) + 
  (Bias × 0.1142) + 
  (Source × 0.1096)
```

## Evaluation Results

| **Model Name** | **Grade** | **Knowledge Accuracy** | **Ethics** | **Bias** | **Source Reliability** |
|----------------|-----------|------------------------|------------|----------|------------------------|
| **GPT-4o** | A+ | 99.33% | 97.50% | 96.00% | 93.75% |
| **GPT-4.5 Preview** | A+ | 98.33% | 97.50% | 96.00% | 95.83% |
| **Claude 3.5 - Sonnet** | A+ | 97.67% | 100.00% | 98.00% | 91.67% |
| **Claude 3.7 - Sonnet** | A+ | 97.67% | 97.50% | 96.00% | 95.83% |
| **Claude 3.5 - Opus** | A+ | 97.33% | 100.00% | 98.00% | 93.75% |
| **Claude 3 - Opus** | A | 95.67% | 100.00% | 98.00% | 91.67% |
| **Gemini 2.0 - Flash** | A | 97.00% | 95.00% | 98.00% | 91.67% |
| **Gemini Flash - 1.5** | C+ | 73.67% | 92.50% | 96.00% | 81.25% |
| **GPT-4 Turbo** | C | 93.67% | 0.00% | 0.00% | 93.75% |

## Running Evaluations

### 1. Setup Environment

```bash
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

### 2. Run a Model Evaluation

```bash
python evaluate_islamic_model.py
```

### 3. Leaderboard Generation

```bash
python run_evaluation.py
```

Generates a comparative ranking based on weighted metrics.

## Repository Structure

```
Islamic-LLM-Eval/
├── .gitignore             # Ignore unnecessary files
├── README.md              # Documentation
├── requirements.txt       # Dependencies
├── evaluate_islamic_model.py  # Core evaluation script
├── models/                # LLM evaluation interfaces
│   ├── openai.py          # GPT model testing
│   ├── google_palm.py     # Gemini model testing
│   ├── huggingface.py     # Open-source model testing
├── data/                  # Evaluation datasets
│   ├── islamic_knowledge.jsonl  # Islamic Q&A
│   ├── ethics.jsonl             # Ethics dataset
└── tasks/                 # Evaluation logic
    ├── islamic_knowledge_task.py
```