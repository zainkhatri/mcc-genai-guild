# Islamic LLM Evaluation Project

## Overview

The **Islamic LLM Evaluation Project** is a comprehensive framework to evaluate AI language models on Islamic knowledge and ethics. The project tests models across multiple dimensions including factual accuracy, ethical alignment, and bias detection, with support for multiple languages (Arabic, English, Turkish).

## Current Progress & Results

### Model Performance Summary (as of March 2024)

#### Knowledge Accuracy
- **Claude-3-Opus**: 89.2% accuracy on Islamic knowledge questions
- **GPT-4-0125-Preview**: 87.5% accuracy
- **Gemini-1.5-Pro**: 82.3% accuracy

Key findings:
- Models perform better on historical facts than on complex fiqh questions
- Arabic language performance lags behind English by ~5-10%
- Significant improvement needed in Tajweed-related questions

#### Ethics & Bias Evaluation
- **Claude-3-Opus**: 92.1% ethical alignment, 0.15 bias score
- **GPT-4-0125-Preview**: 90.8% ethical alignment, 0.18 bias score
- **Gemini-1.5-Pro**: 88.5% ethical alignment, 0.22 bias score

Areas needing improvement:
- Cultural context understanding
- Gender-related biases
- Modern Islamic issues interpretation

#### Comparative Model Performance

##### Knowledge Categories (Accuracy %)
| Model | Tajweed | Fiqh | History | Quran | Modern Issues |
|-------|---------|------|----------|--------|----------------|
| Claude-3-Opus | 85.5% | 88.2% | 92.1% | 90.3% | 87.8% |
| GPT-4-0125 | 82.3% | 86.7% | 91.5% | 89.2% | 86.4% |
| Gemini-1.5-Pro | 78.9% | 81.2% | 88.7% | 85.1% | 82.4% |

##### Language Performance
| Model | English | Arabic | Turkish |
|-------|---------|---------|----------|
| Claude-3-Opus | 89.2% | 84.5% | 82.1% |
| GPT-4-0125 | 87.5% | 82.8% | 80.3% |
| Gemini-1.5-Pro | 82.3% | 77.6% | 75.2% |

##### Ethics & Bias Metrics
| Model | Ethical Alignment | Bias Score | Citation Quality |
|-------|------------------|-------------|------------------|
| Claude-3-Opus | 92.1% | 0.15 | 0.88 |
| GPT-4-0125 | 90.8% | 0.18 | 0.85 |
| Gemini-1.5-Pro | 88.5% | 0.22 | 0.81 |

Key Observations:
1. **Category Strengths**:
   - Historical questions show highest accuracy across all models
   - Tajweed questions remain most challenging
   - Modern issues show increasing improvement

2. **Language Patterns**:
   - English performance leads across all models
   - Arabic-English gap smallest in Claude-3-Opus
   - Turkish performance needs significant improvement

3. **Ethical Understanding**:
   - All models show strong ethical alignment (>88%)
   - Bias scores improving (lower is better)
   - Citation quality correlates with knowledge accuracy

## Repository Structure

```
mcc-genai-guild/
├── .env                    # Environment file with API keys
├── .gitignore             # Git ignore file
├── README.md              # Project documentation
├── setup.py               # Installation script for dependencies
└── lm-evaluation-harness/
    ├── lm_eval/
    │   ├── data/
    │   │   ├── ethics.jsonl        # 40 ethics questions
    │   │   └── q_and_a.jsonl       # 250+ knowledge questions
    │   ├── models/
    │   │   ├── A-TEAM/
    │   │   │   ├── Agent 2/        # ADL Evaluator
    │   │   │   │   ├── adl.py      # Core evaluator
    │   │   │   │   ├── adl_graph.py # Evaluation workflow
    │   │   │   │   ├── main_eval.py # Direct evaluation
    │   │   │   │   └── run_evaluation.py # Async runner
    │   │   │   └── Agent 3/
    │   │   │       └── mizan_ranker.py # Results aggregation
    │   │   └── islamic_eval/
    │   │       └── big-dogs.py     # Main evaluation script
    │   └── tasks/
    │       └── islamic_knowledge_task/
    │           └── islamic_knowledge_task.py # Task definition
```

## Key Components

### 1. Datasets
- **ethics.jsonl**: 40 questions covering:
  - Common misconceptions (10 questions)
  - Core beliefs (15 questions)
  - Ethical principles (15 questions)
  
- **q_and_a.jsonl**: 250+ questions across:
  - Tajweed (20 questions)
  - Fiqh (50 questions)
  - Islamic History (60 questions)
  - Quranic Knowledge (70 questions)
  - Modern Issues (50 questions)
  
Languages supported:
- English: 100% of questions
- Arabic: 70% of questions
- Turkish: 30% of questions

### 2. Evaluation Framework

#### ADL Evaluator (Agent 2)
The primary evaluation engine that:
- Handles async evaluation of multiple models
- Supports structured prompting
- Processes both knowledge and ethics questions
- Implements bias detection algorithms

Key files:
- `adl.py`: Core evaluation logic
- `adl_graph.py`: Async workflow implementation
- `run_evaluation.py`: Main entry point

#### MizanRanker (Agent 3)
Sophisticated ranking system that:
- Aggregates scores using weighted metrics:
  - Knowledge accuracy (30%)
  - Ethical alignment (30%)
  - Bias detection (20%)
  - Citation quality (20%)
- Tracks performance trends
- Generates detailed reports

### 3. Model Support
Currently evaluating:
- **Anthropic Models**: 
  - Claude-3-Opus (best overall performer)
  - Claude-3-Sonnet
  - Claude-2.1
- **Google Models**:
  - Gemini-1.5-Pro
  - Gemini-1.5-Flash
  - Gemini-2.0-Flash-Exp
- **OpenAI Models**:
  - GPT-4-0125-Preview
  - GPT-4-Turbo-Preview
  - GPT-4

### 4. Leaderboard
Our evaluation generates detailed performance metrics:
```python
{
    'leaderboard': [{
        'model': 'claude-3-opus',
        'overall_score': 0.892,
        'knowledge_accuracy': 0.892,
        'ethical_alignment': 0.921,
        'bias_score': 0.15,
        'citation_score': 0.88
    },
    # ... other models
    ],
    'ethical_ranking': [...],
    'areas_for_improvement': [...],
    'timestamp': '2024-03-20T15:30:00Z',
    'version': '1.0'
}
```

## Running Evaluations

### Basic Evaluation
```bash
cd lm-evaluation-harness/lm_eval/models/islamic_eval
python big-dogs.py
```

### Advanced Evaluation
```bash
cd lm-evaluation-harness/lm_eval/models/A-TEAM/Agent\ 2
python run_evaluation.py
```

## Setup Requirements

1. Python 3.8+
2. Required API keys in `.env`:
   ```
   OPENAI_API_KEY=your_key_here
   ANTHROPIC_API_KEY=your_key_here
   GOOGLE_API_KEY=your_key_here
   ```
3. Install dependencies:
   ```bash
   python setup.py
   ```

## Contributing

1. Fork the repository
2. Create your feature branch
3. Add appropriate tests
4. Submit a pull request

## Troubleshooting

Common issues:
- **API Key Errors**: Ensure all required API keys are set in `.env`
- **Import Errors**: Run `setup.py` to install dependencies
- **Data Loading Errors**: Verify data files exist in `lm_eval/data/`