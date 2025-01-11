# Comparative Evaluation of Islamic Knowledge Q&A

This document presents the results of our comparative evaluation of several language models (LMs) on an Islamic Knowledge Q&A task. The goal is to gauge how accurately these models can answer questions related to Islamic topics (e.g., Qur’an, Hadith, and Islamic jurisprudence). 

We used the **lm-evaluation-harness** framework to ensure consistency and reproducibility. Each model received the same 50 questions and was prompted to answer with exactly one letter (A, B, C, or D). The correct answer for each question was known in advance, allowing us to measure straightforward **Accuracy** (correct / total questions).

---

## Evaluation Results

| **Model Name**               | **Accuracy** | **Grade** |
|------------------------------|--------------|-----------|
| Gemini 1.5 Pro               | 96.00%       | A         |
| GPT-4 O1                     | 94.00%       | A         |
| GPT-4 Turbo                  | 92.00%       | A-        |
| GPT-4                        | 92.00%       | A-        |
| Claude 3 Opus                | 92.00%       | A-        |
| Gemini 1.5 Flash             | 84.00%       | B         |
| Claude 3 Sonnet              | 76.00%       | C         |
| Claude 2.1                   | 72.00%       | C-        |
| Gemini 2.0 Flash (Beta)      | 28.00%       | F         |


## Highlights and Observations

- **Gemini 1.5 Pro** achieved the highest overall accuracy (96%), showing strong performance on Islamic Q&A.
- **OpenAI GPT-4** variations clustered between 92% and 94%, indicating consistent accuracy across its versions.
- **Claude 3 Opus** reached 92%, matching GPT-4 Turbo’s performance.
- **Gemini 2.0 Flash (Beta)** scored significantly lower (28%), suggesting it may be early in development and not yet optimized for tasks like this.

---

## Conclusion

This comparative study demonstrates that modern large language models can achieve high accuracy on specialized Islamic Knowledge questions, with **Gemini 1.5 Pro** and **GPT-4 O1** leading the pack. However, results can vary widely depending on the maturity and fine-tuning of each model. 

By sharing these findings, we aim to empower the community to select the best models for Islamic educational and research purposes and to continue improving them through further fine-tuning.