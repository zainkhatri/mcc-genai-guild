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

# MCC Evaluation Project: Steps to the Leaderboard

## **Step 1: Setting Things Up**
- **Creating a Clean Workspace:**  
  We made a virtual environment (venv), which is like a separate room for our project. It keeps everything organized and avoids messing with other parts of the computer.
- **Installing Tools:**  
  We added the tools we needed, like `lm-eval` (to evaluate models), `openai` (to connect to AI), and `python-dotenv` (to manage secret keys).

---

## **Step 2: Organizing the Project**
- **Building the Foundation:**  
  We arranged everything neatly in folders:
  - A script (`evaluate_islamic_model.py`) to run our tests.
  - A special task file (`islamic_knowledge_task.py`) to define how we test the AI.
  - A dataset (`islamic_knowledge.jsonl`) with our questions and answers.

---

## **Step 3: Creating the Testing Framework**
- **Designing the Test:**  
  We built a task that:
  - Reads our list of questions and correct answers.
  - Formats them into multiple-choice questions.
  - Scores how well the AI answers.
- **Integrating It All:**  
  We plugged this task into our testing system so it works seamlessly.

---

## **Step 4: Testing the Models**
- **Connecting to AI Models:**  
  We set up keys to talk to different AI models, like GPT-4 and Gemini.
- **Running the Questions:**  
  Each AI was asked the same set of questions and their answers were collected.
- **Helping the AI Learn:**  
  We gave each model a few example questions to show how it should respond.

---

## **Step 5: Collecting the Results**
- **Scoring the AI:**  
  We calculated how many questions each model got right (Accuracy).
- **Finding the Best:**  
  We noted which models did well (like Gemini 1.5 Pro with 96%) and which struggled (like Gemini 2.0 Beta with 28%).

---

## **Step 6: Creating the Leaderboard**
- **Ranking the Models:**  
  We organized the results into a leaderboard to show how the models compare.
- **Sharing It:**  
  We put the leaderboard online using Hugging Face Spaces so others can see it too.

---

## **In Simple Terms:**
1. We created a clean workspace to keep things organized.
2. We set up the project with tools, scripts, and a dataset.
3. We made a testing system to ask questions and check answers.
4. We tested different AI models and collected their scores.
5. We ranked the models based on their performance.
6. We shared the results online so everyone can see which models are the best.

That’s how we built the leaderboard!

## **Top-Level Files**
- **`.env`**:  
  Stores API keys and sensitive information. This file is excluded from version control for security.

- **`.gitignore`**:  
  Specifies files and folders to exclude from Git tracking, such as `.env` and `venv/`.

---

## **Folder: `lm-evaluation-harness/`**

### **Scripts and Code**
- **`lm_eval/models/evaluate_islamic_model.py`**:  
  The main script that runs model evaluations. It loads AI models, processes Islamic knowledge questions, evaluates answers, and generates results.

- **`lm_eval/tasks/islamic_knowledge_task/islamic_knowledge_task.py`**:  
  Defines the custom task for evaluating Islamic knowledge. It loads questions, formats them, evaluates model responses, and calculates scores.

- **`lm_eval/tasks/islamic_knowledge_task/__init__.py`**:  
  Marks the `islamic_knowledge_task/` folder as a Python package, enabling imports for the evaluation system.

### **Data**
- **`lm_eval/data/islamic_knowledge.jsonl`**:  
  Contains the dataset of Islamic knowledge questions and answers in JSONL format. Each entry includes a question, multiple-choice options, and the correct answer.

---

## **Folder: `venv/`**
- Contains the virtual environment, which includes an isolated Python setup with all required libraries and dependencies for the project.

---

## **Summary:**
These files and folders work together to enable the evaluation of AI models on Islamic knowledge tasks, from dataset preparation to leaderboard generation and deployment.

## Conclusion

This comparative study demonstrates that modern large language models can achieve high accuracy on specialized Islamic Knowledge questions, with **Gemini 1.5 Pro** and **GPT-4 O1** leading the pack. However, results can vary widely depending on the maturity and fine-tuning of each model. 

By sharing these findings, we aim to empower the community to select the best models for Islamic educational and research purposes and to continue improving them through further fine-tuning.gi