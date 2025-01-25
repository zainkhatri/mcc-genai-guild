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