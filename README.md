# Islamic GenAI Guild

Welcome to the **Islamic GenAI Guild**! This project is all about using AI responsibly to support Islamic knowledge and principles. We're building datasets, grading AI tools, and evaluating how well they handle Islamic topics like the Quran, Hadith, ethics, and more.

## What We're Doing (subject for approval/change)

1. **Creating Datasets**  
   - Collecting Q&A pairs about Islamic topics like Fiqh, Aqeedah, and contemporary issues.  
   - Public datasets for training, hidden ones for testing.

2. **Grading AI Tools**  
   - Testing tools like GPT-4 and others on accuracy, ethics, and bias detection.  
   - Giving them grades (A, B, C…) and showing how they can improve.

3. **Making a Leaderboard**  
   - Ranking AI tools based on their performance.  
   - Showing which ones are best for specific topics, like Quranic Tafsir or Islamic history.

## Setting Up the Virtual Environment

Follow these steps to set up and run the virtual environment:

1. **Navigate to the Project Directory**  
   ```bash
   cd lm-evaluation-harness
   ```

2. **Create a Virtual Environment**  
   ```bash
   python3 -m venv venv
   ```

3. **Activate the Virtual Environment**  
   ```bash
   source venv/bin/activate
   ```

4. **Install Required Dependencies**  
   ```bash
   pip install -r requirements.txt
   pip install lm-eval[api]
   pip install anthropic google generativeai openai
   ```

5. **Navigate to the Model Evaluation Directory**  
   ```bash
   cd lm_eval/models
   ```

6. **Run the Islamic Model Evaluation Script**  
   ```bash
   python evaluate_islamic_model.py
   ```

Now you're all set to evaluate Islamic models! Let us know if you encounter any issues or have suggestions to improve the process.
