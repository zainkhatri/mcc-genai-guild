"""DeepSeek model evaluation script with dry run and cost estimation."""
import os
import json
from pathlib import Path
import jsonlines
from datetime import datetime
from dotenv import load_dotenv
import openai
import random
import argparse
import logging
from typing import List, Dict, Any
import time

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(message)s'  # Simplified format for cleaner output
)
logger = logging.getLogger(__name__)

def estimate_tokens(text: str) -> int:
    """Rough token count estimation."""
    return len(text.split()) + len(text) // 4  # Simple heuristic

class DeepSeekEvaluator:
    def __init__(self, api_key: str, dry_run: bool = False):
        """Initialize the evaluator with API key and mode."""
        self.client = openai.OpenAI(
            api_key=api_key,
            base_url="https://api.deepseek.com/v1"
        )
        self.dry_run = dry_run
        self.total_input_tokens = 0
        self.total_output_tokens = 0
        
    def check_credits(self) -> bool:
        """Test API access and credit availability."""
        if self.dry_run:
            return True
            
        try:
            response = self.client.chat.completions.create(
                model="deepseek-chat",
                messages=[{"role": "user", "content": "test"}],
                max_tokens=1
            )
            return True
        except Exception as e:
            if "Insufficient Balance" in str(e):
                logger.error("⚠️ Insufficient balance. Please add credits to your DeepSeek account.")
            else:
                logger.error(f"Error checking credits: {e}")
            return False

    def load_questions(self, data_path: str, sample_size: int = None, language: str = None) -> List[Dict[str, Any]]:
        """Load and optionally sample questions from JSONL file."""
        questions = []
        with jsonlines.open(data_path) as reader:
            for item in reader:
                if language and item.get('language') != language:
                    continue
                questions.append(item)
                
        if sample_size and sample_size < len(questions):
            # Ensure balanced sampling across categories
            categories = {}
            for q in questions:
                cat = q['category']
                if cat not in categories:
                    categories[cat] = []
                categories[cat].append(q)
            
            # Calculate samples per category
            total_cats = len(categories)
            per_cat = max(1, sample_size // total_cats)
            
            # Sample from each category
            sampled = []
            for cat_questions in categories.values():
                sampled.extend(random.sample(cat_questions, min(per_cat, len(cat_questions))))
            
            # If we need more questions to reach sample_size, randomly add more
            while len(sampled) < sample_size and questions:
                remaining = random.choice(questions)
                if remaining not in sampled:
                    sampled.append(remaining)
            
            questions = sampled[:sample_size]
            random.shuffle(questions)
        
        logger.info(f"Loaded {len(questions)} questions" + 
                   (f" in {language}" if language else "") +
                   (f" (sampled from full dataset)" if sample_size else ""))
        
        # In dry run mode, show sample of questions
        if self.dry_run:
            logger.info("\nSample questions:")
            logger.info("="*60)
            for i, q in enumerate(questions[:3], 1):
                logger.info(f"\n{i}. Category: {q['category']}")
                logger.info(f"Question: {q['question']}")
                logger.info(f"Options: {q['options']}")
                logger.info(f"Correct: {q['correct']}")
                logger.info(f"Language: {q.get('language', 'en')}")
            if len(questions) > 3:
                logger.info(f"\n... and {len(questions)-3} more questions")
        
        return questions

    def format_question(self, question: Dict[str, Any]) -> List[Dict[str, str]]:
        """Format a question into chat messages with improved formatting."""
        # Add question mark if missing
        q_text = question['question']
        if not q_text.strip().endswith('?'):
            q_text = q_text + '?'
            
        # Create lettered options with better formatting
        options = question['options'].split(', ')
        options_text = '\n'.join(f"{chr(65 + i)}. {opt.strip()}" 
                               for i, opt in enumerate(options))
        
        messages = [
            {
                "role": "system",
                "content": (
                    "You are evaluating Islamic knowledge questions. Your task is to select "
                    "the correct answer from the multiple choice options provided. Respond "
                    "with ONLY the letter (A, B, C, or D) corresponding to the correct option. "
                    "Do not include any explanation, punctuation, or additional text."
                )
            },
            {
                "role": "user",
                "content": (
                    f"CATEGORY: {question['category']}\n\n"
                    f"QUESTION: {q_text}\n\n"
                    f"OPTIONS:\n{options_text}\n\n"
                    f"Your answer (A/B/C/D):"
                )
            }
        ]
        
        if self.dry_run:
            # Calculate token estimates
            system_tokens = estimate_tokens(messages[0]["content"])
            user_tokens = estimate_tokens(messages[1]["content"])
            total_tokens = system_tokens + user_tokens
            
            logger.info("\nFormatted chat messages:")
            logger.info("-" * 60)
            for msg in messages:
                logger.info(f"\n[{msg['role'].upper()}] ({estimate_tokens(msg['content'])} tokens)")
                logger.info(msg['content'])
            
            logger.info("\nAPI request that would be made:")
            logger.info("-" * 60)
            request = {
                "model": "deepseek-chat",
                "messages": messages,
                "temperature": 0,
                "max_tokens": 1,
                "stop": ["\n", ".", " "]
            }
            logger.info(json.dumps(request, indent=2))
            
            # Show token and cost estimates
            input_cost = total_tokens * 0.14 / 1_000_000  # $0.14 per 1M tokens
            output_cost = 1 * 0.28 / 1_000_000  # $0.28 per 1M tokens
            logger.info("\nEstimated costs:")
            logger.info(f"Input tokens: {total_tokens} (${input_cost:.6f})")
            logger.info(f"Output tokens: 1 (${output_cost:.6f})")
            logger.info(f"Total for this question: ${(input_cost + output_cost):.6f}")
            
            self.total_input_tokens += total_tokens
            self.total_output_tokens += 1
            
        return messages

    def get_correct_option(self, question: Dict[str, Any]) -> str:
        """Get the correct answer letter."""
        options = [opt.strip() for opt in question['options'].split(',')]
        correct = question['correct'].strip()
        
        for i, option in enumerate(options):
            if option == correct:
                return chr(65 + i)
                
        logger.warning(f"Could not find correct answer '{correct}' in options: {options}")
        return 'A'  # Default if not found

    def evaluate_question(self, messages: List[Dict[str, str]], correct_letter: str) -> str:
        """Get model's answer or simulate one in dry run mode."""
        if self.dry_run:
            # In dry run mode, simulate a mix of correct and incorrect answers
            simulated = random.choice([correct_letter, 'A', 'B', 'C', 'D'])
            logger.info(f"\nSimulated response: {simulated}")
            logger.info(f"Correct answer was: {correct_letter}")
            return simulated
            
        for attempt in range(3):
            try:
                response = self.client.chat.completions.create(
                    model="deepseek-chat",
                    messages=messages,
                    temperature=0,
                    max_tokens=1,
                    stop=["\n", ".", " "]
                )
                return response.choices[0].message.content.strip().upper()
            except Exception as e:
                if "Insufficient Balance" in str(e):
                    raise
                logger.warning(f"Attempt {attempt + 1}/3 failed: {e}")
                if attempt < 2:
                    time.sleep(2 ** attempt)  # Exponential backoff
        return ""

    def run_evaluation(self, questions: List[Dict[str, Any]], 
                      output_file: str = None) -> Dict[str, Any]:
        """Run evaluation with progress tracking and cost estimation."""
        if self.dry_run:
            logger.info("\n" + "="*60)
            logger.info("STARTING DRY RUN EVALUATION")
            logger.info("="*60)
        
        evaluated = []
        correct = 0
        category_stats = {}
        total = len(questions)

        if not self.dry_run and not self.check_credits():
            return {}

        for i, question in enumerate(questions, 1):
            logger.info(f"\nProcessing question {i}/{total} "
                       f"[Category: {question['category']}]")
            
            try:
                messages = self.format_question(question)
                correct_letter = self.get_correct_option(question)
                model_answer = self.evaluate_question(messages, correct_letter)
                
                is_correct = model_answer == correct_letter
                if is_correct:
                    correct += 1
                    
                result = {
                    "question": question["question"],
                    "category": question["category"],
                    "language": question.get("language", "en"),
                    "correct_answer": correct_letter,
                    "model_answer": model_answer,
                    "is_correct": is_correct
                }
                evaluated.append(result)
                
                # Update category stats
                cat = question["category"]
                if cat not in category_stats:
                    category_stats[cat] = {"correct": 0, "total": 0}
                category_stats[cat]["total"] += 1
                if is_correct:
                    category_stats[cat]["correct"] += 1
                
            except Exception as e:
                if "Insufficient Balance" in str(e):
                    logger.error("Insufficient balance - saving progress and stopping")
                    break
                logger.error(f"Error processing question {i}: {e}")
                continue

        # Calculate final results
        if evaluated:
            final_accuracy = correct / len(evaluated)
            results = {
                "total_questions": total,
                "questions_evaluated": len(evaluated),
                "correct_answers": correct,
                "accuracy": final_accuracy,
                "category_stats": category_stats,
                "evaluated_questions": evaluated
            }
            
            # Show final results
            logger.info(f"\nEvaluation Results:")
            logger.info("="*60)
            logger.info(f"Overall Accuracy: {final_accuracy:.2%}")
            for cat, stats in category_stats.items():
                cat_acc = stats["correct"] / stats["total"]
                logger.info(f"{cat}: {cat_acc:.2%} ({stats['correct']}/{stats['total']})")
            
            if self.dry_run:
                # Show final cost estimates
                total_input_cost = self.total_input_tokens * 0.14 / 1_000_000
                total_output_cost = self.total_output_tokens * 0.28 / 1_000_000
                total_cost = total_input_cost + total_output_cost
                
                logger.info("\nFinal Cost Estimates:")
                logger.info("="*60)
                logger.info(f"Total input tokens: {self.total_input_tokens} (${total_input_cost:.4f})")
                logger.info(f"Total output tokens: {self.total_output_tokens} (${total_output_cost:.4f})")
                logger.info(f"Total estimated cost: ${total_cost:.4f}")
            
            if output_file and not self.dry_run:
                with open(output_file, 'w') as f:
                    json.dump(results, f, indent=2)
                logger.info(f"\nResults saved to {output_file}")

            return results
        return {}

def main():
    parser = argparse.ArgumentParser(description='Evaluate DeepSeek on Islamic knowledge questions')
    parser.add_argument('--test', type=int, help='Run on a small test set of N questions')
    parser.add_argument('--language', choices=['en', 'tr', 'ar'], help='Evaluate only questions in this language')
    parser.add_argument('--dry-run', action='store_true', help='Preview evaluation without making API calls')
    args = parser.parse_args()

    # Load environment variables
    load_dotenv()
    api_key = os.getenv("DEEPSEEK_API_KEY")
    if not api_key:
        raise ValueError("DeepSeek API key not found in environment variables")

    # Initialize evaluator
    evaluator = DeepSeekEvaluator(api_key, dry_run=args.dry_run)
    
    # Get path to data file
    current_dir = Path(__file__).parent
    data_path = current_dir.parent.parent / "data" / "q_and_a.jsonl"
    
    # Output file for saving progress (only in real run mode)
    output_file = None
    if not args.dry_run:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_file = f"deepseek_evaluation_{timestamp}.json"
    
    # Load questions
    questions = evaluator.load_questions(
        str(data_path), 
        sample_size=args.test,
        language=args.language
    )
    
    if args.dry_run:
        logger.info("\nDRY RUN MODE - No API calls will be made")
    
    # Run evaluation
    logger.info("\nStarting evaluation...")
    evaluator.run_evaluation(questions, output_file)

if __name__ == "__main__":
    main()