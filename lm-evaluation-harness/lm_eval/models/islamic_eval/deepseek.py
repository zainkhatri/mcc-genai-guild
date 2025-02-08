"""DeepSeek evaluation script for Islamic knowledge testing."""

import json
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from datetime import datetime

def test_model(model_name="deepseek-ai/DeepSeek-R1"):
    try:
        print(f"\nTesting {model_name} capabilities...")
        print(f"GPU Available: {torch.cuda.is_available()}")
        if torch.cuda.is_available():
            print(f"GPU Count: {torch.cuda.device_count()}")
            print(f"GPU Names: {[torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())]}")
            print(f"GPU Memory: {[torch.cuda.get_device_properties(i).total_memory/(1024**3) for i in range(torch.cuda.device_count())]} GB")
        
        device = "cuda" if torch.cuda.is_available() else "cpu"
        
        print("Loading tokenizer...")
        tokenizer = AutoTokenizer.from_pretrained(
            model_name,
            trust_remote_code=True,
            use_fast=True
        )
        
        print("Loading model...")
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            trust_remote_code=True,
            torch_dtype=torch.bfloat16,  # DeepSeek uses BF16
            device_map="auto",           # Let HF handle multi-GPU distribution
        )
        model.eval()  # Set to evaluation mode
        
        # Load test data
        print("Loading test data...")
        with open('../../data/q_and_a.jsonl', 'r') as f:
            qa_pairs = [json.loads(line) for line in f]
        
        correct = 0
        total = len(qa_pairs)
        results = []
        
        print(f"\nTesting all {total} questions...")
        start_time = datetime.now()
        
        for i, qa in enumerate(qa_pairs, 1):
            print(f"\nQ{i}/{total}: {qa['question']}")
            print(f"Category: {qa['category']}")
            print(f"Options: {qa['options']}")
            print(f"Correct: {qa['correct']}")
            
            # Format prompt to encourage single-letter response
            prompt = f"""Question: {qa['question']}
Choices: {qa['options']}
Please provide only a single letter (A, B, C, or D) as your answer.
Answer: """

            inputs = tokenizer(prompt, return_tensors="pt").to(device)
            
            with torch.no_grad():
                outputs = model.generate(
                    **inputs,
                    max_new_tokens=5,        # We only need a short answer
                    num_return_sequences=1,
                    temperature=0.1,         # Low temperature for more focused answers
                    do_sample=False,         # Greedy decoding
                    pad_token_id=tokenizer.pad_token_id
                )
            
            response = tokenizer.decode(outputs[0], skip_special_tokens=True)
            
            # Extract just the model's answer - look for first letter after "Answer:"
            try:
                answer = response.split("Answer:")[-1].strip().upper()
                # Take first letter if multiple characters
                answer = ''.join(c for c in answer if c.isalpha())[:1]
            except:
                answer = "ERROR"
            
            print(f"Model answer: {answer}")
            
            # Check if answer matches correct option
            # Convert correct answer to letter (e.g., if correct="qaaf", find its position in options)
            options_list = [opt.strip() for opt in qa['options'].split(',')]
            correct_idx = options_list.index(qa['correct'])
            correct_letter = chr(65 + correct_idx)  # Convert to A, B, C, D
            
            is_correct = answer == correct_letter
            
            if is_correct:
                correct += 1
                print("✓ CORRECT")
            else:
                print("✗ WRONG")
            
            # Store result
            results.append({
                'question_id': i,
                'category': qa['category'],
                'question': qa['question'],
                'options': qa['options'],
                'correct_answer': correct_letter,
                'model_answer': answer,
                'is_correct': is_correct
            })
        
        end_time = datetime.now()
        duration = (end_time - start_time).total_seconds()
        accuracy = (correct/total)*100
        
        print(f"\nFinal Results:")
        print(f"Accuracy: {accuracy:.1f}% ({correct}/{total} correct)")
        print(f"Total time: {duration:.1f} seconds")
        print(f"Average time per question: {duration/total:.1f} seconds")
        
        # Save detailed results
        results_file = f"deepseek_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        with open(results_file, 'w') as f:
            json.dump({
                'model_name': model_name,
                'accuracy': accuracy,
                'total_time': duration,
                'correct_count': correct,
                'total_questions': total,
                'results': results
            }, f, indent=2)
        
        print(f"\nDetailed results saved to {results_file}")
        
    except Exception as e:
        print(f"Error: {str(e)}")
        raise e

if __name__ == "__main__":
    test_model()