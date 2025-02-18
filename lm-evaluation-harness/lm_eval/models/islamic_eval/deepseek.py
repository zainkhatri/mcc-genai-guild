"""DeepSeek-R1 evaluation script for Islamic knowledge testing."""

import json
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, AutoConfig
from datetime import datetime

def force_letter_answer(raw_answer: str) -> str:
    """Convert any answer into A, B, C, or D."""
    answer = raw_answer.strip().upper()
    
    if answer in ['A', 'B', 'C', 'D']:
        return answer
        
    number_map = {
        '1': 'A', '١': 'A', 'ONE': 'A', 'FIRST': 'A',
        '2': 'B', '٢': 'B', 'TWO': 'B', 'SECOND': 'B',
        '3': 'C', '٣': 'C', 'THREE': 'C', 'THIRD': 'C',
        '4': 'D', '٤': 'D', 'FOUR': 'D', 'FOURTH': 'D'
    }
    
    for key in number_map:
        if key in answer:
            return number_map[key]
    
    return 'A'

def test_model(model_name="deepseek-ai/DeepSeek-R1"):
    try:
        print(f"\nTesting {model_name} capabilities...")
        device = "cuda" if torch.cuda.is_available() else "cpu"
        
        print("Loading config...")
        config = AutoConfig.from_pretrained(
            model_name,
            trust_remote_code=True
        )
        # Disable quantization config to avoid FP8 issues
        if hasattr(config, 'quantization_config'):
            delattr(config, 'quantization_config')
        
        print("Loading tokenizer...")
        tokenizer = AutoTokenizer.from_pretrained(
            model_name,
            trust_remote_code=True,
            use_fast=True
        )
        
        print("Loading model...")
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            config=config,
            trust_remote_code=True,
            device_map="auto",           # Let HF handle multi-GPU distribution
            torch_dtype=torch.bfloat16,  # Use bfloat16 instead of fp8
            low_cpu_mem_usage=True,
            offload_folder="offload"     # Use disk offloading if needed
        )
        model.eval()

        with open('../../data/q_and_a.jsonl', 'r') as f:
            qa_pairs = [json.loads(line) for line in f]
        
        correct = 0
        total = len(qa_pairs)
        results = []
        
        print(f"\nEvaluating {total} questions...")
        
        for i, qa in enumerate(qa_pairs, 1):
            print(f"\nQ{i}/{total}: {qa['question']}")
            print(f"Category: {qa['category']}")
            print(f"Options: {qa['options']}")
            print(f"Correct: {qa['correct']}")
            
            prompt = f"""<think>
Question: {qa['question']}
Choices: {qa['options']}
You must respond with exactly one letter: A, B, C, or D. No other response is allowed.
Respond with a single letter only.
</think>
Answer: """

            inputs = tokenizer(prompt, return_tensors="pt").to(device)
            
            with torch.no_grad():
                outputs = model.generate(
                    **inputs,
                    max_new_tokens=5,
                    temperature=0.6,
                    top_p=0.95,
                    pad_token_id=tokenizer.pad_token_id
                )
            
            response = tokenizer.decode(outputs[0], skip_special_tokens=True)
            raw_answer = response.split("Answer:")[-1].strip()
            
            answer = force_letter_answer(raw_answer)
            print(f"Model raw output: {raw_answer}")
            print(f"Forced answer: {answer}")
            
            options_list = [opt.strip() for opt in qa['options'].split(',')]
            try:
                correct_idx = options_list.index(qa['correct'])
                correct_letter = chr(65 + correct_idx)
                
                is_correct = answer == correct_letter
                if is_correct:
                    correct += 1
                    print("✓ CORRECT")
                else:
                    print("✗ WRONG")
                
                results.append({
                    'question_id': i,
                    'category': qa['category'],
                    'question': qa['question'],
                    'options': qa['options'],
                    'correct_answer': correct_letter,
                    'model_answer': answer,
                    'raw_model_output': raw_answer,
                    'is_correct': is_correct
                })
            except ValueError as e:
                print(f"Warning: Could not determine correct answer index - {e}")
                continue
        
        accuracy = (correct/total)*100
        
        print(f"\nFinal Results:")
        print(f"Total questions: {total}")
        print(f"Accuracy: {accuracy:.1f}% ({correct}/{total} correct)")

        results_file = f"deepseek_r1_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        with open(results_file, 'w') as f:
            json.dump({
                'model_name': model_name,
                'total_questions': total,
                'correct_answers': correct,
                'accuracy': accuracy,
                'results': results
            }, f, indent=2)

    except Exception as e:
        print(f"Error: {str(e)}")
        raise e

if __name__ == "__main__":
    test_model()