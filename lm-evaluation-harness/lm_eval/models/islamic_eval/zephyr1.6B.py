import json
import torch
import os
from pathlib import Path
from transformers import AutoModelForCausalLM, AutoTokenizer

def find_data_file(filename):
    """Find the data file in the lm_eval/data directory"""
    current = Path(os.getcwd())
    while current.name != 'lm-evaluation-harness' and current.parent != current:
        current = current.parent
    
    data_path = current / 'lm_eval' / 'data' / filename
    if data_path.exists():
        return str(data_path)
    raise FileNotFoundError(f"Could not find {filename}")

def process_qa_data(json_lines, num_samples=5):
    """Process JSON lines into Q&A format"""
    qa_pairs = []
    for line in json_lines.strip().split('\n')[:num_samples]:
        try:
            data = json.loads(line)
            qa_pairs.append({
                'question': data['question'],
                'options': data['options'],
                'correct': data['correct'],
                'category': data.get('category', 'Unknown')
            })
        except json.JSONDecodeError:
            continue
    return qa_pairs

def test_model(model_name="stabilityai/stablelm-2-zephyr-1_6b", num_samples=5):
    """Test the model with simple Q&A format"""
    try:
        print(f"\nLoading {model_name}...")
        device = "cuda" if torch.cuda.is_available() else "cpu"
        
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
            
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            trust_remote_code=True,
            torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
            device_map="auto" if torch.cuda.is_available() else None
        ).to(device)
        
        print(f"Successfully loaded {model_name}")
        
        # Get QA pairs from the data using correct path
        data_file = find_data_file('q_and_a.jsonl')
        print(f"Found data file at: {data_file}")
        
        with open(data_file, 'r', encoding='utf-8') as f:
            qa_data = f.read()
            qa_pairs = process_qa_data(qa_data, num_samples)
        
        correct = 0
        total = len(qa_pairs)
        
        print(f"\nTesting with {total} questions:")
        
        for i, qa in enumerate(qa_pairs, 1):
            print(f"\nQuestion {i}:")
            print(f"Category: {qa['category']}")
            print(f"Question: {qa['question']}")
            print(f"Options: {qa['options']}")
            print(f"Correct Answer: {qa['correct']}")
            
            prompt = (
                f"Question: {qa['question']}\n"
                f"Options: {qa['options']}\n"
                f"Answer with only the correct option without any explanation: "
            )
            
            inputs = tokenizer(prompt, return_tensors="pt").to(device)
            
            with torch.no_grad():
                outputs = model.generate(
                    **inputs,
                    max_new_tokens=10,
                    do_sample=True,
                    temperature=0.1,
                    top_p=0.95,
                    num_beams=5,
                    pad_token_id=tokenizer.pad_token_id,
                    eos_token_id=tokenizer.eos_token_id,
                )
            
            response = tokenizer.decode(outputs[0], skip_special_tokens=True)
            if response.startswith(prompt):
                response = response[len(prompt):]
            response = response.strip().lower()
            
            # Clean up response to get just the answer
            response = response.split('\n')[0]
            response = response.split('.')[0]
            response = response.strip('" \',.;:')
            
            # Check if answer matches any word in the correct answer
            is_correct = qa['correct'].lower() in response
            if is_correct:
                correct += 1
            
            print(f"Model response (cleaned): {response}")
            print(f"Correct? {'✓' if is_correct else '✗'}")
        
        accuracy = (correct / total) * 100
        print(f"\nAccuracy on {total} samples: {accuracy:.2f}%")
        
    except Exception as e:
        print(f"Error during testing: {str(e)}")
        raise

if __name__ == "__main__":
    test_model(num_samples=300)