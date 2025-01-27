import json
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

def test_model(model_name="HuggingFaceH4/zephyr-7b-beta"):
    try:
        print(f"\nTesting {model_name} raw capabilities...")
        device = "cuda" if torch.cuda.is_available() else "cpu"
        
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            trust_remote_code=True,
            torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
            device_map="auto"
        )
        
        # Load ALL test data
        with open('../../data/q_and_a.jsonl', 'r') as f:
            qa_pairs = [json.loads(line) for line in f]
        
        correct = 0
        total = len(qa_pairs)
        
        print(f"Testing all {total} questions...")
        
        for i, qa in enumerate(qa_pairs, 1):
            print(f"\nQ{i}: {qa['question']}")
            print(f"Category: {qa['category']}")
            print(f"Options: {qa['options']}")
            print(f"Correct: {qa['correct']}")
            
            prompt = f"Q: {qa['question']}\nOptions: {qa['options']}\nA:"
            
            inputs = tokenizer(prompt, return_tensors="pt").to(device)
            outputs = model.generate(**inputs, max_new_tokens=20)
            response = tokenizer.decode(outputs[0]).lower()
            
            # Extract just the model's answer
            try:
                answer = response.split("a:")[1].split("\n")[0].strip()
            except:
                answer = "ERROR: Could not parse answer"
            
            print(f"Model answer: {answer}")
            
            # Check if answer is exactly correct (ignoring case)
            is_correct = qa['correct'].lower().strip() == answer
            
            if is_correct:
                correct += 1
                print("✓ CORRECT")
            else:
                print("✗ WRONG")
        
        print(f"\nFinal accuracy: {(correct/total)*100:.1f}% ({correct}/{total} correct)")
        
    except Exception as e:
        print(f"Error: {e}")

if __name__ == "__main__":
    test_model()