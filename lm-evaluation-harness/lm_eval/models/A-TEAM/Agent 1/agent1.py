import json
import openai
import os
from dotenv import load_dotenv
from langgraph.graph import StateGraph
from typing import Dict, List, Any, TypedDict

# Load environment variables from .env file
load_dotenv()

# Set OpenAI API Key from environment variable
api_key = os.getenv("OPENAI_API_KEY")
if not api_key:
    raise ValueError("❌ OPENAI_API_KEY is missing. Set it in your .env file or environment variables.")

openai.api_key = api_key

# Define the state schema for LangGraph
class TafakkurSynthesizerState(TypedDict):
    dataset_name: str
    num_samples: int
    generated_data: List[Dict[str, Any]]

# Function to generate synthetic Q&A using OpenAI API
def generate_synthetic_data(dataset_name: str, num_samples: int) -> List[Dict[str, Any]]:
    """
    Uses an LLM (GPT-4) to generate synthetic Islamic Q&A data based on user input.
    """
    messages = [
        {"role": "system", "content": "You are an Islamic scholar generating factual Q&A pairs on Islamic teachings. "
                                      "You MUST return output as a valid JSON array. "
                                      "Do NOT include explanations, text, or additional notes—only the JSON array."},
        {"role": "user", "content": f"Generate {num_samples} Q&A pairs for a dataset about {dataset_name}. "
                                    "Format the response as a JSON array of objects, where each object has 'question' and 'answer' fields. "
                                    "Ensure the output is valid JSON. Example output:\n\n"
                                    "[{{\"question\": \"What is Islam?\", \"answer\": \"Islam is a monotheistic religion...\"}}, "
                                    "{{\"question\": \"Who is the last prophet in Islam?\", \"answer\": \"Prophet Muhammad (PBUH) is the last prophet.\"}}]"}
    ]

    response = openai.chat.completions.create(
        model="gpt-4o",
        messages=messages,
        max_tokens=5000
    )

    # Extract response text and parse it as JSON
    try:
        generated_data = json.loads(response.choices[0].message.content)
    except json.JSONDecodeError:
        print("❌ Error: The AI did not return valid JSON. Please check the model output.")
        return []

    return generated_data

# Define Tafakkur Synthesizer class
class TafakkurSynthesizer:
    def __init__(self, dataset_name: str, num_samples: int):
        self.dataset_name = dataset_name
        self.num_samples = num_samples
    
    def synthesize(self) -> List[Dict[str, Any]]:
        return generate_synthetic_data(self.dataset_name, self.num_samples)

# Define the LangGraph node function
def tafakkur_synthesizer_node(state: TafakkurSynthesizerState) -> TafakkurSynthesizerState:
    synthesizer = TafakkurSynthesizer(dataset_name=state["dataset_name"], num_samples=state["num_samples"])
    generated_data = synthesizer.synthesize()

    # Set the save path inside /lm_eval/data/
    dataset_filename = state["dataset_name"].replace(" ", "_").lower() + ".jsonl"
    save_path = os.path.join("/Users/fizaomer/islamic_llm_eval/mcc-genai-guild/lm-evaluation-harness/lm_eval/data/", dataset_filename)

    # Save to a JSONL file
    with open(save_path, "w", encoding="utf-8") as file:
        for entry in generated_data:
            file.write(json.dumps(entry) + "\n")

    print(f"✅ Dataset saved as: {save_path}")

    return {"dataset_name": state["dataset_name"], "num_samples": state["num_samples"], "generated_data": generated_data}

# ✅ Define LangGraph workflow
tafakkur_graph = StateGraph(TafakkurSynthesizerState)

# Add Tafakkur Synthesizer as a node
tafakkur_graph.add_node("tafakkur_synthesizer", tafakkur_synthesizer_node)

# Set entry and finish points
tafakkur_graph.set_entry_point("tafakkur_synthesizer")
tafakkur_graph.set_finish_point("tafakkur_synthesizer")  # ✅ Correct API usage

# Compile the graph
tafakkur_workflow = tafakkur_graph.compile()

# 🔹 Run an example generation
if __name__ == "__main__":
    dataset_name = input("Enter the dataset name (e.g., 'Islamic Ethics', 'Hadith Studies', 'Islamic History'): ")
    num_samples = int(input("Enter the number of Q&A pairs to generate: "))

    result = tafakkur_workflow.invoke({"dataset_name": dataset_name, "num_samples": num_samples})
    print(json.dumps(result, indent=4))  # Pretty-print JSON output
