import json
import pandas as pd
from langgraph.graph import StateGraph
from typing import Dict, List, Any, TypedDict

# Define the state schema for LangGraph
class IlmExtractorState(TypedDict):
    source_type: str
    file_path: str
    extracted_data: List[Dict[str, Any]]

# Function to extract data from JSONL
def extract_data_from_jsonl(file_path: str) -> List[Dict[str, Any]]:
    """
    Reads a JSONL file and returns a list of dictionaries.
    """
    data = []
    with open(file_path, "r", encoding="utf-8") as file:
        for line in file:
            data.append(json.loads(line.strip()))  # Convert each line to a JSON object
    return data

# Define Ilm Extractor class
class IlmExtractor:
    def __init__(self, source_type: str, file_path: str):
        self.source_type = source_type
        self.file_path = file_path
    
    def extract(self) -> List[Dict[str, Any]]:
        if self.source_type == "jsonl":
            return extract_data_from_jsonl(self.file_path)
        else:
            raise ValueError("Unsupported file type. Use 'jsonl'.")

# Define the LangGraph node function
def ilm_extractor_node(state: IlmExtractorState) -> IlmExtractorState:
    extractor = IlmExtractor(source_type=state["source_type"], file_path=state["file_path"])
    extracted_data = extractor.extract()
    return {"source_type": state["source_type"], "file_path": state["file_path"], "extracted_data": extracted_data}

# ✅ Fix: Define the graph with a state schema
ilm_graph = StateGraph(IlmExtractorState)

# Add Ilm Extractor as a node
ilm_graph.add_node("ilm_extractor", ilm_extractor_node)

# Set entry and finish points
ilm_graph.set_entry_point("ilm_extractor")
ilm_graph.set_finish_point("ilm_extractor")  # ✅ Correct API usage

# Compile the graph
ilm_workflow = ilm_graph.compile()

# Run an example extraction
if __name__ == "__main__":
    result = ilm_workflow.invoke({"source_type": "jsonl", "file_path": "/Users/fizaomer/islamic_llm_eval/mcc-genai-guild/lm-evaluation-harness/lm_eval/data/bias_detection.jsonl"})
    print(json.dumps(result, indent=4))  # Pretty-print JSON output
