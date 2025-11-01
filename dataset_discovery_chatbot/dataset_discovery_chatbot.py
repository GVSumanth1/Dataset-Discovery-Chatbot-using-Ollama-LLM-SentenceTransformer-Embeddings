"""


pip install sentence-transformers
python -m pip install sentence-transformers
python use_case_1.pyIm



====================================================================================
Dataset Discovery Chatbot using Ollama LLM + SentenceTransformer Embeddings
====================================================================================
Use Case:
---------
This Python script implements an interactive chatbot that helps users find 
relevant datasets from a given metadata configuration. It combines:
  • Ollama’s LLM (e.g., Llama 3.1:8B) for natural language understanding and response generation.
  • SentenceTransformer embeddings (all-MiniLM-L6-v2) for semantic similarity matching.

Workflow:
---------
1. The chatbot greets the user.
2. It asks the user for their data-related query.
3. It searches through metadata (list of dataset names + descriptions) 
   to find the most semantically relevant datasets.
4. The chatbot suggests top dataset matches to the user or asks for clarification.
5. The user can continue chatting or exit anytime.

TASKS Performed:
-------------

1. Ensure the LLM does NOT suggest any dataset names that do NOT exist in `metadata.json`.
2. Set the temperature to 1 [higher is more creative, lower is more coherent].
3. Set the system prompt for example to define chatbot behavior more clearly. 
4. Try 2 or 3 different LLM models from Ollama and compare results.
5. Log chat history and matched datasets.
    - Store interactions in a simple JSONL or SQLite file for analysis.
6. Adjust similarity thresholds for "perfect" vs "close" matches based on user feedback.

"""


import json
import ollama
import datetime
import jsonlines
from sentence_transformers import SentenceTransformer, util

# ====== Configuration ======
metadata_filename = "config/metadata.json"
prompts_filename = "config/prompts.json"
log_filename = "logs/chat_history.jsonl"  # JSONL file for storing chat history

LLM_MODEL = "llama3.1:8b"  
EMBEDDING_MODEL = SentenceTransformer("all-MiniLM-L6-v2")  # Used for semantic similarity
SYSTEM_PROMPT = """
    You are BB-1, the Berlin dataset information chatbot. You only know about datasets for Berlin, and specifically about public drinking water fountains in Berlin.
    If a user asks about water fountains in any city other than Berlin (for example, Potsdam or Hamburg), you must answer: "I don't know".
    The answer should be the name of a dataset and nothing else, unless the question is about a city other than Berlin, in which case reply: "I don't know".
    If you don't know the answer, just say "I don't know".
    """

# ====== Helper functions ======
def log_interaction(user_input, bot_response, matched_datasets=None):
    """Log chat interactions and matched datasets to JSONL file."""
    try:
        # Create logs directory if it doesn't exist
        import os
        os.makedirs('logs', exist_ok=True)
        
        # Prepare log entry
        log_entry = {
            "timestamp": datetime.datetime.now().isoformat(),
            "user_input": user_input,
            "bot_response": bot_response,
            "matched_datasets": matched_datasets if matched_datasets else []
        }
        
        # Append to JSONL file
        with open(log_filename, 'a', encoding='utf-8') as f:
            json.dump(log_entry, f, ensure_ascii=False)
            f.write('\n')
    except Exception as e:
        print(f"Warning: Could not log interaction: {str(e)}")

def load_metadata():
    """Load metadata from config file."""
    with open(metadata_filename, "r", encoding="utf-8") as file:
        return json.load(file)

def load_prompts():
    """Load prompts from config file."""
    with open(prompts_filename, "r", encoding="utf-8") as file:
        return json.load(file)

def execute_prompt(prompt):
    """Send a prompt to Ollama and return the model's response."""
    # response = ollama.chat(
    #     model=LLM_MODEL,
    #     messages=[{"role": "user", "content": prompt}]
    # )
    # return response["message"]["content"].strip()

    response = ollama.chat(
        model=LLM_MODEL,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": prompt}
        ]
        # options={"temperature": 1}
    )
    return response["message"]["content"].strip()


def greet_user():
    """Generate a greeting using the LLM."""
    prompt = "\n".join(prompts["greet_user"])
    return execute_prompt(prompt)

def get_relevant_datasets(user_query, metadata):
    """Find the most relevant datasets based on user query using embeddings."""
    query_embedding = EMBEDDING_MODEL.encode(user_query, convert_to_tensor=True)
    dataset_scores = []

    for dataset in metadata:
        desc_embedding = EMBEDDING_MODEL.encode(dataset["description"], convert_to_tensor=True)
        similarity_score = util.pytorch_cos_sim(query_embedding, desc_embedding).item()
        dataset_scores.append((dataset["dataset"], dataset["description"], similarity_score))

    # Sort by highest similarity score
    dataset_scores.sort(key=lambda x: x[2], reverse=True)

    # Identify best matches
    perfect_matches = [f"Dataset: {ds[0]} - {ds[1]}" for ds in dataset_scores if ds[2] >= 0.5][:5]
    close_matches = [f"Dataset: {ds[0]} - {ds[1]}" for ds in dataset_scores if 0.3 <= ds[2] < 0.5][:5]

    if perfect_matches:
        return perfect_matches
    
    if close_matches:
        return "No perfect match found. Did you mean:\n" + "\n".join(close_matches)
    
    return "No matching datasets found."

def ask_dataset():
    """Ask user about their query and find the relevant dataset."""
    user_query = input("\nYou: ")  # Take user input
    dataset_info = "\n".join([f"Dataset: {item['dataset']} - {item['description']}" for item in metadata])
        
    prompt = "\n".join(prompts["identify_dataset"]).format(
        datasets=dataset_info, 
        user_input=user_query
    )
    
    response = execute_prompt(prompt)
    
    # Use embeddings to find the relevant dataset(s)
    relevant_datasets = get_relevant_datasets(user_query, metadata)
    
    # Prepare response and log interaction
    final_response = ""
    if isinstance(relevant_datasets, str):  # When it returns a message instead of a list
        if "No matching datasets found" in relevant_datasets:
            final_response = "Chatbot: No matching datasets found. Please try a different request."
        if "Did you mean" in relevant_datasets:
            print(f"\nChatbot: {relevant_datasets}")
            selected_dataset = input("\nChatbot: Please enter the dataset name from the suggestions, or type 'none' to cancel: ")
            if selected_dataset.lower() == "none":
                final_response = "\nChatbot: No dataset selected. Please try again."
            else:
                final_response = f"\nChatbot: You've selected '{selected_dataset}'. Proceeding with this dataset."
    else:
        final_response = f"\nChatbot: Here are the top matching datasets based on your request:\n" + "\n".join(relevant_datasets)
    
    # Log the interaction
    log_interaction(user_query, final_response, relevant_datasets if isinstance(relevant_datasets, list) else [])
    return final_response

# ====== Main interactive chatbot ======
if __name__ == "__main__":
    # Load metadata and prompts
    metadata = load_metadata()
    prompts = load_prompts()

    # Greet user
    greetings = greet_user()
    print("\nChatbot Greeting:", greetings)

    # Ask user for dataset name
    dataset_prompt = ask_dataset()
    print(dataset_prompt)

    # Interactive chat loop
    while True:
        user_input = input("\nYou: ")  # Take user input
        if user_input.lower() in ["exit", "quit", "bye"]:  # Exit condition
            farewell = "\nChatbot: Goodbye! Have a great day! 👋"
            log_interaction(user_input, farewell)
            print(farewell)
            break
        response = execute_prompt(user_input)  # Get AI response
        log_interaction(user_input, response)
        print("\nChatbot:", response)
