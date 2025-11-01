- **Ollama LLM (like Llama 3.1:8B)** → for understanding user intent and natural replies  
- **SentenceTransformer (all-MiniLM-L6-v2)** → for semantic similarity and matching  
- **Few-shot prompt engineering** → to guide the LLM using structured examples  

Everything runs **locally** — no external API calls.

---

##  What this project provides

A local chatbot that helps you discover datasets relevant to your data-related queries.  
It reads dataset metadata and interacts naturally to find the most suitable match.  

---

## Features

- **Local LLM-based chatbot** powered by Ollama  
- **Semantic search** using SentenceTransformer embeddings  
- **Confidence-based match ranking** (perfect vs close matches)  
- **Few-shot prompt engineering** for structured, example-driven responses  
- **Logs every interaction** in `logs/chat_history.jsonl`  
- Prevents hallucination — chatbot never suggests datasets that don’t exist  
- Easy to extend for new datasets or domains  

---

## Why I thought of this project

While exploring open data portals, I realized how difficult it is to find the right dataset manually.  
So I decided to build a **dataset discovery assistant** that understands natural queries like  
> “Show me bike theft data in Berlin”  
and directly points to the right dataset from metadata.

---

## Dataset and Metadata

All dataset info comes from **`config/metadata.json`**.

Example datasets:
- `trinkbrunnen` → Berlin public drinking water fountains  
- `fahrraddiebstähle` → Bike theft incidents in Berlin  

Each dataset entry contains:
- Dataset name and description  
- File and download URL  
- Column metadata (name, type, description)  

Chatbot prompts and instructions are stored in:  
**`config/prompts.json`**

---

## How to Run

1. **Install dependencies**
   ```bash
   pip install sentence-transformers
   python -m pip install sentence-transformers
2. **Run the chatbot**
   python use_case_1.py
---


## Workflow

1. User enters a natural query  
2. Query is encoded using **SentenceTransformer embeddings**  
3. Each dataset description is compared using **cosine similarity**  
4. Datasets are ranked by relevance  
5. **Ollama’s LLM**, guided by **few-shot examples**, formulates the response  
6. Everything is logged in `logs/chat_history.jsonl`

---

## What I Did

- Built a **local interactive chatbot** using Python  
- Integrated **Ollama LLM** and **SentenceTransformer** for dual reasoning  
- Used **few-shot prompt engineering** to define clear example-driven behavior  
- Ensured only valid datasets are ever suggested  
- Logged user queries and matched datasets for evaluation  
- Tuned similarity thresholds for “perfect” and “close” matches  

---

## What I Learned

- Combining **embeddings + LLM reasoning** gives accurate, explainable dataset search  
- **Few-shot prompting** makes LLMs more structured and reliable  
- **SentenceTransformer** provides strong local semantic similarity  
- **Logging** helps improve and debug model behavior  
- **Ollama** allows running modern LLMs efficiently on local machines  

---


