# Geoscience RAG Agent for Scanned PDFs

This project builds a Retrieval-Augmented Generation (RAG) agent designed to extract and clean text from scanned geoscience reports (PDFs), embed them, and answer user queries using a local LLM and Hugging Face embeddings.

---

## Project Structure

```
project-root/
│
├── src/
│   ├── extract_clean_text.py      # Extracts, cleans, chunks, and embeds text
│   ├── build_rag_agent.py          # Builds the RAG QA agent
│   ├── evaluate_responses.py       # Evaluates agent responses using cosine similarity
│
├── run_agent.py                    # Interactive script to chat with the agent
├── query_examples.json             # Sample QA pairs for evaluation
├── results/eval_results.json       # Output of evaluation (created at runtime)
├── pdfreports/                     # Directory to store scanned PDF reports
└── README.md                       # This file
```

---

## Installation

### Requirements
- Python 3.9+
- PyTorch (CUDA recommended)
- Tesseract OCR

### Install dependencies
```bash
pip install -r requirements.txt
```
Make sure `tesseract` is installed on your system and added to the PATH.

---

## Usage

### 1. Prepare Your PDF Reports
Place all scanned PDFs into the `pdfreports/` directory.

### 2. Run the Interactive Agent
```bash
python run_agent.py --pdf_dir pdfreports
```
Ask questions related to the content of the scanned PDFs interactively.

### 3. Evaluate Agent Responses
Create a `query_examples.json` file with query and groundtruth structure:
```json
[
  {
    "query": "What formation lies beneath Fox Lake?",
    "groundtruth": "The formation beneath Fox Lake is the Kiskatinaw Formation."
  }
]
```
Then run:
```bash
python src/evaluate_responses.py
```
Results will be saved to `results/eval_results.json`.

---

## How It Works

### OCR + Cleaning
- Converts scanned PDFs to images using `pdf2image`
- Uses `pytesseract` for OCR
- Proofreads and cleans text using an LLM (OpenHermes/Mistral)

### Embedding + Retrieval
- Splits cleaned text into overlapping chunks
- Embeds using BAAI/bge-base-en-v1.5
- Stores in FAISS for fast similarity search

### QA Agent
- A LangChain `RetrievalQA` chain uses the embedded store and Hugging Face LLM pipeline
- Prompt is highly constrained to prevent hallucination and enforce document-grounded answers

---

## Notes
- The LLM is loaded in 8-bit with quantization to fit on consumer GPUs
- Use short and precise questions for best results
- Make sure PDFs are actually scanned image-based (not already text-based)


