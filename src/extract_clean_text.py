import os
from pdf2image import convert_from_path
import pytesseract
from transformers import pipeline, AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain.vectorstores import FAISS

# Load tokenizer and model for proofreading
bnb_config = BitsAndBytesConfig(load_in_8bit=True)
model_name = "teknium/OpenHermes-2.5-Mistral-7B"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name, device_map="auto", quantization_config=bnb_config)

if tokenizer.pad_token_id is None:
    tokenizer.pad_token_id = tokenizer.eos_token_id

proof_pipeline = pipeline("text-generation", model=model, tokenizer=tokenizer, pad_token_id=tokenizer.pad_token_id)
embedding_model = HuggingFaceEmbeddings(model_name="BAAI/bge-base-en-v1.5")


def extract_text_from_scanned_pdf(pdf_path):
    pages = convert_from_path(pdf_path)
    text_pages = [pytesseract.image_to_string(page) for page in pages]
    return "\n".join(text_pages)

    
def clean_text_with_llm(text):
    prompt = f"""You are a helpful assistant proofreading text extracted from scanned geoscience reports.
Correct OCR errors, fix spelling and grammar, and remove irrelevant symbols or noise.
Do not change scientific terms or paraphrase—only correct errors while preserving meaning.

Text:
{text}

Cleaned Text:"""
    result = proof_pipeline(prompt, do_sample=False, max_new_tokens=1000)
    return result[0]['generated_text'].replace(prompt, "").strip()


def load_pdf_data_and_clean(folder_path):
    texts, metadatas = [], []
    for filename in os.listdir(folder_path):
        if filename.lower().endswith(".pdf"):
            pdf_path = os.path.join(folder_path, filename)
            raw_text = extract_text_from_scanned_pdf(pdf_path)
            print(f"[✓] Extracted: {filename}")
            title = os.path.splitext(filename)[0]

            splitter = RecursiveCharacterTextSplitter(chunk_size=1200, chunk_overlap=100)
            chunks = splitter.split_text(raw_text)

            for chunk in chunks:
                if len(chunk.strip()) > 100:
                    cleaned = clean_text_with_llm(chunk)
                    labeled_chunk = f"[Context from report titled '{title}']\n\n{cleaned}"
                    texts.append(labeled_chunk)
                    metadatas.append({"title": title})

    return texts, metadatas
