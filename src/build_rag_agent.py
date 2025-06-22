import os
from langchain.vectorstores import FAISS
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate
from langchain_huggingface import HuggingFacePipeline, HuggingFaceEmbeddings
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline, BitsAndBytesConfig

from src.extract_clean_text import load_pdf_data_and_clean


def get_prompt_template():
    template = """
You are a precise assistant specialized in interpreting geoscience reports. Use only the information in the context below to answer the question. Do not use any external knowledge or make assumptions.

Step-by-step instructions:
1. Check if the question refers to a specific named location (e.g., a lake, region, or geological area).
2. If that exact location is NOT explicitly found in the context, respond only with:
   "The information to answer this question is not available in the provided document."
3. If the location is found in the context, and the context includes a clear answer, respond with that answer in no more than two sentences.
4. Do not guess, speculate, or infer beyond what is explicitly written in the context.

Context:
{context}

Question:
{question}

Answer:
"""
    return PromptTemplate(template=template, input_variables=["context", "question"])


def load_llm():
    bnb_config = BitsAndBytesConfig(load_in_8bit=True)
    model = AutoModelForCausalLM.from_pretrained(
        "teknium/OpenHermes-2.5-Mistral-7B",
        device_map="auto",
        quantization_config=bnb_config,
    )
    tokenizer = AutoTokenizer.from_pretrained("teknium/OpenHermes-2.5-Mistral-7B")
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id

    generation_pipeline = pipeline(
        "text-generation",
        model=model,
        tokenizer=tokenizer,
        pad_token_id=tokenizer.pad_token_id,
        max_new_tokens=512,
    )
    return HuggingFacePipeline(pipeline=generation_pipeline)


def build_rag_agent(folder_path):
    texts, metadatas = load_pdf_data_and_clean(folder_path)
    embedding_model = HuggingFaceEmbeddings(model_name="BAAI/bge-base-en-v1.5")
    vectorstore = FAISS.from_texts(texts=texts, embedding=embedding_model, metadatas=metadatas)

    retriever = vectorstore.as_retriever(search_type="similarity", search_kwargs={"k": 3})
    llm = load_llm()
    prompt = get_prompt_template()

    qa_chain = RetrievalQA.from_chain_type(
        llm=llm,
        retriever=retriever,
        chain_type="stuff",
        chain_type_kwargs={"prompt": prompt}
    )
    return qa_chain
