import os
import logging

from langchain_core.documents import Document
from langchain_core.prompts import ChatPromptTemplate
from langchain_ollama import ChatOllama
from typing import List, Tuple

# Logging für die Streamlit App
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.StreamHandler()
    ]
)

os.environ["TOKENIZERS_PARALLELISM"] = "false"

FH_PROMPT = """
<|begin_of_text|>
<|start_header_id|>system<|end_header_id|>
Du bist der Assistent der Fachhochschule Südwestfalen (FH-SWF).
Du beantwortest Fragen von Studieninteressierten und Studierenden.

Nutze den bereitgestellten Kontext, um die Frage konkret zu beantworten.
Wenn du die Frage nicht aus dem Kontext beantworten kannst, antworte mit "Das weiß ich leider nicht.", sonst nichts.
Erfinde niemals etwas, was nicht aus dem Kontext hervorgeht.

<|start_header_id|>user<|end_header_id|>
Kontext: {context}


Frage: {query}

<|start_header_id|>assistant<|end_header_id|>
"""


def keep_relevant_metadata(documents: list[Document]) -> Tuple[List[Document], List[str]]:
    relevant_metadata = [
        "standort",
        "studiengang",
        "abschluss"
    ]
    links = []
    for doc in documents:
        if doc.metadata['link'] not in links:
            links.append(doc.metadata['link'])
        doc.metadata = {key: doc.metadata[key] for key in relevant_metadata}
    return documents, links


def format_page_content(documents: list[Document]) -> str:
    formatted_context = ""
    for doc in documents:
        formatted_context += f"Standort: {doc.metadata['standort']}, Studiengang: {doc.metadata['studiengang']}, Abschluss: {doc.metadata['abschluss']}\n{doc.page_content}\n\n"
    return formatted_context


def prepare_context(retrieved_documents: list[Document]) -> Tuple[str, List[str]]:
    documents, links = keep_relevant_metadata(retrieved_documents)
    prepared_context = format_page_content(documents)
    return prepared_context, links


def generate_answer(query: str, context: str, llm: ChatOllama) -> str:
    prompt = ChatPromptTemplate.from_template(template=FH_PROMPT)
    prompt_with_context_and_query = prompt.invoke({"context": context, "query": query})
    response = llm.invoke(prompt_with_context_and_query)
    answer = response.content
    return answer


def create_llm_for_answers():
    llm = ChatOllama(
        model="cyberwald/llama-3.1-sauerkrautlm-8b-instruct:latest",
        temperature=0,
        num_ctx=10000
    )
    return llm
