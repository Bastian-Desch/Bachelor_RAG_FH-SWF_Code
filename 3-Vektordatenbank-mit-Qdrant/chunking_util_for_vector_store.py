import json
import logging
import os

from langchain_core.documents import Document
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_qdrant import QdrantVectorStore
from langchain_text_splitters import SentenceTransformersTokenTextSplitter, RecursiveCharacterTextSplitter
from transformers import AutoTokenizer

'''
Die Funktionen hier sind weitestgehend aus der chunking_util.py Datei übernommen.
Sie wurden teilweise abgeändert, um sie für die Erstellung der Datenbank-Collections zu nutzen.
Der Umfang wurde zudem reduziert, damit nur noch die relevanten Funktionen enthalten sind. (z. B. nur zwei Splitter und keine Funktion für Plots)
'''

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S"
)


def create_tokenizer(model_name: str) -> AutoTokenizer:
    """
    Erzeugt Tokenizer für das übergebene Embedding-Modell.

    :parameter model_name: Name des Embedding-Modells
    :return: Tokenizer für das Modell
    """
    tokenizer = AutoTokenizer.from_pretrained(pretrained_model_name_or_path=model_name)
    return tokenizer


def create_token_based_text_splitter(model_name: str, max_tokens_per_chunk: int,
                                     token_overlap: int) -> SentenceTransformersTokenTextSplitter:
    """
    Erzeugt einen Text-Splitter, der feste Tokenanzahlen berücksichtigt.
    :param model_name: Name des Embedding-Modells für den Tokenizer.
    :param max_tokens_per_chunk: Maximale Anzahl an Tokens pro Chunk.
    :param token_overlap: Anzahl Token für Überlappung zwischen Chunks.
    :return: Konfigurierter Token-Text-Splitter.
    """
    token_based_splitter = SentenceTransformersTokenTextSplitter(
        model_name=model_name,
        tokens_per_chunk=max_tokens_per_chunk,
        chunk_overlap=token_overlap
    )
    return token_based_splitter


def create_recursive_text_splitter(model_name: str, max_tokens_per_chunk: int,
                                   token_overlap: int) -> RecursiveCharacterTextSplitter:
    """
    Erzeugt einen rekursiven Text-Splitter, der die Standard-Liste ["\n\n", "\n", " ", ""] zum Aufteilen nutzt.
    :param model_name: Name des Embedding-Modells für den Tokenizer.
    :param max_tokens_per_chunk: Maximale Anzahl an Tokens pro Chunk.
    :param token_overlap: Anzahl Token für Überlappung zwischen Chunks.
    :return: Konfigurierter Recursive-Text-Splitter.
    """
    recursive_token_based_splitter = RecursiveCharacterTextSplitter.from_huggingface_tokenizer(
        tokenizer=create_tokenizer(model_name),
        chunk_size=max_tokens_per_chunk,
        chunk_overlap=token_overlap
    )
    return recursive_token_based_splitter


def load_json_data(json_path: str) -> dict:
    """
    Lädt eine JSON-Datei (hier nur für die Studiengangsdaten/Dokumente) und gibt ein Dictionary zurück.
    :param json_path: Pfad der Datei mit den Studiengangsdaten.
    :return: Dictionary mit den Studiengangsdaten für weitere Verarbeitung.
    """
    try:
        with open(json_path, 'r', encoding='utf-8') as json_file:
            data = json.load(json_file)
        return data
    except FileNotFoundError:
        logging.error(f"Datei nicht gefunden: {json_path}")
        return {}
    except json.JSONDecodeError:
        logging.error(f"Fehler beim Laden der JSON-Datei: {json_path}")


def load_markdown_content(file_path: str) -> str:
    """
    Lädt den Inhalt einer Markdown-Datei und gibt ihn als String zurück.
    :param file_path: Pfad der Datei, die eingelesen wird.
    :return: Inhalt der Datei.
    """
    if not os.path.exists(file_path):
        logging.warning(f"Datei {file_path} nicht gefunden! Grund dafür prüfen!")
        return None
    try:
        with open(file_path, 'r', encoding='utf-8') as md_file:
            return md_file.read()
    except Exception as e:
        logging.error(f"Fehler beim Lesen der Datei {file_path}: {e}")
        return None


def generate_chunks(md_content: str,
                    splitter: RecursiveCharacterTextSplitter | SentenceTransformersTokenTextSplitter) -> list[Document]:
    """
    Erzeugt Chunks mit RecursiveCharacterTextSplitter oder SentenceTransformersTokenTextSplitter.
    :param md_content: Inhalt einer Markdown-Datei.
    :param splitter: Text-Splitter zum Erzeugen der Chunks bzw. LangChain-Dokumente.
    :return: Liste der Chunks bzw. LangChain-Dokumente.
    """
    if not isinstance(splitter, (RecursiveCharacterTextSplitter, SentenceTransformersTokenTextSplitter)):
        logging.error("Kein gültiger Splitter! Leere Liste wird zurückgegeben.")
        return []

    return splitter.create_documents([md_content])


def create_documents_to_upload(text_splitter: RecursiveCharacterTextSplitter | SentenceTransformersTokenTextSplitter,
                               markdown_dir: str, data: dict) -> list[Document]:
    """
    Erzeugt LangChain-Dokumente für alle Markdown-Dateien in einem Ordner.
    :param text_splitter: Text-Splitter für die Chunk-Erzeugung.
    :param markdown_dir: Ordner mit den Markdown-Dateien.
    :param data: Dictionary mit den Studiengangsdaten.
    :return: Liste der erzeugten LangChain-Dokumente.

    Die Überprüfung, ob eine Datei mit einem bestimmten Hash-Wert bereits verarbeitet wurde, diente lediglich als 
    absicherung, da in der praktischen Ausarbeitung zwischenzeitlich Dateien durcheinander geraten sind.
    Diese Überprüfung ist eigentlich nicht notwendig, da die Hash-Werte in der JSON-Datei eindeutig sind und jede Datei aus dem Verzeichnis,
    indem sie gespeichert sind eingelesen wird.
    """
    all_docs = []
    doc_id = 1
    processed_hashes = set()

    for studiengang_data in data["studiengaenge"]:
        studiengang = studiengang_data["studiengang"]
        abschluss = studiengang_data["abschluss"]
        standort = studiengang_data["standort"]

        for dokument in studiengang_data["dokumente"]:
            link = dokument["link"]
            hash_value = dokument["hash"]

            if hash_value in processed_hashes:
                logging.info(
                    f"Datei mit dem Hashwert: {hash_value} wurde bereits verarbeitet. Sie wird nicht erneut verarbeitet.")
                continue

            md_file = dokument["markdown_datei"]
            file_path = os.path.join(markdown_dir, md_file)

            md_content = load_markdown_content(file_path)
            if md_content is None:
                continue

            documents = generate_chunks(md_content, text_splitter)

            for doc in documents:
                doc.page_content = doc.page_content  # f"passage: {doc.page_content}" wurde entfernt, da es mit encode_kwargs berücksichtigt wird
                doc.metadata.update({
                    "id": doc_id,
                    "studiengang": studiengang,
                    "abschluss": abschluss,
                    "standort": standort,
                    "link": link,
                    "hash": hash_value
                })
                doc_id += 1
                all_docs.append(doc)

    return all_docs


def create_collection(model_name: str, documents: list[Document], collection_name: str, qdrant_url: str) -> None:
    """
    Erstellt eine Collection in Qdrant und lädt die Dokumente hoch.
    :param model_name: Name des Embedding-Modells für die Vektoren.
    :param documents: Liste der Dokumente, die hochgeladen werden.
    :param collection_name: Name der Collection in Qdrant.
    :param qdrant_url: URL des Qdrant-Servers.
    """
    # Für die nutzung der encode_kwargs siehe: ("prompt" Argument ist relevant)
    # https://python.langchain.com/api_reference/huggingface/embeddings/langchain_huggingface.embeddings.huggingface.HuggingFaceEmbeddings.html#langchain_huggingface.embeddings.huggingface.HuggingFaceEmbeddings.query_encode_kwargs
    # https://sbert.net/docs/package_reference/sentence_transformer/SentenceTransformer.html#sentence_transformers.SentenceTransformer.encode
    encode_kwargs = {'prompt': 'passage: '}
    embeddings = HuggingFaceEmbeddings(model_name=model_name, encode_kwargs=encode_kwargs)

    QdrantVectorStore.from_documents(
        documents=documents,
        embedding=embeddings,
        collection_name=collection_name,
        url=qdrant_url,
        prefer_grpc=False,  # Nur HTTP nutzen
        batch_size=4  # Ressourcennutzung limitieren
    )
