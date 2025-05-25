import os
import logging
import time
from chunking_util_for_vector_store import (
    create_token_based_text_splitter,
    create_recursive_text_splitter,
    load_json_data,
    create_documents_to_upload,
    create_collection
)


def main():
    # Relevante Pfade, die bereits angelegt wurden
    JSON_PATH = "../1-Datenbeschaffung-mit-Docling/studiengaenge-und-links-und-markdown-dateien.json"  # JSON mit Quelldaten
    MARKDOWN_DIR = "../1-Datenbeschaffung-mit-Docling/Markdown-Dateien"  # Ordner mit Markdown-Dateien

    # URL des Qdrant-Servers, der im Docker Container läuft
    QDRANT_URL = "http://localhost:6333"
    # Die beiden Embedding-Modelle (Die Enums wurden erst später erstellt, daher wurde hier noch nicht damit gearbeitet)
    embedding_models = [
        "intfloat/multilingual-e5-large",
        "mixedbread-ai/deepset-mxbai-embed-de-large-v1"
    ]
    CHUNK_SIZE = 500
    CHUNK_OVERLAP = 50
    logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s",
                        datefmt="%Y-%m-%d %H:%M:%S")

    # JSON-Datei einlesen (falls nicht vorhanden abbrechen):
    logging.info("Einlesen der JSON-Datei")
    data = load_json_data(JSON_PATH)
    if not data:
        logging.error("Keine Daten geladen.")
        exit(1)

    # Vier Collections anlegen (zwei je Modell und Chunking-Verfahren):
    for model_name in embedding_models:
        token_splitter = create_token_based_text_splitter(model_name=model_name,
                                                          max_tokens_per_chunk=CHUNK_SIZE,
                                                          token_overlap=CHUNK_OVERLAP)

        recursive_splitter = create_recursive_text_splitter(model_name=model_name,
                                                            max_tokens_per_chunk=CHUNK_SIZE,
                                                            token_overlap=CHUNK_OVERLAP)

        logging.info(f"Erstelle Token-basierte Chunks mit dem Embedding-Modell: {model_name}")
        documents_by_token_splitter = create_documents_to_upload(text_splitter=token_splitter,
                                                                 markdown_dir=MARKDOWN_DIR,
                                                                 data=data)

        logging.info(f"Erstelle Recursive Chunks mit dem Embedding-Modell: {model_name}")
        documents_by_recursive_splitter = create_documents_to_upload(text_splitter=recursive_splitter,
                                                                     markdown_dir=MARKDOWN_DIR,
                                                                     data=data)
        
        # Collection-Namen für die beiden Chunking-Verfahren und Modelle:
        collection_name_token = f"{model_name.replace('/', '_')}_token_based_chunks"
        collection_name_recursive = f"{model_name.replace('/', '_')}_recursive_chunks"

        logging.info(f"Erstelle Token-basierte-Collection: {collection_name_token}")
        start_time_token_collection = time.time() # Startzeit Token-Collection
        create_collection(model_name=model_name, documents=documents_by_token_splitter,
                          collection_name=collection_name_token, qdrant_url=QDRANT_URL)
        end_time_token_collection = time.time() # Endzeit Token-Collection
        total_time_token_collection = end_time_token_collection - start_time_token_collection # Dauer der Erstellung
        logging.info(f"Erstellung der Token-basierten Collection dauerte {total_time_token_collection:.2f} Sekunden.")

        logging.info(f"Erstelle Recursive-Collection: {collection_name_recursive}")
        start_time_recursive_collection = time.time() # Startzeit Recursive-Collection
        create_collection(model_name=model_name, documents=documents_by_recursive_splitter,
                          collection_name=collection_name_recursive, qdrant_url=QDRANT_URL)
        end_time_recursive_collection = time.time() # Endzeit Recursive-Collection
        total_time_recursive_collection = end_time_recursive_collection - start_time_recursive_collection # Dauer der Erstellung
        logging.info(f"Erstellung der Recursive-Collection dauerte {total_time_recursive_collection:.2f} Sekunden.")

    logging.info("Alle Collections erstellt.") # Logging hier dient als Ersatz für print-Statements, für Nachvollziehbarkeit des Programmablaufs


if __name__ == "__main__":
    """
    Mit diesem Skript werden die 4 Collections in Qdrant erstellt, die später in der Evaluation genutzt werden.
    Über den Konsolenbefehl "docker compose up -d" in diesem Verzeichnis, wird der Qdrant Server gestartet und die 
    bereits erstellten Collections über ein Bind Mount aus dem Verzeichnis "qdrant_data" geladen.
    
    Die Compose-Datei kann der Installationsanleitung der Qdrant-Website entnommen werden.
    """
    main()
