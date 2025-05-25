import json
import os

from langchain_core.documents import Document
from langchain_text_splitters import SentenceTransformersTokenTextSplitter, RecursiveCharacterTextSplitter, \
    MarkdownHeaderTextSplitter, MarkdownTextSplitter
from transformers import AutoTokenizer
from langchain_experimental.text_splitter import SemanticChunker
from langchain_huggingface import HuggingFaceEmbeddings
from collections import Counter
import matplotlib.pyplot as plt
import logging

logging.basicConfig(
    level=logging.INFO,  # DEBUG, INFO, WARNING, ERROR, CRITICAL
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


def create_markdown_header_text_splitter() -> MarkdownHeaderTextSplitter:
    """
    Erzeugt MarkdownHeaderTextSplitter, der anhand aller Überschriften-Ebenen den Text aufteilt.
    :return: Konfigurierter MarkdownHeaderTextSplitter.

    Header-Ebenen wurden alle übergeben, auch wenn nicht alle in den Dateien vorkommen.
    Bis Ebene 3 kommen Überschriften vor, diese Liste hier hat keine weiteren Auswirkungen nach diesen Ebenen.
    """
    headers_to_split_on = [
        ("#", "Header 1"),
        ("##", "Header 2"),
        ("###", "Header 3"),
        ("####", "Header 4"),
        ("#####", "Header 5"),
        ("######", "Header 6")
    ]
    markdown_splitter = MarkdownHeaderTextSplitter(
        headers_to_split_on=headers_to_split_on,
        strip_headers=False
    )
    return markdown_splitter


'''
MarkdownTextSplitter nicht weiter beachten. Das hat die gleichen Resultate gebracht, wie der 
RecursiveCharacterTextSplitter mit Tokenizer, daher wird die bereits vorhandene Implementierung 
des RecursiveCharacterTextSplitter genutzt.
'''


def create_markdown_text_splitter(model_name: str, max_tokens_per_chunk: int,
                                  chunk_overlap: int) -> MarkdownTextSplitter:
    """
    Erzeugt einen Markdown-Text-Splitter, der anhand der Markdown-Spezifischen Zeichen den Text aufteilt.
    Die Liste dieser Zeichen kann hier eingesehen werden: https://github.com/langchain-ai/langchain/blob/9ef2feb6747f5a69d186bd623b569ad722829a5e/libs/langchain/langchain/text_splitter.py#L1175

            elif language == Language.MARKDOWN:
            return [
                # First, try to split along Markdown headings (starting with level 2)
                "\n#{1,6} ",
                # Note the alternative syntax for headings (below) is not handled here
                # Heading level 2
                # ---------------
                # End of code block
                "```\n",
                # Horizontal lines
                "\n\\*\\*\\*+\n",
                "\n---+\n",
                "\n___+\n",
                # Note that this splitter doesn't handle horizontal lines defined
                # by *three or more* of ***, ---, or ___, but this is not handled
                "\n\n",
                "\n",
                " ",
                "",
            ]

    :param model_name: Name des Embedding-Modells für den Tokenizer.
    :param max_tokens_per_chunk: Maximale Anzahl an Tokens pro Chunk.
    :param chunk_overlap: Anzahl Token für Überlappung zwischen Chunks.
    :return: Konfigurierter Markdown-Text-Splitter.
    """
    markdown_splitter = MarkdownTextSplitter.from_huggingface_tokenizer(
        tokenizer=create_tokenizer(model_name=model_name),
        chunk_size=max_tokens_per_chunk,
        chunk_overlap=chunk_overlap
    )
    return markdown_splitter


def create_semantic_text_splitter(model_name: str) -> SemanticChunker:
    """
    Erzeugt einen Text-Splitter, der auf der semantischen Ähnlichkeit aufeinanderfolgender Sätze basiert.
    Standardmäßig ist dieser nur für Sätze implementiert mit dem sentence_split_regex='(?<=[.?!])\\s+'.
    Ein eigenes Regex wurde hier genutzt, um auch Markdown-Tabellenstrukturen zu erkennen.
    (Kann kein Token-Limit festlegen, oft zu große Chunks, nicht weiter nutzen.)

    :param model_name: Name des Embedding-Modells, mit dem Embeddings erzeugt werden.
    :return: Konfigurierter Semantic-Text-Splitter.

    Es wurden verschieden Kombinationen der Parameter "sentence_split_regex" und "breakpoint_threshold_type"
    und "breakpoint_threshold_amount" getestet. Nur die letzte hier zu sehende Kombination ist in den
    Textdateien zu finden, da sie überschrieben wurden.

    In der Ausarbeitung wird erläutert, dass alle Kombinationen die gleichen Probleme aufwiesen.
    """
    embeddings = HuggingFaceEmbeddings(model_name=model_name)
    semantic_splitter = SemanticChunker(
        embeddings=embeddings,
        sentence_split_regex=r'(?<=[.?!])\s+|(?=\n\|(?:[^|\n]+?\|)+\n)',  # Versuch, Markdown-Tabellen zu berücksichtigen (hatte keinen Erfolg)
        breakpoint_threshold_type="standard_deviation",
        breakpoint_threshold_amount=3)
    return semantic_splitter


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


def count_tokens(text: str, tokenizer: AutoTokenizer) -> int:
    """
    Zählt die Anzahl der Tokens in einem Text.
    :param text: Text, dessen Tokens gezählt werden sollen.
    :param tokenizer: Tokenizer, der die Tokenisierung durchführt.
    :return: Anzahl der Tokens im Text.

    Notwendig zur Überprüfung, ob ein Chunking-Verfahren mit dem Kontextfenster der Embedding-Modelle kompatibel ist.
    """
    return len(tokenizer.encode(text, add_special_tokens=False))


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


def generate_chunks(md_content: str, splitter: any) -> list[Document]:
    """
    Erzeugt Chunks aus dem Markdown-Inhalt mit dem übergebenen Text-Splitter.
    Chunks werden direkt als LangChain-Document erzeugt.
    Je nach Text-Splitter muss dafür eine andere Methode aufgerufen werden.

    :param md_content: Inhalt einer Datei.
    :param splitter: Der jeweilige Text-Splitter für die Chunks.
    :return: Liste der Chunks des Ausgangsdokumentes als LangChain-Documents.
    """
    # Die Unterscheidung hier ist notwendig, da die TextSplitter von LangChain verschiedene Methoden für das Erstellen
    # der Chunks bieten. Es wurden die genutzt, die LangChain-Document Objekte mit metadata und page_content erstellen.
    if isinstance(splitter, (
            RecursiveCharacterTextSplitter, SentenceTransformersTokenTextSplitter, SemanticChunker,
            MarkdownTextSplitter)):
        return splitter.create_documents([md_content])
    elif isinstance(splitter, MarkdownHeaderTextSplitter):
        return splitter.split_text(md_content)
    logging.warning("Kein passender Text-Splitter übergeben. Es wurden keine Chunks generiert.")
    return []


def process_and_plot_chunks(data: dict, markdown_dir: str, text_output_dir: str, splitter: any,
                            tokenizer: AutoTokenizer, plot_function: callable, model_name: str,
                            splitter_name: str) -> None:
    """
    Verarbeitet Markdown-Dateien zu Chunks, reichert sie mit Metadaten an und schreibt sie in Textdateien.
    Zudem werden alle Tokenlängen gespeichert und nach der Verarbeitung geplottet, um zu sehen, wie häufig welche
    Chunk-Größe vorhanden ist.

    :param data: Dictionary mit Studiengängen und Dokumenten.
    :param markdown_dir: Ordner mit den Markdown-Dateien.
    :param text_output_dir: Zielordner für die Textdateien.
    :param splitter: Splitter für die Chunk-Erstellung.
    :param tokenizer: Tokenizer zur Tokenzählung.
    :param plot_function: Funktion, die den Plot der Tokenlängen erstellt.
    :param model_name: Name des Embedding-Modells für den Plot.
    :param splitter_name: Name des Text-Splitters für den Plot.
    """
    os.makedirs(text_output_dir, exist_ok=True)
    doc_id = 1
    all_token_lengths = []
    processed_hashes = set()

    for studiengang_data in data["studiengaenge"]:
        for dokument in studiengang_data["dokumente"]:
            hash_value = dokument["hash"]
            if hash_value in processed_hashes:
                logging.info(f"Datei mit Hash {hash_value} wurde bereits verarbeitet. Überspringe...")
                continue

            file_path = os.path.join(markdown_dir, dokument["markdown_datei"])
            content = load_markdown_content(file_path)
            if content is None:
                continue

            documents = generate_chunks(content, splitter)
            output_file_path = os.path.join(text_output_dir, f"{hash_value}.txt")

            out_put_text = ""
            for i, chunk in enumerate(documents):
                chunk.metadata.update({
                    "id": doc_id,  # Chunk-ID für die Evaluation bzw. die Mappings von Fragen und relevanten Dokumenten
                    "studiengang": studiengang_data["studiengang"],
                    "abschluss": studiengang_data["abschluss"],
                    "standort": studiengang_data["standort"],
                    "link": dokument["link"],
                    "hash": hash_value
                })
                chunk_token_count = count_tokens(chunk.page_content, tokenizer)

                all_token_lengths.append(chunk_token_count)

                # Formatierte Ausgabe der Chunks für die Textdateien
                out_put_text += f"=== Chunk-Nr. {i} ===\n"
                out_put_text += f"Anzahl Token in diesem Chunk: {chunk_token_count}\n\n"
                out_put_text += f"Metadata: {chunk.metadata}\n"
                out_put_text += f"Text: {chunk.page_content}\n\n"

                doc_id += 1  # Hochzählen der globalen doc_id (je Chunking-Verfahren separat)

            with open(output_file_path, 'w', encoding='utf-8') as text_out:
                text_out.write(out_put_text)

            processed_hashes.add(hash_value)
            logging.info(f"Verarbeitung abgeschlossen für Datei {dokument['markdown_datei']}.")

    plot_function(Counter(all_token_lengths), model_name, splitter_name)


def plot_token_length_distribution(token_lengths: Counter, model_name: str, splitter_name: str):
    """
    Erstellt ein Balkendiagramm zur visualisierung der Tokenlängen und ihrer Häufigkeit.
    :param token_lengths: Alle Token-Längen und ihre Häufigkeit.
    :param model_name: Das genutzte Embedding-Modell für die Tokenisierung.
    :param splitter_name: Der genutzte Text-Splitter für die Chunk-Erstellung.
    """
    token_lengths_sorted = sorted(token_lengths.items())
    lengths, frequencies = zip(*token_lengths_sorted)

    plt.figure(figsize=(10, 6))
    plt.bar(lengths, frequencies, width=0.8, color='blue')
    plt.xlabel('Tokenlänge')
    plt.ylabel('Häufigkeit')
    plt.title(f'Tokenlängenverteilung - Splitter: {splitter_name}, Modell: {model_name}')
    plt.grid(True)

    plt.show()
