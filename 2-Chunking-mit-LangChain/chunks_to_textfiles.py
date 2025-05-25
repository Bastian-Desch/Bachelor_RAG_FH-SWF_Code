import os
import logging
from chunking_util import (
    create_tokenizer,
    create_token_based_text_splitter,
    create_recursive_text_splitter,
    create_markdown_header_text_splitter,
    create_markdown_text_splitter,
    create_semantic_text_splitter,
    load_json_data,
    plot_token_length_distribution,
    process_and_plot_chunks
)
# Logging statt Print-Statements für nachvollziehbarkeit während des Ablaufs des Skripts
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S"
)

# Relevante Pfade, die bereits angelegt wurden
JSON_PATH = "../1-Datenbeschaffung-mit-Docling/studiengaenge-und-links-und-markdown-dateien.json"  # JSON mit Quelldaten
MARKDOWN_DIR = "../1-Datenbeschaffung-mit-Docling/Markdown-Dateien"  # Ordner mit Markdown-Dateien
OUTPUT_BASE_DIR = "2-Chunking-mit-LangChain-TextDateien"  # Basisordner für alle Ergebnisse

# Alle Textsplitter, die getestet werden (kann dank lambda Funktionen später einfach mit modell_name aufgerufen werden)
splitter_configs = {
    "token_based": lambda model: create_token_based_text_splitter(model, max_tokens_per_chunk=500, token_overlap=50),
    "recursive": lambda model: create_recursive_text_splitter(model, max_tokens_per_chunk=500, token_overlap=50),
    "markdown_text": lambda model: create_markdown_text_splitter(model, max_tokens_per_chunk=500, chunk_overlap=50),
    "markdown_header": lambda _: create_markdown_header_text_splitter(),
    "semantic": lambda model: create_semantic_text_splitter(model)
}

# Namen der Embedding-Modelle (Enums, die in anderen Dateien genutzt werden, wurden erst im späteren Verlauf erstellt)
embedding_models = [
    "intfloat/multilingual-e5-large",
    "mixedbread-ai/deepset-mxbai-embed-de-large-v1"
]


def main():
    logging.info("Lade JSON-Daten...")
    data = load_json_data(JSON_PATH)

    for model_name in embedding_models:
        logging.info(f"Verarbeite Daten mit Embedding-Modell: {model_name}")

        # Tokenizer für jeweiliges Embedding-Modell erstellen
        tokenizer = create_tokenizer(model_name)

        for splitter_name, splitter_func in splitter_configs.items():
            logging.info(f"Nutze Splitter: {splitter_name}")

            # Splitter erstellen
            splitter = splitter_func(model_name)

            # Ausgabeordner für die Textdateien erstellen
            text_output_dir = os.path.join(OUTPUT_BASE_DIR, f"{splitter_name}_{model_name.replace('/', '_')}")
            os.makedirs(text_output_dir, exist_ok=True)

            # Chunks erstellen, speichern und Token-Längen direkt plotten
            # Übergabe der Plotfunktion als Argument, diente nur zum Testen, wie man eine Funktion übergibt
            # und sie dann in der Funktion aufruft (wäre anders evtl. einfacher gewesen)
            process_and_plot_chunks(data, MARKDOWN_DIR, text_output_dir, splitter, tokenizer,
                                    plot_token_length_distribution, model_name, splitter_name)

    logging.info("Verarbeitung abgeschlossen!")


if __name__ == "__main__":
    """
    Mit diesem Skript werden die Funktionen aus chunking_util.py aufgerufen und die Chunks mit allen Splittern erstellt.
    Die Chunks werden in getrennten Ordnern gespeichert.
    Je Modell und Chunking-Verfahren existiert anschließend ein Ordner.
    Die Textdateien dienten zu Beginn der Auswahl der zwei zu vergleichenden Chunking-Verfahren.
    Für die Evaluation wurden sie genutzt, um die IDs der Chunks zu ermitteln, die relevante Inhalte zu den Fragen enthielten.
    D. h. die Textdateien wurden nach relevanten Informationen durchsucht, um die Chunk-IDs als Dokumenten-IDs den Fragen zuzuordnen (Mappings).
    """
    main()
