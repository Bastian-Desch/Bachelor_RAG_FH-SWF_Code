from docling.document_converter import DocumentConverter
import json
import hashlib
import os
import re
import logging

# Ordner und Dateien, die vorher bereits angelegt wurden
OUTPUT_DIR = "Markdown-Dateien"
JSON_SOURCES = "studiengaenge-und-links.json"
JSON_OUTPUT = "studiengaenge-und-links-und-markdown-dateien.json"

# Logging, um Verarbeitung nachvollziehen zu können und keine print-Statements zu verwenden
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


def get_md_content(url: str) -> str:
    """Gibt den Inhalt eines Dokumentes im Markdown-Format als String zurück."""
    converter = DocumentConverter()
    try:
        result = converter.convert(url)
        md_content = result.document.export_to_markdown(image_placeholder="")
        # Bilder werden quasi entfernt, da sie nicht weiter verarbeitet werden
        return md_content
    except Exception as e:
        logger.error(f"Fehler beim Konvertieren von {url}: {e}")
        return ""


def remove_data_before_first_heading(md_content: str) -> str:
    """
    Entfernt alle Inhalte vor der ersten Überschrift in einem Markdown-String.
    Gibt den bereinigten Markdown-String zurück.
    Wird nur für Webseiten verwendet, da zu Beginn jeder Webseite umfangreiche Verlinkungen aufgelistet werden,
    die keine verwertbaren Informationen für das RAG-System enthalten.
    """
    first_heading = re.search(r"^#+", md_content, re.MULTILINE)
    if first_heading:
        md_content = md_content[first_heading.start():]
    return md_content


def calculate_content_hash(content: str) -> str:
    """
    Berechnet den Hashwert des Markdown-Inhaltes und gibt ihn als String zurück.
    Wird für eindeutige Dateinamen der Markdown-Dateien verwendet.
    """
    return hashlib.sha256(content.encode('utf-8')).hexdigest()


def process_json_source_and_create_md_files(json_source_path: str, output_dir: str, json_output_path: str) -> None:
    """
    Öffnet die JSON-Datei mit Studiengängen und Links, konvertiert die Dokumente in Markdown und speichert sie
    mit dem Hash-Wert als Dateinamen im Zielordner ab. Fügt JSON-Inhalt in neue Datei ein, ergänzt die Felder
    "hash" und "markdown_datei" mit Hash-Wert und Dateinamen.
     """
    logger.info("Öffne JSON Datei mit Studiengängen und Links:")
    with open(json_source_path, "r", encoding="utf-8") as file:
        json_data = json.load(file)

    logger.info("Gehe jeden Studiengang durch:")
    for studiengang in json_data["studiengaenge"]:
        logger.info(f"Studiengang: {studiengang}")

        for dokument in studiengang["dokumente"]:
            logger.info(f"Dokument: {dokument['typ']}, URL: {dokument['link']}")
            doc_type = dokument["typ"]
            link = dokument["link"]
            md_content = get_md_content(link)

            if doc_type == "html":
                md_content = remove_data_before_first_heading(md_content)  # Inhalte der Webseiten bereinigen

            if md_content:
                file_hash = calculate_content_hash(md_content)
                filename = f"{file_hash}.md"
                save_path = os.path.join(output_dir, filename)

                # Abfrage, falls URLs fälschlicherweise doppelt in der JSON eingetragen wurden
                if not os.path.exists(save_path):
                    with open(save_path, 'w', encoding='utf-8') as md_file:
                        md_file.write(md_content)
                else:
                    logger.warning(f"Datei {filename} existiert bereits und wird nicht erneut erstellt oder überschrieben.")
                dokument["hash"] = file_hash
                dokument["markdown_datei"] = filename

    logger.info("Speichere Änderungen in neuer JSON-Datei:")
    with open(json_output_path, "w", encoding="utf-8") as json_file:
        json.dump(json_data, json_file, indent=4)


def main():
    process_json_source_and_create_md_files(JSON_SOURCES, OUTPUT_DIR, JSON_OUTPUT)


if __name__ == "__main__":
    main()