from docling.document_converter import DocumentConverter
import json
import hashlib
import os
import re
import logging

OUTPUT_DIR = "/Users/bastian/Desktop/BachelorProjektAbgabe/1-Datenbeschaffung-mit-Docling/Markdown-Dateien"
JSON_SOURCES = "/Users/bastian/Desktop/BachelorProjektAbgabe/1-Datenbeschaffung-mit-Docling/studiengaenge-und-links.json"
JSON_OUTPUT = "/Users/bastian/Desktop/BachelorProjektAbgabe/1-Datenbeschaffung-mit-Docling/studiengaenge-und-links-und-markdown-dateien.json"

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


def get_md_content(url: str) -> str:
    """Gibt den Inhalt eines Dokumentes im Markdown-Format als String zurück."""
    converter = DocumentConverter()  # Standard Konverter, es werden nur Webseiten und PDF von der Webseite genutzt
    try:  # Bilder werden nicht für die Datenbank für das Retrieval benötigt.
        result = converter.convert(url) # Konvertieren der URL mit Docling
        md_content = result.document.export_to_markdown(image_placeholder="")  # Erkannte Images entfernen ohne Platzhalter
        return md_content  # Inhalt als Markdown zurückgeben
    except Exception as e:
        logger.error(f"Fehler beim Konvertieren von {url}: {e}")
        return ""


def remove_data_before_first_heading(md_content: str) -> str:
    """Entfernt alle Inhalte vor der ersten Überschrift in einem Markdown-String.
       Gibt den bereinigten Markdown-String zurück."""
    # Suchen der ersten Überschrift. Auf Webseiten ist diese immer am Zeilenanfang.
    first_heading = re.search(r"^#+", md_content, re.MULTILINE)  # Betrachtet jeden Zeilenanfang, nicht nur String-Anfang
    if first_heading:
        # Entfernen aller Inhalte vor der ersten Überschrift. Webseiten haben sonst Listen von Links/Aufzählungen ohne Informationsgehalt
        md_content = md_content[first_heading.start():]  # Alles nach der ersten gefundenen Überschrift behalten
    return md_content


def calculate_content_hash(content: str) -> str:
    """Berechnet den Hashwert des Markdown-Inhaltes und gibt ihn als String zurück."""
    # Hashwert des Inhaltes berechnen, um doppelte Inhalte zu vermeiden.
    # Genutzt, um Dateien eindeutig zu benennen und doppelte Inhalte zu vermeiden, wenn Links auf identische Inhalte verweisen.
    return hashlib.sha256(content.encode('utf-8')).hexdigest()  # Hexadezimale Darstellung zurückgeben


def process_json_source_and_create_md_files(json_source_path: str, output_dir: str, json_output_path: str) -> None:
    """Öffnet die JSON-Datei mit Studiengängen und Links, konvertiert die Dokumente in Markdown und speichert sie
     mit dem Hash-Wert als Dateinamen im Zielordner ab. Fügt JSON-Inhalt in neue Datei ein"""
    logger.info("Öffne JSON Datei mit Studiengängen und Links:")
    with open(json_source_path, "r", encoding="utf-8") as file:
        json_data = json.load(file)

    # Jeden Studiengang durchgehen
    logger.info("Gehe jeden Studiengang durch:")
    for studiengang in json_data["studiengaenge"]:
        logger.info(f"Studiengang: {studiengang}")

        # Jedes Dokument durchgehen
        for dokument in studiengang["dokumente"]:
            logger.info(f"Dokument: {dokument['typ']}, URL: {dokument['link']}")
            # Typen nehmen
            doc_type = dokument["typ"]
            # Link nehmen
            link = dokument["link"]
            # Markdown-Inhalt holen
            md_content = get_md_content(link)

            # Inhalt bereinigen, wenn der Typ HTML ist, sonst nicht, da PDF keine störenden Daten am Anfang haben
            if doc_type == "html":
                md_content = remove_data_before_first_heading(md_content)

            # Wenn Inhalt erstellt wurde, Inhalt in Markdown-Datei speichern.
            # Dateiname ist der Hashwert des Inhaltes, um zu verhindern, dass Inhalte doppelt gespeichert werden.
            if md_content:
                # Hash berechnen
                file_hash = calculate_content_hash(md_content)

                # Dateinamen erstellen:
                filename = f"{file_hash}.md"

                # Dateipfad für das Speichern
                save_path = os.path.join(output_dir, filename)

                # Speichern der Datei
                if not os.path.exists(save_path):
                    with open(save_path, 'w', encoding='utf-8') as md_file:
                        md_file.write(md_content)
                else:
                    logger.warning(f"Datei {filename} existiert bereits und wird nicht erneut erstellt oder überschrieben.")
                dokument["hash"] = file_hash
                dokument["markdown_datei"] = filename

    # Speichern der Änderungen in neuer JSON, um Ursprungsdatei zu erhalten, falls später etwas geändert werden soll (z. B. andere Metadaten/Studiengänge)
    logger.info("Speichere Änderungen in neuer JSON-Datei:")
    with open(json_output_path, "w", encoding="utf-8") as json_file:
        json.dump(json_data, json_file, indent=4) # Warnung kann hier ignoriert werden


def main():
    process_json_source_and_create_md_files(JSON_SOURCES, OUTPUT_DIR, JSON_OUTPUT)


if __name__ == "__main__":
    main()