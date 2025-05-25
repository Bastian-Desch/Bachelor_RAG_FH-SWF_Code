import os
import logging
from docling.document_converter import DocumentConverter

# Logging-Konfiguration, damit nicht alles als print-Statement ausgegeben werden muss.
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

DOCUMENT_SOURCES = {
    "webseite": "https://www.fh-swf.de/de/studienangebot/studiengaenge/wirtschaftsinformatik__b_sc__/wirtschaftsinformatik__b_sc__1.php",
    "fpo": "https://www.fh-swf.de/media/neu_np/hv_2/prfungsundnderungsordnungen/fbtbw_1/FPO_2018_WiIng_WiInf_IBE_IBI.pdf",
    "modulhandbuch": "https://www.fh-swf.de/media/neu_np/hv_2/dateien_sg_2_4/verlaufsplaene_modulhandbuecher_1/hagen/tbw_neu/Modulhandbuch_Winf_FPO_2018_Stand_09.2024.pdf",
}

BASE_OUTPUT_DIR = "Docling-Export-Beispiele"


def ensure_directory_exists(path: str) -> None:
    """Erstellt das Verzeichnis, falls es nicht existiert."""
    os.makedirs(path, exist_ok=True)


def convert_document(source: str, output_path: str, format: str) -> None:
    """
    Konvertiert ein Dokument in das Markdown-Format, HTML oder Text
    und speichert es an einem angegebenen Ordner. Unterscheidung ist nur für
    die Export-Methode aus Docling notwendig.
    """
    try:
        converter = DocumentConverter()
        result = converter.convert(source)

        if format == "markdown":
            content = result.document.export_to_markdown()
        elif format == "html":
            content = result.document.export_to_html()
        elif format == "text":
            content = result.document.export_to_text()
        else:
            logging.error(f"Unbekanntes Format: {format}")
            return

        ensure_directory_exists(os.path.dirname(output_path))

        with open(output_path, "w", encoding="utf-8") as file:
            file.write(content)

        logging.info(f"Datei erfolgreich gespeichert: {output_path}")

    except Exception as e:
        logging.error(f"Fehler beim Konvertieren von {source}: {e}")


def generate_files(format: str) -> None:
    """Erstellt alle Dateien in dem übergebenen Format."""
    for doc_name, source in DOCUMENT_SOURCES.items():
        output_path = os.path.join(BASE_OUTPUT_DIR, f"Beispiele-Docling-{doc_name}", f"{doc_name}-{format}.txt")
        convert_document(source, output_path, format)


if __name__ == "__main__":
    """
    Alle Exportformate generieren und in eigenen Ordnern speichern.
    Dient der Überprüfung, wie die Inhalte der verschiedenen ursprünglichen Dokumente der Webseiten/PDFs 
    exportiert werden. Erstellt einen Export in alle Formate für je ein Beispiel der genutzten Dokumente.
    Damit wurde festgestellt, mit welchem Format am besten gearbeitet werden kann.
    """
    generate_files("markdown")
    generate_files("html")
    generate_files("text")
