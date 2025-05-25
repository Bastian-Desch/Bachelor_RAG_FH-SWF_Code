# In dieser README wird der Aufbau des Projektes erklärt.

Das Projekt enthält den Quellcode und Dateien für die praktische Ausarbeitung, die im Rahmen meiner Bachelorarbeit durchgeführt wurde.

# Projektstruktur

# 1. [Datenbeschaffung mit Docling](./1-Datenbeschaffung-mit-Docling)

Dieser Teil enthält den gesamten Code für das **_Kapitel 4.2 Datenbeschaffung mit Docling_** der Bachelorarbeit.<br/>
Das Jupyter Notebook [docling_test.ipynb](./1-Datenbeschaffung-mit-Docling/docling_test.ipynb) enthält
die ersten Beispiele dazu, wie mit Docling die Quellen konvertiert werden können.<br/><br/>

In der Datei [docling_export_examples.py](./1-Datenbeschaffung-mit-Docling/Docling-Export-Beispiele/docling_export_examples.py) wird 
der Export bzw. die Konvertierung in HTML, Text und Markdown umgesetzt und die Ergebnisse in <br/>
Textdateien gespeichert.<br/>
Zweck war es, festzustellen, welches Format für die weitere Verarbeitung am besten geeignet ist.<br/>
<br/>
Dabei hat sich gezeigt, dass der Export in das Text-Format mit Informationsverlusten verbunden ist.<br/>
Das HTML-Format liefert zwar alle Informationen, die HTML-Tags müssten aber gesondert entfernt werden,
wobei Tabellen erhalten bleiben müssten, <br/>
um ihre Struktur für das LLM zu erhalten.<br/>
Mit dem Markdown-Format kann der Export direkt weiterverwendet werden, da Tabellenstrukturen erhalten bleiben.<br/>
<br/><br/>
An dieser Stelle hat sich auch gezeigt, dass auf Webseiten eine Reihe von Links zu Beginn der Seite enthalten ist,<br/>
diese müssen entfernt werden, da sie keine Informationen enthalten, die für die weitere Verarbeitung wichtig sind.<br/>

<hr/>

Die JSON-Datei [studiengaenge-und-links.json](./1-Datenbeschaffung-mit-Docling/studiengaenge-und-links.json) 
enthält die Links zu den Webseiten der Studiengänge. Sie entspricht der in **_Kapitel 4.2, Abbildung 7_** schematisch<br/>
dargestellten JSON-Datei.<br/>
Das Skript [get_source_data.py](./1-Datenbeschaffung-mit-Docling/get_source_data.py) liest die JSON-Datei ein, 
konvertiert die Webseiten und PDF-Dateien in das Markdown-Format und speichert die Ergebnisse <br/>in dem Ordner
[Markdown-Dateien](./1-Datenbeschaffung-mit-Docling/Markdown-Dateien) ab.<br/>
Zusätzlich erstellt es eine neue JSON-Datei [studiengaenge-und-links-und-markdown-dateien.json](./1-Datenbeschaffung-mit-Docling/studiengaenge-und-links-und-markdown-dateien.json),
 welche später für das Erstellen der Chunks und die Zuordnung <br/>
der Metadaten genutzt wird.<br/><br/>
**_Von einer erneuten Ausführung ist abzuraten, da die URLs zu den Webseiten und PDF-Dateien teilweise nicht mehr verfügbar sind._**<br/>
Bereits während der Durchführung dieser Ausarbeitung wurde ein Link der Webseite der FH-SWF entfernt.<br/>

<hr/>

# 2. [Chunking mit LangChain](./2-Chunking-mit-LangChain)

Dieser Abschnitt enthält den Quellcode für Kapitel **_4.4 Chunking-Verfahren_** der Bachelorarbeit.

Die Funktionen für das Erstellen der Chunks sind in der Datei [chunking_util.py](./2-Chunking-mit-LangChain/chunking_util.py) enthalten.<br/>
In der Datei [chunks_to_textfiles.py](./2-Chunking-mit-LangChain/chunks_to_textfiles.py) werden die Chunks unter Verwendung der implementierten Funktionalitäten in Textdateien gespeichert.<br/>
In dem Ordner [2-Chunking-mit-LangChain-TextDateien](./2-Chunking-mit-LangChain/2-Chunking-mit-LangChain-TextDateien) befindet sich zu jedem erprobten Chunking-Verfahren ein Ordner je Embedding-Modell. <br/>
<br/>
Für den SemanticChunker wurden verschieden Kombinationen der Parameter "sentence_split_regex" und "breakpoint_threshold_type"
und "breakpoint_threshold_amount" getestet. <br/>
Nur die letzte genutzte Kombination ist in den Textdateien vorhanden. Die Probleme bei allen Kombinationen waren identisch und sind in der Arbeit beschrieben.


Grund für die Nutzung beider Embedding-Modelle war hier, dass im Vorhinein nicht klar war, ob die Tokenizer beider Modelle 
zu identischen Chunks-Führen würden.<br/>
Dies war jedoch der Fall, wie sich hier in den Textdateien und in den **_Abbildungen 17, 18, 19 und 20_** in der Bachelorarbeit zeigt.<br/>
Diese Abbildungen entstammen ebenfalls aus dem Skript [chunks_to_textfiles.py](./2-Chunking-mit-LangChain/chunks_to_textfiles.py), dort wurden zu jedem Chunking-Verfahren die <br/>
die Tokens pro Chunk gezählt und die Häufigkeit der jeweiligen Länge mitgezählt, um dies anschließend in einem Diagramm darzustellen.<br/>

<hr/>

Die Textdateien enthalten hier bereits die Metadaten und die Anzahl der Tokens pro Chunks zu jedem Chunk.<br/>
Dies wurde später in der Evaluation relevant. Die Dokumente in der Vektordatenbank enthielten ebenfalls die Metadaten,<br/>
wodurch über die Hash-Werte und IDs in den abgerufenen Dokumenten die Inhalte der Chunks in den Textdateien nachvollzogen werden konnten.<br/>
In der Vorbereitung der Evaluation wurden in diesen Textdateien diejenigen Chunks identifiziert, die <br/>
die relevanten Inhalte zu einer Frage enthalten haben. So konnten die IDs der Chunks identifiziert werden, <br/>
um die Qrels Dicitionaries mit den Fragen_IDs und Dokumenten_IDs zu erstellen. <br/>
Eine solche Identifizierung der relevanten Dokumente wäre über die Qdrant-Datenbnak sonst nur schwer möglich gewesen.

<hr/>

# 3. [Vektordatenbank-mit-Qdrant](./3-Vektordatenbank-mit-Qdrant/)

Dieser Abschnitt enthält den Quellcode für Kapitel **_4.5 Aufbau der Datenbank-Collections_** der Bachelorarbeit.

Insbesondere die in **_Kapitel 4.5.2 (Abbildung 21)_** beschriebenen Schritte werden umgesetzt.<br/>
Die [Docker-Compose-Datei](./3-Vektordatenbank-mit-Qdrant/docker-compose.yml) kann der Qdrant-Dokumentation entnommen werden.<br/>
Die Inhalte der Datenbank wurden in dem lokalen Ordner [qdrant_data](./3-Vektordatenbank-mit-Qdrant/qdrant_data/) gespeichert. Mit dem Bind Mount in der Compose-Datei werden die Daten der Datenbank erhalten und wieder eingelesen.<br/>

<hr/>

In der Datei [chunking_util_for_vector_store.py](./3-Vektordatenbank-mit-Qdrant/chunking_util_for_vector_store.py) sind die Funktionalitäten für das Erstellen der Chunks, deren Embeddings und den Upload in die Datenbank implementiert.<br/>
Die Funktionen sind nahezu identisch zu denen, die für das Erstellen und Speichern der Chunks in Textdateien genutzt wurden, sind jedoch an die Nutzung für die Qdrant-Datenbnak angepasst.<br/>
Außerdem sind dort nur die erforderlichen Funktionen (TextSplitter) vorhanden.<br/>

<hr/>

Das Skript [create_collections.py](./3-Vektordatenbank-mit-Qdrant/create_collections.py) greift auf diese Funktionen zurück und erstellt die 4 Collections für die Kombinationen aus Chunking-Verfahren und Embedding-Modellen.<br/>

<hr/>

Das Jupyter-Notebook [query_examples.ipynb](./3-Vektordatenbank-mit-Qdrant/query_examples.ipynb) diente den ersten Versuchen, eine Verbindung zu den Collections aufzubauen und diese zu durchsuchen.<br/>
Die Datei hat sonst keine weitere Relevanz.

<hr/>

**_Eine erneute Ausführung des create_collections.py Skriptes sollte nicht notwendig sein, da die Collections in einem Docker Container aus dem lokalen Ordner geladen werden. <br/>
Ein einfaches docker compose up über die Konsole in diesem Ordnerpfad startet die Datenbank mit diesen Daten._**

# 4. [Retrieval-LangChain-Qdrant](./4-Retrieval-LangChain-Qdrant/)

Dieser Abschnitt diente lediglich der Erprobung des Abrufs von Dokumenten aus der Datenbank anhand von 10 Fragen.<br/>
In diesem Schritt wurde mit den Fragestellungen gearbeitet, die in der Bachelorarbeit als **_"Umformulierte Fragen"_** bezeichnet wurden.<br/>
Dies diente der umfänglichen Überprüfung der Nutzbarkeit des SelfQueryRetrievers. An dieser Stelle war noch nicht klar, dass dieser Retriever aufgrund der in der Arbeit erläuterten Herausforderungen keine Vergleichbarkeit zu den anderen Verfahren (Abruf mit und ohne manuell gesetzte Filter mit den ursprünglichen Fragestellungen) bietet.<br/>
Die Ergebnisse der Abrufe wurden daher in Textdateien gespeichert, um diese nachvollziehen zu können.

<hr/>

Der Abruf erfolgte hier schon für alle Collections, um auch nachvollziehen zu können, wie sich diese unterscheiden.

<hr/>

# 5. [Antworten-LangChain-Qdrant-Ollama](./5-Antworten-LangChain-Qdrant-Ollama/)

Dieser Abschnitt nutzt die umformulierten Fragen und dient ausschließlich der Erprobung, wie Antworten mit abgerufenen Dokumenten generiert werden können,<br/> indem in einem Prompt Platzhalter für Fragen und Kontext gesetzt werden, die erst nachträglich befüllt werden. Der Prompt wird dann über die Ollama-API an das LLM übergeben.


Dieser Abschnitt ist somit nicht weiter relevant für die Evaluation, sondern zeigt nur das grundsätzliche Vorgehen.

<hr/>

# 6. [Evaluation-Retrieval](./6-Evaluation-Retrieval/) und [Evaluation-Vereinfachte-Fragen](./6-Evaluation-Retrieval-Vereinfachte-Fragen/) müssen zusammen betrachtet werden

Dieser Abschnitt enthält den Quellcode für die Kapitel **_4.6 Retrieval mit LangChain und Qdrant_** und **_4.7 Evaluation und Resultate des Retrievals_** der Bachelorarbeit.

Dieser Abschnitt diente der Evaluation des Retrievals mit:

1.  Dem SelfQueryRetriever
2.  Dem Abruf ohne Filter
3.  Dem Abruf mit manuell erstellten Filtern

[6-Evaluation-Retrieval](./6-Evaluation-Retrieval/) erfolgte mit den **_"Umformulierten Fragen"_**<br/>
[6-Evaluation-Vereinfachte-Fragen](./6-Evaluation-Retrieval-Vereinfachte-Fragen/) erfolgte mit den **_"Ursprünglichen Fragen" und lieferte die Ergebnisse, die auch in der Bachelorarbeit betrachtet wurden_**

Hier werden die **ursprünglichen Fragen** als **"Vereinfachte Fragen"** bezeichnet, da sie keine Schlagworte enthalten.

<hr/>

In diesem Abschnitt wurde deutlich, dass keine Vergleichbarkeit des _SelfQueryRetrievers_ mit den ursprünglichen Fragen **(hier vereinfachte Fragen genannt)** und den umformulierten Fragen gegeben ist.<br/>
Die Problematik wird in der Bachelorarbeit beschrieben.<br/>
Die Evaluationsergebnisse mit allen drei Verfahren wurden ermittelt, die des SelfQueryRetrievers wurden jedoch nicht betrachtet.<br/>
Sie können hier dennoch nachvollzogen werden.

<hr/>

Die Zusammenstellung der genutzten Mappings (Qrels) für die Evaluation kann der Markdowndatei [fragen_relevante_inhalte.md](./6-Evaluation-Retrieval/fragen_relevante_inhalte.md) entnommen werden.

<hr/>

Um die Evaluation mit **ranx** nachzuvollziehen, eignen sich die Notebooks:

1.  [infloat_collections_evaluation.ipynb](./6-Evaluation-Retrieval-Vereinfachte-Fragen/infloat_modell/infloat_collections_evaluation.ipynb)
2.  [mixedbread_collections_evaluation.ipynb](./6-Evaluation-Retrieval-Vereinfachte-Fragen/mixedbread_modell/mixedbread_collections_evaluation.ipynb)

<br/>
In beiden Fällen wurden immer die Kennzahlen @5 und @10 ermittelt.<br/>
In der Arbeit wurden nur die Werte @10 betrachtet, da dies auch das Vorgehen in dem MTEB-Benchmark ist, welches die Basis der Auswahl der beiden Embedding-Modelle war.

# 7. [Evaluation-Antworten](./7-Evaluation-Antworten/)

Dieser Abschnitt enthält den Quellcode für die Kapitel **_4.8 Generieren der Antworten_** und **_4.9 Evaluation und Resultate der Antworten_** der Bachelorarbeit.

Dieser Abschnitt generiert die Antworten zu den ursprünglichen Fragestellungen.<br/>
Dafür werden die Informationen aus der Collection abgerufen, die die besten Retrieval-Ergebnisse lieferte.<br/>
Genutzt wurde hier das lokal ausgeführte Llama3.1:8b (4-Bit quantisiert) über die Ollama-API.<br/>

[Code zum Generieren der 10 Antworten](./7-Evaluation-Antworten/antworten_generieren.ipynb)
<br/>
[Code zum Generieren der 5 Antworten zur Bewertung des Aspektes der Negative Rejection](./7-Evaluation-Antworten/negative_rejection_fragen_antworten.ipynb)

Die Antworten zu den 10 Fragen der Evaluation sowie 5 weitere Fragen zur Bewertung des Aspektes der Negative Rejection sind in Textdateien abgespeichert.<br/>
So können sie nachträglich eingesehen und die Bewertungen der Arbeit nachvollzogen werden. Dafür sind sie auch als Anhang an die Arbeit angefügt.

[Antworten auf die 10 Fragen der Evaluation](./7-Evaluation-Antworten/antworten_llama_base_modell)
<br/>
[Antworten auf die 5 Fragen zur Bewertung des Aspektes der Nagtive Rejection](./7-Evaluation-Antworten/antworten_llama_base_model_negative_rejection)

<hr/>

Die Evaluation erfolgte in der Bachelorarbeit rein qualitativ.<br/>
Der Versuch, mit dem _Ragas-Framework_ Kennzahlen zu ermitteln, war erfolglos und wurde daher aus diesem Projekt vollständig entfernt.<br/> 
Dies begründet den Wegfall quantitativer Bewertungen der Antworten.

# 8. [App-Streamlit](./8-App-Streamlit/)

Greift auf die Dateien [enums.py](./enums.py) und [generation_util.py](./generation_util.py) zurück, um mit der 
[fh_qa_app.py](./8-App-Streamlit/fh_qa_app.py) eine Anwendung über den Browser zu betreiben.
<br/>
Diese basiert auf Streamlit und ermöglicht es, drei Filterkriterien zu setzen und Fragen an das System zu stellen.<br/>
**_Die App dient als Unterstützung der Präsentation der Arbeit im Kolloquium_**. Daher findet diese keine Betrachtung in der Bachelorarbeit.
