Hier werden die Fragen erfasst und die relevanten Links.<br/>
Damit können dann in den Text-Dateien aus dem Chunking-mit-LangChain die IDs relevanter Chunks ermittelt werden.<br/>
Mit diesen kann dann ein qrels Dictionary für jede Frage mit den relevanten Chunk-IDs erstellt werden.<br/>
Da beide Modelle mit der gleichen Tokenisierung arbeiten, reicht es die **Chunks einmalig herauszusuchen!**, <br/>
um die Mappings (qrels) zu erstellen.<br/>

<hr>

# Frage 1: Wann kann ich mich für den Master in Elektrotechnik in Meschede einschreiben?
```text
Link zur Quelle: https://www.fh-swf.de/de/studienangebot/studiengaenge/elektrotechnik_m_eng_/index.php

Relevante Chunk-IDs und Relevanzbewertung.
Token-Basierte-Splits
"id": 1021, 10 -> Hat die volle Antwort
"id": 1022, 1 -> Hat überschneidung mit der Antwort

Der Chunks 1022 beginnt mit den letzten 50 Token des 1021. Daher enthält er noch einen Teil der Antwort, ist aber nur minimal relevant.

Die Bewertung der Relevanz sollte auch nur die Reihenfolge vorgeben und ansonsten nicht weiter wichtig sein. Kritisch zu sehen ist hier, die subjektive Einschätzung.
Bei den Anderen Fragen wird das nicht mehr weiter begründet.

Rekursive-Splits:
"id": 1370, 1
"id": 1371, 10
```

<hr>

# Frage 2: Welche Literatur ist relevant für das Modul Mathematik im Bachelorstudiengang Elektrotechnik in Soest?
```text
Token-Basierte-Splits:
Link zur Quelle: https://www.fh-swf.de/media/neu_np/hv_2/dateien_sg_2_4/verlaufsplaene_modulhandbuecher_1/soest_4/fb_eet_3/Modulhandbuch_BA_ETS_ETA_ETP_20_02.2024.pdf
"id": 1301, 10

Rekursive Splits:
Link zur Quelle: https://www.fh-swf.de/media/neu_np/hv_2/dateien_sg_2_4/verlaufsplaene_modulhandbuecher_1/soest_4/fb_eet_3/Modulhandbuch_BA_ETS_ETA_ETP_20_02.2024.pdf
"id": 1866, 10
```

<hr>

# Frage 3: Welche Fachgebiete sind an der FH vertreten?
```text
Token-Basierte-Splits:
Link zur Quelle: https://www.fh-swf.de/de/studienangebot/index.php
"id": 1480, 10
"id": 1481, 2
Link zur Quelle: https://www.fh-swf.de/de/index.php
"id": 1476: 10

Rekursive-Splits:
Link zur Quelle: https://www.fh-swf.de/de/studienangebot/index.php
"id": 2292, 10
"id": 2293, 2
Link zur Quelle: https://www.fh-swf.de/de/index.php
"id": 2286, 10
```

<hr>

# Frage 4: Ist eine Beurlaubung während des Studiums möglich?
```text
Token-Basierte-Splits
Link zur Quelle: https://www.fh-swf.de/de/studierende/studienorganisation/index.php
"id": 1498, 10
"id": 1499, 10
"id": 1500, 2

Rekursive-Splits:
Link zur Quelle: https://www.fh-swf.de/de/studierende/studienorganisation/index.php
"id": 2313, 10
"id": 2314, 10
"id": 2315, 2
```
<hr>

# Frage 5: Welche Studienmodelle werden angeboten?
```text
Token-Basierte-Splits
Link zur Quelle: https://www.fh-swf.de/de/studienangebot/studienmodelle/index.php
"id": 1482, 10
"id": 1483, 10
"id": 1484, 10
"id": 1485, 10


Rekursive-Splits
Link zur Quelle: https://www.fh-swf.de/de/studienangebot/studienmodelle/index.php
"id": 2295, 10
"id": 2296, 10
"id": 2297, 10
"id": 2298, 10
Link zur Quelle: https://www.fh-swf.de/de/studieninteressierte/index.php
"id": 2290, 4
```

<hr>

# Frage 6: Wie viele Seiten muss ich in meiner Bachelorarbeit in Wirtschaftsinformatik in Hagen schreiben?

```text
Token-Basierte-Splits:
Link zur Quelle: https://www.fh-swf.de/media/neu_np/hv_2/prfungsundnderungsordnungen/fbtbw_1/FPO_2018_WiIng_WiInf_IBE_IBI.pdf
"id": 18, 10

Rekursive-Splits:
Link zur Quelle: https://www.fh-swf.de/media/neu_np/hv_2/prfungsundnderungsordnungen/fbtbw_1/FPO_2018_WiIng_WiInf_IBE_IBI.pdf
"id": 25, 10
```

<hr>

# Frage 7: Wie sind Portfolioprüfungen im Bachelorstudiengang Elektrotechnik in Soest aufgebaut?

```text
Token-Basierte-Splits:
Link zur Quelle: https://www.fh-swf.de/media/neu_np/hv_2/dateien_sg_2_4/online_antraege_formulare_soest/Lesefassung_FPO_BA_ETS_20_03.2023.pdf
"id": 1192, 10
"id": 1193, 8

Rekursive-Splits:
Link zur Quelle: https://www.fh-swf.de/media/neu_np/hv_2/dateien_sg_2_4/online_antraege_formulare_soest/Lesefassung_FPO_BA_ETS_20_03.2023.pdf
"id": 1586, 10
```

<hr>

# Frage 8: Was muss ich beachten, wenn ich bei einer Prüfung krank bin?

```text
Token-Basierte-Splits:
Link zur Quelle: https://www.fh-swf.de/media/neu_np/hv_2/dateien_sg_2_4/online_pruefungsorga/Ruecktritt-Info-2023.pdf
"id": 1553, 10
"id": 1554, 10
"id": 1555, 10
Link zur Quelle: https://www.fh-swf.de/media/neu_np/hv_2/prfungsundnderungsordnungen/rahmenpruefungsordnung/RPO_2018_endgueltig.pdf
"id": 1530, 10

Rekursive-Splits:
Link zur Quelle: https://www.fh-swf.de/media/neu_np/hv_2/dateien_sg_2_4/online_pruefungsorga/Ruecktritt-Info-2023.pdf
"id": 2388, 10
"id": 2389, 10
"id": 2390, 10
Link zur Quelle: https://www.fh-swf.de/media/neu_np/hv_2/prfungsundnderungsordnungen/rahmenpruefungsordnung/RPO_2018_endgueltig.pdf
"id": 2356, 8
"id": 2357, 8
```

<hr>

# Frage 9: Was sind die Inhalte im Modul IT-Sicherheit im Bachelorstudiengang Elektrotechnik in Hagen?

```text
Token-Basierte-Splits:
Link zur Quelle: https://www.fh-swf.de/media/neu_np/hv_2/dateien_sg_2_4/verlaufsplaene_modulhandbuecher_1/hagen/e___i/Modulhandbuch_Elektrotechnik_2024_Juli_2024.pdf
"id": 704, 10
 
Rekursive-Splits:
Link zur Quelle: https://www.fh-swf.de/media/neu_np/hv_2/dateien_sg_2_4/verlaufsplaene_modulhandbuecher_1/hagen/e___i/Modulhandbuch_Elektrotechnik_2024_Juli_2024.pdf
"id": 946, 10
```

<hr>

# Frage 10: Wie lange dauern die Klausuren im Bachelorstudiengang Wirtschaftsinformatik in Hagen?

```text
Token-Basierte-Splits:
Link zur Quelle:
"id": 14, 10
"id": 15, 10

Rekursive-Splits:
Link zur Quelle:
"id": 19, 10
```

<hr>

# Tokenbasierte-Splits Qrels mit Fragen-ID und den erwarteten relevanten Dokumenten:

```json
{
  "q_1": {
    "d_1021": 10,
    "d_1022": 1
  },
  "q_2": {
    "d_1301": 10
  },
  "q_3": {
    "d_1476": 10,
    "d_1480": 10,
    "d_1481": 2
  },
  "q_4": {
    "d_1498": 10,
    "d_1499": 10,
    "d_1500": 2
  },
  "q_5": {
    "d_1482": 10,
    "d_1483": 10,
    "d_1484": 10,
    "d_1485": 10
  },
  "q_6": {
    "d_18": 10
  },
  "q_7": {
    "d_1192": 10,
    "d_1193": 8
  },
  "q_8": {
    "d_1530": 10,
    "d_1553": 10,
    "d_1554": 10,
    "d_1555": 10
  },
  "q_9": {
    "d_704": 10
  },
  "q_10": {
    "d_14": 10,
    "d_15": 10
  }
}
```

<hr>

# Rekursive Splits Qrels mit Fragen und relevanten Dokumenten:

```json
{
  "q_1": {
    "d_1371": 10,
    "d_1370": 1
  },
  "q_2": {
    "d_1866": 10
  },
  "q_3": {
    "d_2292": 10,
    "d_2293": 2,
    "d_2286": 10
  },
  "q_4": {
    "d_2313": 10,
    "d_2314": 10,
    "d_2315": 2
  },
  "q_5": {
    "d_2290": 4,
    "d_2295": 10,
    "d_2296": 10,
    "d_2297": 10,
    "d_2298": 10
  },
  "q_6": {
    "d_25": 10
  },
  "q_7": {
    "d_1586": 10
  },
  "q_8": {
    "d_2388": 10,
    "d_2389": 10,
    "d_2390": 10,
    "d_2356": 8,
    "d_2357": 8
  },
  "q_9": {
    "d_946": 10
  },
  "q_10": {
    "d_19": 10
  }
}
```

<hr>

Nach dem Vergleich der Run-Dictionaries, die mit meiner eigenen Funktion (retrieve_run_dict_with_documents_with_scores) erzeugt wurden, und den Abrufen aus dem Ordner 4-Retrieval-LangChain-Qdrant<br>
ist sicher, dass die erhaltenen Dokumente in beiden Fällen identisch sind. Die Reihenfolgen sind ebenfalls identisch.<br>
Die eigene Funktion liefert somit korrekte Ergebnisse und kann in der Evaluation mit Ranx genutzt werden.<br>

Die Funktionen zum Abruf mit score werden später weiter verwendet, auch wenn der score nicht mehr für das Generieren der Antworten relevant ist.


Rückblicken ist eine Relevanzbewertung von 1 bis 10 nicht erforderlich gewesen. Eine Gliederung von 1 bis 5 hätte auch gereicht.<br>
Grund ist, dass in der Regel die Informationen, die zu einer Frage relevant waren, auch nur in einem Chunk waren.<br>
Dadurch entstehen Chunks, die entweder vollständig relevant sind oder nicht.<br>
Einige Chunks hatten auch nur den Token-Overlap eines relevanten Chunks. Diese wurden mit 1 oder 2 bewertet.

<hr>

```text
Wieder eine Inkonsistenz mit Docling gefunden.
Unter https://www.fh-swf.de/de/studienangebot/index.php wird der Abschnitt zu den Studienmodellen nicht mit exportiert.
Danach die Elemente aber schon.
Das zeigt, dass Docling nicht ideal für die Webseiten geeignet ist.


Allgemeines Problem der Webseiten ist, dass die Informationen gleichzeitig redundant vorhanden sind und über unzusammenhängende Seiten verteilt sind.

Retrieval abgerufen werden.
```