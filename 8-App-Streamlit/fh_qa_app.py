import torch
import os
import sys

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))  # Erforderlich, um aus den ..._util.py die Funktionen zu nutzen
import streamlit as st
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage

from generation_util import (
    create_llm_for_answers, prepare_context, generate_answer
)
from retrieval_util import (
    get_embeddings_function, get_qdrant_client, create_filter_from_values, retrieve_documents_with_filter_without_scores
)
from enums import EmbeddingModels, CollectionNames

torch.classes.__path__ = []  # Workaround für Torch-Problem mit Streamlit-Watcher für automatische Updates nach Änderungen am Code

system_first_message = ""

# Streamlit-Seitenkonfiguration
st.set_page_config(page_title="FH-SWF Q&A-Assistent", page_icon="🎓", layout="wide")
st.title("Assistent der Fachhochschule Südwestfalen für deine Fragen zum Studium")

# Sidebar-Filter als Drop-Down-Menüs
st.sidebar.header(
    "Für Fragen zu einem Studiengang, wähle den Studiengang, den Standort und den Abschluss aus.\nFür allgemeine Fragen wähle 'Alle' aus.")
studiengang = st.sidebar.selectbox("Studiengang", ["Alle", "Elektrotechnik", "Wirtschaftsinformatik"])
standort = st.sidebar.selectbox("Standort", ["Alle", "Meschede", "Soest", "Iserlohn", "Hagen", "Lüdenscheid"])
abschluss = st.sidebar.selectbox("Abschluss", ["Alle", "Bachelor", "Master"])

# Verlauf löschen (ist auch in der Sidebar, ist nicht wirklich notwendig, da Chatverlauf nicht an das LLM übergeben wird)
clear_chat = st.sidebar.button("Alte Fragen und Antworten löschen 🗑️")


# Notwendige Ressourcen im Cache speichern
@st.cache_resource
def load_llm():
    return create_llm_for_answers()


@st.cache_resource
def load_embeddings():
    return get_embeddings_function(model_name=EmbeddingModels.MIXEDBREAD.value)


@st.cache_resource
def load_qdrant_client():
    return get_qdrant_client(embeddings=load_embeddings(),
                             collection_name=CollectionNames.MIXEDBREAD_RECURSIVE.value,
                             qdrant_url="http://localhost:6333")


# Nach dem Löschen des Verlaufs wird wieder die erste Nachricht in den Chatverlauf geschrieben
if clear_chat:
    st.session_state.chat_history = [
        SystemMessage(content="Hallo, ich bin der Assistent der Fachhochschule Südwestfalen.\n"
                              "Stelle mir Fragen zum Studium an der FH-SWF.")
    ]

# Ressourcen dem session_state hinzufügen
if "llm" not in st.session_state:
    st.session_state.llm = create_llm_for_answers()

if "embeddings" not in st.session_state:
    st.session_state.embeddings = get_embeddings_function(model_name=EmbeddingModels.MIXEDBREAD.value)

if "qdrant_client" not in st.session_state:
    st.session_state.qdrant_client = get_qdrant_client(embeddings=st.session_state.embeddings,
                                                       collection_name=CollectionNames.MIXEDBREAD_RECURSIVE.value,
                                                       qdrant_url="http://localhost:6333")

# Chat initialisieren
if "chat_history" not in st.session_state:
    st.session_state.chat_history = [
        SystemMessage(content="Hallo, ich bin der Assistent der Fachhochschule Südwestfalen.\n"
                              "Stelle mir Fragen zum Studium an der FH-SWF.")
    ]

# Anzeige der bisherigen Nachrichten
for message in st.session_state.chat_history:
    if isinstance(message, HumanMessage):
        with st.chat_message("Human"):
            st.markdown(message.content)
    else:
        with st.chat_message("AI"):
            st.markdown(message.content)

# Eingabeaufforderung steht als Platzhalter in dem Eingabefenster
user_query = st.chat_input("Stelle hier deine Frage: ")

if user_query is not None and user_query != "":
    # Die Nutzereingabe wird zuerst dem Chatverlauf angehangen
    st.session_state.chat_history.append(HumanMessage(content=user_query))
    with st.chat_message("Human"):
        st.markdown(user_query)

    # Filter setzt sich aus allen drei Werten der Dropdown-Menüs zusammen (logische UND)
    query_filter = create_filter_from_values(studiengang=studiengang, standort=standort, abschluss=abschluss)
    # Abgerufene Chunks zwischenspeichern
    documents = retrieve_documents_with_filter_without_scores(client=st.session_state.qdrant_client,
                                                              max_documents=10,
                                                              filters=query_filter,
                                                              query=user_query)
    # Kontext für den Prompt formatieren
    prepared_context, links = prepare_context(retrieved_documents=documents)
    with st.spinner("Ich generiere die Antwort..."):  # Spinner Anzeigen, bis die Antwort generiert wurde
        ai_response = generate_answer(query=user_query, context=prepared_context, llm=st.session_state.llm)
        link_list = ""
        for link in links:  # Hier nicht nur auf die ersten 2 Links beschränkt, da ggf. auch andere Fragen gestellt werden können
            link_list += f"- {link}\n"  # Links werden nach der Antwort des LLMs ausgegeben (Nicht regelbasiert oder durch LLM verändert)

        ai_response += f"\n\n**Weitere Informationen findest du hier:**\n{link_list}"
        with st.chat_message("AI"):
            st.markdown(ai_response, unsafe_allow_html=True)
        st.session_state.chat_history.append(AIMessage(content=ai_response))  # Antwort + Links dem Chatverlauf anhängen
