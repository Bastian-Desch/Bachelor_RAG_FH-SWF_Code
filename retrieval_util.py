import logging
from typing import List, Dict, Any, Optional

from langchain_huggingface import HuggingFaceEmbeddings
from langchain_qdrant import RetrievalMode, QdrantVectorStore
from qdrant_client.models import Filter, FieldCondition, MatchValue
from langchain_core.documents import Document
from langchain_ollama import ChatOllama
from langchain.retrievers.self_query.base import SelfQueryRetriever
from langchain.chains.query_constructor.schema import AttributeInfo

from enums import CollectionNames, EmbeddingModels, OllamaModels

# Logging für die Streamlit App
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.StreamHandler()
    ]
)

# Quelle: https://python.langchain.com/docs/how_to/add_scores_retriever/#selfqueryretriever
class CustomSelfQueryRetriever(SelfQueryRetriever):
    def _get_docs_with_query(
            self, query: str, search_kwargs: Dict[str, Any]
    ) -> List[Document]:
        """Get docs, adding score information."""
        docs, scores = zip(
            *self.vectorstore.similarity_search_with_score(query, **search_kwargs)
        )
        for doc, score in zip(docs, scores):
            doc.metadata["score"] = score

        return docs


EMBEDDING_MODELS = [
    "mixedbread-ai/deepset-mxbai-embed-de-large-v1",
    "intfloat/multilingual-e5-large"
]

COLLECTIONS = [
    "intfloat_multilingual-e5-large_token_based_chunks",
    "intfloat_multilingual-e5-large_recursive_chunks",
    "mixedbread-ai_deepset-mxbai-embed-de-large-v1_token_based_chunks",
    "mixedbread-ai_deepset-mxbai-embed-de-large-v1_recursive_chunks"
]


def get_qdrant_client(embeddings: HuggingFaceEmbeddings, collection_name: str, qdrant_url: str) -> QdrantVectorStore:
    if collection_name not in COLLECTIONS:
        raise ValueError(f"{collection_name} is not a valid collection name")
    else:
        client = QdrantVectorStore.from_existing_collection(
            embedding=embeddings,
            retrieval_mode=RetrievalMode.DENSE,
            prefer_grpc=False,
            collection_name=collection_name,
            url=qdrant_url
        )
        return client


def get_embeddings_function(model_name: str) -> HuggingFaceEmbeddings:
    encode_kwargs = {"prompt": "query: "}
    if model_name not in EMBEDDING_MODELS:
        raise ValueError(f"{model_name} is not a valid embedding model")
    else:
        return HuggingFaceEmbeddings(model_name=model_name, encode_kwargs=encode_kwargs)


def create_field_condition(field: str, value: str) -> FieldCondition:
    return FieldCondition(
        key=f"metadata.{field}",
        match=MatchValue(value=value)
    )


def create_filter(field_values):
    allowed_fields = {"studiengang", "standort", "abschluss"}

    for field in field_values.keys():
        if field not in allowed_fields:
            raise ValueError(f"Ungültiges Feld: '{field}'. Erlaubte Felder sind: {allowed_fields}")

    must_conditions = [create_field_condition(field, value) for field, value in field_values.items()]

    return Filter(must=must_conditions)


def create_filter_from_values(studiengang: str, standort: str, abschluss: str) -> Filter:
    query_filter = Filter(
        must=[
            create_field_condition(field="studiengang", value=studiengang),
            create_field_condition(field="standort", value=standort),
            create_field_condition(field="abschluss", value=abschluss)
        ])
    logging.info(f"Filter: {query_filter}")
    return query_filter


def retrieve_documents_with_filter_without_scores(client: QdrantVectorStore, max_documents: int,
                                                  filters: Filter, query: str) -> List[Document]:
    documents = client.similarity_search(query=query, k=max_documents, filter=filters)
    return documents


def retrieve_with_score_without_filters(client: QdrantVectorStore,
                                        max_documents: int,
                                        query: str) -> list[tuple[Document, float]]:
    documents_with_score = client.similarity_search_with_score(query=query, k=max_documents)
    return documents_with_score


def retrieve_with_score_with_filters(client: QdrantVectorStore,
                                     max_documents: int,
                                     filters: Filter,
                                     query: str) -> list[tuple[Document, float]]:

    documents_with_score = client.similarity_search_with_score(query=query,
                                                               k=max_documents,
                                                               filter=filters)
    return documents_with_score


def retrieve_with_score_with_self_query_retriever(client: QdrantVectorStore, llm: ChatOllama, search_kwargs: dict,
                                                  query: str) -> list[tuple[Document, float]]:

    docs_description = "Informationen zum Studium an der Fachhochschule Südwestfalen, den Standorten und Studienmodellen."

    metadata_field_info = [
        AttributeInfo(
            name="standort",
            description="Standort des Studiengangs. Einer von ['Alle', 'Iserlohn', 'Hagen', 'Soest', 'Meschede', 'Lüdenscheid']",
            type="string",
        ),
        AttributeInfo(
            name="studiengang",
            description="Der Name des Studiengangs. Einer von ['Alle', 'Wirtschaftsinformatik', 'Elektrotechnik']",
            type="string",
        ),
        AttributeInfo(
            name="abschluss",
            description="Der akademische Grad. Einer von ['Alle', 'Bachelor', 'Master']",
            type="string",
        ),
    ]

    retriever = CustomSelfQueryRetriever.from_llm(
        llm=llm,
        vectorstore=client,
        document_contents=docs_description,
        metadata_field_info=metadata_field_info,
        use_original_query=False,  # Für vereinfachte Fragen auf False setzen
        verbose=True,  # Für Logs der Filter und Frage des LLMs auf True setzen
        search_kwargs=search_kwargs
    )

    documents_with_score = retriever.invoke(query)
    # Veränderter Rückgabewert, damit es identisch zu den anderen Abrufen ist und ich einheitlich den Score abrufen kann
    list_of_tuples = []
    for doc in documents_with_score:
        score = float(doc.metadata.get("score"))
        list_of_tuples.append((doc, score))

    return list_of_tuples


def create_llm_for_self_query_retriever() -> ChatOllama:
    query_llm = ChatOllama(
        model=OllamaModels.LLAMA.value,
        temperature=0,
        num_ctx=16000
    )
    return query_llm


def retrieve_run_dict_with_documents_with_scores(client: QdrantVectorStore, fragen_dict: Dict[int, str], mode: str,
                                                 filter_dict: Optional[Dict[int, Filter]] = None,
                                                 llm: Optional[ChatOllama] = None,
                                                 search_kwargs: Optional[Dict[str, str]] = None,
                                                 max_documents: Optional[int] = 10) -> Dict[str, Dict[str, float]]:
    """
    Ruft die Dokumente mit ihren Scores zu allen Fragen ab und gibt ein Dictionary zurück, welches das Format für die Run-Dicts aus ranx hat.
    :param client: Der Qdrant-Client.
    :param fragen_dict: Die Fragen mit ihrer ID.
    :param mode: Modus für die Abfrage. Mögliche Werte: "no_filters", "optimal_filters", "self_query_retriever".
    :param filter_dict: Filter für den "optimal_filters"-Modus.
    :param llm: LLM für den "self_query_retriever"-Modus.
    :param search_kwargs: Key-Word-Arguments für die Suche im "self_query_retriever"-Modus.
    :param max_documents: Anzahl abzurufender Dokumente.
    :return: Dictionary mit den Fragen-IDs und zu jeder Fragen-ID ein Dictionary mit den Dokumenten-IDs und den Scores als Wert.
    """
    ergebnisse = {}

    for q_id, frage in fragen_dict.items():
        if mode == "no_filters":
            docs = retrieve_with_score_without_filters(client=client, max_documents=max_documents, query=frage)
        elif mode == "optimal_filters":
            docs = retrieve_with_score_with_filters(client=client, max_documents=max_documents,
                                                    filters=filter_dict[q_id], query=frage)
        elif mode == "self_query_retriever":
            docs = retrieve_with_score_with_self_query_retriever(client=client, llm=llm, search_kwargs=search_kwargs,
                                                                 query=frage)
        else:
            raise ValueError(f"Falschen Modus übergeben! mode: {mode}.")

        doc_scores = {f"d_{doc.metadata['id']}": score for doc, score in docs}
        ergebnisse[f"q_{q_id}"] = doc_scores

    return ergebnisse
