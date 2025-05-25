from enum import Enum


class CollectionNames(Enum):
    MIXEDBREAD_TOKEN_BASED = "mixedbread-ai_deepset-mxbai-embed-de-large-v1_token_based_chunks"
    MIXEDBREAD_RECURSIVE = "mixedbread-ai_deepset-mxbai-embed-de-large-v1_recursive_chunks"
    INFLOAT_TOKEN_BASED = "intfloat_multilingual-e5-large_token_based_chunks"
    INFLOAT_RECURSIVE = "intfloat_multilingual-e5-large_recursive_chunks"


class EmbeddingModels(Enum):
    MIXEDBREAD = "mixedbread-ai/deepset-mxbai-embed-de-large-v1"
    INFLOAT = "intfloat/multilingual-e5-large"


class OllamaModels(Enum):
    LLAMA = "llama3.1:8b"
    SAUERKRAUT = "cyberwald/llama-3.1-sauerkrautlm-8b-instruct:latest"
