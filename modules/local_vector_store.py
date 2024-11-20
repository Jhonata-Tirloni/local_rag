from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
import chromadb


def initialize_vector_store() -> Chroma:
    """
     Realiza a criação e inicialização de um client para o ChromaDB,
     configura o modelo de embedding a ser usado e instancia uma collection no
     banco de dados Chroma.

     :returns:
         Cliente chroma que irá ser responsável por realizar o embedding e a inclusão
         dos textos no banco de dados de vetores.
    """
    persistent_client = chromadb.PersistentClient()

    embeddings = HuggingFaceEmbeddings(
        model_name="CAMINHO-PARA-O-MODELO-DE-EMBEDDING",
        model_kwargs={"device": "cpu"}, # Pode mudar conforme o modelo
        encode_kwargs={"normalize_embeddings": False}, # Pode mudar conforme o modelo
    )

    vector_store = Chroma(
        client=persistent_client,
        collection_name="main_pdf_collection",
        embedding_function=embeddings,
    )

    return vector_store
