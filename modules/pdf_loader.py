from random import randint
from uuid import uuid4
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from local_vector_store import initialize_vector_store
from langchain_core.documents import Document


vector_store = initialize_vector_store()

loader = PyPDFLoader(
    r"C:\Users\jhonata_tirloni\Documents\Github\local_rag\example_pdf\d&d_guia_solo.pdf"
)
docs = loader.load()

text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000,
    chunk_overlap=0,
    separators=["\n\n", "\n", ". ", " ", ""]
)
splits: list[Document] = text_splitter.split_documents(docs)

docs_list: list = []
for doc in docs:
    docs_list.append(
        Document(
            page_content=doc.page_content,
            metadata={"source": "pdf_file"},
            id=randint(1, 1000000),
        )
    )

uuids = [str(uuid4()) for _ in range(len(docs_list))]

try:
    vector_store.add_documents(documents=docs_list, ids=uuids)
except Exception as e:
    print(e)
