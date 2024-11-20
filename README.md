# local_rag
---------

## O que é?
App RAG (Retrieve Augmented Generation) para execução e conversação local. Utiliza-se dos modelos Llama-3.2-1b-Instruct como LLM e all-mpnet-base-v2 como embedding. 

## Como usar? 
Geralmente, um modelo LLM text-to-text pode se comportar como uma pessoa ou algum outro tipo de diretriz. Este comportamento pode ser configurado no arquivo search_vectorstore.py, na linha 30:
```
        {
            "role": "system",
            "content": "You will be shown the user's question, and the relevant information from a document of that topic. Answer the user's question using this information and also give some tips based on your knowledge (as long as it represents just 5% of the whole context of the answer)", <-- Substitua pelo comportamento desejado.
        },
        {
            "role": "user",
            "content": f"Question: {question_prompt}. \n Information: {info}",
        },
    ]
```
Além disso, essa diretriz também pode ser usada para configurar como o modelo pode se comportar ao usar o documento de PDF como base para a resposta.
Caso não queira um comportamento específico, o arquivo já vem preenchido com um comportamento padrão.

Depois disso, é necessário importar algum arquivo para o modelo. No caso, já temos um arquivo exemplo na pasta example_pdf (manual solo de D&D) que já está dentro do nosso banco de dados, pronto para ser usado pelo modelo. Caso queira importar um novo arquivo, siga os passos abaixo:

## Importando PDF's
  1. Delete a pasta "chroma";
  2. Entre na pasta modules;
  3. Abra o arquivo "pdf_loader.py" e altere a linha 12 para o caminho aonde está seu pdf
        ```
            from uuid import uuid4
            from langchain_community.document_loaders import PyPDFLoader
            from langchain_text_splitters import RecursiveCharacterTextSplitter
            from local_vector_store import initialize_vector_store
            from langchain_core.documents import Document
            
            
            vector_store = initialize_vector_store()
            
            loader = PyPDFLoader(
                r"CAMIHO-PARA-O-ARQUIVO-PDF" <-- Altere aqui
            )
            docs = loader.load()
        ```
  5. Execute o arquivo via terminal com o comando:
     ```
     python pdf_loader.py
     ```
  Isso deve criar uma nova pasta chamada "chroma" com os dados salvos. O correto é importar um PDF por vez, para cada execução do script pdf_loader.py.

## Instalando os pacotes necessários
1. Instale os pacotes necessários
```
  pip install -r requirements.txt
```

## Configurando os modelos
1. Crie uma conta no chat abaixo:
https://huggingface.co/
2. Procure algum modelo text-to-text
3. Encontre e clone o repositório
Como por exemplo: meta-llama/Llama-3.2-1B-Instruct
4. Entre no arquivo search_vectorstore.py e altere o caminho da linha seis para a pasta que clonou o modelo do hugginface
   ```
     from transformers import pipeline
    from langchain_core.documents import Document
    from modules.local_vector_store import initialize_vector_store
    
    
    modelPath: str = r"CAMINHO-PARA-O-MODELO-LLM" <-- Altere aqui
    pipe = pipeline("text-generation",
                    model=modelPath,
                    device_map="auto")
   ```
6. Faça a mesma coisa mas, dessa vez, baixe um modelo de embedding (sugiro usar o descrito no inicio do readme), e altere o caminho no arquivo local_vectorstore.py:
   ```
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
        model_name="CAMINHO-PARA-O-MODELO-DE-EMBEDDING", <-- Altere aqui
        model_kwargs={"device": "cpu"},
        encode_kwargs={"normalize_embeddings": False},
    )
   ```

## Usando o app
Por fim, execute a linha abaixo via terminal.
```
  python search_vectorstore.py
```

## O que é o Hugging face?
https://huggingface.co/huggingface

O Hugging Face é um site colaborativo onde diversas empresas postam seus modelos de aprendizado de máquina com qualidade mínima para produção. Lá se encontram os mais variados tipos de modelos open-source, desde os modelos Llama da META (Facebook) e muitas outras organizações. Além do modelo text-to-text usado neste app, existem outros inúmeros tipos de modelos disponíveis, que vão desde criação de imagens e videos, a sintetização de texto para audio. Tire um tempo para dar uma explorada!

## Qual a vantagem de rodar o aplicativo localmente?
LLM's rodados localmente não possuem acesso algum a rede externa, assim seu conteúdo não é compartilhado com terceiros, e ainda permite a customização do modelo, como por exemplo o retreinamento em outros tipos de arquivos locais. 
