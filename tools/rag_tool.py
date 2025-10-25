# /tools/rag_tool.py

import os
import streamlit as st
from langchain_chroma import Chroma
from langchain_core.tools import tool
from langchain_openai.embeddings import OpenAIEmbeddings
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter

VECTORSTORE_DIR = "vectorstore_db"

@st.cache_resource
def setup_vectorstore(pdf_files):
    """
    Cria e persiste um banco de dados vetorial a partir de arquivos PDF.
    Usa o cache do Streamlit para evitar recriar o banco a cada execução.
    """
    if not pdf_files:
        return None
    
    temp_dir = "temp_pdf_storage"
    if not os.path.exists(temp_dir):
        os.makedirs(temp_dir)

    # Salva arquivos temporariamente para o loader poder acessá-los
    for uploaded_file in pdf_files:
        # with open(os.path.join(temp_dir, uploaded_file.name), "mb") as f:
        with open(os.path.join(temp_dir, uploaded_file.name), "wb") as f:
            f.write(uploaded_file.getbuffer())

    # Carrega os documentos
    docs = []
    for file_name in os.listdir(temp_dir):
        loader = PyPDFLoader(os.path.join(temp_dir, file_name))
        docs.extend(loader.load())

    # Divide os documentos em chunks
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    splits = text_splitter.split_documents(docs)

    # Cria o ChromaDB e o persiste
    vectorstore = Chroma.from_documents(
        documents=splits,
        embedding=OpenAIEmbeddings(),
        persist_directory=VECTORSTORE_DIR
    )
    return vectorstore

@tool
def knowledge_base_search(query: str) -> str:
    """
    Use esta ferramenta para responder perguntas que necessitam de conhecimento externo
    ou teórico sobre análise de dados, estatística ou o domínio do problema.
    A entrada deve ser uma pergunta clara sobre o tópico que você precisa pesquisar.
    """
    if not os.path.exists(VECTORSTORE_DIR):
        return "A base de conhecimento não foi inicializada. Não é possível pesquisar."
    
    vectorstore = Chroma(
        persist_directory=VECTORSTORE_DIR,
        embedding_function=OpenAIEmbeddings()
    )
    # retriever = vectorstore.as_retriever()
    # Buscando metadados da fonte
    retriever = vectorstore.as_retriever(search_kwargs={"k": 3})
    docs = retriever.invoke(query)
    
    # context = "\n\n".join([doc.page_content for doc in docs])
    # Formatando a saíde com a fonte de dados
    context = ""
    for doc in docs:
        # Extrair nome do arquivo
        source = doc.metadata.get('source', 'N/A').split('/')[-1]
        page = doc.metadata.get('page', 'N/A')
        context += f"Fonte: {source}, Página: {page}\n"
        context += f"Conteúdo: {doc.page_content}\n\n"
    if not context:
        return "Nenhum contexto relevante encontrado na base de conhecimento."
    return f"Contexto encontrado na base de conhecimento:\n{context}"