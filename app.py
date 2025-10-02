# /ui/app.py
import sys
import os
import streamlit as st
import pandas as pd
import base64
import re
from llm.llm_factory import LLMFactory
from graph.eda_graph import create_eda_graph
from tools.rag_tool import setup_vectorstore
from langchain_core.messages import AIMessage, HumanMessage

st.set_page_config(page_title="Agente de Análise de CSV", layout="wide")

st.title("Agente para Análise de Dados (E.D.A.)")
st.markdown("Desenvolvido com `LangGraph` para análises de alta acurácia.")

# --- Inicialização do Estado da Sessão ---
def init_session_state():
    if "messages" not in st.session_state:
        st.session_state.messages = []
    if "graph_runner" not in st.session_state:
        st.session_state.graph_runner = None

init_session_state()

# --- BARRA LATERAL (CONFIGURAÇÕES) ---
with st.sidebar:
    st.header("Configurações")
    
    uploaded_csv = st.file_uploader("1. Carregue seu arquivo CSV", type="csv")
    
    uploaded_pdfs = st.file_uploader(
        "2. (Opcional) Carregue PDFs para a Base de Conhecimento (RAG)",
        type="pdf",
        accept_multiple_files=True
    )
    
    llm_provider = st.selectbox("3. Escolha o provedor de LLM", ("Gemini", "GPT", "LocalLM"))
    
    api_key = None
    if llm_provider in ["GPT", "Gemini"]:
        api_key = st.text_input(f"Insira a chave de API do {llm_provider}", type="password", help="Não é obrgiatório o uso de senha para o Gemini.")

    if st.button("🚀 Iniciar Agente"):
        if uploaded_csv is not None:
            with st.spinner("Processando arquivos e construindo o grafo..."):
                try:
                    df = pd.read_csv(uploaded_csv)
                    
                    if uploaded_pdfs:
                        setup_vectorstore(uploaded_pdfs)
                        st.success("Base de conhecimento (RAG) criada com sucesso!")
                    
                    llm = LLMFactory.create_llm(llm_provider, api_key)
                    st.session_state.graph_runner = create_eda_graph(llm, df)
                    
                    st.session_state.messages = []
                    st.success(f"Agente inicializado com {llm_provider}. Pronto para análise!")
                except Exception as e:
                    st.error(f"Erro na inicialização: {e}")
        else:
            st.warning("Por favor, carregue um arquivo CSV para começar.")

# --- ÁREA PRINCIPAL DO CHAT ---
def display_chat_history():
    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            # Expander para detalhes de raciocínio
            if "details" in msg:
                with st.expander("Ver Raciocínio do Agente."):
                    st.markdown("##### Plano de Análise")
                    st.markdown(msg["details"]["plan"], unsafe_allow_html=True)

                    st.markdown("##### Código Executado")
                    st.code(msg["details"]["code"], language="python")
                    
                    st.markdown("##### Resultado Bruto")
                    # Remoção plot do gráfico
                    raw_result = re.sub(r'\[PLOT_DATA:.*?\]', '[Visualização gerada com sucesso]', msg["details"]["result"])
                    # st.text(raw_result)
                    st.code(raw_result, language="text")
            # Resposta do modelo
            st.markdown(msg["content"])
            
            # Gráfico gerado
            if "image" in msg:
                st.image(base64.b64decode(msg["image"]))


if st.session_state.graph_runner:
    st.info(f"Agente pronto! Faça perguntas sobre o arquivo `{uploaded_csv.name}`.")
    display_chat_history()
    
    if prompt := st.chat_input("Ex: 'Qual a correlação entre as colunas V1 e Amount?'"):
        st.session_state.messages.append({"role": "user", "content": prompt, "lc_message": HumanMessage(content=prompt)})
        with st.chat_message("user"):
            st.markdown(prompt)
        
        with st.chat_message("assistant"):
            with st.spinner("O agente está executando o workflow de análise..."):
                # chat_history = [msg["lc_message"] for msg in st.session_state.messages[:-1]]

                # final_state = st.session_state.graph_runner(prompt, chat_history)

                # conclusion = final_state.get("conclusion", "Não foi possível gerar uma conclusão.")

                # result_text = str(final_state.get("execution_result", ""))

                final_state = st.session_state.graph_runner(prompt, st.session_state.get("messages", []))

                conclusion = final_state.get("conclusion", "Não foi possível gerar uma conclusão.")

                execution_details = {
                    "plan": final_state.get("plan", "Plano não disponível."),
                    "code": final_state.get("code_to_execute", "Código não disponível."),
                    "result": str(final_state.get("execution_result", "Resultado não disponível."))
                }

                result_text = execution_details["result"]
                plot_match = re.search(r'\[PLOT_DATA:(.*?)\]', result_text)

                # assistant_message = {"role": "assistant", "content": conclusion, "lc_message": AIMessage(content=conclusion)}
                assistant_message = {"role": "assistant", "content": conclusion, "details": execution_details}
                
                if plot_match:
                    img_data = plot_match.group(1)
                    st.markdown(conclusion)
                    st.image(base64.b64decode(img_data))
                    assistant_message["image"] = img_data
                else:
                    st.markdown(conclusion)
                
                st.session_state.messages.append(assistant_message)
                st.rerun()
else:
    st.info("Por favor, configure o agente na barra lateral (\">>>\") para começar a análise.")