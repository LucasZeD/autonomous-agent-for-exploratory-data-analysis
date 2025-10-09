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
from PIL import UnidentifiedImageError

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
    st.header("⚙️ Configurações")
    
    uploaded_csv = st.file_uploader("1. Faça o upload do seu CSV", type="csv")
    
    uploaded_pdfs = st.file_uploader(
        "2. (Opcional) Adicione PDFs de contexto (RAG)",
        type="pdf",
        accept_multiple_files=True,
        help="Forneça documentos PDF que contenham informações sobre o seu CSV. O agente usará esses arquivos para dar respostas mais ricas e contextuais."
    )
    
    llm_provider = st.selectbox("3. Escolha o modelo de IA", ("Gemini", "GPT", "LocalLM"))
    
    api_key = None
    if llm_provider in ["GPT", "Gemini"]:
        api_key = st.text_input(f"Insira a chave de API do {llm_provider}", type="password", help="Não é obrigatório o uso de chave para o Gemini, o desenvolvedor já forneceu uma.")

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
            if "image" in msg and msg["image"]:
                try:
                    st.image(base64.b64decode(msg["image"]))
                except Exception as e:
                    st.warning(f"Falha ao renderizar imagem: {e}")


if st.session_state.graph_runner:
    st.info(f"Agente pronto! Faça perguntas sobre o arquivo `{uploaded_csv.name}`.")
    display_chat_history()
    
    if prompt := st.chat_input("Faça uma pergunta sobre seus dados..."):
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
                
                # if plot_match:
                #     img_data = plot_match.group(1)
                #     st.markdown(conclusion)
                #     st.image(base64.b64decode(img_data))
                #     assistant_message["image"] = img_data
                # else:
                #     st.markdown(conclusion)
                st.markdown(conclusion)
                if plot_match:
                    img_data = plot_match.group(1).strip()
                    assistant_message["image"] = img_data
                    # Tenta renderizar a imagem, tratando possíveis erros
                    try:
                        # Verifica se img_data não está vazio antes de tentar decodificar
                        if img_data:
                            st.image(base64.b64decode(img_data))
                        else:
                            st.warning("O agente indicou a geração de um gráfico, mas os dados da imagem estão ausentes.")
                    except (base64.binascii.Error, UnidentifiedImageError):
                        st.warning("Não foi possível renderizar a visualização. O agente pode ter retornado um resultado textual em vez de um gráfico.")
                
                st.session_state.messages.append(assistant_message)
                st.rerun()
else:
    st.markdown("### Bem-vindo ao Agente de Análise de Dados!")
    st.markdown("Esta ferramenta permite que você converse com seus dados em arquivos CSV para extrair insights de forma rápida e intuitiva.")
    st.markdown("#### Como Começar:")
    st.markdown("1.  **Abra a barra lateral** no canto superior esquerdo (ícone `>>>`).")
    st.markdown("2.  **Faça o upload** do seu arquivo CSV.")
    st.markdown("3.  **Escolha o modelo** de IA que deseja usar.")
    st.markdown("4.  **Insira a chave API** da IA que deseja usar.")
    st.markdown("5.  Clique em **'Iniciar Agente'** e comece a fazer perguntas!")