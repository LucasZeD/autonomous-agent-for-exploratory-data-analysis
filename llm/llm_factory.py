# /llm/llm_factory.py

from langchain_openai import ChatOpenAI
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.language_models.chat_models import BaseChatModel
from dotenv import load_dotenv
import os
import streamlit as st

load_dotenv()

class LLMFactory:
    """
    Fábrica responsável por criar instâncias de modelos de linguagem (LLMs).
    Utiliza o Factory Pattern para desacoplar a criação de LLMs do resto da aplicação.
    """
    @staticmethod
    def create_llm(provider: str, api_key: str = None) -> BaseChatModel:
        """
        Cria e retorna uma instância de um LLM com base no provedor especificado.

        Args:
            provider (str): O nome do provedor ('GPT', 'Gemini', 'LocalLM').
            api_key (str, optional): A chave de API para o serviço.

        Returns:
            BaseChatModel: Uma instância do modelo de chat.
        
        Raises:
            ValueError: Se o provedor for desconhecido.
        """
        if provider.upper() == 'GEMINI':
            key = api_key or os.getenv("GOOGLE_API_KEY")
            if not key:
                try:
                    import streamlit as st
                    key = st.secrets.get("GOOGLE_API_KEY")
                except (ImportError, AttributeError):
                    pass
            if not key:
                raise ValueError("Chave de API do Google não encontrada.")
            """
            To check the model names available curl:
            ```powershell
            curl -H "Content-Type: application/json" "https://generativelanguage.googleapis.com/v1/models?key=YOUR_API_KEY"
            ```
            ```linux/macos
            curl -H "Content-Type: application/json" \ "https://generativelanguage.googleapis.com/v1/models?key=YOUR_API_KEY"
            ```
            
            Current available models (as of 2024-10):
            models/gemini-2.5-flash
            models/gemini-2.5-pro
            models/gemini-2.0-flash
            models/gemini-2.0-flash-001
            models/gemini-2.0-flash-lite-001
            models/gemini-2.0-flash-lite
            models/gemini-2.0-flash-preview-image-generation
            models/gemini-2.5-flash-lite
            models/embedding-001
            models/text-embedding-004
            """
            return ChatGoogleGenerativeAI(google_api_key=key, temperature=0, model="gemini-2.5-pro", convert_system_message_to_human=True)
        
        elif provider.upper() == 'GPT':
            key = api_key or os.getenv("OPENAI_API_KEY")
            if not key:
                try:
                    import streamlit as st
                    key = st.secrets.get("OPENAI_API_KEY")
                except ImportError:
                    pass
            if not key:
                raise ValueError("Chave de API da OpenAI não encontrada.")
            return ChatOpenAI(api_key=key, temperature=0, model_name="gpt-4")

        elif provider.upper() == 'LOCALLM':
            # Exemplo para um LLM local (Ollama) servido via API compatível com OpenAI
            return ChatOpenAI(
                base_url="http://localhost:11434/v1",
                api_key="ollama", # A API key pode ser qualquer string para Ollama
                model_name="llama3" # Nome do modelo que você está servindo
            )
        
        else:
            raise ValueError(f"Provedor de LLM desconhecido: {provider}")