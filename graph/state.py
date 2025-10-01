# /graph/state.py

from typing import TypedDict, List, Annotated
import operator
import pandas as pd
from langchain_core.messages import BaseMessage

class EdaGraphState(TypedDict):
    """
    Representa o estado do nosso grafo de análise.

    Atributos:
        question: A pergunta original do usuário.
        df_head: As primeiras linhas do DataFrame para dar contexto ao LLM.
        classification: A classificação da pergunta (ex: 'plot', 'descritivo').
        plan: O plano de execução gerado pelo LLM.
        code_to_execute: O snippet de código Python gerado para a etapa atual.
        execution_result: O resultado (texto ou imagem base64) da execução do código.
        conclusion: A conclusão final gerada para o usuário.
        chat_history: O histórico da conversa.
    """
    question: str
    df_head: str
    classification: str
    plan: str
    code_to_execute: str
    execution_result: str
    conclusion: str
    # A anotação permite que a chave `chat_history` acumule mensagens
    chat_history: Annotated[List[BaseMessage], operator.add]