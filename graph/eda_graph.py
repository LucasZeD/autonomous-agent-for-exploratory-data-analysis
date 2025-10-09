# /graph/eda_graph.py

from langchain_core.prompts import PromptTemplate
from .state import EdaGraphState
from tools.pandas_tool import PythonExecutorTool
from tools.rag_tool import knowledge_base_search
from langgraph.graph import StateGraph, END
from langchain_core.messages import HumanMessage
import pandas as pd
import re

# --- DEFINIÇÃO DOS NÓS DO GRAFO ---

def plan_node(state: EdaGraphState, llm):
    """Nó que gera um plano de análise com base na pergunta."""
    prompt = PromptTemplate.from_template(
        """Você é um planejador especialista em análise de dados. Dada a pergunta do usuário,
        o histórico da nossa conversa anterior e as primeiras linhas de um DataFrame,
        crie um plano passo a passo conciso para responder à pergunta.
        Leve em conta as análises já realizadas no histórico para evitar repetições.

        Histórico da Conversa:
        {chat_history}

        Pergunta do Usuário: {question}
        Cabeçalho do DataFrame:
        {df_head}

        Plano:"""
    )
    chain = prompt | llm
    plan = chain.invoke({
        "question": state["question"],
        "df_head": state["df_head"],
        "chat_history": state["chat_history"]  # <-- Adicione esta linha
    }).content
    return {"plan": plan}

def code_generation_node(state: EdaGraphState, llm):
    """Nó que gera código Python para executar o plano."""
    prompt = PromptTemplate.from_template(
        """Você é um programador Python sênior, especialista em pandas, matplotlib e seaborn.
        Sua tarefa é gerar um único script Python para executar o plano de análise de dados abaixo.
        O script deve ser completo e autossuficiente para responder à pergunta do usuário.

        **--- REGRAS CRÍTICAS DE SAÍDA ---**
        - **Para `sklearn.manifold.TSNE`**: Use o parâmetro `max_iter` em vez do obsoleto `n_iter` (ex: `TSNE(..., max_iter=1000)`).
        
        Seu script DEVE produzir uma ou ambas as seguintes variáveis como resultado final:
        1.  `result_data`: Use esta variável para armazenar qualquer resultado numérico ou textual final (ex: um DataFrame, uma contagem, uma correlação).
            - Exemplo: `result_data = df['coluna'].describe()`
        2.  **Para Gráficos**: 
            **NUNCA use `plt.show()`**.
            Use plt.figure() para iniciar um novo gráfico.
            `fig_base64`: Se o plano envolver a criação de um gráfico, esta variável DEVE conter a imagem do gráfico codificada em base64.
            - Siga este exemplo de código para gerar `fig_base64`:
                ```python
                import io
                import base64
                import matplotlib.pyplot as plt
                plt.figure()
                # ... seu código de plotagem ...
                buf = io.BytesIO()
                plt.savefig(buf, format='png')
                buf.seek(0)
                # Apenas codifique para base64. A ferramenta cuidará da decodificação.
                fig_base64 = base64.b64encode(buf.read())
                plt.close()
                ```

        **--- REGRAS DE SEGURANÇA ---**
        - NÃO use bibliotecas que interajam com o sistema, como `os`, `sys`, `subprocess`, `open()`.
        - Opere exclusivamente em memória. O DataFrame já está carregado na variável `df`.

        **--- CONTEXTO ---**
        Plano de Análise:
        {plan}

        Cabeçalho do DataFrame:
        {df_head}
        
        **--- SCRIPT PYTHON ---**
        Gere um único bloco de código Python que implemente o plano completo, seguindo TODAS as regras acima. O código deve ser limpo, sem comentários ou markdown.
        """
    )
    chain = prompt | llm
    code = chain.invoke({"plan": state["plan"], "df_head": state["df_head"]}).content
    # Limpa o código de blocos de markdown
    match = re.search(r"```python\n(.*?)\n```", code, re.DOTALL)
    if match:
        clean_code = match.group(1).strip()
    else:
        # If no markdown block is found, assume the whole response is code (fallback)
        clean_code = code.strip().replace("```python", "").replace("```", "")
    return {"code_to_execute": clean_code}

def code_execution_node(state: EdaGraphState, pandas_tool):
    """Nó que executa o código Python gerado."""
    code = state["code_to_execute"]
    result = pandas_tool.invoke({"code": code})
    return {"execution_result": result}

def conclusion_node(state: EdaGraphState, llm):
    """Nó que gera a conclusão final para o usuário."""
    prompt = PromptTemplate.from_template(
        """Você é um analista de dados especialista. Com base na pergunta original,
        no plano executado, nos resultados obtidos e no histórico da nossa conversa,
        formule uma conclusão clara, objetiva e concisa para o usuário.
        Se a pergunta for sobre conclusões gerais, use o histórico para sintetizar as descobertas.

        Histórico da Conversa:
        {chat_history}

        Pergunta Original: {question}
        Plano Executado: {plan}
        Resultado da Execução: {result}

        Conclusão Final:"""
    )
    chain = prompt | llm
    conclusion = chain.invoke({
        "question": state["question"],
        "plan": state["plan"],
        "result": state["execution_result"],
        "chat_history": state["chat_history"]  # <-- Adicione esta linha
    }).content
    return {"conclusion": conclusion}

def create_eda_graph(llm: object, df: pd.DataFrame):
    pandas_tool = PythonExecutorTool(df=df)
    
    # RAG tool accessible by the agent.
    tools = [pandas_tool, knowledge_base_search]

    # Define o workflow
    workflow = StateGraph(EdaGraphState)

    # Adiciona os nós
    workflow.add_node("planner", lambda state: plan_node(state, llm))
    workflow.add_node("code_generator", lambda state: code_generation_node(state, llm))
    workflow.add_node("code_executor", lambda state: code_execution_node(state, pandas_tool))
    workflow.add_node("concluder", lambda state: conclusion_node(state, llm))
    
    # Define as arestas (o fluxo)
    workflow.set_entry_point("planner")
    workflow.add_edge("planner", "code_generator")
    workflow.add_edge("code_generator", "code_executor")
    workflow.add_edge("code_executor", "concluder")
    workflow.add_edge("concluder", END)

    # Compila o grafo em um objeto executável
    app = workflow.compile()
    
    def run_graph(question: str, chat_history: list):
        df_head = df.head().to_string()
        inputs = {
            "question": question,
            "df_head": df_head,
            "chat_history": chat_history + [HumanMessage(content=question)]
        }
        return app.invoke(inputs)

    return run_graph