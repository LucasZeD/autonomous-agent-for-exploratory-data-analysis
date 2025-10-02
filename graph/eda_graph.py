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
        """Você é um planejador especialista em análise de dados. Dada a pergunta do usuário
        e as primeiras linhas de um DataFrame, crie um plano passo a passo conciso para
        responder à pergunta. Descreva cada passo de forma clara.

        Pergunta: {question}
        Cabeçalho do DataFrame:
        {df_head}

        Plano:"""
    )
    chain = prompt | llm
    plan = chain.invoke({"question": state["question"], "df_head": state["df_head"]}).content
    return {"plan": plan}

def code_generation_node(state: EdaGraphState, llm):
    """Nó que gera código Python para executar o plano."""
    prompt = PromptTemplate.from_template(
        """Você é um programador Python especialista em pandas, matplotlib e seaborn.
        Com base no plano de análise, gere o código Python necessário para executar o PRÓXIMO passo.
        Use a variável 'df' para se referir ao DataFrame.

        **--- DIRETRIZES DE SAÍDA ---**
        1. **Para Gráficos**: 
            - Use plt.figure() para iniciar um novo gráfico.
            - **NÃO use `plt.show()`**.
            - Salve o gráfico em base64 na variável `fig_base64` como no exemplo:
                ```python
                import io
                import base64
                import matplotlib.pyplot as plt
                plt.figure()
                # ... seu código de plotagem ...
                buf = io.BytesIO()
                plt.savefig(buf, format='png')
                buf.seek(0)
                fig_base64 = base64.b64encode(buf.read()).decode('utf-8')
                plt.close()
                ```
        2. **Para Cálculos Numéricos**:
            Se você calcular valores (contagens, médias, correlações, etc.),
            armazene o resultado final (seja um DataFrame, uma Série ou um dicionário) em uma variável chamada `result_data`.
            Exemplo: `result_data = df['Class'].value_counts()`

        **--- REGRAS DE SEGURANÇA CRÍTICAS ---**
        - **NUNCA** use bibliotecas ou funções que interajam com o sistema de arquivos ou sistema operacional, como `os`, `sys`, `subprocess`, `open()`, etc.
        - Todo o código deve operar exclusivamente em memória usando as bibliotecas permitidas (pandas, matplotlib, seaborn, numpy).
        - Não tente carregar ou salvar arquivos. O DataFrame 'df' já está em memória.
        
        Plano: {plan}
        Cabeçalho do DataFrame:
        {df_head}

        Gere apenas o código Python, sem comentários, para o próximo passo do plano, respeitando TODAS as regras acima:"""
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
        no plano executado e nos resultados obtidos, formule uma conclusão clara e concisa
        para o usuário. Se os resultados incluírem um gráfico, mencione-o.

        Pergunta Original: {question}
        Plano: {plan}
        Resultado da Execução: {result}

        Conclusão:"""
    )
    chain = prompt | llm
    conclusion = chain.invoke({
        "question": state["question"],
        "plan": state["plan"],
        "result": state["execution_result"]
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