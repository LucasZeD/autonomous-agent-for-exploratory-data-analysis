# /agents/pandas_agent.py

import pandas as pd
from langchain.agents import AgentExecutor, create_react_agent
from langchain.prompts import PromptTemplate
from langchain_core.language_models.chat_models import BaseChatModel
from langchain.memory import ConversationBufferMemory

from tools.pandas_tool import PythonExecutorTool

def create_pandas_agent(df: pd.DataFrame, llm: BaseChatModel) -> AgentExecutor:
    """
    Cria um agente conversacional para análise de dados em um DataFrame Pandas.
    """
    
    # Template do prompt que instrui o agente sobre seu papel, ferramentas e formato de resposta
    prompt_template = """
    Você é um assistente de IA especialista em análise de dados. Seu nome é 'CSV-Analyst'.
    Você está trabalhando com um DataFrame do pandas em Python, chamado 'df'.
    
    Você tem acesso às seguintes ferramentas:
    {tools}

    Para responder às perguntas do usuário, siga estas etapas:
    1.  **Thought (Pensamento)**: Analise a pergunta do usuário e pense em qual código Python (usando pandas, matplotlib, seaborn) é necessário para respondê-la. O DataFrame está disponível como a variável `df`.
    2.  **Action (Ação)**: Especifique a ferramenta a ser usada (sempre 'python_pandas_executor').
    3.  **Action Input (Entrada da Ação)**: Forneça o código Python a ser executado como um JSON no formato `{{"code": "seu código aqui"}}`.
    4.  **Observation (Observação)**: O resultado da execução do código será retornado.
    5.  Repita as etapas 1-4 conforme necessário para coletar informações suficientes para responder à pergunta.
    6.  Quando tiver a resposta final, termine com "Final Answer: [sua resposta textual aqui]".

    **Diretrizes importantes para a geração de código**:
    - O DataFrame já está carregado na variável `df`. **NÃO** tente carregar os dados.
    - Sempre que gerar um gráfico com matplotlib ou seaborn, faça o seguinte:
        1. Importe as bibliotecas necessárias (`import matplotlib.pyplot as plt`, `import seaborn as sns`).
        2. Crie a figura (`plt.figure()`).
        3. Gere o gráfico.
        4. **NÃO use `plt.show()`**. Em vez disso, salve o gráfico em um buffer de bytes e codifique-o em base64, atribuindo-o a uma variável chamada `fig_base64`.
        Exemplo para salvar gráfico:
        ```python
        import io
        import base64
        import matplotlib.pyplot as plt
        
        plt.figure()
        # ... seu código de plotagem aqui ...
        
        buf = io.BytesIO()
        plt.savefig(buf, format='png')
        buf.seek(0)
        fig_base64 = base64.b64encode(buf.read()).decode('utf-8')
        plt.close()
        ```
    - Forneça respostas claras e concisas baseadas nos resultados observados.

    **Histórico da Conversa:**
    {chat_history}

    **Pergunta do Usuário:**
    {input}

    **Seu Rascunho (Thought, Action, Action Input, Observation...):**
    {agent_scratchpad}
    """
    
    prompt = PromptTemplate.from_template(prompt_template)
    
    tools = [PythonExecutorTool(df=df)]
    
    # Implementa a memória para o agente ter contexto das conversas anteriores [cite: 46]
    memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)
    
    agent = create_react_agent(llm, tools, prompt)
    
    agent_executor = AgentExecutor(
        agent=agent,
        tools=tools,
        memory=memory,
        verbose=True,  # Coloque como False em produção
        handle_parsing_errors=True, # Lida com erros de formatação do LLM
        max_iterations=8 # Previne loops infinitos
    )
    
    return agent_executor