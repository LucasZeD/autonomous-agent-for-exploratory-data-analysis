# Agente Autônomo para Análise Exploratória de Dados (EDA)

## Visão Geral

Este projeto implementa um agente de inteligência artificial projetado para realizar Análise Exploratória de Dados (EDA) em conjuntos de dados tabulares (arquivos `.csv`). Utilizando uma arquitetura baseada em LangGraph, o agente executa um fluxo de trabalho determinístico para gerar, avaliar e executar código Python, respondendo a perguntas do usuário sobre os dados. A solução integra uma interface intuitiva com Streamlit, suporte a múltiplos provedores de LLM e um mecanismo de busca contextual (RAG) para enriquecer a análise com conhecimento externo.

## Principais Características

* **Workflow Controlado com LangGraph**: Substitui a abordagem de agente ReAct por um grafo de estados definido (Planejar -> Gerar Código -> Executar -> Concluir), garantindo maior previsibilidade, consistência e confiabilidade nos resultados da análise.
* **Análise Contextual com RAG**: Permite o carregamento de documentos PDF para uma base de conhecimento vetorial. O agente pode consultar estes documentos para obter contexto adicional, resultando em insights mais aprofundados e informados.
* **Arquitetura Flexível de LLMs**: Utiliza o padrão de projeto *Factory* para abstrair a criação de instâncias de LLMs, permitindo a troca facilitada entre diferentes provedores como Google (Gemini), OpenAI (GPT) e modelos locais (via Ollama).
* **Execução Segura de Código**: A ferramenta de execução de código Python opera em um escopo controlado, analisando o código gerado para bloquear importações de bibliotecas potencialmente perigosas (`os`, `subprocess`, etc.), seguindo o princípio de *Security by Design*.
* **Interface Intuitiva com Streamlit**: Oferece uma interface de usuário simples para upload de arquivos e interação via chat, facilitando o uso da ferramenta por diferentes públicos.

## Arquitetura e Design

A solução é fundamentada em uma arquitetura modular que promove a separação de responsabilidades:

1.  **Controle de Fluxo (LangGraph)**: O núcleo da lógica de análise é orquestrado por um grafo de estados. Isso força o agente a seguir uma metodologia estruturada, aumentando a acurácia e alinhando os resultados com as melhores práticas de EDA.

2.  **Estado Centralizado (`EdaGraphState`)**: O estado do grafo serve como um repositório central para todas as informações relevantes (plano de análise, código gerado, resultados), que são passadas explicitamente entre os nós. Esta abordagem torna o processo transparente e facilita a depuração.

3.  **Contexto com RAG (`rag_tool`)**: A capacidade de busca em documentos externos (PDFs) enriquece a análise. O agente pode correlacionar os dados numéricos com informações textuais, como documentações ou relatórios, para gerar conclusões mais ricas.

4.  **Abstração de LLM (`LLMFactory`)**: O padrão *Factory* desacopla a lógica de negócio da implementação específica de um modelo de linguagem, permitindo a fácil integração de novos provedores sem alterar o núcleo da aplicação.

5.  **Ferramentas Seguras (`pandas_tool`)**: A principal ferramenta é um executor de código Python que, em vez de um REPL genérico, implementa uma estratégia segura que valida e executa o código em um ambiente isolado, contendo apenas o DataFrame e bibliotecas seguras de análise e visualização.

6.  **Interface do Usuário (Streamlit)**: A camada de apresentação é construída com Streamlit, provendo os componentes necessários para a interatividade do usuário, como upload de arquivos e interface de chat.

## Estrutura de Arquivos

```text
/exploratory_data_agent
|
|-- /agents
|   |-- __init__.py
|   |-- pandas_agent.py
|
|-- /graph
|   |-- __init__.py
|   |-- eda_graph.py
|   |-- state.py
|
|-- /llm
|   |-- __init__.py
|   |-- llm_factory.py
|
|-- /tools
|   |-- __init__.py
|   |-- pandas_tool.py
|
|-- /ui
|   |-- __init__.py
|
|-- /utils
|   |-- __init__.py
|   |-- security.py
|
|-- app.py
|-- .env
|-- requirements.txt
```

## Como Executar

### 1. Pré-requisitos
* NA

### 2. Ambiente Virtual
É altamente recomendável criar e ativar um ambiente virtual para isolar as dependências do projeto.

```bash
# Criar o ambiente virtual
python -m venv env

# Ativar environment
.\env\Scripts\activate # Windows
source env/bin/activate # Linux e macOS
```

### 3. Instalação de Dependências
Com o ambiente virtual ativado, instale as bibliotecas necessárias:
```bash
pip install -r requirements.txt
```

### 4. Configuração das Chaves de API
Crie um arquivo chamado `.env` na raiz do projeto, utilizando o seguinte modelo:
```text
OPENAI_API_KEY="sua-chave-de-api-da-openai"
GOOGLE_API_KEY="sua-chave-de-api-do-google"
```
Insira as chaves de API correspondentes aos serviços que pretende utilizar.

- **Nota sobre LLMs Locais:**
    Para usar a opção `LocalLM`, certifique-se de que um serviço como o Ollama esteja em execução. O endpoint padrão configurado é `http://localhost:11434/v1`.

### 5. Execução da Aplicação
Execute o seguinte comando no terminal, a partir da raiz do projeto:
```bash
streamlit run app.py
```
A aplicação será aberta automaticamente em seu navegador.

## Guia de Uso

Após iniciar a aplicação, siga os passos na barra lateral esquerda:

1.  **Carregue o arquivo de dados**: Faça o upload do arquivo `.csv` que será o objeto da análise. Este é o DataFrame principal que o agente utilizará.
2.  **(Opcional) Carregue a base de conhecimento**: Faça o upload de um ou mais arquivos `.pdf`. Estes documentos serão processados e vetorizados para servirem como uma base de conhecimento contextual para o agente (RAG).
3.  **Selecione o Provedor LLM**: Escolha o modelo de linguagem que deseja utilizar.
4.  **Insira a Chave de API** (se aplicável).
5.  **Inicie o Agente**: Clique no botão "Iniciar Agente".

Após a inicialização, o chat estará pronto para receber suas perguntas.

### Exemplos de Perguntas
Fonte de dados: https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud

#### Descrição dos Dados
* Quais são os tipos de dados (numéricos, categóricos)?
* Qual a distribuição de cada variável (histogramas, distribuições)?
* Qual o intervalo de cada variável (mínimo, máximo)?
* Quais são as medidas de tendência central (média, mediana)?
* Qual a variabilidade dos dados (desvio padrão, variância)?
* *Quantas linhas e colunas o dataset possui?*
* *Mostre as primeiras 5 linhas dos dados.*
* *Quais são os tipos de dados de cada coluna?*
* *Qual a distribuição da variável 'Class'? Me mostre em um gráfico de barras.*
* *Descreva as principais medidas estatísticas do dataset.*

#### Relações entre Variáveis
* *Existe alguma correlação entre 'Amount' e 'Time'?*
* *Gere um gráfico de dispersão entre as colunas X e Y.*
* *Quais variáveis possuem a maior correlação com a variável alvo?*

#### Perguntas Baseadas em Contexto (RAG)
* *Com base no documento fornecido, explique o que a coluna 'V1' representa.*
* *Resuma o processo de negócio descrito no relatório PDF.*

#### Identificação de Padrões e Tendências:
* Existem padrões ou tendências temporais?
* Quais os valores mais frequentes ou menos frequentes?
* Existem agrupamentos (clusters) nos dados?

#### Detecção de Anomalias (Outliers):
* Existem valores atípicos nos dados?
* Como esses outliers afetam a análise?
* Esses outliers podem ser removidos, transformados ou investigados?

#### Relações entre Variáveis:
* Como as variáveis estão relacionadas umas com as outras? (Gere Gráficos de dispersão, tabelas cruzadas)
* Existe correlação entre as variáveis?
* Quais variáveis parecem ter maior ou menor influência sobre outras?

## Próximos Passos: IA Responsável

Este projeto possui um roteiro claro para a implementação de práticas de IA Responsável.

<!-- CONCLUIDO ### 1. Transparência e Explicabilidade (XAI)
* **Status Atual**: O fluxo de análise é interno (caixa-preta), e o usuário vê apenas a conclusão final.
* **Próximos Passos**:
    * **Exibir Raciocínio**: Utilizar um componente `st.expander` na interface para mostrar ao usuário o plano de análise, o código Python executado e o resultado bruto que fundamentou a resposta final.
    * **Citação de Fontes**: Aprimorar a ferramenta de RAG para retornar a fonte da informação (nome do arquivo, página), citando-a explicitamente na resposta quando o conhecimento externo for utilizado. -->

### 1. Justiça e Mitigação de Viés (Fairness & Bias)
* **Status Atual**: O agente analisa os dados "como estão" (`as is`), sem uma avaliação crítica sobre vieses inerentes.
* **Próximos Passos**:
    * **Prompt Engineering**: Incluir instruções explícitas no prompt do sistema para que o LLM avalie e mencione potenciais vieses em suas conclusões.
    * **Ferramenta de Detecção de Viés**: Desenvolver uma nova ferramenta, utilizando bibliotecas como `Fairlearn`, para realizar uma análise preliminar de desequilíbrios estatísticos em atributos sensíveis do dataset.

### 2. Privacidade (Privacy)
* **Status Atual**: O DataFrame é processado em memória e as chaves de API são gerenciadas de forma segura.
* **Próximos Passos**:
    * **Ferramenta de Anonimização**: Adicionar um nó opcional no início do grafo para identificar e anonimizar Informações de Identificação Pessoal (PII) antes da análise

### 3. Confiabilidade e Robustez (Reliability & Robustness)
* **Status Atual**: Em caso de erro na execução do código, o sistema retorna a mensagem de erro bruta ao usuário e para.
* **Próximos Passos**:
    * **Ciclo de Auto-Correção**: Implementar um ciclo no grafo onde, em caso de falha, o nó de execução retorne ao nó de geração de código, informando a mensagem de erro. O LLM seria então instruído a corrigir o código anterior com base no erro.
    * **Validação de Código com AST**: Evoluir a sanitização de código de `regex` para uma análise via Árvore de Sintaxe Abstrata (`ast`), permitindo a criação de regras de segurança mais granulares e robustas.

### 4. Logging e Auditoria
* **Status Atual**: O feedback de erros é reativo e exibido diretamente na interface em caso de falha. Não há um sistema persistente de logs para análise posterior ou auditoria.
* **Próximos Passos**:
    * **Implementar Logging Estruturado**: Integrar uma biblioteca de logging para capturar eventos-chave da aplicação, como o início e o fim de cada nó do grafo, as decisões do planejador e o resultado (sucesso ou falha) da execução de ferramentas.
    * **Criar Trilha de Auditoria**: Utilizar os logs para construir uma trilha de auditoria completa para cada consulta do usuário. Isso é crucial para a depuração de comportamentos inesperados e para aumentar a transparência sobre as operações do agente.

### 5. Adicionar Multiplos Agentes
* **Status Atual**: O problema reside puramente na capacidade do modelo de linguagem local (gpt-oss:20b) de executar a tarefa final de síntese no `conclusion_node`. Mesmo recebendo o contexto perfeito (pergunta, plano e dados numéricos), o modelo falhou na tarefa de "formular uma conclusão". Em vez de interpretar os dados fornecidos, ele se perdeu, identificou um padrão estatístico vago e alucinou uma resposta completamente desconexa sobre distribuição binomial. Isso acontece porque modelos menores ou menos refinados, como muitos LLMs locais, são excelentes em tarefas estruturadas (como gerar código a partir de um prompt claro), mas podem ter dificuldade com tarefas que exigem um raciocínio mais abstrato (como "interprete estes resultados e escreva um parágrafo sobre eles").
* **Próximos Passos**:
    * **Usar um Modelo Mais Forte para a Conclusão**: Em sistemas de agentes complexos, é comum usar modelos diferentes para tarefas diferentes (`agentic orchestration`). Como já foi implementado o LLMFactory será mais facil de implementar este processo, logo usando o modelo local (gpt-oss:20b) para as tarefas mais baratas e estruturadas (planejamento, geração de código) e um modelo mais poderoso para o conclusion_node, que exige mais raciocínio. Para um sistema completamente offline, utilizar gpts de diferentes tamanhos pode trazer um bom resultado.