# /tools/pandas_tool.py

import pandas as pd
import io
from contextlib import redirect_stdout
from typing import Type, Any
from langchain_core.tools import BaseTool
from pydantic import BaseModel, Field

from utils.security import sanitize_code, SecurityException

class PythonExecutorTool(BaseTool):
    """
    Ferramenta segura para executar código Python para análise de dados com Pandas.
    O código é executado em um ambiente controlado para mitigar riscos de segurança.
    """
    name: str = "python_pandas_executor"
    description: str = (
        "Executa código Python para analisar um DataFrame do pandas chamado 'df'. "
        "Use esta ferramenta para responder a perguntas sobre os dados, realizar cálculos e gerar visualizações. "
        "O código deve imprimir resultados ou salvar gráficos para visualização."
    )
    df: pd.DataFrame
    
    class ToolInput(BaseModel):
        code: str = Field(description="O código Python a ser executado para analisar o DataFrame 'df'.")

    args_schema: Type[BaseModel] = ToolInput

    def _run(self, code: str) -> str:
        """Executa o código após a sanitização."""
        try:
            sanitized_code = sanitize_code(code)
            
            # Prepara o ambiente de execução local com as bibliotecas permitidas
            local_scope = {
                'df': self.df,
                'pd': pd,
                'io': io
                # Bibliotecas de plotagem serão importadas dentro do exec se necessário
            }
            
            # Redireciona a saída padrão (prints) para uma string
            buffer = io.StringIO()
            with redirect_stdout(buffer):
                exec(sanitized_code, globals(), local_scope)
            
            output = buffer.getvalue()

            # Verifica se uma imagem foi gerada e a retorna
            if 'fig_base64' in local_scope:
                 return f"Plot gerado com sucesso.\n[PLOT_DATA:{local_scope['fig_base64']}]"
            
            return output if output else "Código executado com sucesso, sem saída de texto."

        except SecurityException as e:
            return f"Erro de Segurança: {e}"
        except Exception as e:
            return f"Erro de Execução: {type(e).__name__} - {e}"

    def _arun(self, *args: Any, **kwargs: Any) -> Any:
        # A execução assíncrona não é implementada para esta ferramenta
        raise NotImplementedError("A execução assíncrona não é suportada por esta ferramenta.")