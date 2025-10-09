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
            
            output_parts = []

            # 1. Captura a saída de prints
            print_output = buffer.getvalue()
            if print_output:
                output_parts.append(print_output)
            # 2. Captura o resultado númerico da variável 'result_data'
            if 'result_data' in local_scope:
                data_str = str(local_scope['result_data'])
                output_parts.append(data_str)
            # 3. Captura o grafico, se existir
            # if 'fig_base64' in local_scope:
            #     fig_data = local_scope['fig_base64']
            #     fig_data_str = ""

            #     # Garante que o dado final seja uma string decodificada.
            #     if isinstance(fig_data, bytes):
            #         fig_data_str = fig_data.decode('utf-8')
            #     elif isinstance(fig_data, str):
            #         # Se já for string (devido a um ambiente anômalo), usa diretamente.
            #         fig_data_str = fig_data
            #     else:
            #         # Se for outro tipo, converte para string de forma segura.
            #         fig_data_str = str(fig_data)
                
            #     output_parts.append(f"Plot gerado com sucesso.\n[PLOT_DATA:{fig_data_str}]")
            if 'fig_base64' in local_scope:
                fig_data = local_scope['fig_base64']
                
                # VERIFICAÇÃO ADICIONAL: Só processa se fig_data não for None e for do tipo bytes ou str
                if fig_data and isinstance(fig_data, (bytes, str)):
                    fig_data_str = ""

                    if isinstance(fig_data, bytes):
                        fig_data_str = fig_data.decode('utf-8')
                    else: # Já é uma string
                        fig_data_str = fig_data
                    
                    # Garante que a string não esteja vazia antes de adicionar o placeholder
                    if fig_data_str.strip():
                        output_parts.append(f"Plot gerado com sucesso.\n[PLOT_DATA:{fig_data_str}]")
            
            if not output_parts:
                return "Código executado com sucesso, sem saída visual ou textual."
            
            return "\n\n".join(output_parts)

        except SecurityException as e:
            return f"Erro de Segurança: {e}"
        except Exception as e:
            return f"Erro de Execução: {type(e).__name__} - {e}"

    def _arun(self, *args: Any, **kwargs: Any) -> Any:
        raise NotImplementedError("A execução assíncrona não é suportada por esta ferramenta.")