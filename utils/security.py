# /utils/security.py

import re

# Lista de bibliotecas permitidas para importação no código gerado pelo LLM.
ALLOWED_IMPORTS = {
    "pandas", "numpy", "matplotlib", "seaborn", "io", "base64"
}

# Palavras-chave e funções que são bloqueadas para evitar acesso ao sistema de arquivos ou execução de comandos.
FORBIDDEN_KEYWORDS = {
    "os", "sys", "subprocess", "eval", "execfile", "open", "input",
    "__import__", "shutil", "glob", "socket", "requests"
}

def sanitize_code(code: str) -> str:
    """
    Analisa o código gerado pelo LLM para garantir que ele não contenha
    importações ou palavras-chave proibidas.
    Levanta uma exceção de segurança se uma violação for encontrada.
    """
    # Verifica importações
    imports_found = re.findall(r'^\s*(?:import|from)\s+([a-zA-Z0-9_.]+)', code, re.MULTILINE)
    for lib in imports_found:
        if lib.split('.')[0] not in ALLOWED_IMPORTS:
            raise SecurityException(f"Importação proibida detectada: '{lib}'. Apenas as seguintes importações são permitidas: {ALLOWED_IMPORTS}")

    # Verifica palavras-chave proibidas
    for keyword in FORBIDDEN_KEYWORDS:
        # Usa word boundaries (\b) para evitar correspondências parciais (ex: 'closet' contendo 'os')
        if re.search(r'\b' + keyword + r'\b', code):
            raise SecurityException(f"Palavra-chave ou função proibida detectada: '{keyword}'.")

    return code

class SecurityException(Exception):
    """Exceção customizada para violações de segurança."""
    pass