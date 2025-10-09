# /utils/security.py

import tokenize
import io
import re

# Lista de bibliotecas permitidas para importação no código gerado pelo LLM.
ALLOWED_IMPORTS = {
    "pandas", "numpy", "matplotlib", "seaborn", "io", "base64", "sklearn"
}

# Palavras-chave e funções que são bloqueadas para evitar acesso ao sistema de arquivos ou execução de comandos.
FORBIDDEN_KEYWORDS = {
    "os", "sys", "subprocess", "eval", "execfile", "open", "input",
    "__import__", "shutil", "glob", "socket", "requests"
}

def remove_comments(code):
    # Converte a string de código em um fluxo de bytes que o tokenize pode ler
    code_io = io.BytesIO(code.encode('utf-8'))
    
    tokens = tokenize.tokenize(code_io.readline)
    result = []
    
    for toknum, tokval, _, _, _ in tokens:
        # Mantém todos os tokens, exceto os de comentário e os de encoding
        if toknum != tokenize.COMMENT and toknum != tokenize.ENCODING:
            result.append((toknum, tokval))
            
    # Reconstitui o código a partir dos tokens filtrados
    untokenized_code = tokenize.untokenize(result)
    
    # Adiciona verificação de tipo para lidar com inconsistências ambientais
    if isinstance(untokenized_code, bytes):
        return untokenized_code.decode('utf-8')
    else:
        return untokenized_code


def sanitize_code(code: str) -> str:
    """
    Analisa o código gerado pelo LLM para garantir que ele não contenha
    importações ou palavras-chave proibidas.
    Levanta uma exceção de segurança se uma violação for encontrada.
    """
    # Remove comentários de linha única antes de qualquer verificação
    code_without_comments = remove_comments(code)

    # Verifica importações
    imports_found = re.findall(r'^\s*(?:import|from)\s+([a-zA-Z0-9_.]+)', code_without_comments, re.MULTILINE)
    for lib in imports_found:
        if lib.split('.')[0] not in ALLOWED_IMPORTS:
            raise SecurityException(f"Importação proibida detectada: '{lib}'. Apenas as seguintes importações são permitidas: {ALLOWED_IMPORTS}")

    # Verifica palavras-chave proibidas
    for keyword in FORBIDDEN_KEYWORDS:
        # Usa word boundaries (\b) para evitar correspondências parciais (ex: 'closet' contendo 'os')
        if re.search(r'\b' + keyword + r'\b', code_without_comments):
            raise SecurityException(f"Palavra-chave ou função proibida detectada: '{keyword}'.")

    return code

class SecurityException(Exception):
    """Exceção customizada para violações de segurança."""
    pass