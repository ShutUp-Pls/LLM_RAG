PLANTILLA_QWEN_ESTANDAR = """
Contexto recuperado de los documentos:
{contexto}

Pregunta del usuario:
{pregunta}
"""

PLANTILLA_LLAMA_ESTRICTA = """
Documentos de referencia:
{contexto}

Responde a la siguiente consulta de forma concisa basándote en los documentos:
{pregunta}
"""

PLANTILLA_DEEPSEEK_RAZONAMIENTO = """
Analiza el siguiente contexto paso a paso:
{contexto}

Considerando la información anterior, resuelve:
{pregunta}
"""


def seleccionar_plantilla_usuario(identificador_modelo: str) -> str:

    def __validar_y_extraer_plantilla(mapa: dict, clave: str) -> str:
        if clave not in mapa:
            modelos_soportados = ", ".join(mapa.keys())
            raise ValueError(
                f"El modelo '{clave}' no tiene una plantilla de usuario asignada. "
                f"Modelos soportados: [{modelos_soportados}]"
            )
        return mapa[clave]

    mapa_disponible = {
        "qwen": PLANTILLA_QWEN_ESTANDAR,
        "llama": PLANTILLA_LLAMA_ESTRICTA,
        "deepseek": PLANTILLA_DEEPSEEK_RAZONAMIENTO
    }
    return __validar_y_extraer_plantilla(mapa_disponible, identificador_modelo)