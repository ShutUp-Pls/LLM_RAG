PROMPT_BASE_ENRUTADOR = """
Eres un analizador semántico y clasificador experto.
Tu única tarea es analizar la consulta del usuario y devolver las clasificaciones solicitadas.
No respondas a la consulta del usuario. Responde ÚNICAMENTE con el formato requerido.
"""

CRITERIO_RAG = """
[ANALISIS_RAG]
Catalogo de temas: [{catalogo_temas}].
Si la consulta incluye algún tema en el catalogo de temas responde "SI", en cualquier otro caso responde "NO",
Estructura OBLIGATORIA de tu respuesta (Línea por línea):
RAZONAMIENTO: <Escribe aquí una breve justificación de 1 o 2 líneas explicando por qué requiere o no usar el catálogo, y el motivo de su intención>
RAG: <SI o NO>".
"""

CRITERIO_INTENCION = """
[ANALISIS_INTENCION]
Clasifica la intención principal de la consulta en una de las siguientes categorías: [PREGUNTA, COMANDO, SALUDO, DESCONOCIDO].
Estructura OBLIGATORIA de tu respuesta (Línea por línea):
INTENCION: <CATEGORIA>.
"""

PROMPT_REFUERZO_ENRUTADOR = """
Responde únicamente con la estructura OBLIGATORIA línea por línea especificadas anteriormente.
"""

def construir_prompt_enrutador(
    evaluar_rag: bool = False,
    evaluar_intencion: bool = False,
    catalogo_temas: str = "Conocimiento general"
) -> str:
    
    fragmentos_activos = []

    fragmentos_activos.append(PROMPT_BASE_ENRUTADOR)
    if evaluar_rag: fragmentos_activos.append(CRITERIO_RAG.format(catalogo_temas=catalogo_temas))
    if evaluar_intencion: fragmentos_activos.append(CRITERIO_INTENCION)       
    fragmentos_activos.append(PROMPT_REFUERZO_ENRUTADOR)
    
    return "\n".join(fragmentos_activos)