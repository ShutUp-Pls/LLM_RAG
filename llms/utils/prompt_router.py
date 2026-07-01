PROMPT_BASE_ENRUTADOR = """
Eres un analizador semántico y clasificador experto.
Tu única tarea es analizar la consulta del usuario y devolver las clasificaciones solicitadas.
No respondas a la consulta del usuario. Responde ÚNICAMENTE con el formato requerido.
"""

CRITERIO_RAG_QWEN = """
[ANALISIS_RAG]
Catálogo de temas permitidos: [{catalogo_temas}].
Evalúa de forma estricta y lógica si la consulta requiere obligatoriamente recuperar información de uno de los temas del catálogo.
Reglas críticas:
1. Si la consulta es un saludo (ej. "Hola"), una charla trivial, o una frase sin intención de búsqueda, el resultado DEBE ser "NO".
2. NO fuerces conexiones. No asocies síntomas genéricos (estrés, hambre, mareos) con temas médicos complejos si la consulta no los menciona directamente. Si dudas, el resultado DEBE ser "NO".
3. Solo si existe una relación semántica directa, evidente e innegable con al menos un tema, el resultado DEBE ser "SI".

Estructura OBLIGATORIA de tu respuesta (Línea por línea):
RAZONAMIENTO: <Explica paso a paso si existe una relación directa u obvia, o si es una consulta trivial/genérica>
RAG: <SI o NO>
"""

CRITERIO_RAG_OPENAI = """
[ANALISIS_RAG]
Catálogo de temas permitidos: [{catalogo_temas}].
Evalúa si el núcleo de la consulta del usuario se relaciona semánticamente con alguno de los temas del catálogo para extraer contexto útil.
Reglas críticas:
1. Si la consulta es un saludo, charla trivial, o no busca información (ej. comandos de programación puros), el resultado DEBE ser "NO".
2. Si la consulta aborda un tema general que está contenido o estrechamente relacionado con un tema del catálogo (ej. consultan sobre "obesidad y dieta" y el catálogo tiene "Tratamiento Obesidad"), el resultado DEBE ser "SI".
3. Tolera datos secundarios: Si la consulta menciona conceptos extra que no están en el catálogo, pero el tema principal SÍ está, prioriza la coincidencia temática y responde "SI".

Estructura OBLIGATORIA de tu respuesta (Línea por línea):
RAZONAMIENTO: <Explica brevemente la conexión temática principal o el motivo de rechazo>
RAG: <SI o NO>
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
    catalogo_temas: str = "Conocimiento general",
    modelo: str = "qwen"
) -> str:
    
    fragmentos_activos = []

    fragmentos_activos.append(PROMPT_BASE_ENRUTADOR)
    
    if evaluar_rag:
        if modelo.lower() == "openai":
            fragmentos_activos.append(CRITERIO_RAG_OPENAI.format(catalogo_temas=catalogo_temas))
        else:
            fragmentos_activos.append(CRITERIO_RAG_QWEN.format(catalogo_temas=catalogo_temas))
            
    if evaluar_intencion: 
        fragmentos_activos.append(CRITERIO_INTENCION)
               
    fragmentos_activos.append(PROMPT_REFUERZO_ENRUTADOR)
    
    return "\n".join(fragmentos_activos)