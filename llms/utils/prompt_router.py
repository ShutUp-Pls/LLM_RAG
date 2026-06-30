PROMPT_BASE_ENRUTADOR = """
Eres un analizador semántico y clasificador experto.
Tu única tarea es analizar la consulta del usuario y devolver las clasificaciones solicitadas.
No respondas a la consulta del usuario. Responde ÚNICAMENTE con el formato requerido.
"""

CRITERIO_RAG = """
[ANALISIS_RAG]
Tu sistema tiene acceso a una base de conocimiento externa con información sobre: [{catalogo_temas}].
¿La consulta aborda, directa o indirectamente, alguno de los temas antes mencionados (RAG: SI)?
En cualquier otro caso -> "RAG: NO"
Responde estrictamente: "RAG: SI" o "RAG: NO".
"""

CRITERIO_INTENCION = """
[ANALISIS_INTENCION]
Clasifica la intención principal de la consulta en una de las siguientes categorías: [PREGUNTA, COMANDO, SALUDO, DESCONOCIDO].
Responde estrictamente: "INTENCION: <CATEGORIA>".
"""

PROMPT_REFUERZO_ENRUTADOR = """
Analiza la siguiente consulta y entrega tus resultados línea por línea:
"""

'''
- Para ahorrar computación y sobre todo tokens, era necesario que el modelo sepa discriminar cuando usar o no el sistema RAG,
llevandonos a un enrutador semantico, ya que para saber si una consulta debe o no usar información adicional hay que entender
la consulta y su sentido, esto se puede lograr usando el mismo modelo de lenguaje que se pretende usar para responder.
- A pesar de que por ahora lo escencial era entender la consulta para saber si usar o no el RAG, podrían haber otros criterios
semanticos interesantes que abordar.
- Por ende este prompt está pensado para acumular criterios semanticos en una sola consulta.
'''

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