PLANTILLA_CON_CONTEXTO = """
Contexto recuperado de los documentos:
{contexto}

Consulta del usuario:
{consulta}
"""

PLANTILLA_SIN_CONTEXTO = """
{consulta}
"""

'''
- Por ahora hemos detectado solo 2 tipos de consultas.

    1) Con contexto -> Cuando se usa el RAG.
    2) Sin contexto -> Todas las demás.

- Puede parecer una función innecesaria, pero está aquí para mantener la estructura y arquitectura del codigo.
- Además, podría existir la posibilidad de detectar otra arquitectura de consulta en un futuro.
'''

def seleccionar_plantilla_usuario(tipo_prompt: str) -> str:

    mapa_disponible = {
        "con_contexto": PLANTILLA_CON_CONTEXTO,
        "sin_contexto": PLANTILLA_SIN_CONTEXTO,
    }

    if tipo_prompt not in mapa_disponible:
        arquitecturas_soportadas = ", ".join(mapa_disponible.keys())
        raise ValueError(
            f"La arquitectura '{tipo_prompt}' no tiene una plantilla asignada. "
            f"Prompts disponibles: [{arquitecturas_soportadas}]"
        )
    
    return mapa_disponible[tipo_prompt]