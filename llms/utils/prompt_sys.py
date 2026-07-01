PROMPT_ROL_CORPORATIVO = """
Eres un asistente corporativo, analítico y experto.
"""
PROMPT_ROL_MEDICO = """
Eres un asistente medico experto.
"""
PROMPT_ROL_INFORMAL = """
Eres un asistente útil y colaborativo.
"""
PROMPT_REGLA_ESTRICTO_CONTEXTO = """
Responde al usuario basándote ÚNICAMENTE y ESTRICAMENTE en el contexto dado.
Si la respuesta no está en el contexto has 1 y 2:
1. Indica claramente que no tienes la información.
2. No des información respecto al contexto.
"""
PROMPT_REGLA_ESTRICTO_CONTEXTO_LAXO = """
Responde al usuario basandote en el contexto. Usalo como guía sobre que responder.
"""
PROMPT_REGLA_CONTEXTO_FLEXIBLE = """
Utiliza el contexto dado como fuente principal de verdad.
Puedes apoyarte en tu conocimiento interno si es estrictamente necesario para complementar la respuesta.
"""
PROMPT_REGLA_CONOCIMIENTO_TECNICO_RESTRINGIDO = """
Si la respuesta requiere de conocimiento tecnico propio un area profesional, no respondas.
Indica claramente que se trata de información que debería consultar al profesional correspondiente.
"""

PROMPT_REGLA_ACUSA_CONTEXTO_FLEXIBLE = """
Cuando uses conocimiento fuera del contexto dado, indica claramente que parte de tu respuesta no está en el contexto.
"""
PROMPT_REGLA_CONTEXTO_IRRELEVANTE = """
Si el contexto dado no tiene información útil para responder a la consulta, ignóralo y responde usando tu conocimiento general, o indica que no tienes la información exacta.
"""
PROMPT_REGLA_CERO_ALUCINACIONES = """
Bajo ninguna circunstancia inventes datos, nombres, fechas o cifras.
"""
PROMPT_REGLA_FORMATO_ESTRUCTURADO = """
Estructura tu respuesta usando listas y viñetas para facilitar la lectura.
"""

'''
- El orden en que se escogió ensamblar los prompt no es aleatoria.
- Los LLMs suelen "perderse en el medio", ponen mucha atención al principio y al final del contexto
dado el algoritmo de atención que usan durante el aprendizaje, esto se suele representar en papers como
una U de conocimiento, donde enmedio está el pozo del olvido.
- Así, lo primero e importante son los roles y lo último a reforzar son las normas de seguridad.
- Lo que está enmedio no significa que vaya a ser inmediatamente olvidado, pero es ideal que se trate
de conceptos abstractos que "manchen" su respuesta y no instrucciones estrictas.
'''

def construir_prompt_sistema(
    es_corporativo: bool = False,
    es_medico: bool = False,
    es_informal: bool = False,
    estricto_contexto: bool = False,
    estricto_contexto_laxo: bool = False,
    contexto_irrelevante: bool = False,
    contexto_flexible: bool = False,
    acusa_contexto_flexible: bool = False,
    evitar_alucinaciones: bool = False,
    formato_estructurado: bool = False,
    conocimiento_tecnico_restringido: bool = False
) -> str:

    def __recolectar_roles() -> list:
        roles = []
        if es_corporativo:          roles.append(PROMPT_ROL_CORPORATIVO)
        if es_informal:             roles.append(PROMPT_ROL_INFORMAL)
        if es_medico:               roles.append(PROMPT_ROL_MEDICO)
        return roles

    def __recolectar_reglas_contexto() -> list:
        reglas = []
        if estricto_contexto:       reglas.append(PROMPT_REGLA_ESTRICTO_CONTEXTO)
        if estricto_contexto_laxo:  reglas.append(PROMPT_REGLA_ESTRICTO_CONTEXTO_LAXO)
        if contexto_irrelevante:    reglas.append(PROMPT_REGLA_CONTEXTO_IRRELEVANTE)
        if contexto_flexible:       reglas.append(PROMPT_REGLA_CONTEXTO_FLEXIBLE)
        return reglas
    
    def __recolectar_reglas_formato() -> list:
        reglas = []
        if acusa_contexto_flexible:             reglas.append(PROMPT_REGLA_ACUSA_CONTEXTO_FLEXIBLE)
        if conocimiento_tecnico_restringido:    reglas.append(PROMPT_REGLA_CONOCIMIENTO_TECNICO_RESTRINGIDO)
        if formato_estructurado:                reglas.append(PROMPT_REGLA_FORMATO_ESTRUCTURADO)
        return reglas

    def __recolectar_reglas_seguridad() -> list:
        reglas = []
        if evitar_alucinaciones:    reglas.append(PROMPT_REGLA_CERO_ALUCINACIONES)
        return reglas

    def __ensamblar_directrices(fragmentos: list) -> str:
        return " ".join(fragmentos)

    fragmentos_activos = []
    fragmentos_activos.extend(__recolectar_roles())
    fragmentos_activos.extend(__recolectar_reglas_contexto())
    fragmentos_activos.extend(__recolectar_reglas_formato())
    fragmentos_activos.extend(__recolectar_reglas_seguridad())

    return __ensamblar_directrices(fragmentos_activos)