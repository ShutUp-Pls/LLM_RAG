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
PROMPT_REGLA_CONTEXTO_FLEXIBLE = """
Utiliza el contexto dado como fuente principal de verdad.
Puedes apoyarte en tu conocimiento interno si es estrictamente necesario para complementar la respuesta.
"""
PROMPT_REGLA_ACUSA_CONTEXTO_FLEXIBLE = """
Cuando uses conocimiento fuera del contexto dado, indica claramente que parte de tu respuesta no está en el contexto.
"""
PROMPT_REGLA_CERO_ALUCINACIONES = """
Bajo ninguna circunstancia inventes datos, nombres, fechas o cifras.
"""
PROMPT_REGLA_FORMATO_ESTRUCTURADO = """
Estructura tu respuesta usando listas y viñetas para facilitar la lectura.
"""

def construir_prompt_sistema(
    es_corporativo: bool = False,
    es_medico: bool = False,
    es_informal: bool = False,
    estricto_contexto: bool = False,
    contexto_flexible: bool = False,
    acusa_contexto_flexible: bool = False,
    evitar_alucinaciones: bool = False,
    formato_estructurado: bool = False
) -> str:

    def __recolectar_roles() -> list:
        roles = []
        if es_corporativo: roles.append(PROMPT_ROL_CORPORATIVO)
        if es_informal: roles.append(PROMPT_ROL_INFORMAL)
        if es_medico: roles.append(PROMPT_ROL_MEDICO)
        return roles

    def __recolectar_reglas_contexto() -> list:
        reglas = []
        if estricto_contexto: reglas.append(PROMPT_REGLA_ESTRICTO_CONTEXTO)
        if contexto_flexible: reglas.append(PROMPT_REGLA_CONTEXTO_FLEXIBLE)
        if acusa_contexto_flexible: reglas.append(PROMPT_REGLA_ACUSA_CONTEXTO_FLEXIBLE)
        return reglas

    def __recolectar_reglas_seguridad() -> list:
        reglas = []
        if evitar_alucinaciones: reglas.append(PROMPT_REGLA_CERO_ALUCINACIONES)
        return reglas

    def __recolectar_reglas_formato() -> list:
        reglas = []
        if formato_estructurado: reglas.append(PROMPT_REGLA_FORMATO_ESTRUCTURADO)
        return reglas

    def __ensamblar_directrices(fragmentos: list) -> str:
        return " ".join(fragmentos)

    fragmentos_activos = []
    fragmentos_activos.extend(__recolectar_roles())
    fragmentos_activos.extend(__recolectar_reglas_contexto())
    fragmentos_activos.extend(__recolectar_reglas_seguridad())
    fragmentos_activos.extend(__recolectar_reglas_formato())

    return __ensamblar_directrices(fragmentos_activos)