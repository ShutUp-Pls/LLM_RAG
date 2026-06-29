import sys
from llms.qwen import QwenLocal

def ejecutar_diagnostico_aislado() -> None:
    print("Iniciando carga de prueba para QwenLocal...")
    llm = QwenLocal()
    llm.inicializar_modelo()
    
    contexto_test = "Chile es un país ubicado en el extremo sudoeste de América del Sur. Su capital oficial es Santiago."
    print(f"Contexto simulado: {contexto_test}")

    pregunta_test_uno = "¿Cuál es la capital oficial de Chile?"
    pregunta_test_dos = "¿Cuál es la capital oficial de Argentina?"
    
    print(f"\nPregunta de prueba: {pregunta_test_uno}")
    respuesta = llm.generar_respuesta(pregunta_test_uno, contexto_test)
    print(f"Respuesta obtenida del LLM:\n{respuesta}\n")

    print(f"\nPregunta de prueba: {pregunta_test_dos}")
    respuesta = llm.generar_respuesta(pregunta_test_dos, contexto_test)
    print(f"Respuesta obtenida del LLM:\n{respuesta}\n")

if __name__ == "__main__":
    ejecutar_diagnostico_aislado()