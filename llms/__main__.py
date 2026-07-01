import sys
from llms.agente import Agente
from rag.rag_retrieval import Retrieval
from rag.utils.logger import configurar_sistema_registros

def obtener_modelo(nombre_modelo: str):
    if nombre_modelo.lower() == "qwen":
        from llms.qwen import QwenLocal
        return QwenLocal()
    elif nombre_modelo.lower() == "openai":
        from llms.openai_llm import OpenAILLM
        return OpenAILLM()
    else:
        print(f"Error: El modelo '{nombre_modelo}' no es válido. Usa 'qwen' u 'openai'.")
        sys.exit(1)

def ejecutar_diagnostico_integrado(nombre_modelo: str) -> None:
    llm = obtener_modelo(nombre_modelo)
    llm.inicializar_modelo()
    
    retriever = Retrieval()
    retriever.preparar_recuperador()
    
    agente = Agente(llm, retriever)

    consultas_test = [
        "¡Hola buenas tardes! ¿Qué tal todo por ahí?", 
        "¿Cuáles son los principales hallazgos o datos sobre la obesidad?",
        "Escribe una función de ordenamiento burbuja en Python.",
        "Mi perro tiene un gato mascota."
    ]
    
    for consulta in consultas_test:
        print("=" * 70)
        print(f"Consulta entrante: '{consulta}'")
        respuesta, _ = agente.procesar_consulta(consulta)
        print(f"\n-> Respuesta final del sistema:\n{respuesta}\n")

    retriever.cerrar_recursos()

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Uso: python3 -m llms <qwen|openai>")
        sys.exit(1)
        
    configurar_sistema_registros(f"agente_{sys.argv[1]}")
    ejecutar_diagnostico_integrado(sys.argv[1])