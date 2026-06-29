import sys
from llms.agente import Agente
from llms.qwen import QwenLocal

from rag.rag_retrieval import Retrieval
from rag.utils.logger import configurar_sistema_registros

def ejecutar_diagnostico_integrado() -> None:
    llm = QwenLocal()
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
    configurar_sistema_registros("agente")
    ejecutar_diagnostico_integrado()