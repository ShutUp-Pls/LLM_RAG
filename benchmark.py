import csv
import logging
import sys

from llms.agente import Agente
from rag.rag_retrieval import Retrieval
from rag.utils.logger import configurar_sistema_registros

def obtener_modelo(nombre_modelo: str):
    if nombre_modelo.lower() == "qwen":
        from llms.qwen import QwenLocal
        return QwenLocal()
    elif nombre_modelo.lower() == "openai":
        from llms.openai import OpenAILLM
        return OpenAILLM()
    else:
        logging.error(f"El modelo '{nombre_modelo}' no es válido. Usa 'qwen' u 'openai'.")
        sys.exit(1)

def ejecutar_benchmark(nombre_modelo: str) -> None:
    logging.info(f"Cargando modelo {nombre_modelo}...")
    llm = obtener_modelo(nombre_modelo)
    llm.inicializar_modelo()
    
    logging.info("Inicializando sistema RAG y Agente...")
    retriever = Retrieval()
    retriever.preparar_recuperador()
    agente = Agente(llm, retriever)

    archivo_entrada = "preguntas.csv"
    archivo_salida = f"MarcoDelgado_RAG_{nombre_modelo}.csv"
    
    preguntas = []
    try:
        with open(archivo_entrada, mode='r', encoding='utf-8') as f_in:
            lector = csv.DictReader(f_in)
            for fila in lector:
                preguntas.append({
                    "id": fila['id'],
                    "consulta": fila['pregunta']
                })
    except FileNotFoundError:
        logging.error(f"No se encontró el archivo: {archivo_entrada}")
        sys.exit(1)

    logging.info(f"Se cargaron {len(preguntas)} preguntas para evaluar.")

    with open(archivo_salida, mode='w', encoding='utf-8', newline='') as f_out:
        f_out.write("id,answer,context\n")
        escritor = csv.writer(f_out, quoting=csv.QUOTE_NONNUMERIC)
        
        for item in preguntas:
            print("-" * 50)
            logging.info(f"Procesando ID {item['id']}: {item['consulta']}")
            
            respuesta, contexto = agente.procesar_consulta(item['consulta'])

            respuesta_limpia = respuesta.replace('\n', ' ').replace('\r', ' ').strip() if respuesta else ""
            contexto_limpio = contexto.replace('\n', ' ').replace('\r', ' ').strip() if contexto else ""
            id_numerico = int(item['id'])
            
            escritor.writerow([id_numerico, respuesta_limpia, contexto_limpio])

    logging.info(f"\n¡Benchmark finalizado! Archivo de entrega guardado en: {archivo_salida}")
    retriever.cerrar_recursos()

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Uso: python3 benchmark.py <qwen|openai>")
        sys.exit(1)
        
    modelo_seleccionado = sys.argv[1]
    configurar_sistema_registros(f"benchmark_{modelo_seleccionado}")
    ejecutar_benchmark(modelo_seleccionado)