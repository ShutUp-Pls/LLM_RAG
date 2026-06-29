import csv
import logging
import sys

from llms.agente import Agente
from llms.qwen import QwenLocal
from rag.rag_retrieval import Retrieval
from rag.utils.logger import configurar_sistema_registros

def ejecutar_benchmark() -> None:
    logging.info("Cargando modelo QwenLocal...")
    llm = QwenLocal()
    llm.inicializar_modelo()
    
    logging.info("Inicializando sistema RAG y Agente...")
    retriever = Retrieval()
    retriever.preparar_recuperador()
    agente = Agente(llm, retriever)

    archivo_entrada = "preguntas.csv"
    archivo_salida = "MarcoDelgado_RAG_1.csv"
    
    preguntas = []
    try:
        with open(archivo_entrada, mode='r', encoding='utf-8') as f_in:
            lector = csv.DictReader(f_in)
            for fila in lector:
                preguntas.append({
                    "id": fila['numero'],
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
    configurar_sistema_registros("benchmark")
    ejecutar_benchmark()