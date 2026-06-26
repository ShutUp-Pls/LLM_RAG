import logging
from logger import configurar_sistema_registros

from rag_ingesta import aplicar_ingesta_de_documentos
from rag_embedding import ejecutar_pipeline_ingesta_nativo

PREFIJO_LOG = "pipeline_principal"

def ejecutar_pipeline_completo():
    configurar_sistema_registros(PREFIJO_LOG)
    logging.info("INICIANDO PIPELINE COMPLETO DE PROCESAMIENTO RAG")
    try:
        logging.info(">>> INICIANDO FASE 1: Transformación e Ingesta de Documentos")
        aplicar_ingesta_de_documentos()
        configurar_sistema_registros(PREFIJO_LOG)
        logging.info("<<< FASE 1 COMPLETADA EXITOSAMENTE")

        logging.info(">>> INICIANDO FASES 2 y 3: Chunking (Padre-Hijo) y Embedding")
        coleccion, modelo = ejecutar_pipeline_ingesta_nativo()
        configurar_sistema_registros(PREFIJO_LOG)
        logging.info("<<< FASE 2 y 3 COMPLETADA EXITOSAMENTE")

        if coleccion and modelo:
            return True
        else:
            logging.error("El pipeline terminó, pero ocurrió un error durante el Chunking o Embedding.")
            return False

    except Exception as error_critico:
        configurar_sistema_registros(PREFIJO_LOG)
        logging.error(f"Fallo crítico, ejecución del pipeline principal interrumpida: {error_critico}")
        return False

if __name__ == "__main__":
    ejecutar_pipeline_completo()