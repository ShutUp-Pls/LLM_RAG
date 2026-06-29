import logging

from rag.rag_parsing import Parsing
from rag.rag_chunking import Chunking
from rag.rag_embedding import Embedding

class Ingestion:
    def __init__(self):
        self.parsing = Parsing()
        self.chunking = Chunking()
        self.embedding = Embedding()

    def ejecutar_pipeline_completo(self) -> bool:
        logging.info("INICIANDO PIPELINE COMPLETO DE PROCESAMIENTO RAG")
        
        try:
            logging.info(">>> INICIANDO FASE 1: Parsing (Transformación a MD)")
            self.parsing.aplicar_parsing_de_documentos()
            logging.info("<<< FASE 1 COMPLETADA EXITOSAMENTE")

            logging.info(">>> INICIANDO FASE 2: Chunking (Fragmentación Padre-Hijo en SQLite)")
            self.chunking.ejecutar_exportacion_a_sqlite()
            logging.info("<<< FASE 2 COMPLETADA EXITOSAMENTE")

            logging.info(">>> INICIANDO FASE 3: Embedding (Vectorización a ChromaDB)")
            self.embedding.ejecutar_pipeline_ingesta()
            logging.info("<<< FASE 3 COMPLETADA EXITOSAMENTE")

            return True

        except Exception as error_critico:
            logging.error(f"Fallo crítico, ejecución del pipeline principal interrumpida: {error_critico}")
            return False