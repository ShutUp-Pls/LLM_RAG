import argparse
from rag.utils.logger import configurar_sistema_registros

INGESTION = "ingestion"
CHUNKING = "chunking"
EMBEDDING = "embedding"
RETRIEVAL = "retrieval"
PARSING = "parsing"

def main():
    parser = argparse.ArgumentParser(description="Pipeline RAG: Ejecución de módulos.")
    parser.add_argument(
        "modulo", 
        choices=[INGESTION, PARSING, CHUNKING, EMBEDDING, RETRIEVAL],
        help="El submódulo que deseas ejecutar de manera aislada o completa."
    )
    
    args = parser.parse_args()

    if args.modulo == INGESTION:
        from rag import Ingestion
        configurar_sistema_registros(INGESTION)
        pipeline = Ingestion()
        pipeline.ejecutar_pipeline_completo()

    elif args.modulo == PARSING:
        from rag.rag_parsing import Parsing, DIR_ENTRADA_ORG, DIR_SALIDA_MD
        configurar_sistema_registros(PARSING)
        parsing = Parsing(DIR_ENTRADA_ORG, DIR_SALIDA_MD)
        parsing.aplicar_parsing_de_documentos()

    elif args.modulo == CHUNKING:
        from rag.rag_chunking import Chunking
        configurar_sistema_registros(CHUNKING)
        chunking = Chunking()
        chunking.ejecutar_exportacion_a_sqlite()

    elif args.modulo == EMBEDDING:
        from rag.rag_embedding import Embedding
        configurar_sistema_registros(EMBEDDING)
        embedding = Embedding()
        embedding.ejecutar_pipeline_ingesta()
        embedding.realizar_prueba_integridad()

    elif args.modulo == RETRIEVAL:
        from rag.rag_retrieval import Retrieval
        configurar_sistema_registros(RETRIEVAL)
        retrieval = Retrieval()
        retrieval.realizar_prueba_recuperacion()

if __name__ == "__main__":
    main()