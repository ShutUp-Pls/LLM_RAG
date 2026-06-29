import logging

from llms.qwen import QwenLocal
from rag.rag_retrieval import Retrieval

class Agente:
    def __init__(self, llm: QwenLocal, retriever: Retrieval):
        self.llm = llm
        self.retriever = retriever
        self.catalogo = self.retriever.obtener_catalogo_temas()

        logging.info("Inicializado correctamente. Catálogo detectado: [%s]", self.catalogo)

    def procesar_consulta(self, consulta: str) -> tuple[str, str]:
        logging.info("Consulta entrante: '%s'", consulta)

        analisis = self.llm.enrutar_consulta(
            consulta, 
            evaluar_rag=True, 
            evaluar_intencion=True, 
            catalogo_temas=self.catalogo
        )

        logging.info("Decisiones del Router: %s", analisis)
        
        contexto = ""
        if analisis.get("requiere_rag", False):
            contexto_crudo = self.retriever.obtener_contexto(consulta)
            logging.info("Contexto crudo recuperado. Tamaño: %d caracteres.", len(contexto_crudo))
            
            logging.info("Iniciando compresión semántica del contexto...")
            contexto = self.llm.condensar_contexto(consulta, contexto_crudo)
            logging.info("Contexto condensado con éxito. Nuevo tamaño: %d caracteres.", len(contexto))

        logging.debug("Iniciando generación de respuesta final...")    
        respuesta = self.llm.generar_respuesta(
            consulta=consulta,
            contexto=contexto
        )
        
        return respuesta, contexto