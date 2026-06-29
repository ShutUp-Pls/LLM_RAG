import json
import logging
import chromadb
import warnings
import transformers
import huggingface_hub

from pathlib import Path

from sentence_transformers import SentenceTransformer

from rag.utils.sqlite import GestorSQLite

from rag import (
    DIR_CHROMA_DB, 
    DB_SQLITE_PADRES, 
    DB_SQLITE_HIJOS,
    EMBEDDING_NOMBRE_COLECCION, 
    EMBEDDING_MODELO, 
    EMBEDDING_TAMANO_LOTE, 
    EMBEDDING_DISPOSITIVO, 
    EMBEDDING_SILENCIAR_ADVERTENCIAS
)

class Embedding:
    QUERY_OBTENER_HIJOS = "SELECT id, contenido, metadatos FROM documentos_hijo"
    QUERY_OBTENER_PADRE_POR_ID = "SELECT contenido FROM documentos_padre WHERE id = ?"

    def __init__(
        self,
        dir_chroma_db: str | Path = DIR_CHROMA_DB,
        nombre_coleccion: str = EMBEDDING_NOMBRE_COLECCION,
        modelo_embeddings: str = EMBEDDING_MODELO,
        tamano_lote_insercion: int = EMBEDDING_TAMANO_LOTE,
        dispositivo_ejecucion: str = EMBEDDING_DISPOSITIVO,
        silenciar_advertencias: bool = EMBEDDING_SILENCIAR_ADVERTENCIAS
    ):
        self.tamano_lote = tamano_lote_insercion
        self.db_padres = GestorSQLite(Path(dir_chroma_db) / DB_SQLITE_PADRES)
        self.db_hijos = GestorSQLite(Path(dir_chroma_db) / DB_SQLITE_HIJOS)
        
        self.__aplicar_silenciadores_entorno(silenciar_advertencias)
        
        logging.info(f"Conectando a ChromaDB en: {dir_chroma_db}")
        self.cliente_chroma = chromadb.PersistentClient(path=str(dir_chroma_db))
        self.coleccion = self.cliente_chroma.get_or_create_collection(name=nombre_coleccion)
        
        logging.info(f"Cargando modelo SentenceTransformer: {modelo_embeddings}")
        self.modelo_vectorial = SentenceTransformer(modelo_embeddings, device=dispositivo_ejecucion)

    def __aplicar_silenciadores_entorno(self, activar_silencio: bool) -> None:
        if activar_silencio:
            warnings.filterwarnings("ignore")
            transformers.logging.set_verbosity_error()
            huggingface_hub.utils.logging.set_verbosity_error()
            huggingface_hub.utils.disable_progress_bars()

    def __extraer_y_formatear_hijos(self) -> tuple[list, list, list]:
        registros_crudos = self.db_hijos.obtener_todos(self.QUERY_OBTENER_HIJOS)
        
        textos_extraidos, metadatos_extraidos, ids_extraidos = [], [], []
        for id_hijo, contenido, metadatos_json in registros_crudos:
            textos_extraidos.append(contenido)
            metadatos_extraidos.append(json.loads(metadatos_json))
            ids_extraidos.append(id_hijo)

        return textos_extraidos, metadatos_extraidos, ids_extraidos

    def __vectorizar_e_ingestar_por_lotes(self, textos: list, metadatos: list, ids: list) -> None:
        total_elementos = len(textos)
        logging.info(f"Vectorizando hacia ChromaDB. Total de fragmentos: {total_elementos}")
        
        for indice in range(0, total_elementos, self.tamano_lote):
            lote_textos = textos[indice:indice + self.tamano_lote]
            lote_metadatos = metadatos[indice:indice + self.tamano_lote]
            lote_ids = ids[indice:indice + self.tamano_lote]
            
            try:
                vectores = self.modelo_vectorial.encode(lote_textos).tolist()
                self.coleccion.add(documents=lote_textos, embeddings=vectores, metadatas=lote_metadatos, ids=lote_ids)
                logging.info(f"Lote insertado: {min(indice + self.tamano_lote, total_elementos)}/{total_elementos}")
            except Exception as error_insercion:
                logging.error(f"Fallo crítico insertando lote en índice {indice}: {error_insercion}")
                raise

    def ejecutar_pipeline_ingesta(self) -> None:
        if not self.db_hijos.ruta_db.exists():
            logging.error("Base de datos de hijos no encontrada. Abortando.")
            return

        try:
            textos, metadatos, ids = self.__extraer_y_formatear_hijos()
            if not textos:
                logging.warning("Tabla de hijos vacía.")
                return

            self.__vectorizar_e_ingestar_por_lotes(textos, metadatos, ids)
            logging.info("Sincronización finalizada con éxito.")
        except Exception as error_pipeline:
            logging.error(f"Ingesta interrumpida: {error_pipeline}")

    def realizar_prueba_integridad(self) -> None:
        logging.info("Iniciando prueba de integridad relacional (Hijo -> Padre)...")
        
        query_hijo_azar = "SELECT id, parent_id, contenido FROM documentos_hijo ORDER BY RANDOM() LIMIT 1"
        resultado_hijo = self.db_hijos.obtener_uno(query_hijo_azar)
        
        if not resultado_hijo:
            logging.warning("No hay registros en la base de datos de hijos para realizar la prueba.")
            return
            
        id_hijo, parent_id, contenido_hijo = resultado_hijo
        logging.info(f"--- MATCH EN SQLITE (HIJO ID: {id_hijo}) ---")
        logging.info(f"Extracto: {contenido_hijo[:150]}...") 
        
        if not parent_id:
            logging.error("Inconsistencia: El documento hijo no tiene un parent_id asignado.")
            return

        resultado_padre = self.db_padres.obtener_uno(self.QUERY_OBTENER_PADRE_POR_ID, (parent_id,))
        
        if resultado_padre:
            contenido_padre = resultado_padre[0]
            logging.info(f"--- MATCH EN SQLITE (PADRE ID: {parent_id}) ---")
            logging.info(f"Texto íntegro verificado: {contenido_padre[:300]}...")
            logging.info("Prueba de integridad superada exitosamente.")
        else:
            logging.error(f"Inconsistencia: El parent_id {parent_id} no existe en la tabla documentos_padre.")