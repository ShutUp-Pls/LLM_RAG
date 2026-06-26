import logging
import uuid
import warnings
import json
import sqlite3
from pathlib import Path

import chromadb
import transformers
import huggingface_hub
from sentence_transformers import SentenceTransformer

from rag.rag_chunking import procesar_directorio_completo_markdown
from ..logger import configurar_sistema_registros

PREFIJO_LOG = "embedding"
DIRECTORIO_BASE_DATOS = "./chroma_db"
RUTA_ALMACEN_PADRES = "./chroma_db/padres.sqlite3"
NOMBRE_COLECCION = "conocimiento_qwen"
MODELO_EMBEDDINGS_LOCAL = "intfloat/multilingual-e5-small"
TAMANO_LOTE_INSERCION = 50
DISPOSITIVO_EJECUCION = "cpu"
SILENCIAR_ADVERTENCIAS_LIBRERIAS = True

def aplicar_silenciadores_librerias():
    if SILENCIAR_ADVERTENCIAS_LIBRERIAS:
        warnings.filterwarnings("ignore")
        transformers.logging.set_verbosity_error()
        huggingface_hub.utils.logging.set_verbosity_error()
        huggingface_hub.utils.disable_progress_bars()

def inicializar_cliente_base_datos():
    logging.info(f"Conectando a ChromaDB persistente en: {DIRECTORIO_BASE_DATOS}")
    cliente = chromadb.PersistentClient(path=DIRECTORIO_BASE_DATOS)
    coleccion = cliente.get_or_create_collection(name=NOMBRE_COLECCION)
    return coleccion

def instanciar_modelo_embeddings():
    logging.info(f"Cargando modelo SentenceTransformer: {MODELO_EMBEDDINGS_LOCAL} en {DISPOSITIVO_EJECUCION}")
    modelo = SentenceTransformer(MODELO_EMBEDDINGS_LOCAL, device=DISPOSITIVO_EJECUCION)
    return modelo

def guardar_padres_sqlite(padres_langchain, ruta_salida):
    Path(ruta_salida).parent.mkdir(parents=True, exist_ok=True)
    conexion = sqlite3.connect(ruta_salida)
    cursor = conexion.cursor()
    
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS documentos_padre (
            id TEXT PRIMARY KEY,
            contenido TEXT NOT NULL,
            metadatos TEXT
        )
    ''')
    
    padres_insertados = 0
    for padre in padres_langchain:
        id_padre = padre.metadata.get("parent_id")
        if id_padre:
            metadatos_json = json.dumps(padre.metadata, ensure_ascii=False)
            cursor.execute('''
                INSERT OR REPLACE INTO documentos_padre (id, contenido, metadatos)
                VALUES (?, ?, ?)
            ''', (id_padre, padre.page_content, metadatos_json))
            padres_insertados += 1
            
    conexion.commit()
    conexion.close()
    logging.info(f"Almacen SQLite actualizado: {padres_insertados} padres guardados en {ruta_salida}")

def desensamblar_chunks_langchain(chunks_langchain):
    textos = []
    metadatos = []
    ids = []
    
    for chunk in chunks_langchain:
        textos.append(chunk.page_content)
        metadata_limpia = chunk.metadata if chunk.metadata else {"fuente": "desconocida"}
        metadatos.append(metadata_limpia)
        ids.append(str(uuid.uuid4()))
        
    return textos, metadatos, ids

def ingestar_por_lotes(coleccion, modelo, textos, metadatos, ids):
    total_textos = len(textos)
    logging.info(f"Iniciando vectorizacion nativa por lotes. Total de fragmentos hijos: {total_textos}")
    
    for indice in range(0, total_textos, TAMANO_LOTE_INSERCION):
        lote_textos = textos[indice:indice + TAMANO_LOTE_INSERCION]
        lote_metadatos = metadatos[indice:indice + TAMANO_LOTE_INSERCION]
        lote_ids = ids[indice:indice + TAMANO_LOTE_INSERCION]
        
        try:
            vectores = modelo.encode(lote_textos).tolist()
            coleccion.add(
                documents=lote_textos,
                embeddings=vectores,
                metadatas=lote_metadatos,
                ids=lote_ids
            )
            logging.info(f"Lote insertado exitosamente: {indice + len(lote_textos)}/{total_textos}")
        except Exception as error_lote:
            logging.error(f"Fallo critico al insertar el lote {indice}: {error_lote}")
            raise

def ejecutar_pipeline_ingesta_nativo():
    configurar_sistema_registros(PREFIJO_LOG)
    aplicar_silenciadores_librerias()
    logging.info("Iniciando motor de vectorizacion y almacenamiento Parent-Child (SQLite).")
    
    padres_extraidos, hijos_extraidos = procesar_directorio_completo_markdown()
    
    if not padres_extraidos or not hijos_extraidos:
        logging.warning("No se recibieron fragmentos suficientes. Abortando proceso.")
        return None, None

    try:
        guardar_padres_sqlite(padres_extraidos, RUTA_ALMACEN_PADRES)
        
        textos, metadatos, ids = desensamblar_chunks_langchain(hijos_extraidos)
        coleccion = inicializar_cliente_base_datos()
        modelo = instanciar_modelo_embeddings()
        
        ingestar_por_lotes(coleccion, modelo, textos, metadatos, ids)
        
        logging.info("Pipeline de vectorizacion y almacenamiento Parent-Child finalizado con exito.")
        return coleccion, modelo
    except Exception as error_pipeline:
        logging.error(f"El proceso de ingesta fue interrumpido: {error_pipeline}")
        return None, None

def realizar_prueba_recuperacion(coleccion, modelo, pregunta_prueba="obesidad"):
    logging.info(f"Iniciando prueba de recuperacion. Query: '{pregunta_prueba}'")
    
    vector_pregunta = modelo.encode([pregunta_prueba]).tolist()
    
    resultados = coleccion.query(
        query_embeddings=vector_pregunta,
        n_results=1, 
        include=["documents", "metadatas", "distances"]
    )
    
    documentos_hijos = resultados["documents"][0]
    metadatos_hijos = resultados["metadatas"][0]
    
    if not documentos_hijos:
        logging.warning("No se encontraron resultados de busqueda en ChromaDB.")
        return
        
    try:
        conexion = sqlite3.connect(RUTA_ALMACEN_PADRES)
        cursor = conexion.cursor()
    except Exception as error_db:
        logging.error(f"Fallo al conectar con la base de datos de Padres: {error_db}")
        return

    for i, (doc_hijo, meta_hijo) in enumerate(zip(documentos_hijos, metadatos_hijos)):
        logging.info(f"--- MATCH ENCONTRADO EN LA CAPA FIJA (HIJO) ---")
        logging.info(f"Metadata Hijo: {meta_hijo}")
        logging.info(f"Extracto del Hijo: {doc_hijo[:150]}...") 
        
        parent_id = meta_hijo.get("parent_id")
        
        if parent_id:
            cursor.execute('SELECT contenido, metadatos FROM documentos_padre WHERE id = ?', (parent_id,))
            resultado_padre = cursor.fetchone()
            
            if resultado_padre:
                contenido_padre = resultado_padre[0]
                metadatos_padre = json.loads(resultado_padre[1])
                logging.info(f"--- RECUPERACION DE CAPA SEMANTICA (PADRE) EXITOSA ---")
                logging.info(f"Metadata recuperada del DB: {metadatos_padre}")
                logging.info(f"Extracto del Padre (El texto que ira a Qwen): {contenido_padre[:300]}...")
            else:
                logging.error(f"Inconsistencia referencial: El parent_id {parent_id} no existe en SQLite.")
        else:
            logging.error("El hijo recuperado no posee un parent_id valido.")

    conexion.close()

if __name__ == "__main__":
    coleccion_cargada, modelo_cargado = ejecutar_pipeline_ingesta_nativo()
    
    if coleccion_cargada and modelo_cargado:
        realizar_prueba_recuperacion(
            coleccion=coleccion_cargada, 
            modelo=modelo_cargado, 
            pregunta_prueba="obesidad" 
        )