import logging
import sqlite3
import chromadb
from sentence_transformers import SentenceTransformer

def conectar_bases_de_datos(directorio_db, ruta_sqlite, nombre_coleccion):
    logging.info("Inicializando conexiones a bases de datos de recuperacion...")
    
    cliente_chroma = chromadb.PersistentClient(path=directorio_db)
    coleccion = cliente_chroma.get_collection(name=nombre_coleccion)
    
    conexion_sqlite = sqlite3.connect(ruta_sqlite)
    cursor_sqlite = conexion_sqlite.cursor()
    
    logging.info("Bases de datos conectadas exitosamente.")
    return coleccion, conexion_sqlite, cursor_sqlite

def cargar_modelo_embeddings(ruta_emb, disp_emb):
    logging.info(f"Cargando modelo de embeddings: {ruta_emb} en {disp_emb}")
    modelo_emb = SentenceTransformer(ruta_emb, device=disp_emb)
    logging.info("Modelo de embeddings cargado exitosamente.")
    return modelo_emb

def recuperar_contexto_desde_pregunta(pregunta, coleccion, modelo_emb, cursor_sqlite, top_k):
    logging.info("Vectorizando pregunta del usuario...")
    vector_pregunta = modelo_emb.encode([pregunta]).tolist()
    
    logging.info("Buscando similitud en ChromaDB (Capa Fija - Hijos)...")
    resultados = coleccion.query(
        query_embeddings=vector_pregunta,
        n_results=top_k,
        include=["metadatas"]
    )
    
    metadatos_hijos = resultados["metadatas"][0]
    
    if not metadatos_hijos:
        logging.warning("No se encontro informacion relevante en la base de datos vectorial.")
        return ""

    contextos_recuperados = []
    ids_procesados = set()
    
    logging.info("Recuperando secciones completas en SQLite (Capa Semantica - Padres)...")
    for meta in metadatos_hijos:
        parent_id = meta.get("parent_id")
        
        if parent_id and parent_id not in ids_procesados:
            ids_procesados.add(parent_id)
            cursor_sqlite.execute('SELECT contenido FROM documentos_padre WHERE id = ?', (parent_id,))
            resultado_padre = cursor_sqlite.fetchone()
            
            if resultado_padre:
                contextos_recuperados.append(resultado_padre[0])
                logging.info(f"Contexto anadido desde parent_id: {parent_id}")
                
    texto_contexto_final = "\n\n---\n\n".join(contextos_recuperados)
    return texto_contexto_final