import logging
import sqlite3
import chromadb
from pathlib import Path
from sentence_transformers import SentenceTransformer
from rag import (
    DIR_CHROMA_DB,
    DB_SQLITE_PADRES,
    EMBEDDING_NOMBRE_COLECCION,
    EMBEDDING_MODELO,
    EMBEDDING_DISPOSITIVO,
    INFERENCIA_TOP_K
)

class Retrieval:
    def __init__(
        self,
        dir_chroma: Path = DIR_CHROMA_DB,
        db_padres: str = DB_SQLITE_PADRES,
        nombre_coleccion: str = EMBEDDING_NOMBRE_COLECCION,
        modelo_embeddings: str = EMBEDDING_MODELO,
        dispositivo: str = EMBEDDING_DISPOSITIVO
    ):
        self.dir_chroma = Path(dir_chroma)
        self.ruta_sqlite = self.dir_chroma / db_padres
        self.nombre_coleccion = nombre_coleccion
        self.nombre_modelo = modelo_embeddings
        self.dispositivo = dispositivo
        
        self.cliente_chroma = None
        self.coleccion = None
        self.conexion_sqlite = None
        self.cursor_sqlite = None
        self.modelo_vectorial = None

    def __inicializar_conexiones(self) -> None:
        self.cliente_chroma = chromadb.PersistentClient(path=str(self.dir_chroma))
        self.coleccion = self.cliente_chroma.get_collection(name=self.nombre_coleccion)
        self.conexion_sqlite = sqlite3.connect(str(self.ruta_sqlite))
        self.cursor_sqlite = self.conexion_sqlite.cursor()

    def __cargar_modelo_embeddings(self) -> None:
        self.modelo_vectorial = SentenceTransformer(self.nombre_modelo, device=self.dispositivo)

    def __vectorizar_pregunta(self, pregunta: str) -> list:
        return self.modelo_vectorial.encode([pregunta]).tolist()

    def __consultar_vectores_hijos(self, vector_pregunta: list, top_k: int) -> list:
        resultados = self.coleccion.query(
            query_embeddings=vector_pregunta,
            n_results=top_k,
            include=["metadatas"]
        )
        return resultados.get("metadatas", [[]])[0]

    def __consultar_contenido_padre(self, parent_id: str) -> str | None:
        self.cursor_sqlite.execute('SELECT contenido FROM documentos_padre WHERE id = ?', (parent_id,))
        resultado = self.cursor_sqlite.fetchone()
        return resultado[0] if resultado else None

    def __reunir_texto_padres(self, metadatos_hijos: list) -> list:
        contextos = []
        ids_procesados = set()
        for meta in metadatos_hijos:
            parent_id = meta.get("parent_id")
            if parent_id and parent_id not in ids_procesados:
                ids_procesados.add(parent_id) if hasattr(self, 'ids_processed') else ids_procesados.add(parent_id)
                contenido = self.__consultar_contenido_padre(parent_id)
                if contenido:
                    contextos.append(contenido)
        return contextos

    def __unificar_bloques_contexto(self, contextos: list) -> str:
        return "\n\n---\n\n".join(contextos)

    def preparar_recuperador(self) -> None:
        self.__inicializar_conexiones()
        self.__cargar_modelo_embeddings()

    def obtener_contexto(self, pregunta: str, top_k: int = INFERENCIA_TOP_K) -> str:
        vector = self.__vectorizar_pregunta(pregunta)
        metadatos_hijos = self.__consultar_vectores_hijos(vector, top_k)
        if not metadatos_hijos:
            return ""
        contextos_lista = self.__reunir_texto_padres(metadatos_hijos)
        return self.__unificar_bloques_contexto(contextos_lista)

    def cerrar_recursos(self) -> None:
        if self.conexion_sqlite:
            self.conexion_sqlite.close()

    def realizar_prueba_recuperacion(self, pregunta_prueba: str = "obesidad") -> None:
        """Prueba interna de recuperación semántica que aísla la lógica de presentación del main."""
        logging.info(f"Iniciando prueba de recuperación semántica. Query: '{pregunta_prueba}'")
        try:
            self.preparar_recuperador()
            contexto_recuperado = self.obtener_contexto(pregunta_prueba)
            
            logging.info("--- PRUEBA DE RECUPERACIÓN ---")
            if contexto_recuperado:
                logging.info(f"Contexto recuperado de forma semántica:\n{contexto_recuperado}")
            else:
                logging.warning("No se encontró contexto relevante para la consulta de prueba.")
        except Exception as error:
            logging.error(f"Fallo durante la ejecución de la prueba de recuperación: {error}")
        finally:
            self.cerrar_recursos()