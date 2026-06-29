import uuid
import json
import logging

from pathlib import Path

from langchain_text_splitters import (
    MarkdownHeaderTextSplitter,
    RecursiveCharacterTextSplitter
)

from rag.utils.sqlite import GestorSQLite

from rag import (
    DIR_SALIDA_MD,
    DIR_CHROMA_DB,
    DB_SQLITE_PADRES,
    DB_SQLITE_HIJOS,
    CHUNKING_LIMITE_CARACTERES,
    CHUNKING_SUPERPOSICION,
    CHUNKING_JERARQUIA_ENCABEZADOS
)

class Chunking:
    
    QUERY_CREAR_TABLA_PADRES = '''
        CREATE TABLE IF NOT EXISTS documentos_padre (
            id TEXT PRIMARY KEY, fuente TEXT NOT NULL, contenido TEXT NOT NULL, metadatos TEXT
        )
    '''
    QUERY_CREAR_TABLA_HIJOS = '''
        CREATE TABLE IF NOT EXISTS documentos_hijo (
            id TEXT PRIMARY KEY, parent_id TEXT NOT NULL, fuente TEXT NOT NULL, contenido TEXT NOT NULL, metadatos TEXT
        )
    '''
    QUERY_INSERTAR_PADRE = '''
        INSERT OR REPLACE INTO documentos_padre (id, fuente, contenido, metadatos) VALUES (?, ?, ?, ?)
    '''
    QUERY_INSERTAR_HIJO = '''
        INSERT OR REPLACE INTO documentos_hijo (id, parent_id, fuente, contenido, metadatos) VALUES (?, ?, ?, ?, ?)
    '''

    def __init__(
        self, 
        dir_entrada: str | Path = DIR_SALIDA_MD,
        dir_salida: str | Path = DIR_CHROMA_DB,
        limite_caracteres_chunk: int = CHUNKING_LIMITE_CARACTERES,
        caracteres_superposicion: int = CHUNKING_SUPERPOSICION
    ):
        self.dir_entrada = Path(dir_entrada)
        self.db_padres = GestorSQLite(Path(dir_salida) / DB_SQLITE_PADRES)
        self.db_hijos = GestorSQLite(Path(dir_salida) / DB_SQLITE_HIJOS)
        
        self.jerarquia_encabezados = CHUNKING_JERARQUIA_ENCABEZADOS
        
        self.divisor_semantico = MarkdownHeaderTextSplitter(
            headers_to_split_on=self.jerarquia_encabezados, 
            strip_headers=False
        )
        self.divisor_recursivo = RecursiveCharacterTextSplitter(
            chunk_size=limite_caracteres_chunk, 
            chunk_overlap=caracteres_superposicion, 
            length_function=len
        )

    def __inicializar_esquemas_db(self) -> None:
        self.db_padres.ejecutar_consulta(self.QUERY_CREAR_TABLA_PADRES)
        self.db_hijos.ejecutar_consulta(self.QUERY_CREAR_TABLA_HIJOS)

    def __leer_contenido_archivo(self, ruta_archivo: Path) -> str:
        with open(ruta_archivo, "r", encoding="utf-8") as archivo:
            return archivo.read()

    def __procesar_jerarquia_documental(self, texto_crudo: str, nombre_fuente: str) -> tuple[list[tuple], list[tuple]]:
        bloques_semanticos = self.divisor_semantico.split_text(texto_crudo)
        parametros_padres = []
        parametros_hijos = []

        for bloque_padre in bloques_semanticos:
            id_padre = str(uuid.uuid4())
            bloque_padre.metadata["parent_id"] = id_padre
            bloque_padre.metadata["fuente"] = nombre_fuente
            
            metadatos_padre_json = json.dumps(bloque_padre.metadata, ensure_ascii=False)
            parametros_padres.append((id_padre, nombre_fuente, bloque_padre.page_content, metadatos_padre_json))
            
            bloques_hijos = self.divisor_recursivo.split_documents([bloque_padre])
            
            for hijo in bloques_hijos:
                id_hijo = str(uuid.uuid4())
                hijo.metadata["id"] = id_hijo
                hijo.metadata["parent_id"] = id_padre
                hijo.metadata["fuente"] = nombre_fuente
                
                metadatos_hijo_json = json.dumps(hijo.metadata, ensure_ascii=False)
                parametros_hijos.append((id_hijo, id_padre, nombre_fuente, hijo.page_content, metadatos_hijo_json))

        return parametros_padres, parametros_hijos

    def __procesar_archivo_individual(self, ruta_archivo: Path) -> None:
        try:
            texto_crudo = self.__leer_contenido_archivo(ruta_archivo)
            padres_tuplas, hijos_tuplas = self.__procesar_jerarquia_documental(texto_crudo, ruta_archivo.name)
            
            self.db_padres.ejecutar_insercion_masiva(self.QUERY_INSERTAR_PADRE, padres_tuplas)
            self.db_hijos.ejecutar_insercion_masiva(self.QUERY_INSERTAR_HIJO, hijos_tuplas)
            
            logging.info(f"Fragmentación exitosa: {ruta_archivo.name} ({len(padres_tuplas)} padres, {len(hijos_tuplas)} hijos)")
        except Exception as error_fragmentacion:
            logging.error(f"Error procesando {ruta_archivo.name}: {error_fragmentacion}")

    def ejecutar_exportacion_a_sqlite(self) -> None:
        if not self.dir_entrada.exists():
            logging.error(f"Directorio de entrada no encontrado: {self.dir_entrada}")
            return

        self.__inicializar_esquemas_db()
        archivos_markdown = list(self.dir_entrada.glob("*.md"))
        
        logging.info(f"Iniciando fragmentación de {len(archivos_markdown)} archivos a SQLite.")
        for archivo in archivos_markdown: self.__procesar_archivo_individual(archivo)
        logging.info("Proceso de chunking finalizado.")