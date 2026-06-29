import fitz
import logging

from pathlib import Path

from rag.utils.sys_os import crear_directorios
from rag.utils.marker import convertir_mediante_marker
from rag.utils.docling import convertir_mediante_docling

from rag import (
    DIR_ENTRADA_ORG, 
    DIR_SALIDA_MD, 
    PARSING_EXTENSIONES_SOPORTADAS
)

'''
- Mientras un texto plano solo provee información, markdown provee información y estructura, lo cual
para un LLM y, en especial, para un sistema RAG que requiere determinar jerarquía en la información
este formato se vuelve sumamente beneficioso.
- Sin embargo, la información puede venir en tantos formatos distintos que la traducción a markdown
no es un proceso trivial ni facilmente estandarizable.
- Para hacernos cargo en una primera instancia de esta variabilidad, hemos decidido aplicar 2 procesos:

    1) Docling: Computacionalmente barato, pero debil frente a archivos con información visual.

    2) Marker: Computacionalmente caro, pero robusto frente a archivos con información visual.

- Así, la idea es simple. Si usamos Docling y el resultado es malo, reintentamos con Marker.
'''

class Parsing:

    def __init__(self, dir_entrada: Path = DIR_ENTRADA_ORG, dir_salida: Path = DIR_SALIDA_MD):
        self.dir_entrada = Path(dir_entrada)
        self.dir_salida = Path(dir_salida)

    def __inicializar_entorno_directorios(self) -> None:
        crear_directorios([self.dir_entrada, self.dir_salida])

    def __obtener_listado_archivos_validos(self) -> list:
        directorio = self.dir_entrada.iterdir()
        return [archivo for archivo in directorio if archivo.is_file() and not archivo.name.startswith('.')]

    def __calcular_promedio_caracteres_por_pagina(self, documento: fitz.Document) -> float:
        longitud_total = sum(len(pagina.get_text().strip()) for pagina in documento)
        return longitud_total / len(documento)

    def __es_pdf_carente_de_capa_texto(self, ruta_archivo: Path) -> bool:
        try:
            documento = fitz.open(ruta_archivo)
            if len(documento) == 0: 
                return False
            
            promedio_caracteres = self.__calcular_promedio_caracteres_por_pagina(documento)
            documento.close()
            return promedio_caracteres < 50
        
        except Exception as error_lectura:
            logging.error(f"Fallo al leer PDF con PyMuPDF {ruta_archivo.name}: {error_lectura}")
            return True

    def __intentar_conversion_docling_con_respaldo(self, archivo: Path, ruta_salida: Path) -> None:
        try: 
            convertir_mediante_docling(archivo, ruta_salida)
        except ValueError as error_validacion:
            logging.warning(f"Docling descartó el archivo {archivo.name}: {error_validacion}")
            convertir_mediante_marker(archivo, ruta_salida)

    def __procesar_documento_formato_pdf(self, archivo: Path, ruta_salida: Path) -> None:
        es_imagen_escaneada = self.__es_pdf_carente_de_capa_texto(archivo)
        if es_imagen_escaneada:
            logging.info(f"PDF sin capa de texto detectado. Derivando a Marker: {archivo.name}")
            convertir_mediante_marker(archivo, ruta_salida)
        else:
            self.__intentar_conversion_docling_con_respaldo(archivo, ruta_salida)

    def __enrutar_documento_segun_formato(self, archivo: Path, ruta_salida: Path) -> None:
        extension = archivo.suffix.lower()
        if extension in PARSING_EXTENSIONES_SOPORTADAS:
            convertir_mediante_docling(archivo, ruta_salida)
        elif extension == '.pdf':
            self.__procesar_documento_formato_pdf(archivo, ruta_salida)
        else: 
            logging.info(f"Extensión ignorada: {archivo.name}")

    def __procesar_archivo_individual(self, archivo: Path) -> None:
        ruta_salida = self.dir_salida / f"{archivo.stem}.md"
        if ruta_salida.exists():
            logging.info(f"Archivo omitido por preexistencia: {archivo.name}")
            return

        try:
            self.__enrutar_documento_segun_formato(archivo, ruta_salida)
            logging.info(f"Procesamiento exitoso: {archivo.name}")
        except Exception as error_fatal:
            logging.error(f"Error fatal procesando {archivo.name}: {error_fatal}")

    def __ejecutar_procesamiento_por_lotes(self, archivos_pendientes: list) -> None:
        logging.info(f"Se encontraron {len(archivos_pendientes)} archivos para revisión inicial.")
        for archivo in archivos_pendientes:
            self.__procesar_archivo_individual(archivo)

    def aplicar_parsing_de_documentos(self) -> None:
        self.__inicializar_entorno_directorios()
        archivos_pendientes = self.__obtener_listado_archivos_validos()
        self.__ejecutar_procesamiento_por_lotes(archivos_pendientes)