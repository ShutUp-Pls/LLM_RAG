import logging
import fitz
import subprocess
from pathlib import Path
from docling.document_converter import DocumentConverter

from logger import configurar_sistema_registros

PREFIJO_LOG = "ingesta"
DIRECTORIO_ENTRADA = "./org"
DIRECTORIO_SALIDA = "./md"
EXTENSIONES_SOPORTADAS_NATIVAMENTE = ['.docx', '.pptx', '.html', '.md', '.txt']

def crear_directorios_trabajo():
    Path(DIRECTORIO_ENTRADA).mkdir(parents=True, exist_ok=True)
    Path(DIRECTORIO_SALIDA).mkdir(parents=True, exist_ok=True)

def listar_archivos_pendientes():
    ruta_entrada = Path(DIRECTORIO_ENTRADA)
    return [archivo for archivo in ruta_entrada.iterdir() if archivo.is_file() and not archivo.name.startswith('.')]

def evaluar_densidad_texto_pdf(ruta_archivo):
    try:
        documento = fitz.open(ruta_archivo)
        if len(documento) == 0: return False
        longitud_total = sum(len(pagina.get_text().strip()) for pagina in documento)
        promedio_caracteres = longitud_total / len(documento)
        documento.close()
        return promedio_caracteres < 50
    except Exception as error:
        logging.error(f"Fallo al leer PDF con PyMuPDF {ruta_archivo.name}: {error}")
        return True

def convertir_mediante_docling(ruta_archivo, ruta_salida):
    logging.info(f"Iniciando Docling para: {ruta_archivo.name}")
    convertidor = DocumentConverter()
    resultado = convertidor.convert(str(ruta_archivo))
    texto_markdown = resultado.document.export_to_markdown()
    
    if len(texto_markdown.strip()) < 100:
        raise ValueError("Extraccion insuficiente con Docling")
        
    with open(ruta_salida, "w", encoding="utf-8") as archivo_destino:
        archivo_destino.write(texto_markdown)

def convertir_mediante_marker(ruta_archivo):
    logging.info(f"Iniciando Marker para: {ruta_archivo.name}")
    comando = ["marker_single", str(ruta_archivo), str(DIRECTORIO_SALIDA)]
    resultado = subprocess.run(comando, capture_output=True, text=True)
    
    if resultado.returncode != 0:
        raise RuntimeError(f"Fallo critico en Marker: {resultado.stderr}")
        
    nombre_base = ruta_archivo.stem
    carpeta_temporal_marker = Path(DIRECTORIO_SALIDA) / nombre_base
    archivo_generado_marker = carpeta_temporal_marker / f"{nombre_base}.md"
    
    if archivo_generado_marker.exists():
        ruta_final_md = Path(DIRECTORIO_SALIDA) / f"{nombre_base}.md"
        archivo_generado_marker.rename(ruta_final_md)
    else:
        raise FileNotFoundError("Marker no genero el archivo md esperado")

def enrutar_documento_segun_caracteristicas(archivo, ruta_salida):
    extension = archivo.suffix.lower()
    try:
        if extension in EXTENSIONES_SOPORTADAS_NATIVAMENTE:
            convertir_mediante_docling(archivo, ruta_salida)
        elif extension == '.pdf':
            es_imagen_escaneada = evaluar_densidad_texto_pdf(archivo)
            if es_imagen_escaneada:
                logging.info(f"PDF sin capa de texto detectado. Derivando: {archivo.name}")
                convertir_mediante_marker(archivo)
            else:
                try:
                    convertir_mediante_docling(archivo, ruta_salida)
                except ValueError as error_validacion:
                    logging.warning(f"Docling descarto el archivo {archivo.name}: {error_validacion}")
                    convertir_mediante_marker(archivo)
        else:
            logging.info(f"Extension ignorada: {archivo.name}")
            
        logging.info(f"Procesamiento exitoso: {archivo.name}")
    except Exception as error_fatal:
        logging.error(f"Error fatal procesando {archivo.name}: {error_fatal}")

def procesar_lote_documentos(archivos_pendientes):
    logging.info(f"Se encontraron {len(archivos_pendientes)} archivos para revision inicial.")
    for archivo in archivos_pendientes:
        ruta_salida = Path(DIRECTORIO_SALIDA) / f"{archivo.stem}.md"
        if ruta_salida.exists():
            logging.info(f"Archivo omitido por preexistencia: {archivo.name}")
            continue
        enrutar_documento_segun_caracteristicas(archivo, ruta_salida)

def aplicar_ingesta_de_documentos():
    configurar_sistema_registros(PREFIJO_LOG)
    crear_directorios_trabajo()
    archivos = listar_archivos_pendientes()
    procesar_lote_documentos(archivos)

if __name__ == "__main__":
    aplicar_ingesta_de_documentos()