import logging
import uuid
from pathlib import Path
from langchain_text_splitters import MarkdownHeaderTextSplitter
from langchain_text_splitters import RecursiveCharacterTextSplitter

from ..logger import configurar_sistema_registros

PREFIJO_LOG = "chunking"
DIRECTORIO_ENTRADA_MD = "./md"

JERARQUIA_ENCABEZADOS = [
    ("#", "Header_1"),
    ("##", "Header_2"),
    ("###", "Header_3"),
    ("####", "Header_4"),
]
CONSERVAR_SIMBOLOS_MARKDOWN = False
LIMITE_CARACTERES_POR_CHUNK = 1000
CARACTERES_DE_SUPERPOSICION = 150

def instanciar_divisores_texto():
    divisor_semantico = MarkdownHeaderTextSplitter(
        headers_to_split_on=JERARQUIA_ENCABEZADOS,
        strip_headers=CONSERVAR_SIMBOLOS_MARKDOWN
    )
    divisor_recursivo = RecursiveCharacterTextSplitter(
        chunk_size=LIMITE_CARACTERES_POR_CHUNK,
        chunk_overlap=CARACTERES_DE_SUPERPOSICION,
        length_function=len,
    )
    return divisor_semantico, divisor_recursivo

def procesar_markdown_cascada(ruta_archivo_md, divisor_semantico, divisor_recursivo):
    try:
        with open(ruta_archivo_md, "r", encoding="utf-8") as archivo_origen:
            texto_crudo = archivo_origen.read()
            
        bloques_semanticos = divisor_semantico.split_text(texto_crudo)
        
        padres_generados = []
        hijos_generados = []
        
        # Proceso Parent-Child -> Iteramos sobre cada bloque semántico (Padre)
        for bloque_padre in bloques_semanticos:
            # Generamos un ID único para el Padre
            id_padre = str(uuid.uuid4())
            
            # Inyectamos el ID y la fuente en el Padre
            metadata_padre = bloque_padre.metadata.copy()
            metadata_padre["parent_id"] = id_padre
            metadata_padre["fuente"] = ruta_archivo_md.name
            bloque_padre.metadata = metadata_padre
            padres_generados.append(bloque_padre)
            
            # Cortamos ESTE Padre específico en Hijos
            bloques_hijos = divisor_recursivo.split_documents([bloque_padre])
            
            # Aseguramos que los Hijos hereden el ID de su Padre
            for hijo in bloques_hijos:
                # LangChain suele heredar metadatos, pero lo forzamos por seguridad
                hijo.metadata["parent_id"] = id_padre
                hijo.metadata["fuente"] = ruta_archivo_md.name
                hijos_generados.append(hijo)
        
        logging.info(f"Fragmentacion exitosa: {ruta_archivo_md.name} ({len(padres_generados)} padres, {len(hijos_generados)} hijos)")
        return padres_generados, hijos_generados
        
    except Exception as error_fragmentacion:
        logging.error(f"Error fragmentando {ruta_archivo_md.name}: {error_fragmentacion}")
        return [], []

def procesar_directorio_completo_markdown():
    configurar_sistema_registros(PREFIJO_LOG)
    ruta_directorio = Path(DIRECTORIO_ENTRADA_MD)
    
    if not ruta_directorio.exists():
        logging.error(f"Directorio de entrada no encontrado: {DIRECTORIO_ENTRADA_MD}")
        return [], []

    divisor_semantico, divisor_recursivo = instanciar_divisores_texto()
    archivos_markdown = list(ruta_directorio.glob("*.md"))
    
    todos_los_padres = []
    todos_los_hijos = []

    logging.info(f"Iniciando fragmentacion de {len(archivos_markdown)} archivos markdown.")
    
    for archivo in archivos_markdown:
        padres, hijos = procesar_markdown_cascada(archivo, divisor_semantico, divisor_recursivo)
        todos_los_padres.extend(padres)
        todos_los_hijos.extend(hijos)

    logging.info(f"Fragmentacion global completada. Padres: {len(todos_los_padres)} | Hijos: {len(todos_los_hijos)}")
    return todos_los_padres, todos_los_hijos

if __name__ == "__main__":
    padres_totales, hijos_totales = procesar_directorio_completo_markdown()