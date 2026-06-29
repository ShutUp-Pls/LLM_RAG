import logging

from pathlib import Path
from docling.document_converter import DocumentConverter

from rag import DOCLING_TOLERANCIA_DEFECTO

'''
- Al ser docling más barato computacionalmente, será el primer método bajo el cual
se intentará transformar archivos de información a archivos markdown.
- Aún así, este método es debil frente a documentos con imagenes, de las cuales
no puede extraer ningún tipo de información.
- Para solucionar lo anterior, se implementó un sistema de tolerancia dinamica según el peso del archivo:

    1) Archivo liviano  -> la tolerancia necesaria para considerarse una buena extracción se reduce.
        (Extraer pocos caracteres no significa que la extracción haya sido mala en archivos pequeños)

    2) Archivo pesado   -> la tolerancia necesaria exigidos se topa en un límite superior fijo.
        (Extraer pocos caracteres en un archivo pesado la mayoría de las veces indica que se trata de imagenes, no texto)

    3) Siempre se exige al menos 1 carácter válido. Nunca se darán por aprobados documentos que devuelvan un texto 100% vacío.

    4) En caso de que el texto extraído no alcance la tolerancia dinámica descrita anteriormente, se levanta una excepción.
    
- Así, por 1), 2), 3) y 4) decidimos si la extracción de información fue satifactoria o de lo contrario levantamos una excepción.
- La intención es que un fallo en docling derive ese documento a un método secundario más robusto (Marker para nuestro caso).
- Y si no falla, entonces consideramos el proceso lo suficientemente exitoso para mantener el archivo generado por docling.
- Esta función no se asegura de la existencia ni formato de los paraemtros que pide. Si algún tipo de transformación es necesaria
para cumplir con los parametros de esta función, han de ser manejados previo a su uso en una instancia superior de ejecución.
'''

def convertir_mediante_docling(
        ruta_archivo: Path,
        ruta_salida: Path,
        limite_tolerancia: int = DOCLING_TOLERANCIA_DEFECTO
    ) -> None:
    
    peso_archivo_kb = ruta_archivo.stat().st_size / 1024 if ruta_archivo.exists() else 0
    convertidor_documentos = DocumentConverter()

    def validar_existencia_archivo_origen() -> None:
        if not ruta_archivo.is_file():
            raise FileNotFoundError(f"El archivo origen no existe: {ruta_archivo}")

    def calcular_umbral_tolerancia_dinamica() -> int:
        tolerancia_calculada = min(limite_tolerancia, int(peso_archivo_kb))
        return max(1, tolerancia_calculada)

    def ejecutar_extraccion_markdown() -> str:
        try:
            resultado_conversion = convertidor_documentos.convert(str(ruta_archivo))
            return resultado_conversion.document.export_to_markdown().strip()
        except Exception as error_ejecucion:
            logging.error(f"Fallo crítico en Docling al procesar {ruta_archivo.name}: {error_ejecucion}")
            raise

    def auditar_calidad_extraccion(texto_generado: str, umbral_minimo: int) -> None:
        largo_texto = len(texto_generado)
        if largo_texto < umbral_minimo:
            raise ValueError(
                f"Extracción insuficiente en {ruta_archivo.name}: "
                f"Se extrajeron {largo_texto} caracteres, pero se requerían al menos {umbral_minimo}."
            )
        logging.info(f"Conversión exitosa: {ruta_salida.name} ({largo_texto} caracteres extraídos)")

    def guardar_resultados_en_disco(texto_generado: str) -> None:
        with open(ruta_salida, "w", encoding="utf-8") as archivo_destino:
            archivo_destino.write(texto_generado)

    logging.info(f"Iniciando Docling para: {ruta_archivo.name}")
    validar_existencia_archivo_origen()
    tolerancia_exigida = calcular_umbral_tolerancia_dinamica()
    texto_extraido = ejecutar_extraccion_markdown()
    auditar_calidad_extraccion(texto_extraido, tolerancia_exigida)
    guardar_resultados_en_disco(texto_extraido)