import logging
import subprocess

from pathlib import Path

from rag import MARKER_TIEMPO_BASE_SEGUNDOS, MARKER_MULTIPLICADOR_PESO

'''
- Marker es más caro computacionalmente puesto que es una arquitectura DeepLearning, por lo cual se tratará
de nuestro método secundario de conversión en caso de que Docling falle o no extraiga suficiente info.
- A diferencia del proceso de Docling este no es tan robusto ni resciliente puesto que, al tratarse de
la última posibilidad de traducción, las opciones son simples: O resulta, o falla, pero no hay plan C para este Plan B.

    1) Si resulta       -> Todos felices, tenemos un .md que capturó la información de los archivos.

    2) Si no resulta    -> Toca revisar el documento. Es muy dificil que a traves de visión por computadora
    el proceso falle, a menos que haya un problema con el archivo mismo o el sistema que ejecuta el proceso.

- La unica salvaguarda implementada corresponde a un timeout dinamico que da un tiempo limite dinamico basado en el
peso del archivo para ser generosos en tiempo con archivos grandes y tacaño en tiempo con los archivos pequeños.
- Así, si un archivo corrupto secuestra el proceso de análisis (Ya nos sucedió) el mismo programa
se detendrá y no se quedará infinitamente análisando un archivo que no puede ser análisado.
'''

def convertir_mediante_marker(
        ruta_archivo: Path,
        ruta_salida: Path
    ) -> None:

    peso_archivo_mb = ruta_archivo.stat().st_size / (1024 * 1024) if ruta_archivo.exists() else 0
    nombre_base_archivo = ruta_archivo.stem
    comando_subprocess = ["marker_single", str(ruta_archivo), str(ruta_salida)]

    def calcular_tiempo_limite_ejecucion() -> int:
        tiempo_adicional_por_peso = int(peso_archivo_mb * MARKER_MULTIPLICADOR_PESO)
        return MARKER_TIEMPO_BASE_SEGUNDOS + tiempo_adicional_por_peso

    def ejecutar_proceso_conversion(tiempo_limite: int) -> None:
        try:
            resultado_proceso = subprocess.run(
                comando_subprocess, 
                capture_output=True, 
                text=True, 
                timeout=tiempo_limite
            )
            if resultado_proceso.returncode != 0:
                raise RuntimeError(f"Fallo critico en Marker: {resultado_proceso.stderr}")
        except subprocess.TimeoutExpired:
            raise RuntimeError(f"Marker se congeló (excedió el límite de {tiempo_limite}s)")

    def extraer_y_renombrar_resultado_final() -> None:
        carpeta_temporal_generada = ruta_salida / nombre_base_archivo
        archivo_markdown_generado = carpeta_temporal_generada / f"{nombre_base_archivo}.md"
        ruta_destino_final = ruta_salida / f"{nombre_base_archivo}.md"
        
        if archivo_markdown_generado.exists():
            archivo_markdown_generado.rename(ruta_destino_final)
        else:
            raise FileNotFoundError("Marker no generó el archivo md esperado en la ruta temporal")

    logging.info(f"Iniciando Marker para: {ruta_archivo.name}")
    tiempo_maximo_permitido = calcular_tiempo_limite_ejecucion()
    ejecutar_proceso_conversion(tiempo_maximo_permitido)
    extraer_y_renombrar_resultado_final()