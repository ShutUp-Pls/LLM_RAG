import os
import logging

from pathlib import Path
from datetime import datetime

DIRECTORIO_LOGS = "./logs"
LIMITE_ARCHIVOS_REGISTRO = 100

def gestionar_retencion_registros(prefijo_log):
    patron_busqueda = f"{prefijo_log}_*.txt"
    ruta_logs = Path(DIRECTORIO_LOGS)
    
    if not ruta_logs.exists(): return
        
    archivos_registro = sorted(ruta_logs.glob(patron_busqueda), key=os.path.getmtime, reverse=True)
    if len(archivos_registro) > LIMITE_ARCHIVOS_REGISTRO:
        for archivo_obsoleto in archivos_registro[LIMITE_ARCHIVOS_REGISTRO:]:
            archivo_obsoleto.unlink()

def configurar_sistema_registros(prefijo_log, nivel=logging.DEBUG):
    Path(DIRECTORIO_LOGS).mkdir(parents=True, exist_ok=True)
    gestionar_retencion_registros(prefijo_log)
    marca_tiempo = datetime.now().strftime("%Y%m%d_%H%M%S")
    ruta_registro = Path(DIRECTORIO_LOGS) / f"{prefijo_log}_{marca_tiempo}.txt"
    logger = logging.getLogger()
    logger.setLevel(nivel)
    manejador = logging.FileHandler(ruta_registro, encoding='utf-8')
    formato = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
    manejador.setFormatter(formato)
    
    if logger.hasHandlers():
        logger.handlers.clear()
        
    logger.addHandler(manejador)