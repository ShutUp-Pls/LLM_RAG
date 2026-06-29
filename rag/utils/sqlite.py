import sqlite3
import logging
from pathlib import Path

class GestorSQLite:
    def __init__(self, ruta_db: str | Path):
        self.ruta_db = Path(ruta_db)
        self.ruta_db.parent.mkdir(parents=True, exist_ok=True)

    def ejecutar_consulta(self, query: str, parametros: tuple = ()) -> None:
        """Ejecuta una consulta SQL simple (DDL o escritura única)."""
        try:
            with sqlite3.connect(self.ruta_db) as conexion:
                conexion.execute(query, parametros)
        except Exception as error_db:
            logging.error(f"Error ejecutando consulta en {self.ruta_db.name}: {error_db}")
            raise

    def ejecutar_insercion_masiva(self, query: str, lista_parametros: list[tuple]) -> None:
        """Ejecuta una inserción masiva optimizada (batch)."""
        try:
            with sqlite3.connect(self.ruta_db) as conexion:
                conexion.executemany(query, lista_parametros)
        except Exception as error_db:
            logging.error(f"Error en inserción masiva en {self.ruta_db.name}: {error_db}")
            raise

    def obtener_todos(self, query: str, parametros: tuple = ()) -> list[tuple]:
        """Obtiene todos los registros que coincidan con la consulta."""
        try:
            with sqlite3.connect(self.ruta_db) as conexion:
                cursor = conexion.cursor()
                cursor.execute(query, parametros)
                return cursor.fetchall()
        except Exception as error_db:
            logging.error(f"Error obteniendo registros en {self.ruta_db.name}: {error_db}")
            return []

    def obtener_uno(self, query: str, parametros: tuple = ()) -> tuple | None:
        """Obtiene un único registro que coincida con la consulta."""
        try:
            with sqlite3.connect(self.ruta_db) as conexion:
                cursor = conexion.cursor()
                cursor.execute(query, parametros)
                return cursor.fetchone()
        except Exception as error_db:
            logging.error(f"Error obteniendo registro único en {self.ruta_db.name}: {error_db}")
            return None