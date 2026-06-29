from pathlib import Path
from typing import Union, Iterable

def crear_directorios(rutas: Union[str, Path, Iterable[Union[str, Path]]]) -> None:
    """Crea uno o varios directorios dados."""
    if isinstance(rutas, (str, Path)): rutas = [rutas]
    for ruta in rutas: Path(ruta).mkdir(parents=True, exist_ok=True)