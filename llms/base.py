from abc import ABC, abstractmethod

class BaseLLM(ABC):
    
    @abstractmethod
    def inicializar_modelo(self) -> None:
        pass

    @abstractmethod
    def enrutar_consulta(self, consulta: str, **kwargs) -> dict:
        pass

    @abstractmethod
    def generar_respuesta(self, consulta: str, contexto: str = "", **kwargs) -> str:
        pass