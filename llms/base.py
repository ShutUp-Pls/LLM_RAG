from abc import ABC, abstractmethod

class BaseLLM(ABC):
    
    @abstractmethod
    def inicializar_modelo(self) -> None:
        pass

    @abstractmethod
    def generar_respuesta(self, pregunta: str, contexto: str) -> str:
        pass