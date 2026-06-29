import torch
import warnings
import logging
import huggingface_hub

from transformers import AutoModelForCausalLM, AutoTokenizer

from llms.utils.sys_prompt import construir_prompt_sistema
from llms.utils.user_prompt import seleccionar_plantilla_usuario
from llms.base import BaseLLM

from llms import (
    LLM_MAX_TOKENS,
    LLM_TEMPERATURA,
    LLM_TOP_P
)

class QwenLocal(BaseLLM):
    def __init__(
        self, 
        ruta_modelo: str = "Qwen/Qwen2.5-1.5B-Instruct", 
        dispositivo: str = "cuda" if torch.cuda.is_available() else "cpu",
        silenciar: bool = True
    ):
        self.ruta_modelo = ruta_modelo
        self.dispositivo = dispositivo
        self.silenciar = silenciar
        
        self.tokenizer = None
        self.modelo = None
        
        self.max_tokens = LLM_MAX_TOKENS
        self.temperatura = LLM_TEMPERATURA
        self.top_p = LLM_TOP_P
        
        self.prompt_sistema = construir_prompt_sistema(
            es_corporativo=True,
            estricto_contexto=False,
            acusa_contexto_flexible=True,
            evitar_alucinaciones=True
        )
        self.prompt_usuario = seleccionar_plantilla_usuario("qwen")

    def __aplicar_configuracion_silencio(self) -> None:
        if self.silenciar:
            warnings.filterwarnings("ignore")
            logging.getLogger("transformers").setLevel(logging.ERROR)
            huggingface_hub.utils.logging.set_verbosity_error()
            huggingface_hub.utils.disable_progress_bars()

    def __cargar_componentes_modelo(self) -> None:
        self.tokenizer = AutoTokenizer.from_pretrained(self.ruta_modelo)
        self.modelo = AutoModelForCausalLM.from_pretrained(
            self.ruta_modelo,
            torch_dtype="auto",
            device_map=self.dispositivo
        )

    def __formatear_estructura_mensajes(self, pregunta: str, contexto: str) -> list:
        contenido_usuario = self.prompt_usuario.format(contexto=contexto, pregunta=pregunta)
        return [
            {"role": "system", "content": self.prompt_sistema},
            {"role": "user", "content": contenido_usuario}
        ]

    def __convertir_mensajes_a_texto(self, mensajes: list) -> str:
        return self.tokenizer.apply_chat_template(
            mensajes,
            tokenize=False,
            add_generation_prompt=True
        )

    def __preparar_tensores_entrada(self, texto_formateado: str) -> dict:
        return self.tokenizer([texto_formateado], return_tensors="pt").to(self.modelo.device)

    def __ejecutar_generacion(self, tensores_entrada: dict) -> torch.Tensor:
        return self.modelo.generate(
            **tensores_entrada,
            max_new_tokens=self.max_tokens,
            temperature=self.temperatura,
            top_p=self.top_p,
            do_sample=True
        )

    def __extraer_y_decodificar_respuesta(self, tensores_entrada: dict, tensores_salida: torch.Tensor) -> str:
        longitud_prompt = tensores_entrada["input_ids"].shape[1]
        return self.tokenizer.decode(tensores_salida[0][longitud_prompt:], skip_special_tokens=True)

    def inicializar_modelo(self) -> None:
        self.__aplicar_configuracion_silencio()
        self.__cargar_componentes_modelo()

    def generar_respuesta(self, pregunta: str, contexto: str) -> str:
        if not contexto.strip():
            return "Lo siento, no he encontrado información en mis documentos para responder a esta pregunta."
        
        mensajes = self.__formatear_estructura_mensajes(pregunta, contexto)
        texto_formateado = self.__convertir_mensajes_a_texto(mensajes)
        tensores_entrada = self.__preparar_tensores_entrada(texto_formateado)
        tensores_salida = self.__ejecutar_generacion(tensores_entrada)
        
        return self.__extraer_y_decodificar_respuesta(tensores_entrada, tensores_salida)