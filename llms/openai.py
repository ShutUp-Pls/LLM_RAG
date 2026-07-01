import os
import re
import logging
from openai import OpenAI
from dotenv import load_dotenv

from llms.utils.prompt_router import construir_prompt_enrutador
from llms.utils.prompt_sys import construir_prompt_sistema
from llms.utils.prompt_user import seleccionar_plantilla_usuario
from llms.base import BaseLLM

from llms import (
    LLM_MAX_TOKENS,
    LLM_TEMPERATURA,
    LLM_TOP_P
)

class OpenAILLM(BaseLLM):
    def __init__(
        self, 
        modelo: str = "gpt-4o-mini"
    ):
        load_dotenv()
        self.modelo = modelo
        self.cliente = None
        
        self.max_tokens = LLM_MAX_TOKENS
        self.temperatura = LLM_TEMPERATURA
        self.top_p = LLM_TOP_P
        
        self.prompt_sistema = construir_prompt_sistema(
            es_corporativo=True,
            estricto_contexto=True
        )

    def inicializar_modelo(self) -> None:
        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            raise ValueError("No se encontró la variable de entorno OPENAI_API_KEY.")
        self.cliente = OpenAI(api_key=api_key)
        logging.info("Modelo OpenAI inicializado correctamente.")

    def __ejecutar_llamada(self, mensajes: list, max_tokens: int = None) -> str:
        limite = max_tokens if max_tokens else self.max_tokens
        respuesta = self.cliente.chat.completions.create(
            model=self.modelo,
            messages=mensajes,
            max_tokens=limite,
            temperature=self.temperatura,
            top_p=self.top_p
        )
        return respuesta.choices[0].message.content

    def __procesar_respuesta_enrutador(self, respuesta_cruda: str, evaluar_rag: bool, evaluar_intencion: bool) -> dict:
        resultados = {}
        match_razonamiento = re.search(r'RAZONAMIENTO:\s*(.*?)(?=\nRAG:|$)', respuesta_cruda, re.IGNORECASE | re.DOTALL)
        if match_razonamiento:
            logging.debug("Razonamiento del Router: %s", match_razonamiento.group(1).strip())

        respuesta_mayus = respuesta_cruda.upper()
        
        if evaluar_rag:
            match_decision = re.search(r'RAG:\s*(SI|SÍ|NO)', respuesta_mayus)
            if match_decision:
                decision = match_decision.group(1)
                resultados["requiere_rag"] = False if "NO" in decision else True
            else:
                ultima_linea = respuesta_mayus.split('\n')[-1]
                menciona_rag = "RAG" in ultima_linea
                menciona_no = "NO" in ultima_linea
                menciona_si = "SI" in ultima_linea or "SÍ" in ultima_linea
                resultados["requiere_rag"] = not (menciona_rag and menciona_no and not menciona_si)
            
        if evaluar_intencion:
            if "PREGUNTA" in respuesta_mayus: resultados["intencion"] = "PREGUNTA"
            elif "COMANDO" in respuesta_mayus: resultados["intencion"] = "COMANDO"
            elif "SALUDO" in respuesta_mayus: resultados["intencion"] = "SALUDO"
            else: resultados["intencion"] = "DESCONOCIDO"
            
        return resultados

    def enrutar_consulta(self, consulta: str, evaluar_rag: bool = True, evaluar_intencion: bool = False, catalogo_temas: str = "No hay temas disponibles") -> dict:
        prompt_sistema_router = construir_prompt_enrutador(evaluar_rag, evaluar_intencion, catalogo_temas, modelo="openai")
        plantilla_router = seleccionar_plantilla_usuario("sin_contexto")
        
        mensajes = [
            {"role": "system", "content": prompt_sistema_router},
            {"role": "user", "content": plantilla_router.format(consulta=consulta)}
        ]
        
        limite_tokens = 150 + (25 * sum([evaluar_rag, evaluar_intencion]))
        respuesta_cruda = self.__ejecutar_llamada(mensajes, max_tokens=limite_tokens)
        
        return self.__procesar_respuesta_enrutador(respuesta_cruda, evaluar_rag, evaluar_intencion)

    def filtrar_catalogo_temas(self, consulta: str, catalogo_completo: str) -> str:
        return catalogo_completo

    def condensar_contexto(self, consulta: str, contexto_crudo: str) -> str:
        return contexto_crudo

    def generar_respuesta(self, consulta: str, contexto: str = "") -> str:
        requiere_rag = bool(contexto.strip())

        if requiere_rag:
            prompt_sistema_dinamico = construir_prompt_sistema(
                es_corporativo=True,
                estricto_contexto_laxo=True
            )
        else:
            prompt_sistema_dinamico = construir_prompt_sistema(
                es_corporativo=True,
                conocimiento_tecnico_restringido=True
            )

        if requiere_rag and not contexto.strip():
            return "Lo siento, no he encontrado información en mis documentos para responder a esta consulta."
        
        tipo_prompt = "con_contexto" if requiere_rag else "sin_contexto"
        plantilla = seleccionar_plantilla_usuario(tipo_prompt)
        contenido_usuario = plantilla.format(contexto=contexto, consulta=consulta) if requiere_rag else plantilla.format(consulta=consulta)
        
        mensajes = [
            {"role": "system", "content": prompt_sistema_dinamico},
            {"role": "user", "content": contenido_usuario}
        ]
        return self.__ejecutar_llamada(mensajes)