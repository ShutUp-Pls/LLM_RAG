import re
import torch
import warnings
import logging
import huggingface_hub

from transformers import AutoModelForCausalLM, AutoTokenizer

from llms.utils.prompt_router import construir_prompt_enrutador
from llms.utils.prompt_sys import construir_prompt_sistema
from llms.utils.prompt_user import seleccionar_plantilla_usuario
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
            estricto_contexto=True
        )

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

    def __formatear_mensaje(self, consulta: str, contexto: str, requiere_rag: bool) -> list:
        tipo_prompt = "con_contexto" if requiere_rag else "sin_contexto"
        plantilla = seleccionar_plantilla_usuario(tipo_prompt)
        
        if requiere_rag:
            contenido_usuario = plantilla.format(contexto=contexto, consulta=consulta)
        else:
            contenido_usuario = plantilla.format(consulta=consulta)
            
        return [
            {"role": "system", "content": self.prompt_sistema},
            {"role": "user", "content": contenido_usuario}
        ]
    
    def __formatear_mensajes_enrutador(self, consulta: str, evaluar_rag: bool, evaluar_intencion: bool, catalogo_temas: str) -> list:
        prompt_sistema_router = construir_prompt_enrutador(
            evaluar_rag=evaluar_rag, 
            evaluar_intencion=evaluar_intencion,
            catalogo_temas=catalogo_temas
        )
        plantilla_router = seleccionar_plantilla_usuario("sin_contexto")
        contenido_usuario = plantilla_router.format(consulta=consulta)
        
        return [
            {"role": "system", "content": prompt_sistema_router},
            {"role": "user", "content": contenido_usuario}
        ]

    def __convertir_mensajes_a_texto(self, mensajes: list) -> str:
        return self.tokenizer.apply_chat_template(
            mensajes,
            tokenize=False,
            add_generation_prompt=True
        )

    def __preparar_tensores_entrada(self, texto_formateado: str) -> dict:
        return self.tokenizer(
            [texto_formateado], 
            return_tensors="pt",
            truncation=True,
            max_length=4096
        ).to(self.modelo.device)

    def __ejecutar_generacion(self, tensores_entrada: dict, max_new_tokens: int = None) -> torch.Tensor:
        limite_tokens = max_new_tokens if max_new_tokens is not None else self.max_tokens
        return self.modelo.generate(
            **tensores_entrada,
            max_new_tokens=limite_tokens,
            temperature=self.temperatura,
            top_p=self.top_p,
            do_sample=True
        )

    def __extraer_y_decodificar_respuesta(self, tensores_entrada: dict, tensores_salida: torch.Tensor) -> str:
        longitud_prompt = tensores_entrada["input_ids"].shape[1]
        return self.tokenizer.decode(tensores_salida[0][longitud_prompt:], skip_special_tokens=True)

    def __calcular_limite_tokens_enrutador(self, evaluar_rag: bool, evaluar_intencion: bool) -> int:
        return 150 + (25 * sum([evaluar_rag, evaluar_intencion]))

    def __procesar_respuesta_enrutador(
            self,
            respuesta_cruda: str,
            evaluar_rag: bool,
            evaluar_intencion: bool
        ) -> dict:
        resultados = {}

        match_razonamiento = re.search(r'RAZONAMIENTO:\s*(.*?)(?=\nDECISION:|$)', respuesta_cruda, re.IGNORECASE | re.DOTALL)
        if match_razonamiento:
            razonamiento_texto = match_razonamiento.group(1).strip()
            logging.debug("Razonamiento del Router: %s", razonamiento_texto)
        else:
            logging.debug("Razonamiento del Router: (No se detectó la etiqueta esperada) %s", respuesta_cruda)

        respuesta_mayus = respuesta_cruda.upper()
        
        if evaluar_rag:

            match_decision = re.search(r'DECISION:\s*RAG:\s*(SI|SÍ|NO)', respuesta_mayus)
            
            if match_decision:
                decision = match_decision.group(1)
                resultados["requiere_rag"] = False if "NO" in decision else True
            else:
                menciona_rag = "RAG" in respuesta_mayus
                menciona_no = "NO" in respuesta_mayus
                menciona_si = "SI" in respuesta_mayus or "SÍ" in respuesta_mayus
                if menciona_rag and menciona_no and not menciona_si: resultados["requiere_rag"] = False
                else: resultados["requiere_rag"] = True
            
        if evaluar_intencion:
            if "PREGUNTA" in respuesta_mayus: resultados["intencion"] = "PREGUNTA"
            elif "COMANDO" in respuesta_mayus: resultados["intencion"] = "COMANDO"
            elif "SALUDO" in respuesta_mayus: resultados["intencion"] = "SALUDO"
            else: resultados["intencion"] = "DESCONOCIDO"
            
        return resultados
    
    def inicializar_modelo(self) -> None:
        self.__aplicar_configuracion_silencio()
        self.__cargar_componentes_modelo()
    
    def enrutar_consulta(
            self,
            consulta: str,
            evaluar_rag: bool = True,
            evaluar_intencion: bool = False,
            catalogo_temas: str = "No hay temas disponibles"
        ) -> dict:
        mensajes = self.__formatear_mensajes_enrutador(consulta, evaluar_rag, evaluar_intencion, catalogo_temas)
        texto_formateado = self.__convertir_mensajes_a_texto(mensajes)
        
        logging.debug("Prompt enviado al enrutador:\n%s", texto_formateado)
        
        tensores_entrada = self.__preparar_tensores_entrada(texto_formateado)
        
        limite_tokens = self.__calcular_limite_tokens_enrutador(evaluar_rag, evaluar_intencion)
        tensores_salida = self.__ejecutar_generacion(tensores_entrada, max_new_tokens=limite_tokens)
        
        respuesta_cruda = self.__extraer_y_decodificar_respuesta(tensores_entrada, tensores_salida)

        logging.debug("Respuesta cruda del LLM (Router): '%s'", respuesta_cruda)
        
        return self.__procesar_respuesta_enrutador(respuesta_cruda, evaluar_rag, evaluar_intencion)
    
    def filtrar_catalogo_temas(self, consulta: str, catalogo_completo: str) -> str:
        prompt_sistema_filtro = (
            "Eres un extractor de información experto y analítico.\n"
            "Tu tarea es cruzar la consulta del usuario con un catálogo de temas disponibles.\n"
            "Extrae ÚNICAMENTE los temas del catálogo que tengan relación semántica con la consulta.\n"
            "Reglas críticas:\n"
            "1. Responde SOLO con una lista de los temas encontrados, separados por comas.\n"
            "2. No agregues introducciones, explicaciones, saltos de línea ni puntos finales.\n"
            "3. Si la consulta es un saludo, una orden de programación, o no tiene relación con ningún tema, responde exactamente: 'Sin coincidencia'.\n"
        )

        prompt_usuario = (
            f"CATÁLOGO DE TEMAS:\n{catalogo_completo}\n\n"
            f"CONSULTA DEL USUARIO:\n{consulta}\n"
        )
        
        mensajes = [
            {"role": "system", "content": prompt_sistema_filtro},
            {"role": "user", "content": prompt_usuario}
        ]
        
        texto_formateado = self.__convertir_mensajes_a_texto(mensajes)
        tensores_entrada = self.__preparar_tensores_entrada(texto_formateado)
        
        tensores_salida = self.__ejecutar_generacion(tensores_entrada, max_new_tokens=64)
        respuesta_cruda = self.__extraer_y_decodificar_respuesta(tensores_entrada, tensores_salida)
        
        catalogo_filtrado = respuesta_cruda.strip().strip('.')
        
        return catalogo_filtrado if catalogo_filtrado else "Sin coincidencia"
    
    def condensar_contexto(self, consulta: str, contexto_crudo: str) -> str:
        if not contexto_crudo.strip(): return ""

        prompt_sistema_compresor = (
            "Eres un extractor de información quirúrgico y ultra preciso.\n"
            "Tu única tarea es filtrar el contexto proporcionado y extraer EXCLUSIVAMENTE los fragmentos, "
            "cifras, nombres o citas que sirvan directamente para responder a la consulta del usuario.\n"
            "Reglas críticas:\n"
            "1. Elimina explicaciones redundantes, introducciones, paja e historias irrelevantes.\n"
            "2. Conserva la fidelidad exacta de los datos del texto. No inventes nada.\n"
            "3. Responde directamente con los datos filtrados reunidos. No agregues comentarios como 'Aquí está el resumen'.\n"
            "4. Si ninguna parte del contexto sirve para la consulta, devuelve el texto: 'Sin contexto relevante'.\n"
        )

        prompt_usuario = (
            f"CONTEXTO CRUDO:\n{contexto_crudo}\n\n"
            f"CONSULTA DEL USUARIO:\n{consulta}\n"
        )

        texto_formateado = prompt_sistema_compresor + prompt_usuario
        tensores_entrada = self.__preparar_tensores_entrada(texto_formateado)
        tensores_salida = self.__ejecutar_generacion(tensores_entrada, max_new_tokens=256)
        
        contexto_condensado = self.__extraer_y_decodificar_respuesta(tensores_entrada, tensores_salida)
        return contexto_condensado.strip()

    def generar_respuesta(self, consulta: str, contexto: str = "") -> str:
        requiere_rag = bool(contexto.strip())

        if requiere_rag and not contexto.strip():
            return "Lo siento, no he encontrado información en mis documentos para responder a esta consulta."
        
        mensajes = self.__formatear_mensaje(consulta, contexto, requiere_rag)
        texto_formateado = self.__convertir_mensajes_a_texto(mensajes)
        tensores_entrada = self.__preparar_tensores_entrada(texto_formateado)
        tensores_salida = self.__ejecutar_generacion(tensores_entrada)
        
        return self.__extraer_y_decodificar_respuesta(tensores_entrada, tensores_salida)