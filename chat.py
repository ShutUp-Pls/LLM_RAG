import torch
import logging
import warnings
import huggingface_hub

from transformers import AutoModelForCausalLM, AutoTokenizer
from logger import configurar_sistema_registros
from rag.rag_inferencia import (
    conectar_bases_de_datos, 
    cargar_modelo_embeddings, 
    recuperar_contexto_desde_pregunta
)

PREFIJO_LOG = "chat_interactivo"

DIRECTORIO_BASE_DATOS = "./rag/chroma_db"
RUTA_ALMACEN_PADRES = "./rag/chroma_db/padres.sqlite3"
NOMBRE_COLECCION = "conocimiento_qwen"

MODELO_EMBEDDINGS_LOCAL = "intfloat/multilingual-e5-small"
MODELO_LLM_LOCAL = "Qwen/Qwen2.5-1.5B-Instruct"

DISPOSITIVO_EMBEDDINGS = "cpu"
DISPOSITIVO_LLM = "cuda" if torch.cuda.is_available() else "cpu"
SILENCIAR_ADVERTENCIAS_LIBRERIAS = True
TOP_K_RESULTADOS = 2

MAX_NUEVOS_TOKENS = 512
TEMPERATURA_GENERACION = 0.3
TOP_P_GENERACION = 0.8

BANNER_INCIO = '''
=============================================================
Sistema de Chat Local RAG Iniciado (Qwen2.5 + Parent-Child)
Escribe tu pregunta para consultar tus documentos.
Para salir del sistema, escribe exactamente: salir
=============================================================
'''

RESPUESTA_NO_SABE = "Lo siento, no he encontrado información en mis documentos para responder a esta pregunta."

PROMPT_ACTITUD = (
    "Eres un asistente corporativo y experto. Tu tarea es responder a la pregunta del usuario "
    "basándote ÚNICAMENTE en el contexto proporcionado. Si la respuesta no está en el contexto, "
    "indica claramente que no tienes la información. No inventes datos."
)

PROMPT_USUARIO = (
    "Contexto recuperado de los documentos:\n"
    "{contexto}\n\n"
    "Pregunta del usuario: {pregunta}"
)

def aplicar_silenciadores_librerias():
    if SILENCIAR_ADVERTENCIAS_LIBRERIAS:
        warnings.filterwarnings("ignore")
        logging.getLogger("transformers").setLevel(logging.ERROR)
        huggingface_hub.utils.logging.set_verbosity_error()
        huggingface_hub.utils.disable_progress_bars()

def cargar_modelo_generativo(ruta_llm, disp_llm):
    logging.info(f"Cargando tokenizador y LLM: {ruta_llm} en {disp_llm}")
    tokenizador_llm = AutoTokenizer.from_pretrained(ruta_llm)
    modelo_llm = AutoModelForCausalLM.from_pretrained(
        ruta_llm,
        torch_dtype="auto", 
        device_map=disp_llm
    )
    logging.info("Modelo generativo cargado en memoria y listo.")
    return tokenizador_llm, modelo_llm

def generar_respuesta_con_llm(pregunta, contexto, modelo, tokenizer):
    logging.info("Iniciando generacion de respuesta LLM...")
    if not contexto.strip(): return RESPUESTA_NO_SABE
    
    mensajes = [
        {"role": "system", "content": PROMPT_ACTITUD},
        {"role": "user", "content": PROMPT_USUARIO.format(contexto=contexto, pregunta=pregunta)}
    ]
    
    texto_formateado = tokenizer.apply_chat_template(
        mensajes,
        tokenize=False,
        add_generation_prompt=True
    )
    
    inputs = tokenizer([texto_formateado], return_tensors="pt").to(modelo.device)
    
    outputs = modelo.generate(
        **inputs,
        max_new_tokens=MAX_NUEVOS_TOKENS,
        temperature=TEMPERATURA_GENERACION,
        top_p=TOP_P_GENERACION,
        do_sample=True
    )
    
    longitud_prompt = inputs.input_ids.shape[1]
    respuesta_generada = tokenizer.decode(outputs[0][longitud_prompt:], skip_special_tokens=True)
    
    logging.info("Inferencia completada exitosamente.")
    return respuesta_generada

def iniciar_bucle_conversacional():
    configurar_sistema_registros(PREFIJO_LOG)
    aplicar_silenciadores_librerias()
    
    print("\nIniciando sistema de chat... Por favor espera mientras se cargan los modelos.\n")
    
    try:
        coleccion, conexion_sqlite, cursor_sqlite = conectar_bases_de_datos(
            DIRECTORIO_BASE_DATOS, 
            RUTA_ALMACEN_PADRES, 
            NOMBRE_COLECCION
        )
        
        modelo_emb = cargar_modelo_embeddings(
            MODELO_EMBEDDINGS_LOCAL, 
            DISPOSITIVO_EMBEDDINGS
        )
        
        tokenizador_llm, modelo_llm = cargar_modelo_generativo(
            MODELO_LLM_LOCAL,
            DISPOSITIVO_LLM
        )
        
        print(BANNER_INCIO)
        
    except Exception as error_inicializacion:
        logging.critical(f"Fallo crítico al iniciar el sistema: {error_inicializacion}")
        print("\nOcurrió un error fatal al iniciar el sistema. Revisa los logs.")
        return
    
    while True:
        try:
            pregunta_usuario = input("\nUsuario: ")
            
            if pregunta_usuario.strip().lower() == "salir":
                print("\nCerrando el sistema de chat... ¡Hasta la próxima!")
                break
                
            if not pregunta_usuario.strip():
                continue
                
            contexto_recuperado = recuperar_contexto_desde_pregunta(
                pregunta_usuario, 
                coleccion, 
                modelo_emb, 
                cursor_sqlite, 
                TOP_K_RESULTADOS
            )
            
            respuesta = generar_respuesta_con_llm(
                pregunta_usuario, 
                contexto_recuperado, 
                modelo_llm, 
                tokenizador_llm
            )
            
            print(f"\nQwen: {respuesta}\n")
            
        except KeyboardInterrupt:
            print("\nCerrando el sistema por interrupción de teclado... ¡Hasta la próxima!")
            break

        except Exception as error_ejecucion:
            logging.error(f"Error durante el ciclo de chat: {error_ejecucion}")
            print("\nLo siento, ocurrió un error interno al procesar tu solicitud.")
            
    conexion_sqlite.close()

if __name__ == "__main__":
    iniciar_bucle_conversacional()