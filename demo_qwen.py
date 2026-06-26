import logging
import sqlite3
import warnings
import torch

from transformers import AutoModelForCausalLM, AutoTokenizer
from sentence_transformers import SentenceTransformer
import chromadb
import huggingface_hub

from logger import configurar_sistema_registros

PREFIJO_LOG = "inferencia_qwen"
DIRECTORIO_BASE_DATOS = "./rag_chroma_db"
RUTA_ALMACEN_PADRES = "./rag_chroma_db/padres.sqlite3"
NOMBRE_COLECCION = "conocimiento_qwen"

# Modelos
MODELO_EMBEDDINGS_LOCAL = "intfloat/multilingual-e5-small"
MODELO_LLM_LOCAL = "Qwen/Qwen2.5-1.5B-Instruct"

# Configuracion de ejecucion
DISPOSITIVO_EMBEDDINGS = "cpu"
# Si tienes GPU compatible (NVIDIA/MPS), cambia "cpu" a "cuda" o "mps" para el LLM
DISPOSITIVO_LLM = "cuda" if torch.cuda.is_available() else "cpu" 
SILENCIAR_ADVERTENCIAS_LIBRERIAS = True
TOP_K_RESULTADOS = 2

def aplicar_silenciadores_librerias():
    if SILENCIAR_ADVERTENCIAS_LIBRERIAS:
        warnings.filterwarnings("ignore")
        logging.getLogger("transformers").setLevel(logging.ERROR)
        huggingface_hub.utils.logging.set_verbosity_error()
        huggingface_hub.utils.disable_progress_bars()

def inicializar_infraestructura_rag():
    logging.info("Inicializando conexiones a bases de datos...")
    
    cliente_chroma = chromadb.PersistentClient(path=DIRECTORIO_BASE_DATOS)
    coleccion = cliente_chroma.get_collection(name=NOMBRE_COLECCION)
    
    conexion_sqlite = sqlite3.connect(RUTA_ALMACEN_PADRES)
    cursor_sqlite = conexion_sqlite.cursor()
    
    logging.info("Bases de datos conectadas exitosamente.")
    return coleccion, conexion_sqlite, cursor_sqlite

def cargar_modelos_ia():
    logging.info(f"Cargando modelo de embeddings: {MODELO_EMBEDDINGS_LOCAL} en {DISPOSITIVO_EMBEDDINGS}")
    modelo_emb = SentenceTransformer(MODELO_EMBEDDINGS_LOCAL, device=DISPOSITIVO_EMBEDDINGS)
    
    logging.info(f"Cargando tokenizador y LLM: {MODELO_LLM_LOCAL} en {DISPOSITIVO_LLM}")
    tokenizador_llm = AutoTokenizer.from_pretrained(MODELO_LLM_LOCAL)
    modelo_llm = AutoModelForCausalLM.from_pretrained(
        MODELO_LLM_LOCAL,
        torch_dtype="auto", 
        device_map=DISPOSITIVO_LLM
    )
    
    logging.info("Modelos de IA cargados y listos para inferencia.")
    return modelo_emb, tokenizador_llm, modelo_llm

def recuperar_contexto_jerarquico(pregunta, coleccion, modelo_emb, cursor_sqlite):
    logging.info("Vectorizando pregunta del usuario...")
    vector_pregunta = modelo_emb.encode([pregunta]).tolist()
    
    logging.info("Buscando similitud en ChromaDB (Capa Fija - Hijos)...")
    resultados = coleccion.query(
        query_embeddings=vector_pregunta,
        n_results=TOP_K_RESULTADOS,
        include=["metadatas"]
    )
    
    metadatos_hijos = resultados["metadatas"][0]
    
    if not metadatos_hijos:
        logging.warning("No se encontro informacion relevante en la base de datos vectorial.")
        return ""

    contextos_recuperados = []
    ids_procesados = set()
    
    logging.info("Recuperando secciones completas en SQLite (Capa Semantica - Padres)...")
    for meta in metadatos_hijos:
        parent_id = meta.get("parent_id")
        
        # Evitamos agregar el mismo Padre dos veces si dos de sus Hijos hicieron match
        if parent_id and parent_id not in ids_procesados:
            ids_procesados.add(parent_id)
            cursor_sqlite.execute('SELECT contenido FROM documentos_padre WHERE id = ?', (parent_id,))
            resultado_padre = cursor_sqlite.fetchone()
            
            if resultado_padre:
                contextos_recuperados.append(resultado_padre[0])
                logging.info(f"Contexto anadido desde parent_id: {parent_id}")
                
    texto_contexto_final = "\n\n---\n\n".join(contextos_recuperados)
    return texto_contexto_final

def generar_respuesta_llm(pregunta, contexto, modelo, tokenizer):
    if not contexto.strip():
        return "Lo siento, no he encontrado informacion en mis documentos para responder a esta pregunta."

    mensaje_sistema = (
        "Eres un asistente corporativo y experto. Tu tarea es responder a la pregunta del usuario "
        "basandote UNICAMENTE en el contexto proporcionado. Si la respuesta no esta en el contexto, "
        "indica claramente que no tienes la informacion. No inventes datos."
    )
    
    prompt_usuario = f"Contexto recuperado de los documentos:\n{contexto}\n\nPregunta del usuario: {pregunta}"
    
    mensajes = [
        {"role": "system", "content": mensaje_sistema},
        {"role": "user", "content": prompt_usuario}
    ]
    
    logging.info("Aplicando plantilla de chat de Qwen2.5 y tokenizando...")
    texto_formateado = tokenizer.apply_chat_template(
        mensajes,
        tokenize=False,
        add_generation_prompt=True
    )
    
    inputs = tokenizer([texto_formateado], return_tensors="pt").to(modelo.device)
    
    logging.info("Generando respuesta con el LLM...")
    outputs = modelo.generate(
        **inputs,
        max_new_tokens=512,
        temperature=0.3, # Temperatura baja para que sea mas analitico y menos creativo
        top_p=0.8,
        do_sample=True
    )
    
    longitud_prompt = inputs.input_ids.shape[1]
    respuesta_generada = tokenizer.decode(outputs[0][longitud_prompt:], skip_special_tokens=True)
    
    logging.info("Respuesta generada exitosamente.")
    return respuesta_generada

def iniciar_bucle_conversacional():
    configurar_sistema_registros(PREFIJO_LOG)
    aplicar_silenciadores_librerias()
    
    print("\nIniciando sistema RAG... Por favor espera mientras se cargan los modelos (esto puede tomar unos segundos).\n")
    
    try:
        coleccion, conexion_sqlite, cursor_sqlite = inicializar_infraestructura_rag()
        modelo_emb, tokenizador_llm, modelo_llm = cargar_modelos_ia()
    except Exception as error_inicializacion:
        logging.critical(f"Fallo critico al iniciar el sistema: {error_inicializacion}")
        print("\nOcurrio un error fatal al iniciar el sistema. Revisa los logs.")
        return

    print("="*60)
    print(" Sistema RAG Local Iniciado: Qwen2.5-1.5B (Parent-Child)")
    print(" Escribe tu pregunta para consultar tus documentos.")
    print(" Para salir del sistema, escribe exactamente: Qwen2.5")
    print("="*60)
    
    while True:
        try:
            pregunta_usuario = input("\nUsuario: ")
            
            if pregunta_usuario.strip() == "Qwen2.5":
                print("\nCerrando el sistema RAG... ¡Hasta la proxima!")
                break
                
            if not pregunta_usuario.strip():
                continue
                
            contexto = recuperar_contexto_jerarquico(pregunta_usuario, coleccion, modelo_emb, cursor_sqlite)
            respuesta = generar_respuesta_llm(pregunta_usuario, contexto, modelo_llm, tokenizador_llm)
            
            print(f"\nQwen: {respuesta}\n")
            
        except KeyboardInterrupt:
            print("\nCerrando el sistema RAG por interrupcion de teclado... ¡Hasta la proxima!")
            break
        except Exception as error_ejecucion:
            logging.error(f"Error durante la inferencia: {error_ejecucion}")
            print("\nLo siento, ocurrio un error interno al procesar tu solicitud.")
            
    # Limpieza final
    conexion_sqlite.close()

if __name__ == "__main__":
    iniciar_bucle_conversacional()