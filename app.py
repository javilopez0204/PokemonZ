import streamlit as st
import os
from pypdf import PdfReader
from typing import List
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_groq import ChatGroq
from langchain_community.vectorstores import FAISS
from langchain.chains import ConversationalRetrievalChain
from langchain.memory import ConversationBufferMemory
from langchain.prompts import PromptTemplate
from langchain_core.vectorstores import VectorStore

# --- Constantes y Configuración ---
MODEL_NAME = "llama-3.3-70b-versatile" # Modelo con ventana de contexto enorme (128k tokens)
EMBEDDING_MODEL = "all-MiniLM-L6-v2"
PAGE_ICON = "⚡"
PAGE_TITLE = "Pokédex Z - Profesor Z"
LOCAL_PDF_PATH = "GuiaPokemonZ.pdf"  # <--- ¡IMPORTANTE! Nombre exacto de tu archivo

st.set_page_config(page_title=PAGE_TITLE, page_icon=PAGE_ICON, layout="wide")

# --- Capa de Lógica (Backend) ---

def get_text_from_local_pdf(path: str) -> str:
    """Lee el PDF del disco duro."""
    try:
        pdf_reader = PdfReader(path)
        text = ""
        for page in pdf_reader.pages:
            content = page.extract_text()
            if content:
                text += content + "\n" # Añadimos salto de línea por página
        return text
    except Exception as e:
        st.error(f"Error leyendo el PDF local: {e}")
        return ""

def get_text_chunks(text: str) -> List[str]:
    """
    Divide el texto con una estrategia agresiva de solapamiento 
    para no perder contexto entre cortes.
    """
    if not text.strip():
        return []
    
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=800,       # Tamaño mediano para precisión
        chunk_overlap=300,    # Mucho solapamiento para mantener frases unidas
        length_function=len,
        separators=["\n\n", "\n", " ", ""] # Prioriza cortar por párrafos
    )
    return text_splitter.split_text(text)

@st.cache_resource
def get_embeddings_model():
    return HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL)

@st.cache_resource(show_spinner="Despertando al Profesor Z y leyendo la guía...") 
def process_local_knowledge_base(file_path: str):
    """Procesa el PDF solo una vez y lo guarda en memoria RAM."""
    if not os.path.exists(file_path):
        return None
    
    raw_text = get_text_from_local_pdf(file_path)
    if not raw_text:
        return None

    chunks = get_text_chunks(raw_text)
    embeddings = get_embeddings_model()
    
    # Creamos la base de datos vectorial
    vectorstore = FAISS.from_texts(texts=chunks, embedding=embeddings)
    return vectorstore

def get_conversation_chain(vectorstore: VectorStore, groq_api_key: str):
    """Configura el cerebro del bot con super-memoria."""
    llm = ChatGroq(
        groq_api_key=groq_api_key,
        model_name=MODEL_NAME,
        temperature=0.1 # Temperatura baja = Más fiel a los hechos, menos creativo
    )

    template = """
    Eres el "Profesor Z", la autoridad máxima en Pokémon Z.
    Tienes acceso total a la guía del juego.
    
    INSTRUCCIONES CLAVE:
    1. Usa la información del "Contexto" abajo para responder.
    2. Busca exhaustivamente. Si la respuesta está en el texto, ENCUÉNTRALA.
    3. Si la información es parcial, intenta deducir la respuesta basándote en el contexto.
    4. Responde siempre en Español y con formato claro (listas, negritas).
    
    Contexto recuperado de la guía:
    {context}

    Historial del chat:
    {chat_history}

    Pregunta del entrenador: {question}
    
    Respuesta del Profesor Z:"""

    prompt = PromptTemplate(
        input_variables=["context", "chat_history", "question"],
        template=template
    )

    memory = ConversationBufferMemory(
        memory_key='chat_history', 
        return_messages=True, 
        output_key='answer'
    )

    return ConversationalRetrievalChain.from_llm(
        llm=llm,
        # --- AQUÍ ESTÁ LA MAGIA ---
        # search_type="mmr": Busca diversidad en los resultados (no solo palabras repetidas)
        # k=20: Recupera 20 fragmentos de texto (mucha información)
        retriever=vectorstore.as_retriever(
            search_type="mmr", 
            search_kwargs={"k": 20, "fetch_k": 50, "lambda_mult": 0.5}
        ),
        memory=memory,
        combine_docs_chain_kwargs={"prompt": prompt}
    )

# --- Frontend (Interfaz) ---

def initialize_session():
    if "messages" not in st.session_state:
        st.session_state.messages = [{"role": "assistant", "content": "¡Hola! Soy el Profesor Z. He memorizado la guía completa. Pregúntame sobre ubicaciones, evoluciones o estadísticas."}]
    if "conversation" not in st.session_state:
        st.session_state.conversation = None
    if "api_key" not in st.session_state:
        st.session_state.api_key = ""

def sidebar_logic():
    with st.sidebar:
        st.header("⚙️ Estado del Sistema")
        
        # Gestión de API Key
        env_key = None
        try:
            env_key = st.secrets.get("GROQ_API_KEY")
        except:
            pass

        if env_key:
            st.session_state.api_key = env_key
            st.success("✅ Clave de Groq activa")
        else:
            user_key = st.text_input("Groq API Key:", type="password")
            if user_key:
                st.session_state.api_key = user_key

        st.divider()

        # Estado del archivo
        if os.path.exists(LOCAL_PDF_PATH):
            st.info(f"📚 Guía cargada: {LOCAL_PDF_PATH}")
            
            # Herramienta de depuración para ver si lee bien
            with st.expander("🔍 Ver texto extraído (Debug)"):
                try:
                    debug_text = get_text_from_local_pdf(LOCAL_PDF_PATH)
                    st.text_area("Primeros 2000 caracteres:", debug_text[:2000], height=200)
                except:
                    st.error("No se pudo leer para debug.")
        else:
            st.error(f"❌ FALTA EL ARCHIVO: {LOCAL_PDF_PATH}")
            st.warning("Por favor, sube el archivo PDF a la carpeta de tu proyecto.")

        if st.button("Reiniciar Memoria"):
             st.session_state.messages = []
             if st.session_state.conversation:
                 st.session_state.conversation.memory.clear()
             st.rerun()

def chat_logic():
    st.title(f"{PAGE_ICON} Asistente Pokémon Z")

    # Inicialización automática del cerebro
    if st.session_state.api_key and st.session_state.conversation is None:
        if os.path.exists(LOCAL_PDF_PATH):
            vectorstore = process_local_knowledge_base(LOCAL_PDF_PATH)
            if vectorstore:
                st.session_state.conversation = get_conversation_chain(vectorstore, st.session_state.api_key)
            else:
                st.error("Error crítico procesando la guía.")
        else:
            st.warning("⚠️ Esperando archivo de guía...")

    # Chat UI
    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])

    if prompt := st.chat_input("Ej: ¿Cómo evoluciona Eevee a Sylveon?"):
        if not st.session_state.conversation:
            st.warning("⏳ Introduce tu API Key y asegúrate de que el PDF está en la carpeta.")
            return

        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)

        with st.chat_message("assistant"):
            with st.spinner("Consultando la guía..."):
                try:
                    response = st.session_state.conversation.invoke({'question': prompt})
                    answer = response['answer']
                    st.markdown(answer)
                    st.session_state.messages.append({"role": "assistant", "content": answer})
                except Exception as e:
                    st.error(f"Error: {e}")

def main():
    initialize_session()
    sidebar_logic()
    chat_logic()

if __name__ == '__main__':
    main()