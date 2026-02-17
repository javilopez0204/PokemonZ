import streamlit as st
import os
import time
from pypdf import PdfReader
from typing import List, Optional
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_community.vectorstores import FAISS
from langchain_community.retrievers import BM25Retriever
from langchain.retrievers import EnsembleRetriever
from langchain.chains import ConversationalRetrievalChain
from langchain.memory import ConversationBufferMemory
from langchain_core.prompts import PromptTemplate
from langchain_core.vectorstores import VectorStore

# ---------------------------------------------------------------------------
# Constantes
# ---------------------------------------------------------------------------
MODEL_NAME      = "gemini-2.5-flash-lite"
EMBEDDING_MODEL = "all-MiniLM-L6-v2"
PAGE_ICON       = "⚡"
PAGE_TITLE      = "Pokédex Z – Profesor Z (Gemini Edition)"
LOCAL_PDF_PATH  = "GuiaPokemonZ.pdf"

SYSTEM_TEMPLATE = """Eres el "Profesor Z", la máxima autoridad en Pokémon Z.
Tienes acceso completo a la guía oficial del juego.

INSTRUCCIONES:
1. Usa ÚNICAMENTE el "Contexto" para responder; si no hay información suficiente dilo claramente.
2. Responde siempre en **Español**.
3. Usa formato claro: listas con viñetas, negritas para nombres propios y datos clave.
4. Si la pregunta es ambigua, pide aclaraciones.
5. Nunca inventes datos; si no lo sabes, dilo con honestidad.
6. Revisa TODO el contexto proporcionado antes de decir que no tienes la información.

---
Contexto recuperado de la guía:
{context}

Historial del chat:
{chat_history}

Pregunta del entrenador: {question}

Respuesta del Profesor Z:"""

# ---------------------------------------------------------------------------
# Configuración de página
# ---------------------------------------------------------------------------
st.set_page_config(page_title=PAGE_TITLE, page_icon=PAGE_ICON, layout="wide")

# ---------------------------------------------------------------------------
# Backend – procesamiento de PDF
# ---------------------------------------------------------------------------

def extract_text_from_pdf(path: str) -> str:
    """Extrae texto de todas las páginas del PDF, anotando el número de página."""
    try:
        reader = PdfReader(path)
        pages = []
        for i, page in enumerate(reader.pages):
            content = page.extract_text() or ""
            if content.strip():
                pages.append(f"[Página {i+1}]\n{content}")
        return "\n\n".join(pages)
    except Exception as exc:
        st.error(f"❌ Error leyendo el PDF: {exc}")
        return ""


def split_text(text: str) -> List[str]:
    """Chunks más pequeños = recuperación más precisa."""
    if not text.strip():
        return []
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=600,
        chunk_overlap=150,
        length_function=len,
        separators=["\n\n", "\n", ". ", " ", ""],
    )
    return splitter.split_text(text)


@st.cache_resource(show_spinner=False)
def load_embeddings() -> HuggingFaceEmbeddings:
    return HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL)


@st.cache_resource(show_spinner="📖 El Profesor Z está memorizando la guía...")
def build_knowledge_base(file_path: str) -> Optional[tuple]:
    """Procesa el PDF y devuelve (vectorstore, chunks, raw_text)."""
    if not os.path.exists(file_path):
        return None
    text = extract_text_from_pdf(file_path)
    if not text:
        return None
    chunks = split_text(text)
    embeddings = load_embeddings()
    vectorstore = FAISS.from_texts(texts=chunks, embedding=embeddings)
    return vectorstore, chunks, text


def build_chain(vectorstore: VectorStore, chunks: List[str], api_key: str) -> ConversationalRetrievalChain:
    """Cadena RAG con retriever híbrido: semántico (FAISS) + por palabras clave (BM25)."""
    llm = ChatGoogleGenerativeAI(
        model=MODEL_NAME,
        google_api_key=api_key,
        temperature=0.05,
        convert_system_message_to_human=True,
    )

    prompt = PromptTemplate(
        input_variables=["context", "chat_history", "question"],
        template=SYSTEM_TEMPLATE,
    )

    memory = ConversationBufferMemory(
        memory_key="chat_history",
        return_messages=True,
        output_key="answer",
    )

    # FAISS: búsqueda semántica pura (sin MMR para no filtrar resultados relevantes)
    semantic_retriever = vectorstore.as_retriever(
        search_type="similarity",
        search_kwargs={"k": 20},
    )

    # BM25: búsqueda exacta por palabras clave (ideal para nombres de Pokémon, objetos, etc.)
    bm25_retriever = BM25Retriever.from_texts(chunks)
    bm25_retriever.k = 20

    # Retriever híbrido: combina ambos con peso 50/50
    hybrid_retriever = EnsembleRetriever(
        retrievers=[semantic_retriever, bm25_retriever],
        weights=[0.5, 0.5],
    )

    return ConversationalRetrievalChain.from_llm(
        llm=llm,
        retriever=hybrid_retriever,
        memory=memory,
        return_source_documents=True,
        combine_docs_chain_kwargs={"prompt": prompt},
    )


# ---------------------------------------------------------------------------
# Frontend – estado de sesión
# ---------------------------------------------------------------------------

def init_session():
    defaults = {
        "messages": [
            {
                "role": "assistant",
                "content": (
                    "¡Hola, entrenador! Soy el **Profesor Z** ⚡\n\n"
                    "He memorizado la guía completa de Pokémon Z. "
                    "¡Pregúntame lo que quieras!"
                ),
            }
        ],
        "conversation": None,
        "api_key": "",
        "total_queries": 0,
        "chunk_count": 0,
        "raw_text_len": 0,
        "debug_mode": False,
    }
    for key, val in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = val


# ---------------------------------------------------------------------------
# Frontend – sidebar
# ---------------------------------------------------------------------------

def render_sidebar():
    with st.sidebar:
        st.image(
            "https://upload.wikimedia.org/wikipedia/commons/9/98/International_Pok%C3%A9mon_logo.svg",
            use_container_width=True,
        )
        st.header("⚙️ Configuración")

        # API Key: secrets > env var > input manual
        api_key = (
            st.secrets.get("GOOGLE_API_KEY", None)
            or os.environ.get("GOOGLE_API_KEY", None)
        )
        if api_key:
            st.session_state.api_key = api_key
            st.success("✅ Google API Key configurada")
        else:
            typed_key = st.text_input(
                "Google API Key",
                type="password",
                placeholder="AIza...",
                help="Obtén tu clave en https://aistudio.google.com/",
            )
            if typed_key:
                st.session_state.api_key = typed_key
                st.success("✅ Clave introducida")

        st.divider()

        # Estado y diagnóstico del PDF
        if os.path.exists(LOCAL_PDF_PATH):
            size_mb = os.path.getsize(LOCAL_PDF_PATH) / 1_048_576
            st.info(f"📚 **{LOCAL_PDF_PATH}** · {size_mb:.1f} MB")
        else:
            st.error(f"❌ Falta el archivo `{LOCAL_PDF_PATH}`")

        if st.session_state.chunk_count > 0:
            st.success(f"🧩 **{st.session_state.chunk_count}** fragmentos indexados")
            st.caption(f"📝 {st.session_state.raw_text_len:,} caracteres extraídos")
            if st.session_state.chunk_count < 50:
                st.warning(
                    "⚠️ Muy pocos fragmentos. El PDF puede estar escaneado "
                    "(imágenes en vez de texto). En ese caso se necesita OCR."
                )

        st.divider()

        # Toggle de modo debug — clave para diagnosticar problemas
        st.session_state.debug_mode = st.toggle(
            "🔍 Modo Debug",
            value=st.session_state.debug_mode,
            help=(
                "Muestra los fragmentos exactos que el sistema recuperó. "
                "Actívalo para ver si la info está en el contexto o no llega al modelo."
            ),
        )

        st.divider()

        col1, col2 = st.columns(2)
        col1.metric("💬 Mensajes", len(st.session_state.messages))
        col2.metric("🔍 Consultas", st.session_state.total_queries)

        if st.button("🗑️ Reiniciar conversación", use_container_width=True):
            st.session_state.messages = []
            st.session_state.total_queries = 0
            if st.session_state.conversation:
                st.session_state.conversation.memory.clear()
            st.rerun()

        st.divider()
        st.caption(
            f"Modelo: `{MODEL_NAME}`  \n"
            f"Embeddings: `{EMBEDDING_MODEL}`  \n"
            f"Retriever: Híbrido FAISS + BM25"
        )


# ---------------------------------------------------------------------------
# Frontend – chat principal
# ---------------------------------------------------------------------------

def render_chat():
    st.title(f"{PAGE_ICON} Asistente Pokémon Z")
    st.caption("Powered by Google Gemini · Retriever híbrido semántico + palabras clave")

    # Inicializar RAG
    if st.session_state.api_key and st.session_state.conversation is None:
        if os.path.exists(LOCAL_PDF_PATH):
            with st.spinner("⚙️ Inicializando sistema RAG..."):
                kb = build_knowledge_base(LOCAL_PDF_PATH)
                if kb:
                    vectorstore, chunks, raw_text = kb
                    st.session_state.chunk_count = len(chunks)
                    st.session_state.raw_text_len = len(raw_text)
                    st.session_state.conversation = build_chain(
                        vectorstore, chunks, st.session_state.api_key
                    )
                    st.rerun()
                else:
                    st.error("No se pudo procesar la guía. Revisa el archivo PDF.")
        else:
            st.warning("⚠️ El archivo `GuiaPokemonZ.pdf` no se encontró.")

    # Historial de mensajes
    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])
            if st.session_state.debug_mode and msg.get("sources"):
                with st.expander("🔍 Fragmentos recuperados"):
                    for i, src in enumerate(msg["sources"], 1):
                        st.markdown(f"**Fragmento {i}:**")
                        st.code(src, language=None)

    # Input
    if user_input := st.chat_input("Ej: ¿Cómo evoluciona Eevee a Sylveon?"):
        if not st.session_state.conversation:
            st.warning("⏳ Introduce tu Google API Key y asegúrate de que el PDF está en su sitio.")
            return

        st.session_state.messages.append({"role": "user", "content": user_input})
        with st.chat_message("user"):
            st.markdown(user_input)

        with st.chat_message("assistant"):
            placeholder = st.empty()
            placeholder.markdown("⚡ _El Profesor Z está consultando la guía…_")
            t0 = time.perf_counter()
            try:
                result = st.session_state.conversation.invoke({"question": user_input})
                answer = result["answer"]
                sources = [doc.page_content for doc in result.get("source_documents", [])]
                elapsed = time.perf_counter() - t0

                placeholder.markdown(answer)
                st.caption(f"⏱️ {elapsed:.1f}s · {len(sources)} fragmentos consultados")

                if st.session_state.debug_mode and sources:
                    with st.expander("🔍 Fragmentos recuperados"):
                        for i, src in enumerate(sources, 1):
                            st.markdown(f"**Fragmento {i}:**")
                            st.code(src, language=None)

                st.session_state.messages.append(
                    {"role": "assistant", "content": answer, "sources": sources}
                )
                st.session_state.total_queries += 1

            except Exception as exc:
                placeholder.error(f"❌ Error: {exc}")


# ---------------------------------------------------------------------------
# Punto de entrada
# ---------------------------------------------------------------------------

def main():
    init_session()
    render_sidebar()
    render_chat()


if __name__ == "__main__":
    main()