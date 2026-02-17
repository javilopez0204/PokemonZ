import streamlit as st
import os
import time
import re
from pypdf import PdfReader
from typing import List, Optional
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_community.vectorstores import FAISS
from langchain_community.retrievers import BM25Retriever
from langchain.retrievers import EnsembleRetriever
from langchain.retrievers.multi_query import MultiQueryRetriever
from langchain.chains import ConversationalRetrievalChain
from langchain.memory import ConversationBufferMemory
from langchain_core.prompts import PromptTemplate
from langchain_core.vectorstores import VectorStore

# ---------------------------------------------------------------------------
# Constantes
# ---------------------------------------------------------------------------
MODEL_NAME      = "gemini-2.5-flash-lite-preview-06-17"
EMBEDDING_MODEL = "all-MiniLM-L6-v2"
PAGE_ICON       = "⚡"
PAGE_TITLE      = "Pokédex Z – Profesor Z (Gemini Edition)"
LOCAL_PDF_PATH  = "GuiaPokemonZ.pdf"

QA_TEMPLATE = """Eres el "Profesor Z", la máxima autoridad en Pokémon Z.
Tienes acceso completo a la guía oficial del juego.

INSTRUCCIONES CRÍTICAS:
1. La respuesta SIEMPRE está en el "Contexto" de abajo. Léelo entero con atención.
2. Busca keywords, sinónimos y variantes ortográficas antes de rendirte.
3. Si encuentras la información aunque sea parcial, dala.
4. Solo di "no encuentro esa información" si el contexto está completamente vacío o irrelevante.
5. Responde siempre en **Español**, con listas y negritas.

---
Contexto de la guía (léelo completo):
{context}

Historial:
{chat_history}

Pregunta: {question}

Respuesta del Profesor Z:"""

CONDENSE_TEMPLATE = """Dado el historial de conversación y la nueva pregunta del usuario,
reformula la pregunta para que sea autónoma y mantenga TODOS los nombres propios,
términos de Pokémon, objetos y lugares exactamente como aparecen.
Si la pregunta ya es autónoma, devuélvela SIN cambios.

Historial:
{chat_history}

Pregunta original: {question}

Pregunta reformulada (conserva todos los términos clave):"""

# ---------------------------------------------------------------------------
# Configuración de página
# ---------------------------------------------------------------------------
st.set_page_config(page_title=PAGE_TITLE, page_icon=PAGE_ICON, layout="wide")

# ---------------------------------------------------------------------------
# Backend – procesamiento de PDF
# ---------------------------------------------------------------------------

def normalize_text(text: str) -> str:
    """
    Normaliza el texto para mejorar la búsqueda:
    - Convierte encabezados en MAYÚSCULAS a Title Case para que BM25/embeddings
      los relacionen con preguntas en minúsculas.
    - Elimina signos de interrogación/exclamación al inicio de línea (¿¡).
    - Colapsa espacios múltiples.
    """
    lines = []
    for line in text.split("\n"):
        stripped = line.strip()
        # Líneas completamente en mayúsculas (títulos/encabezados) → Title Case
        if stripped and stripped == stripped.upper() and len(stripped) > 4:
            stripped = stripped.title()
        # Quitar ¿ y ¡ de inicio para no confundir al tokenizador
        stripped = re.sub(r"^[¿¡]+", "", stripped)
        lines.append(stripped)
    text = "\n".join(lines)
    # Colapsar espacios y líneas vacías múltiples
    text = re.sub(r" {2,}", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text


def extract_text_from_pdf(path: str) -> str:
    try:
        reader = PdfReader(path)
        pages = []
        for i, page in enumerate(reader.pages):
            content = page.extract_text() or ""
            if content.strip():
                normalized = normalize_text(content)
                pages.append(f"[Página {i+1}]\n{normalized}")
        return "\n\n".join(pages)
    except Exception as exc:
        st.error(f"❌ Error leyendo el PDF: {exc}")
        return ""


def split_text(text: str) -> List[str]:
    if not text.strip():
        return []
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=100,
        length_function=len,
        separators=["\n\n", "\n", ". ", " ", ""],
    )
    return splitter.split_text(text)


@st.cache_resource(show_spinner=False)
def load_embeddings() -> HuggingFaceEmbeddings:
    return HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL)


@st.cache_resource(show_spinner="📖 El Profesor Z está memorizando la guía...")
def build_knowledge_base(file_path: str) -> Optional[tuple]:
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
    llm = ChatGoogleGenerativeAI(
        model=MODEL_NAME,
        google_api_key=api_key,
        temperature=0.0,
        convert_system_message_to_human=True,
    )

    qa_prompt = PromptTemplate(
        input_variables=["context", "chat_history", "question"],
        template=QA_TEMPLATE,
    )

    condense_prompt = PromptTemplate(
        input_variables=["chat_history", "question"],
        template=CONDENSE_TEMPLATE,
    )

    memory = ConversationBufferMemory(
        memory_key="chat_history",
        return_messages=True,
        output_key="answer",
    )

    # ── Retriever semántico con MultiQuery ────────────────────────────────────
    # MultiQueryRetriever genera 3 variantes de la pregunta y busca con cada una.
    # Así "caramelos raros" también busca "cómo conseguir caramelos raros",
    # "caramelos raros infinitos", etc., cubriendo los títulos del PDF.
    base_semantic = vectorstore.as_retriever(
        search_type="similarity",
        search_kwargs={"k": 15},
    )
    multi_query_retriever = MultiQueryRetriever.from_llm(
        retriever=base_semantic,
        llm=llm,
    )

    # ── Retriever BM25 (palabras clave exactas) ───────────────────────────────
    bm25_retriever = BM25Retriever.from_texts(chunks)
    bm25_retriever.k = 15

    # ── Híbrido: MultiQuery semántico + BM25 ─────────────────────────────────
    hybrid_retriever = EnsembleRetriever(
        retrievers=[multi_query_retriever, bm25_retriever],
        weights=[0.6, 0.4],
    )

    return ConversationalRetrievalChain.from_llm(
        llm=llm,
        retriever=hybrid_retriever,
        memory=memory,
        return_source_documents=True,
        combine_docs_chain_kwargs={"prompt": qa_prompt},
        condense_question_prompt=condense_prompt,
        verbose=False,
    )


def direct_search(query: str, vectorstore: VectorStore, chunks: List[str], k: int = 10) -> List[str]:
    """Búsqueda directa sin LLM para diagnóstico."""
    sem_docs = vectorstore.similarity_search(query, k=k)
    sem_results = [d.page_content for d in sem_docs]

    bm25 = BM25Retriever.from_texts(chunks)
    bm25.k = k
    bm25_docs = bm25.get_relevant_documents(query)
    bm25_results = [d.page_content for d in bm25_docs]

    seen, combined = set(), []
    for r in sem_results + bm25_results:
        if r not in seen:
            seen.add(r)
            combined.append(r)
    return combined


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
        "vectorstore": None,
        "chunks": [],
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

        if os.path.exists(LOCAL_PDF_PATH):
            size_mb = os.path.getsize(LOCAL_PDF_PATH) / 1_048_576
            st.info(f"📚 **{LOCAL_PDF_PATH}** · {size_mb:.1f} MB")
        else:
            st.error(f"❌ Falta el archivo `{LOCAL_PDF_PATH}`")

        if st.session_state.chunk_count > 0:
            st.success(f"🧩 **{st.session_state.chunk_count}** fragmentos indexados")
            st.caption(f"📝 {st.session_state.raw_text_len:,} caracteres extraídos")
            if st.session_state.chunk_count < 50:
                st.warning("⚠️ Muy pocos fragmentos — posible PDF escaneado.")

        st.divider()

        st.session_state.debug_mode = st.toggle(
            "🔍 Modo Debug",
            value=st.session_state.debug_mode,
            help="Muestra los fragmentos exactos enviados al modelo.",
        )

        # Búsqueda directa para diagnóstico
        if st.session_state.vectorstore is not None:
            st.divider()
            st.markdown("#### 🧪 Búsqueda directa en la guía")
            st.caption("Si aparece la info aquí pero el chat no la usa → problema de prompt. Si no aparece → problema de retriever.")
            search_query = st.text_input("Buscar:", placeholder="caramelos raros")
            if search_query:
                results = direct_search(
                    search_query,
                    st.session_state.vectorstore,
                    st.session_state.chunks,
                    k=8,
                )
                if results:
                    for i, r in enumerate(results[:6], 1):
                        with st.expander(f"Resultado {i}"):
                            st.code(r, language=None)
                else:
                    st.warning("Sin resultados.")

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
            f"Retriever: MultiQuery + BM25"
        )


# ---------------------------------------------------------------------------
# Frontend – chat principal
# ---------------------------------------------------------------------------

def render_chat():
    st.title(f"{PAGE_ICON} Asistente Pokémon Z")
    st.caption("Powered by Google Gemini · MultiQuery + BM25 + normalización de texto")

    if st.session_state.api_key and st.session_state.conversation is None:
        if os.path.exists(LOCAL_PDF_PATH):
            with st.spinner("⚙️ Inicializando sistema RAG..."):
                kb = build_knowledge_base(LOCAL_PDF_PATH)
                if kb:
                    vectorstore, chunks, raw_text = kb
                    st.session_state.vectorstore = vectorstore
                    st.session_state.chunks = chunks
                    st.session_state.chunk_count = len(chunks)
                    st.session_state.raw_text_len = len(raw_text)
                    st.session_state.conversation = build_chain(
                        vectorstore, chunks, st.session_state.api_key
                    )
                    st.rerun()
                else:
                    st.error("No se pudo procesar la guía.")
        else:
            st.warning("⚠️ El archivo `GuiaPokemonZ.pdf` no se encontró.")

    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])
            if st.session_state.debug_mode and msg.get("sources"):
                with st.expander("🔍 Fragmentos enviados al modelo"):
                    for i, src in enumerate(msg["sources"], 1):
                        st.markdown(f"**Fragmento {i}:**")
                        st.code(src, language=None)

    if user_input := st.chat_input("Ej: ¿Dónde consigo caramelos raros?"):
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
                    with st.expander("🔍 Fragmentos enviados al modelo"):
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