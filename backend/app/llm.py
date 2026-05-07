"""Cadena conversacional Gemini con prompt del Profesor Z."""
from __future__ import annotations

from typing import List

from langchain.chains import ConversationalRetrievalChain
from langchain.memory import ConversationBufferMemory
from langchain_core.prompts import PromptTemplate
from langchain_google_genai import ChatGoogleGenerativeAI

from .schemas import ChatMessage
from .settings import Settings


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


def make_chain(retriever, settings: Settings) -> ConversationalRetrievalChain:
    llm = ChatGoogleGenerativeAI(
        model=settings.llm_model,
        google_api_key=settings.google_api_key,
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
        memory_key="chat_history", return_messages=True, output_key="answer"
    )
    return ConversationalRetrievalChain.from_llm(
        llm=llm,
        retriever=retriever,
        memory=memory,
        return_source_documents=True,
        combine_docs_chain_kwargs={"prompt": qa_prompt},
        condense_question_prompt=condense_prompt,
        verbose=False,
    )


def hydrate_memory(chain: ConversationalRetrievalChain, history: List[ChatMessage]) -> None:
    """Carga el historial recibido del cliente en la memoria de la cadena."""
    chain.memory.clear()
    pair: List[str] = []
    for msg in history:
        if msg.role == "user":
            pair = [msg.content]
        elif msg.role == "assistant" and pair:
            chain.memory.save_context({"question": pair[0]}, {"answer": msg.content})
            pair = []
