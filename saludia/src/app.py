"""
app.py
------
Interfaz de usuario (Streamlit) para el Agente RAG "SaludIA".
VERSIÓN BLINDADA: 100% independiente de langchain.chains, con extracción segura de texto
y auto-generación de base vectorial en la nube.
"""

import os
from pathlib import Path

import streamlit as st
from dotenv import load_dotenv

# --- Embeddings y vectorstore ---
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma

# --- LangChain Core (API Blindada, inmune a errores de módulos) ---
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.messages import HumanMessage, AIMessage
from langchain_core.runnables import RunnableLambda
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter

# --------------------------------------------------------------------------
# CONFIGURACIÓN INICIAL
# --------------------------------------------------------------------------

load_dotenv()  # Carga variables desde .env

BASE_DIR = Path(__file__).resolve().parent.parent
PERSIST_DIR = BASE_DIR / "chroma_db"
COLLECTION_NAME = "saludia_knowledge_base"
EMBEDDING_MODEL_NAME = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"

LLM_PROVIDER = os.getenv("LLM_PROVIDER", "google").lower()
MODEL_NAME = os.getenv("MODEL_NAME", "gemini-1.5-flash")
TEMPERATURE = float(os.getenv("LLM_TEMPERATURE", "0.1"))

DISCLAIMER_TEXT = (
    "⚕️ **SaludIA** es un asistente administrativo e informativo. "
    "**No sustituye el consejo o diagnóstico médico.** "
    "En caso de emergencia, contacte a urgencias (guardia 24hs)."
)

QA_SYSTEM_PROMPT = """Eres SaludIA, el asistente virtual administrativo e informativo de una clínica de salud.

REGLAS ESTRICTAS QUE DEBES CUMPLIR SIEMPRE:
1. Responde ÚNICAMENTE utilizando la información contenida en el CONTEXTO proporcionado a continuación.
2. Si la respuesta no se encuentra en el CONTEXTO, di textualmente: "No cuento con esa información en mi base de conocimiento. Te recomiendo consultar directamente con recepción o el 0800-333-7253."
3. NUNCA des diagnósticos médicos, indiques tratamientos, dosis de medicamentos ni interpretes síntomas. Ese tipo de consultas deben derivarse siempre a un profesional médico o a la guardia.
4. Sé claro, concreto y amable. Si es útil, resume la información en viñetas.
5. Si la pregunta describe una posible urgencia o emergencia médica, indica de inmediato que debe contactar a la guardia/urgencias, independientemente de lo que diga el contexto.

CONTEXTO:
{context}"""

# --------------------------------------------------------------------------
# CARGA DE RECURSOS Y AUTO-GENERACIÓN
# --------------------------------------------------------------------------

@st.cache_resource(show_spinner="Cargando o generando base de conocimiento de SaludIA...")
def load_vector_store():
    # 1. Instanciamos el modelo de Embeddings
    embeddings = HuggingFaceEmbeddings(
        model_name=EMBEDDING_MODEL_NAME,
        model_kwargs={"device": "cpu"},
        encode_kwargs={"normalize_embeddings": True},
    )

    # 2. Si la base vectorial no existe (como pasa en la nube), la creamos leyendo los .txt
    if not PERSIST_DIR.exists():
        st.info("Generando base vectorial por primera vez en la nube... (esto tomará unos segundos)")
        
        DATA_DIR = BASE_DIR / "data"
        if not DATA_DIR.exists():
            st.error(f"No se encontró la carpeta de datos en: {DATA_DIR}")
            st.stop()
            
        # Leer todos los archivos .txt usando Python nativo
        documents = []
        for filepath in DATA_DIR.glob("*.txt"):
            with open(filepath, "r", encoding="utf-8") as f:
                text = f.read()
                documents.append(Document(page_content=text, metadata={"source": filepath.name}))
        
        if not documents:
            st.error("No se encontraron documentos de texto para procesar en la carpeta data/")
            st.stop()
            
        # Dividir en fragmentos (chunks)
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
        docs = text_splitter.split_documents(documents)
        
        # Crear y guardar en ChromaDB de forma persistente
        vector_store = Chroma.from_documents(
            documents=docs,
            embedding=embeddings,
            collection_name=COLLECTION_NAME,
            persist_directory=str(PERSIST_DIR)
        )
        return vector_store

    # 3. Si ya existe (en tu PC o tras la primera carga), simplemente la abrimos
    vector_store = Chroma(
        collection_name=COLLECTION_NAME,
        embedding_function=embeddings,
        persist_directory=str(PERSIST_DIR),
    )
    return vector_store


def get_llm():
    if LLM_PROVIDER == "google":
        from langchain_google_genai import ChatGoogleGenerativeAI
        api_key = os.getenv("GOOGLE_API_KEY")
        if not api_key:
            st.error("Falta configurar GOOGLE_API_KEY en los Secrets de Streamlit.")
            st.stop()
        
        return ChatGoogleGenerativeAI(
            model="gemini-1.5-flash", 
            temperature=TEMPERATURE, 
            google_api_key=api_key, 
            max_output_tokens=1024
        )

    elif LLM_PROVIDER == "anthropic":
        from langchain_anthropic import ChatAnthropic
        api_key = os.getenv("ANTHROPIC_API_KEY")
        if not api_key:
            st.error("Falta configurar ANTHROPIC_API_KEY")
            st.stop()
        return ChatAnthropic(model=MODEL_NAME, temperature=TEMPERATURE, anthropic_api_key=api_key, max_tokens=1024)

    elif LLM_PROVIDER == "openai":
        from langchain_openai import ChatOpenAI
        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            st.error("Falta configurar OPENAI_API_KEY")
            st.stop()
        return ChatOpenAI(model=MODEL_NAME, temperature=TEMPERATURE, openai_api_key=api_key)

    else:
        st.error(f"LLM_PROVIDER inválido: '{LLM_PROVIDER}'.")
        st.stop()


@st.cache_resource(show_spinner="Inicializando motor RAG (Modo Blindado)...")
def build_rag_chain(_vector_store):
    llm = get_llm()
    retriever = _vector_store.as_retriever(search_kwargs={"k": 4})

    qa_prompt = ChatPromptTemplate.from_messages([
        ("system", QA_SYSTEM_PROMPT),
        MessagesPlaceholder("chat_history"),
        ("human", "{input}"),
    ])

    def process_rag(inputs):
        user_input = inputs["input"]
        chat_history = inputs.get("chat_history", [])

        docs = retriever.invoke(user_input)
        formatted_context = "\n\n".join(doc.page_content for doc in docs)

        prompt_value = qa_prompt.invoke({
            "context": formatted_context,
            "chat_history": chat_history,
            "input": user_input
        })

        llm_response = llm.invoke(prompt_value)
        
        answer_text = ""
        if hasattr(llm_response, 'content'):
            content = llm_response.content
            if isinstance(content, str):
                answer_text = content
            elif isinstance(content, list) and len(content) > 0 and isinstance(content[0], dict):
                answer_text = content[0].get("text", str(content))
            elif isinstance(content, dict):
                answer_text = content.get("text", str(content))
            else:
                answer_text = str(content)
        else:
            answer_text = str(llm_response)

        return {
            "context": docs,
            "answer": answer_text
        }

    return RunnableLambda(process_rag)


def build_chat_history_messages(display_messages):
    history = []
    for msg in display_messages[:-1]:
        if msg["role"] == "user":
            history.append(HumanMessage(content=msg["content"]))
        else:
            history.append(AIMessage(content=msg["content"]))
    return history


# --------------------------------------------------------------------------
# PÁGINAS
# --------------------------------------------------------------------------

def render_inicio():
    st.title("⚕️ SaludIA")
    st.caption("Asistente virtual de la clínica — turnos, convenios y políticas.")

    st.markdown(
        "### 👋 ¡Bienvenido/a a SaludIA!\n"
        "Somos el asistente virtual administrativo de la clínica, disponible "
        "para acompañarte con tus consultas frecuentes las 24 horas."
    )

    st.subheader("¿Qué puedes hacer?")
    st.markdown(
        "- 📅 Consultar información sobre **turnos** (solicitud, reprogramación y cancelaciones).\n"
        "- 🩺 Resolver dudas sobre **convenios y coberturas** con obras sociales o prepagas.\n"
        "- 📄 Conocer nuestras **políticas** de privacidad, cancelación e instrucciones pre y post consulta.\n"
        "- 💬 Chatear en lenguaje natural y recibir respuestas basadas en la base de conocimiento oficial de la clínica."
    )
    st.info("👉 Seleccioná **'Asistente IA'** en el menú lateral para comenzar a chatear con SaludIA.")


def render_asistente_ia():
    st.title("⚕️ SaludIA — Asistente IA")
    st.caption("Preguntá sobre turnos, convenios y políticas de la clínica.")
    st.warning(DISCLAIMER_TEXT)

    vector_store = load_vector_store()
    rag_chain = build_rag_chain(vector_store)

    if "messages" not in st.session_state:
        st.session_state.messages = [{
            "role": "assistant",
            "content": "¡Hola! Soy SaludIA 👋. Puedo ayudarte con preguntas sobre turnos, cancelaciones, convenios e instrucciones de consultas. ¿En qué te ayudo?"
        }]

    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])

    user_input = st.chat_input("Escribí tu consulta aquí...")

    if user_input:
        st.session_state.messages.append({"role": "user", "content": user_input})
        with st.chat_message("user"):
            st.markdown(user_input)

        with st.chat_message("assistant"):
            with st.spinner("Consultando la base de conocimiento..."):
                try:
                    chat_history = build_chat_history_messages(st.session_state.messages)
                    result = rag_chain.invoke({
                        "input": user_input,
                        "chat_history": chat_history
                    })
                    answer = result["answer"]
                except Exception as e:
                    answer = "Ocurrió un error al procesar tu consulta. Por favor, intentá nuevamente."
                    st.error(f"Detalle técnico: {e}")

                st.markdown(answer)

        st.session_state.messages.append({"role": "assistant", "content": answer})

    st.divider()
    if st.button("🗑️ Limpiar conversación"):
        st.session_state.messages = []
        st.rerun()


def render_acerca_del_proyecto():
    st.title("ℹ️ Acerca de SaludIA")
    st.subheader("Asistente inteligente de atención clínica")
    st.markdown("Asistente virtual desarrollado con IA para optimizar la atención administrativa y resolver dudas frecuentes del paciente.")

    st.divider()
    st.markdown("**Tecnologías utilizadas:**")
    st.markdown(
        "- 🐍 Python\n"
        "- 🎈 Streamlit\n"
        "- 🔗 LangChain Core\n"
        "- 🗄️ ChromaDB\n"
        "- 🤗 HuggingFace Embeddings\n"
        "- ✨ Google Gemini"
    )
    st.divider()
    st.caption(f"Proveedor de LLM configurado: **{LLM_PROVIDER}**")


# --------------------------------------------------------------------------
# INTERFAZ STREAMLIT
# --------------------------------------------------------------------------

def main():
    st.set_page_config(page_title="SaludIA", page_icon="⚕️", layout="centered")

    if "messages" not in st.session_state:
        st.session_state.messages = [{
            "role": "assistant",
            "content": "¡Hola! Soy SaludIA 👋. Puedo ayudarte con preguntas sobre turnos, cancelaciones, convenios e instrucciones de consultas. ¿En qué te ayudo?"
        }]

    with st.sidebar:
        st.subheader("⚕️ SaludIA")
        pagina = st.radio(
            "Navegación",
            options=["Inicio", "Asistente IA", "Acerca del proyecto"],
            index=0,
        )
        st.divider()
        st.caption(f"Proveedor: **{LLM_PROVIDER}** (Modelo forzado en código)")

    if pagina == "Inicio":
        render_inicio()
    elif pagina == "Asistente IA":
        render_asistente_ia()
    elif pagina == "Acerca del proyecto":
        render_acerca_del_proyecto()

if __name__ == "__main__":
    main()
