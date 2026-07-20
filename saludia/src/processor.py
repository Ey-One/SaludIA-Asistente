"""
processor.py
------------
Módulo encargado de la ingesta de la Base de Conocimiento de SaludIA.

Flujo:
1. Carga todos los documentos (.pdf y .txt) de la carpeta /data.
2. Divide el contenido en fragmentos ("chunks") manejables para el LLM.
3. Genera embeddings locales y gratuitos con HuggingFace (sentence-transformers).
4. Persiste los vectores en una base de datos local ChromaDB (./chroma_db).

Este script se ejecuta UNA SOLA VEZ (o cada vez que se actualicen los documentos
de /data) para (re)construir el índice vectorial. La app de Streamlit (app.py)
solo LEE de esta base, no la reconstruye.

Uso:
    python src/processor.py
"""

import os
import sys
import logging
from pathlib import Path

from langchain_community.document_loaders import PyPDFLoader, TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma

# --------------------------------------------------------------------------
# CONFIGURACIÓN GENERAL
# --------------------------------------------------------------------------

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
)
logger = logging.getLogger("SaludIA.processor")

# Rutas base del proyecto (relativas a la raíz del repositorio)
BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data"
PERSIST_DIR = BASE_DIR / "chroma_db"

# Modelo de embeddings local y gratuito (multilingüe, funciona bien en español)
EMBEDDING_MODEL_NAME = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"

# Parámetros de fragmentación del texto
CHUNK_SIZE = 800
CHUNK_OVERLAP = 120

# Nombre de la colección dentro de ChromaDB
COLLECTION_NAME = "saludia_knowledge_base"


# --------------------------------------------------------------------------
# CARGA DE DOCUMENTOS
# --------------------------------------------------------------------------

def load_documents(data_dir: Path) -> list:
    """
    Recorre la carpeta /data y carga todos los archivos .pdf y .txt
    utilizando el loader apropiado para cada extensión.
    """
    if not data_dir.exists():
        logger.error(f"La carpeta de datos no existe: {data_dir}")
        sys.exit(1)

    documents = []
    archivos = sorted(data_dir.glob("*"))

    if not archivos:
        logger.warning(f"No se encontraron archivos en {data_dir}. "
                        f"Colocá allí los .txt/.pdf de SaludIA.")

    for file_path in archivos:
        try:
            if file_path.suffix.lower() == ".pdf":
                loader = PyPDFLoader(str(file_path))
                docs = loader.load()
            elif file_path.suffix.lower() == ".txt":
                loader = TextLoader(str(file_path), encoding="utf-8")
                docs = loader.load()
            else:
                logger.info(f"Ignorando archivo no soportado: {file_path.name}")
                continue

            # Enriquecemos los metadatos con el nombre del archivo fuente,
            # útil para citar la fuente de la respuesta en el chat.
            for doc in docs:
                doc.metadata["source_file"] = file_path.name

            documents.extend(docs)
            logger.info(f"Cargado: {file_path.name} ({len(docs)} página/s)")

        except Exception as e:
            logger.error(f"Error cargando {file_path.name}: {e}")

    return documents


# --------------------------------------------------------------------------
# FRAGMENTACIÓN (CHUNKING)
# --------------------------------------------------------------------------

def split_documents(documents: list) -> list:
    """
    Divide los documentos en fragmentos más pequeños (chunks), con
    superposición entre ellos para no perder contexto en los bordes.
    """
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        separators=["\n\n", "\n", ". ", " ", ""],
    )
    chunks = splitter.split_documents(documents)
    logger.info(f"Documentos fragmentados en {len(chunks)} chunks "
                f"(chunk_size={CHUNK_SIZE}, overlap={CHUNK_OVERLAP})")
    return chunks


# --------------------------------------------------------------------------
# VECTORIZACIÓN Y PERSISTENCIA EN CHROMADB
# --------------------------------------------------------------------------

def build_vector_store(chunks: list, persist_dir: Path) -> Chroma:
    """
    Genera los embeddings de cada chunk y los persiste en una base
    ChromaDB local. Si ya existe una base previa, se sobreescribe
    para reflejar el estado actual de /data.
    """
    logger.info(f"Cargando modelo de embeddings: {EMBEDDING_MODEL_NAME} "
                f"(puede tardar la primera vez, se descarga localmente)...")

    embeddings = HuggingFaceEmbeddings(
        model_name=EMBEDDING_MODEL_NAME,
        model_kwargs={"device": "cpu"},
        encode_kwargs={"normalize_embeddings": True},
    )

    logger.info(f"Generando y persistiendo vectores en: {persist_dir}")

    vector_store = Chroma.from_documents(
        documents=chunks,
        embedding=embeddings,
        collection_name=COLLECTION_NAME,
        persist_directory=str(persist_dir),
    )

    # En versiones nuevas del paquete langchain_chroma, la persistencia en
    # disco es automática (usa un PersistentClient de chromadb por debajo).
    # Se intenta llamar a .persist() solo por compatibilidad con versiones
    # previas; si el método no existe, se ignora sin romper la ejecución.
    if hasattr(vector_store, "persist"):
        try:
            vector_store.persist()
        except Exception:
            pass

    logger.info("Base vectorial de SaludIA creada/actualizada exitosamente.")
    return vector_store


# --------------------------------------------------------------------------
# PUNTO DE ENTRADA
# --------------------------------------------------------------------------

def main():
    logger.info("=== Iniciando ingesta de la Base de Conocimiento SaludIA ===")

    documents = load_documents(DATA_DIR)
    if not documents:
        logger.error("No hay documentos para procesar. Abortando.")
        sys.exit(1)

    chunks = split_documents(documents)
    build_vector_store(chunks, PERSIST_DIR)

    logger.info("=== Proceso finalizado. La app.py ya puede consultar la base. ===")


if __name__ == "__main__":
    main()
