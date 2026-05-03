from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_openai import OpenAIEmbeddings
from langchain_openai import ChatOpenAI

from langchain_classic.chains import RetrievalQA
from langchain_core.prompts import PromptTemplate

import streamlit as st
import os


if "OPENAI_API_KEY" in st.secrets:
    os.environ["OPENAI_API_KEY"] = st.secrets["OPENAI_API_KEY"]


st.set_page_config(page_title="CFR Seguros - App Interna", layout="wide")

st.image("logo.png", width=180)
st.markdown("## Asistente Interno de Consulta Profesional")
st.write("Consultas técnicas sobre curso, leyes, circulares y material profesional cargado.")

def load_docs():
    docs = []
    for root, dirs, files in os.walk("docs"):
        for file in files:
            if file.lower().endswith(".pdf"):
                path = os.path.join(root, file)
                loader = PyPDFLoader(path)
                loaded_docs = loader.load()

                carpeta = os.path.basename(root).lower()

                if "ley" in carpeta:
                    source_type = "ley"
                elif "curso" in carpeta:
                    source_type = "curso"
                elif "circular" in carpeta:
                    source_type = "circular"
                else:
                    source_type = "otro"

                for d in loaded_docs:
                    d.metadata["source_type"] = source_type
                    d.metadata["source_file"] = file

                docs.extend(loaded_docs)

    return docs

embeddings = OpenAIEmbeddings(openai_api_key=st.secrets["OPENAI_API_KEY"])


db = FAISS.load_local("faiss_index_interna", embeddings, allow_dangerous_deserialization=True)

prompt_template = """
Sos un asistente interno para Claudio, Productor Asesor de Seguros en Argentina.

Tu función es ayudarlo a evacuar dudas profesionales usando:
- curso de productor asesor
- leyes cargadas
- circulares
- documentación técnica
- criterio profesional general cuando corresponda

Respondé de forma clara, precisa y útil para trabajar.

REGLAS:
- No prometas coberturas sin revisar póliza, condiciones particulares, exclusiones y cláusulas.
- Si algo depende de la compañía o de la póliza, aclaralo.
- Diferenciá entre norma legal, material de curso, circular y criterio profesional.
- Si falta información, indicá qué dato habría que revisar.
- Priorizá utilidad práctica.

FORMATO:
1. Respuesta directa
2. Explicación práctica
3. Puntos a revisar
4. Tipo de fundamento: Ley / Curso / Circular / Criterio profesional
5. Recomendación operativa para Claudio

Contexto:
{context}

Pregunta:
{question}

Respuesta:
"""

PROMPT = PromptTemplate(
    template=prompt_template,
    input_variables=["context", "question"]
)
if st.button("🔄 Recrear base (cuando agrego PDFs)"):
    if os.path.exists("faiss_index_interna"):
        import shutil
        shutil.rmtree("faiss_index_interna")
    st.success("Base eliminada. Cerrá y abrí la app para regenerarla.")

query = st.text_input("Hacé tu consulta interna:")

if query:
    qa = RetrievalQA.from_chain_type(
        llm=ChatOpenAI(model="gpt-4o-mini"),
        retriever=db.as_retriever(search_kwargs={"k": 2}),
        return_source_documents=True,
        chain_type_kwargs={"prompt": PROMPT}
    )

    with st.spinner("Analizando documentación..."):
        result = qa({"query": query})

    st.subheader("Respuesta")
    st.write(result["result"])

    st.subheader("📄 Documentación consultada")
    st.caption(f"Se utilizaron {len(result['source_documents'])} fragmentos de documentación")

    fuentes_ordenadas = {
        "ley": [],
        "curso": [],
        "circular": [],
        "otro": []
    }

    for doc in result["source_documents"]:
        archivo = doc.metadata.get("source_file", "Desconocido")
        tipo = doc.metadata.get("source_type", "otro")

        if tipo not in fuentes_ordenadas:
            tipo = "otro"

        if archivo not in fuentes_ordenadas[tipo]:
            fuentes_ordenadas[tipo].append(archivo)

    if fuentes_ordenadas["ley"]:
        st.markdown("**⚖️ Leyes:**")
        for f in fuentes_ordenadas["ley"]:
            st.write("- " + f)

    if fuentes_ordenadas["curso"]:
        st.markdown("**📘 Curso:**")
        for f in fuentes_ordenadas["curso"]:
            st.write("- " + f)

    if fuentes_ordenadas["circular"]:
        st.markdown("**📄 Circulares:**")
        for f in fuentes_ordenadas["circular"]:
            st.write("- " + f)

    if fuentes_ordenadas["otro"]:
        st.markdown("**📄 Otros:**")
        for f in fuentes_ordenadas["otro"]:
            st.write("- " + f)

from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_openai import OpenAIEmbeddings
from langchain_openai import ChatOpenAI

from langchain_classic.chains import RetrievalQA
from langchain_core.prompts import PromptTemplate

import streamlit as st
import os


if "OPENAI_API_KEY" in st.secrets:
    os.environ["OPENAI_API_KEY"] = st.secrets["OPENAI_API_KEY"]


st.set_page_config(page_title="CFR Seguros - App Interna", layout="wide")

st.image("logo.png", width=180)
st.markdown("## Asistente Interno de Consulta Profesional")
st.write("Consultas técnicas sobre curso, leyes, circulares y material profesional cargado.")

def load_docs():
    docs = []
    for root, dirs, files in os.walk("docs"):
        for file in files:
            if file.lower().endswith(".pdf"):
                path = os.path.join(root, file)
                loader = PyPDFLoader(path)
                loaded_docs = loader.load()

                carpeta = os.path.basename(root).lower()

                if "ley" in carpeta:
                    source_type = "ley"
                elif "curso" in carpeta:
                    source_type = "curso"
                elif "circular" in carpeta:
                    source_type = "circular"
                else:
                    source_type = "otro"

                for d in loaded_docs:
                    d.metadata["source_type"] = source_type
                    d.metadata["source_file"] = file

                docs.extend(loaded_docs)

    return docs

embeddings = OpenAIEmbeddings(openai_api_key=st.secrets["OPENAI_API_KEY"])

docs = load_docs()

if not docs:
    st.warning("No hay PDFs cargados en la carpeta docs. Subí documentación para poder consultar.")
    st.stop()

splitter = RecursiveCharacterTextSplitter(chunk_size=900, chunk_overlap=120)
chunks = splitter.split_documents(docs)

if not chunks:
    st.warning("No se pudo extraer texto de los PDFs cargados.")
    st.stop()

db = FAISS.from_documents(chunks, embeddings)
db.save_local("faiss_index_interna")

prompt_template = """
Sos un asistente interno para Claudio, Productor Asesor de Seguros en Argentina.

Tu función es ayudarlo a evacuar dudas profesionales usando:
- curso de productor asesor
- leyes cargadas
- circulares
- documentación técnica
- criterio profesional general cuando corresponda

Respondé de forma clara, precisa y útil para trabajar.

REGLAS:
- No prometas coberturas sin revisar póliza, condiciones particulares, exclusiones y cláusulas.
- Si algo depende de la compañía o de la póliza, aclaralo.
- Diferenciá entre norma legal, material de curso, circular y criterio profesional.
- Si falta información, indicá qué dato habría que revisar.
- Priorizá utilidad práctica.

FORMATO:
1. Respuesta directa
2. Explicación práctica
3. Puntos a revisar
4. Tipo de fundamento: Ley / Curso / Circular / Criterio profesional
5. Recomendación operativa para Claudio

Contexto:
{context}

Pregunta:
{question}

Respuesta:
"""

PROMPT = PromptTemplate(
    template=prompt_template,
    input_variables=["context", "question"]
)
if st.button("🔄 Recrear base (cuando agrego PDFs)"):
    if os.path.exists("faiss_index_interna"):
        import shutil
        shutil.rmtree("faiss_index_interna")
    st.success("Base eliminada. Cerrá y abrí la app para regenerarla.")

query = st.text_input("Hacé tu consulta interna:")

if query:
    qa = RetrievalQA.from_chain_type(
        llm=ChatOpenAI(model="gpt-4o-mini"),
        retriever=db.as_retriever(search_kwargs={"k": 2}),
        return_source_documents=True,
        chain_type_kwargs={"prompt": PROMPT}
    )

    with st.spinner("Analizando documentación..."):
        result = qa({"query": query})

    st.subheader("Respuesta")
    st.write(result["result"])

    st.subheader("📄 Documentación consultada")
    st.caption(f"Se utilizaron {len(result['source_documents'])} fragmentos de documentación")

    fuentes_ordenadas = {
        "ley": [],
        "curso": [],
        "circular": [],
        "otro": []
    }

    for doc in result["source_documents"]:
        archivo = doc.metadata.get("source_file", "Desconocido")
        tipo = doc.metadata.get("source_type", "otro")

        if tipo not in fuentes_ordenadas:
            tipo = "otro"

        if archivo not in fuentes_ordenadas[tipo]:
            fuentes_ordenadas[tipo].append(archivo)

    if fuentes_ordenadas["ley"]:
        st.markdown("**⚖️ Leyes:**")
        for f in fuentes_ordenadas["ley"]:
            st.write("- " + f)

    if fuentes_ordenadas["curso"]:
        st.markdown("**📘 Curso:**")
        for f in fuentes_ordenadas["curso"]:
            st.write("- " + f)

    if fuentes_ordenadas["circular"]:
        st.markdown("**📄 Circulares:**")
        for f in fuentes_ordenadas["circular"]:
            st.write("- " + f)

    if fuentes_ordenadas["otro"]:
        st.markdown("**📁 Otros:**")
        for f in fuentes_ordenadas["otro"]:
            st.write("- " + f)