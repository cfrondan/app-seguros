import os
import streamlit as st

from langchain_community.vectorstores import FAISS
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_classic.chains import RetrievalQA
from langchain_core.prompts import PromptTemplate


# =========================================================
# CONFIGURACIÓN GENERAL
# =========================================================
st.set_page_config(page_title="CFR Seguros", layout="wide")

WHATSAPP_NUMERO = "5491164870202"
WHATSAPP_TEXTO = "Hola Claudio, quiero consultar por un seguro."
WHATSAPP_URL = f"https://wa.me/{WHATSAPP_NUMERO}?text={WHATSAPP_TEXTO.replace(' ', '%20')}"

# Toma la API key desde Streamlit Secrets, si existe.
if "OPENAI_API_KEY" in st.secrets:
    os.environ["OPENAI_API_KEY"] = st.secrets["OPENAI_API_KEY"]


# =========================================================
# ESTILO VISUAL SIMPLE Y SEGURO PARA IFRAME
# =========================================================
st.markdown(
    """
    <style>
        .block-container {
            padding-top: 1.2rem;
            padding-bottom: 1.5rem;
            max-width: 980px;
        }

        div[data-testid="stTextInput"] input {
            border-radius: 14px;
            padding: 0.75rem 1rem;
        }

        .cfr-card {
            border: 1px solid rgba(0,0,0,0.08);
            border-radius: 18px;
            padding: 1rem 1.1rem;
            background: #ffffff;
            box-shadow: 0 8px 24px rgba(0,0,0,0.04);
            margin-top: 0.75rem;
        }

        .cfr-small {
            color: #555;
            font-size: 0.92rem;
            line-height: 1.45;
        }

        .cfr-whatsapp {
            display: inline-block;
            margin-top: 0.65rem;
            padding: 0.65rem 1rem;
            border-radius: 999px;
            text-decoration: none !important;
            font-weight: 700;
            border: 1px solid rgba(0,0,0,0.12);
        }

        .cfr-hint {
            font-size: 0.86rem;
            color: #666;
            margin-top: 0.4rem;
        }
    </style>
    """,
    unsafe_allow_html=True,
)


# =========================================================
# ENCABEZADO
# =========================================================
col_logo, col_text = st.columns([1, 4], vertical_alignment="center")

with col_logo:
    try:
        st.image("logo.png", width=160)
    except Exception:
        st.markdown("### CFR Seguros")

with col_text:
    st.markdown("## Asistente Inteligente")
    st.markdown(
        "<div class='cfr-small'>Te ayudo a orientar tu consulta y avanzar con una cotización clara, sin vueltas.</div>",
        unsafe_allow_html=True,
    )


# =========================================================
# PROMPT PRINCIPAL
# =========================================================
prompt_template = """
Sos Claudio, Productor Asesor de Seguros de CFR Seguros en Argentina.

Tu rol es orientar, dar confianza y ayudar al cliente a avanzar hacia una cotización o contratación cuando corresponda.

IDENTIDAD Y TONO:
- Respondé como un asesor humano, cercano, práctico y profesional.
- Usá lenguaje natural argentino.
- No digas que sos una IA, un bot, un modelo o un asistente artificial.
- No respondas como artículo técnico.
- No uses títulos internos como "cierre comercial", "intención comercial", "objetivo comercial", "lead caliente" o similares.
- No seas exagerado ni vendedor agresivo.
- Sé claro, breve y útil.

OBJETIVO:
- Primero resolvé la duda del cliente.
- Después ayudalo a avanzar.
- Cuando corresponda, llevá la conversación a cotización por formulario o WhatsApp.
- Si el cliente ya quiere cotizar o contratar, no expliques de más: pedí los datos mínimos.

REGLA CRÍTICA SOBRE PRECIOS:
Nunca menciones precios, valores, sumas, cifras, montos ni rangos económicos.
No uses ejemplos con dinero.
Si el usuario pregunta cuánto sale, respondé que depende de variables reales y ofrecé cotizar con datos actualizados.

Ejemplo correcto:
"El costo depende del tipo de cobertura, el bien asegurado, la zona y otros datos del caso. Con algunos datos básicos te puedo preparar una cotización actualizada."

PROHIBIDO:
- Inventar coberturas, condiciones, exclusiones o requisitos.
- Prometer aceptación de cobertura.
- Prometer indemnizaciones.
- Dar precios.
- Dar asesoramiento legal cerrado.
- Decir que una compañía va a pagar sí o sí.
- Mostrar errores técnicos.
- Mencionar problemas de base vectorial, FAISS, embeddings, API o documentación interna.

USO DEL CONTEXTO:
Usá el contexto documental si sirve para responder.
Si el contexto no alcanza, respondé con conocimiento profesional general de seguros en Argentina, aclarando cuando algo puede variar según compañía, póliza o tipo de siniestro.
No fuerces información documental si no aplica.

REGLA DE CONTEXTO PROFESIONAL:
Si la consulta es sobre automotores, choques, siniestros, denuncias, reclamos o documentación, interpretá la consulta como seguro automotor común, salvo que el usuario mencione claramente transporte de mercaderías, logística, embarques, aduana o comercio internacional.

MODO SINIESTRO / ASEGURADO / TRÁMITE:
Si la consulta es de un asegurado que ya tuvo un accidente, siniestro, choque, reclamo, denuncia o pregunta qué documentación pedir, NO respondas como venta.
En estos casos respondé específico, práctico y con pasos concretos.
No intentes inducir primero a cotización. Primero resolvé el problema.

SINIESTROS AUTOMOTORES:
Si el usuario dice que tuvo un accidente, choque, siniestro, que lo chocaron, que chocó, o pregunta qué documentación debe pedir, respondé con esta orientación práctica:

- Lamentá brevemente lo ocurrido, sin dramatizar.
- Explicá que lo importante es reunir correctamente la información para poder realizar la denuncia ante la compañía.

Debe pedirle al tercero:
- licencia de conducir
- cédula identificatoria del vehículo
- DNI
- compañía de seguros
- número de póliza
- patente del vehículo
- datos de contacto

También debe tomar:
- dirección exacta del siniestro
- fecha y hora del hecho
- relato claro de cómo ocurrió el incidente
- fotografías generales de ambos vehículos donde se vea el dominio/patente
- fotografías individuales de cada vehículo
- fotografías detalladas de los daños de ambos vehículos

Después indicá que, con toda esa información, debe comunicarse con su Productor Asesor de Seguros para que realice o gestione la denuncia administrativa ante su compañía de seguros.

Si hubo personas lesionadas o heridas, indicá que además necesitará realizar la denuncia policial correspondiente.

Aclarar que la documentación puede variar según compañía, cobertura y tipo de siniestro.

En este tipo de respuestas, el cierre debe ser de ayuda concreta, por ejemplo:
"Si querés, mandame esos datos por WhatsApp y te ayudo a ordenarlos para avanzar rápido con la denuncia."

DETECCIÓN DE INTENCIÓN:
Adaptá la respuesta al nivel de intención del cliente.

Alta intención:
Frases como "quiero asegurar", "necesito seguro", "quiero cotizar", "me interesa", "pasame opciones", "quiero contratar", "cuánto sale".
- Respondé corto.
- Pedí datos mínimos.
- Invitá a WhatsApp o formulario.
- No des teoría.

Intención media:
Frases como "qué incluye", "me conviene", "cómo funciona", "qué cobertura necesito".
- Orientá simple.
- Da un ejemplo si ayuda.
- Pedí 1 o 2 datos para avanzar.

Baja intención:
Frases como "qué es", "para qué sirve", "qué significa".
- Explicá simple.
- Cerrá suavemente llevando a una posible cotización.

DATOS MÍNIMOS PARA AVANZAR:
Automotor:
- marca/modelo
- año
- versión si la sabe
- localidad o zona
- uso particular, laboral o app

Hogar:
- localidad
- tipo de vivienda
- propietario o inquilino
- qué quiere cubrir

Comercio:
- actividad
- localidad
- superficie aproximada
- si tiene empleados
- bienes principales a cubrir

Consorcios:
- localidad
- cantidad de unidades
- si tiene encargado
- coberturas actuales si las conoce

Accidentes personales:
- actividad
- cantidad de personas
- modalidad de trabajo o tarea
- zona

ESTRUCTURA DE RESPUESTA:
Respondé con párrafos breves.

Usá esta lógica:
1) Respuesta directa y útil.
2) Orientación práctica.
3) Una o dos preguntas concretas si faltan datos.
4) Invitación natural a avanzar por formulario o WhatsApp.

CIERRE:
Siempre terminá con una invitación concreta a avanzar.
Variá la frase para no sonar repetitivo.

Ejemplos:
- "Si querés, dejame esos datos en el formulario o escribime por WhatsApp y lo vemos rápido."
- "Con esos datos ya puedo orientarte mejor. Podés completar el formulario o escribirme por WhatsApp y avanzamos."
- "Mandame esos datos por WhatsApp y lo resolvemos sin vueltas."
- "Si preferís, completá el formulario y te preparo una propuesta clara."

Contexto disponible:
{context}

Consulta del usuario:
{question}

Respuesta:
"""

PROMPT = PromptTemplate(
    template=prompt_template,
    input_variables=["context", "question"],
)


# =========================================================
# CARGA SEGURA DE MODELO Y BASE FAISS
# =========================================================
@st.cache_resource(show_spinner=False)
def cargar_modelo():
    return ChatOpenAI(model="gpt-4o-mini", temperature=0.2)


@st.cache_resource(show_spinner=False)
def cargar_faiss():
    try:
        embeddings = OpenAIEmbeddings()
        return FAISS.load_local(
            "faiss_index_interna",
            embeddings,
            allow_dangerous_deserialization=True,
        )
    except Exception:
        return None


llm = cargar_modelo()
db = cargar_faiss()


def responder_sin_faiss(pregunta: str) -> str:
    """Respuesta de respaldo: mantiene la app funcionando aunque no cargue la documentación."""
    return llm.invoke(PROMPT.format(context="", question=pregunta)).content


def responder_con_faiss(pregunta: str) -> tuple[str, bool]:
    """Respuesta con documentación si FAISS está disponible. Si algo falla, vuelve al modo seguro."""
    if db is None:
        return responder_sin_faiss(pregunta), False

    try:
        retriever = db.as_retriever(search_kwargs={"k": 4})

        qa = RetrievalQA.from_chain_type(
            llm=llm,
            retriever=retriever,
            return_source_documents=True,
            chain_type_kwargs={"prompt": PROMPT},
        )

        result = qa.invoke({"query": pregunta})
        respuesta = result.get("result", "").strip()
        source_documents = result.get("source_documents", [])

        if not respuesta:
            return responder_sin_faiss(pregunta), False

        uso_documentacion = bool(source_documents)
        return respuesta, uso_documentacion

    except Exception:
        return responder_sin_faiss(pregunta), False


# =========================================================
# INTERFAZ DE CONSULTA
# =========================================================
st.markdown("---")

query = st.text_input(
    "Hacé tu consulta:",
    placeholder="Ej: Quiero cotizar un seguro para mi auto...",
)

if query:
    with st.spinner("Analizando tu consulta..."):
        respuesta, uso_documentacion = responder_con_faiss(query)

    st.markdown("<div class='cfr-card'>", unsafe_allow_html=True)
    st.subheader("Respuesta")
    st.write(respuesta)

    if uso_documentacion:
        st.caption("Fuente: Documentación cargada y conocimiento profesional del asesor.")
    else:
        st.caption("Fuente: Conocimiento profesional del asesor.")

    st.markdown(
        f"""
        <a class="cfr-whatsapp" href="{WHATSAPP_URL}" target="_blank">
            Escribir por WhatsApp
        </a>
        <div class="cfr-hint">
            También podés completar el formulario de la página para que te contacten.
        </div>
        """,
        unsafe_allow_html=True,
    )
    st.markdown("</div>", unsafe_allow_html=True)

else:
    st.markdown(
        """
        <div class="cfr-card">
            <b>Podés preguntarme por:</b>
            <div class="cfr-small">
                seguro automotor, hogar, comercio, consorcios, accidentes personales,
                documentación para siniestros o qué datos hacen falta para cotizar.
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )
