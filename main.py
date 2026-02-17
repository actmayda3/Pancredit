import os
import base64
import numpy as np
import joblib
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from openai import OpenAI

# Helper to get OpenAI API key (prefer st.secrets, fallback to env)
def get_openai_key():
    try:
        key = st.secrets.get("OPENAI_API_KEY")
    except Exception:
        key = None
    if not key:
        key = os.environ.get("OPENAI_API_KEY")
    return key

# ==========================================
# CONFIG
# ==========================================

# Cabecera fija con logo PanCredit: busca archivos con extensiones comunes o permite subirlo
BASE_LOGO_NAME = "pancredit_logo"

def _encode_image_to_base64_bytes(b: bytes) -> str:
    try:
        return base64.b64encode(b).decode("utf-8")
    except Exception:
        return ""

def _find_logo_on_disk(base_name: str):
    # Buscar variaciones con y sin extensión (comúnmente .png, .jpg, .jpeg)
    exts = ["", ".png", ".jpg", ".jpeg", ".PNG", ".JPG", ".JPEG"]
    for e in exts:
        candidate = base_name + e
        if os.path.exists(candidate):
            return candidate
    return None

img_tag = ""
found = _find_logo_on_disk(BASE_LOGO_NAME)
if found:
    try:
        with open(found, "rb") as f:
            data = f.read()
            img_b64 = _encode_image_to_base64_bytes(data)
            if img_b64:
                # respetar tipo de imagen por extensión
                mime = "image/png"
                if found.lower().endswith(".jpg") or found.lower().endswith(".jpeg"):
                    mime = "image/jpeg"
                img_tag = f'<img src="data:{mime};base64,{img_b64}" style="width:150px;height:200px;max-height:200px;object-fit:contain;margin-right:28px;display:block;"/>'
    except Exception:
        img_tag = ""

# Mostrar la cabecera fija 
st.markdown(
    """
    <style>
        .pc-header{position:fixed;top:0;left:0;width:100%;display:flex;align-items:center;padding:10px 20px;background:rgba(255,255,255,0.98);z-index:9999;box-shadow:0 1px 6px rgba(0,0,0,0.08);}
        .pc-header .title{font-size:44px;font-weight:900;color:#0b2545;margin-left:18px;white-space:nowrap;letter-spacing:-1px;line-height:1.1;}
        .pc-header img{width:150px;height:200px;object-fit:contain;display:block}
        .pc-header .slogan{font-size:10px;font-style:italic;font-weight:400;color:#0b2545;text-align:center;margin-top:8px;margin-bottom:0px;}
        /* Forzar el contenido principal a ocupar todo el ancho */
        .block-container, .main, .css-1d391kg, .css-18e3th9, .css-1outpf7, .css-1v0mbdj, .css-1wrcr25, .css-1oe5cao, .css-1vq4p4l, .css-1dp5vir, .css-1c7y2kd, .css-1kyxreq, .css-1v3fvcr, .css-1b2g4y0, .css-1r6slb0, .css-1q8dd3e, .css-1c7y2kd, .css-1v3fvcr, .css-1b2g4y0, .css-1r6slb0, .css-1q8dd3e {
            max-width: 95vw !important;
            width: 95vw !important;
            margin-left: 2.5vw !important;
            margin-right: 2.5vw !important;
            padding-left: 0 !important;
            padding-right: 0 !important;
        }
    </style>
    <div class="pc-header">
        <div style="display:flex;flex-direction:column;align-items:flex-start;">
            {img}
            <div class="slogan" style="font-size:12px;font-style:italic;font-weight:400;color:#0b2545;text-align:left;margin-top:2px;margin-bottom:0px;line-height:1.3;">
                "Financiamiento inteligente<br>para decisiones inteligentes."
            </div>
        </div>
        <div class="title" style="width:100%;text-align:center;">🏦 Score de Crédito de Originación</div>
        <div style="flex:1"></div>
    </div>
    """.replace("{img}", img_tag),
    unsafe_allow_html=True,
)

# Spacer para que el contenido no quede debajo de la cabecera fija
st.markdown("<div style='height:170px'></div>", unsafe_allow_html=True)

# Mensaje destacado de ayuda bajo la cabecera
st.markdown(
    '<div style="font-size:22px;line-height:1.45;color:#0b2545;background:#f6f8fa;padding:22px 0;border-radius:10px;margin-bottom:18px;max-width:95vw;width:95vw;box-shadow:0 2px 8px #0001;text-align:justify;">'
    '¿Quieres un crédito pero no sabes si calificas? Aquí puedes ingresar tu NIT, revisar tu información, modificarla si es necesario y ver tu score de crédito al instante. Además, te ayudamos con recomendaciones personalizadas para mejorar tu perfil crediticio. <strong>¡Vamos a descubrirlo juntos!</strong>'
    '</div>',
    unsafe_allow_html=True
)

MODEL_PATH = "mejor_modelo_calibrado.pkl"
DATA_PATH = "base_clientes_modelo.csv"

# ==========================================
# AYUDAS
# ==========================================
def safe_get(d: dict, key: str, default):
    """Evita KeyError cuando falte una columna en el CSV."""
    v = d.get(key, default)
    try:
        if pd.isna(v):
            return default
    except Exception:
        pass
    return v

def to_int(x, default=0):
    try:
        return int(x)
    except Exception:
        return int(default)

def to_float(x, default=0.0):
    try:
        return float(x)
    except Exception:
        return float(default)

# ==========================================
# LOAD MODEL
# ==========================================
@st.cache_resource
def load_model():

    if not os.path.exists(MODEL_PATH):
        st.error(f"No se encontró el archivo de modelo en: {MODEL_PATH}")
        return None

    # 🔧 Parche de compatibilidad ANTES de joblib.load
    try:
        from sklearn.compose import _column_transformer

        # Si la versión actual de sklearn no tiene esa clase, la creamos
        if not hasattr(_column_transformer, "_RemainderColsList"):
            class _RemainderColsList(list):
                """Lista de columnas de 'remainder' para modelos antiguos."""
                pass

            _column_transformer._RemainderColsList = _RemainderColsList

    except Exception as e:
        # No es crítico, pero lo avisamos en la interfaz
        st.warning(f"No se pudo aplicar el parche de compatibilidad de sklearn: {e}")

    # Ahora sí, intentamos cargar el modelo
    try:
        model = joblib.load(MODEL_PATH)
        return model
    except Exception as e:
        st.error(f"Error cargando modelo después del parche: {e}")
        return None
    
model = load_model()

# En la interfaz, mostrar el path del modelo cargado (útil para debugging)
#import sys
#st.write("🔧 Python en uso:", sys.executable)

# ==========================================
# CONSULTAR CLIENTE POR NIT
# ==========================================
def consultar_cliente(nit: str):
    if not os.path.exists(DATA_PATH):
        return None

    df = pd.read_csv(DATA_PATH, encoding="latin-1")

    try:
        nit_int = int(nit)
    except Exception:
        return None

    if "NIT" not in df.columns:
        return None

    cliente = df[df["NIT"] == nit_int]
    if cliente.empty:
        return None

    return cliente.iloc[0].to_dict()

# ==========================================
# GUARDAR ACTUALIZACIÓN EN CSV
# ==========================================
def guardar_cliente_actualizado(cliente_actualizado: dict):
    if not os.path.exists(DATA_PATH):
        st.error("No se encontró el archivo base_clientes_modelo.csv")
        return

    df = pd.read_csv(DATA_PATH, encoding="latin-1")

    nit = cliente_actualizado.get("NIT", None)
    if nit is None:
        st.error("No se pudo identificar el NIT del cliente para actualizar.")
        return

    try:
        nit_int = int(nit)
    except Exception:
        st.error("El NIT del cliente actualizado no es válido.")
        return

    if "NIT" not in df.columns:
        st.error("El archivo no contiene la columna NIT.")
        return

    mask = df["NIT"] == nit_int
    if not mask.any():
        st.warning("No se encontró el NIT en el archivo, no se guardó la actualización.")
        return

    idx = df[mask].index[0]

    for k, v in cliente_actualizado.items():
        if k in df.columns:
            # Actualizar PERIODO/Periodo siempre con el valor actual
            if k == "PERIODO":
                if "PERIODO" in df.columns:
                    df.at[idx, "PERIODO"] = cliente_actualizado["PERIODO"]
                elif "Periodo" in df.columns:
                    df.at[idx, "Periodo"] = cliente_actualizado["PERIODO"]
            else:
                df.at[idx, k] = v

    df.to_csv(DATA_PATH, index=False, encoding="latin-1")
    st.success("Actualización guardada.")

# ==========================================
# CALCULAR VARIABLES (ingenieria de variables)
# ==========================================
def calcular_variables(df: pd.DataFrame) -> pd.DataFrame:
    # 1) INGRESO_POR_DEP
    df["INGRESO_POR_DEP"] = df["INGRESO_CONSOLIDADO"] / df["FINODEPEND"].replace(0, np.nan)
    df["INGRESO_POR_DEP"] = df["INGRESO_POR_DEP"].fillna(df["INGRESO_CONSOLIDADO"])

    # 2) RATIO_SALDO_TOTAL
    df["RATIO_SALDO_TOTAL"] = df["SALDO_ACTUAL_ACT"] / df["MONTO_ORIGINAL_ACT"].replace(0, np.nan)
    df["RATIO_SALDO_TOTAL"] = df["RATIO_SALDO_TOTAL"].replace([np.inf, -np.inf], np.nan).fillna(0)

    # 3) TIENE_SERV
    df["TIENE_SERV"] = (df["MONTO_ORIGINAL_ACT_SERV"] > 0).astype(int)

    # 4) ID_UNICO
    for col in ["FISOLICITUDID", "FIIDTIENDA", "FIIDCLIENTE"]:
        if col not in df.columns:
            df[col] = ""
    if "NIT" not in df.columns:
        df["NIT"] = 0
    df["ID_UNICO"] = (
        df["FISOLICITUDID"].astype(str)
        + df["FIIDTIENDA"].astype(str)
        + df["FIIDCLIENTE"].astype(str)
    )
    df["ID_UNICO"] = df["ID_UNICO"].replace("", np.nan).fillna(df["NIT"].astype(str))

    # 5) RATIO_PAGO_TOTAL
    df["RATIO_PAGO_TOTAL"] = df["IMPORTE_PAGO_ACT"] / df["SALDO_ACTUAL_ACT"].replace(0, np.nan)
    df["RATIO_PAGO_TOTAL"] = df["RATIO_PAGO_TOTAL"].replace([np.inf, -np.inf], np.nan).fillna(0)

    return df

# ==========================================
# SCORE
# ==========================================
def score_with_model(model, X: pd.DataFrame) -> dict:
    proba_default = float(model.predict_proba(X)[:, 1][0])

    score = int(round(850 - 550 * proba_default))
    score = max(300, min(850, score))

    if proba_default < 0.10:
        decision = "Aprobación Automática"
        comentario = "Riesgo muy bajo."
    elif proba_default < 0.20:
        decision = "Aprobación Rápida"
        comentario = "Aprobación con monitoreo adicional."
    elif proba_default < 0.30:
        decision = "Revisar manualmente"
        comentario = "Información adicional y visita domiciliaria"
    else:
        if proba_default < 0.40:
            decision = "Rechazar"
            comentario = "Alto riesgo, riesgo significativo."
        else:
            decision = "Rechazar"
            comentario = "Muy Alto riesgo, morosidad crítica."

    return {
        "proba_default": proba_default,
        "score": score,
        "decision": decision,
        "comentario": comentario
    }

# ==========================================
# Velocímetro + PERFILAMIENTO
# ==========================================
def plot_credit_score_gauge(score: int, proba: float):
    fig, ax = plt.subplots(figsize=(3.2, 1.5), subplot_kw={'aspect': 'equal'})
    ax.set_xlim(-1.2, 1.2)
    ax.set_ylim(-0.1, 1.2)
    ax.axis('off')

    min_score, max_score = 300, 850


    sections = [
            {"label": "Muy bajo",  "min": 795, "max": 850, "color": "#2CA02C"},
            {"label": "Bajo",      "min": 740, "max": 795, "color": "#A1BF7F"},
            {"label": "Moderado",  "min": 685, "max": 740, "color": "#F2FF00"},
            {"label": "Alto",      "min": 600, "max": 685, "color": "#FFA500"},
            {"label": "Muy alto",  "min": 300, "max": 600, "color": "#E65100"},
        ]

    for sec in sections:
        start_angle = (sec["min"] - min_score) / (max_score - min_score) * 180
        end_angle   = (sec["max"] - min_score) / (max_score - min_score) * 180

        wedge = patches.Wedge(
            center=(0, 0),
            r=1,
            theta1=180 - end_angle,
            theta2=180 - start_angle,
            facecolor=sec["color"],
            edgecolor="white"
        )
        ax.add_patch(wedge)

        mid_angle = 180 - (start_angle + end_angle) / 2
        x_label = 0.62 * np.cos(np.radians(mid_angle))
        y_label = 0.62 * np.sin(np.radians(mid_angle))
        ax.text(x_label, y_label, sec["label"], ha="center", va="center", fontsize=4)

    angle = (score - min_score) / (max_score - min_score) * 180
    x = 0.9 * np.cos(np.radians(180 - angle))
    y = 0.9 * np.sin(np.radians(180 - angle))
    ax.annotate('', xy=(x, y), xytext=(0, 0),
                arrowprops=dict(facecolor="#1126AA", edgecolor="#1126AA",
                                width=2, headwidth=8))
    ax.plot(0, 0, 'ko', markersize=8)

    # Determinar color del título según la franja donde cae el score
    title_color = "#000000"
    for sec in sections:
        if score >= sec["min"] and score <= sec["max"]:
            title_color = sec["color"]
            break

    # Título más pequeño y compacto
    ax.set_title(
        f"Score: {score}  |  Prob: {proba*100:.1f}%",
        fontsize=7,
        color=title_color,
        fontweight="bold",
        pad=2
    )
    # Centrar la gráfica en toda la pantalla
    st.markdown('<div style="display:flex;justify-content:center;align-items:center;width:100vw;">', unsafe_allow_html=True)
    st.pyplot(fig, use_container_width=True)
    st.markdown('</div>', unsafe_allow_html=True)
    plt.close(fig)

def colored_box(text: str, bg: str, text_color: str = "#ffffff"):
    st.markdown(
        f'<div style="background-color:{bg};padding:10px;border-radius:6px;color:{text_color};font-weight:700;">{text}</div>',
        unsafe_allow_html=True,
    )

# ==========================================
# PERFILAMIENTO
# ==========================================

def mostrar_perfilamiento(score: int, proba: float):
    st.subheader("🔍 Perfilamiento Crediticio")

    # Muy bajo: [0.00, 0.10)
    if proba < 0.10:
        colored_box("💎 <strong>Cliente Premium (Muy bajo riesgo)</strong>", bg="#065406", text_color="#afe82b")
        st.markdown("""
        - **Probabilidad de incumplimiento:** Menor a 10%  
        - **Descripción:** Excelente perfil. Accede a los mejores productos financieros sin restricciones.  
        - **Productos recomendados:**  
          - 🟢 Tarjeta de crédito  
          - 🟢 Préstamo Personal  
          - 🟢 Prestamo de Servicios
          - 🤝 **Estamos aquí para acompañarte. Hablemos para definir juntos las mejores condiciones para tu crédito.**
        """)

    # Bajo: [0.10, 0.20)
    elif proba < 0.20:
        colored_box("✅ <strong>Cliente Confiable (Bajo riesgo)</strong>", bg="#095219", text_color="#6E8A56")
        st.markdown("""
        - **Probabilidad de incumplimiento:** Entre 10% y 20%  
        - **Descripción:** Buen perfil. Elegible para productos estándar, con monitoreo adicional.  
        - **Productos recomendados:**  
          - 🟢 Préstamo Personal
          - 🟢 Prestamo de Servicios 
          - 🟡 Tarjeta de crédito sujeta a límite bajo.
          - 🌟 **Nos entusiasma apoyarte. Conversemos para encontrar el plan que mejor se ajuste a lo que necesitas ahora.**
        """)

    # Moderado: [0.20, 0.30)
    elif proba < 0.30:
        colored_box("🟡 <strong>Cliente en Revisión (Riesgo moderado)</strong>", bg="#F2FF00", text_color="#818806")
        st.markdown("""
        - **Probabilidad de incumplimiento:** Entre 20% y 30%  
        - **Descripción:** Perfil con mucho potencial, requiere revisión manual.  
        - **Acciones sugeridas:**  
          - 🟢 Prestamo de Servicios 
          - 🟡 Préstamo Personal condicionado a revisión manual
          - 🟢 Ajustar monto o plazo  
          - 🟢 Considerar garantías o aval  
          - ✨ **¡Queremos que lo logres! Un asesor te ayudará a ajustar las piezas para que tu crédito sea una realidad.**
        """)

    # Alto y muy alto: [0.30, 1.00]
    else:
        colored_box("🔴 <strong>Cliente No Elegible (Riesgo alto/muy alto)</strong>", bg="#913331", text_color="#EC100D")
        st.markdown("""
        - **Probabilidad de incumplimiento:** Mas del 30% 
        - **Descripción:** Riesgo inaceptable por el momento, muy baja probabilidad de recuperación.  
        - **Acciones sugeridas:**  
          - 🔴 No recomendable adquirir nuevos créditos en este momento  
          - 🔴 Reestructurar deudas actuales  
          - 🔴 Mantener pagos puntuales y constantes  
          - 🔴 Fortalecer estabilidad laboral e ingresos 
          - 💡 **¡No te desanimes! Trabajemos juntos hoy para que tu próximo resultado sea un "SÍ"**
          - 🤝 **¿No sabes por dónde empezar?, No te preocupes, estamos para guiarte**
          - 📞 **Contáctanos para una asesoría personalizada y diseñemos juntos un plan que te llevará a tu próximo crédito.**
        """) 

# ==========================================
# OPENAI: RESUMEN + RECOMENDACIONES
# ==========================================
def explain_with_openai(df_modelo: pd.DataFrame, result: dict) -> str:
    """
    Usa la API de OpenAI para devolver:
    1) Resumen en lenguaje sencillo
    2) Recomendaciones para mejorar su perfil créditicio
    """
    key = get_openai_key()
    if not key:
        return "No se encontró OPENAI_API_KEY. Configúrala en .streamlit/secrets.toml o como variable de entorno."

    client = OpenAI(api_key=key)
    row = df_modelo.iloc[0]

    prompt = f"""
Eres un asesor financiero de un banco.

Explica el resultado del score de crédito y la decisión en lenguaje sencillo.
No uses términos técnicos ni menciones modelos internos.
No hables de regulaciones ni políticas.
Se amable, demustrando empatía por la situación del cliente, se cortes.
Que el cliente entienda claramente su situación actual y qué puede hacer para mejorar su perfil crediticio.
Que cualquier duda contacte al banco para recibir asesoría personalizada.


Datos del cliente:
- Edad: {row.get('EDAD', 'N/D')}
- Sexo: {row.get('FISEXO', 'N/D')}
- Número de dependientes: {row.get('FINODEPEND', 'N/D')}
- Antigüedad laboral (meses): {row.get('ANT_LAB_MESES', 'N/D')}
- Ingreso mensual: {row.get('FNINGMES', 'N/D')}
- Ingreso deudor adicional: {row.get('FNINGDEUMES', 'N/D')}
- Ingreso consolidado: {row.get('INGRESO_CONSOLIDADO', 'N/D')}
- ID Plazo: {row.get('IDPLAZO', 'N/D')}
- ID Tipo Vivienda: {row.get('IDTIPOVIVIENDA', 'N/D')}
- Código postal del domicilio (IDCCP): {row.get('IDCCP', 'N/D')}
- Región geográfica o comercial (IDREGION): {row.get('IDREGION', 'N/D')} (1=Norte, 2=Sur)
- ID AceptApp: {row.get('IDACEPTAPP', 'N/D')}
- Saldo actual en cuentas activas: {row.get('SALDO_ACTUAL_ACT', 'N/D')}
- Monto original de las cuentas activas: {row.get('MONTO_ORIGINAL_ACT', 'N/D')}
- Importe del pago mensual actual: {row.get('IMPORTE_PAGO_ACT', 'N/D')}
- Monto original de servicios: {row.get('MONTO_ORIGINAL_ACT_SERV', 'N/D')}
- Periodo de análisis (PERIODO): {row.get('PERIODO', 'N/D')}
- Semana de solicitud (SEMANA): {row.get('SEMANA', 'N/D')}


Resultado:
- Puntuación: {result['score']}
- Probabilidad de incumplimiento: {result['proba_default']:.3f}
- Decisión: {result['decision']}
- Comentario: {result['comentario']}

Devuelve exactamente:

1) Resumen en lenguaje sencillo
2) Recomendaciones para mejorar su perfil crediticio
"""

    response = client.responses.create(model="gpt-5.2", input=prompt)

    return response.output_text.strip()

# ==========================================
# UI PRINCIPAL
# ==========================================

# El título principal se muestra en la cabecera fija; dejamos un pequeño espacio
st.markdown("<div style='height:6px'></div>", unsafe_allow_html=True)

if model is None:
    st.warning("No encontré model/score_pipeline.pkl. El cálculo real no funcionará hasta que esté disponible.")

st.subheader("🔎 Buscar Cliente")
nit_input = st.text_input("Ingrese Número de Identificación Tributaria (NIT) a 7 dígitos")

if st.button("Buscar"):
    # Limpiar estado previo antes de cargar nuevo cliente
    st.session_state.cliente = None
    st.session_state.cliente_encontrado = False
    cliente = consultar_cliente(nit_input)
    if cliente:
        st.session_state.cliente = cliente
        st.session_state.cliente_encontrado = True
        st.success("Cliente cargado correctamente.")
    else:
        st.error("Cliente no encontrado o NIT inválido.")

# ==========================================
# FORMULARIO
# ==========================================
if (
    "cliente" in st.session_state
    and st.session_state.cliente is not None
    and st.session_state.get("cliente_encontrado", False)
):
    cliente = st.session_state.cliente

    # Defaults seguros (por si falta columna en CSV)
    default_EDAD           = to_int  (safe_get(cliente, "EDAD", 30), 30)
    default_FISEXO         = to_int  (safe_get(cliente, "FISEXO", 1), 1)
    default_FINODEPEND     = to_int  (safe_get(cliente, "FINODEPEND", 0), 0)
    default_ANT_LAB_MESES  = to_int  (safe_get(cliente, "ANT_LAB_MESES", 12), 12)
    default_FNINGMES       = to_float(safe_get(cliente, "FNINGMES", 25000.0), 25000.0)
    default_FNINGDEUMES    = to_float(safe_get(cliente, "FNINGDEUMES", 0.0), 0.0)
    default_IDPLAZO        = to_int  (safe_get(cliente, "IDPLAZO", 12), 12)

    default_IDTIPOVIVIENDA = to_int  (safe_get(cliente, "IDTIPOVIVIENDA", 1), 1)
    default_IDACEPTAPP     = to_int  (safe_get(cliente, "IDACEPTAPP", 1), 1)
    default_IDREGION = to_int(safe_get(cliente, "IDREGION", 1), 1)
    default_IDCCP = to_int(safe_get(cliente, "IDCCP", 0), 0)

    default_INGRESO_CONSOLIDADO      = to_float(safe_get(cliente, "INGRESO_CONSOLIDADO", default_FNINGMES), default_FNINGMES)
    default_SALDO_ACTUAL_ACT         = to_float(safe_get(cliente, "SALDO_ACTUAL_ACT", 0.0), 0.0)
    default_MONTO_ORIGINAL_ACT       = to_float(safe_get(cliente, "MONTO_ORIGINAL_ACT", 0.0), 0.0)
    default_IMPORTE_PAGO_ACT         = to_float(safe_get(cliente, "IMPORTE_PAGO_ACT", 0.0), 0.0)
    default_MONTO_ORIGINAL_ACT_SERV  = to_float(safe_get(cliente, "MONTO_ORIGINAL_ACT_SERV", 0.0), 0.0)

    st.subheader("🧾 Formulario")

    with st.form("form_edicion"):
        # Mostrar logo de la empresa solo si existe, sin uploader
        logo_path = "pancredit_logo.png"
        if os.path.exists(logo_path):
            st.image(logo_path, width=220)
            st.markdown('<div style="text-align:center;font-size:18px;color:#0b2545;font-weight:500;margin-top:8px;">Financiamiento inteligente para decisiones inteligentes.</div>', unsafe_allow_html=True)
        c1, c2, c3 = st.columns(3)

        # Se carga en el formulario
        with c1:
            st.markdown("### 👤 Datos personales")
            FECHA_SOLICITUD = st.date_input("Fecha de solicitud", key="fecha_solicitud")
            EDAD = st.number_input("Edad (EDAD)", min_value=23, max_value=83, value=int(default_EDAD))
            sexo_dict = {1: "Masculino", 2: "Femenino"}
            sexo_labels = [sexo_dict[k] for k in sorted(sexo_dict)]
            sexo_values = [k for k in sorted(sexo_dict)]
            selected_sexo = st.selectbox("Sexo (FISEXO)", sexo_labels, index=sexo_values.index(default_FISEXO))
            FISEXO = sexo_values[sexo_labels.index(selected_sexo)]
            FINODEPEND = st.number_input("Número de dependientes (FINODEPEND)", min_value=0, max_value=15, value=int(default_FINODEPEND))
            ANT_LAB_MESES = st.number_input("Antigüedad laboral en meses (ANT_LAB_MESES)", min_value=5, max_value=65, value=int(default_ANT_LAB_MESES))


        with c2:
            st.markdown("### 📈 Ingresos")
            FNINGMES = st.number_input("$ Ingreso mensual (FNINGMES)", min_value=0.0, value=float(default_FNINGMES), step=100.0, format="%.2f")
            FNINGDEUMES = st.number_input("$ Ingreso deudor adicional (FNINGDEUMES)", min_value=0.0, value=float(default_FNINGDEUMES), step=100.0, format="%.2f")
            INGRESO_CONSOLIDADO = st.number_input("$ Ingreso consolidado (INGRESO_CONSOLIDADO)", min_value=0.0, value=float(default_INGRESO_CONSOLIDADO), step=100.0, format="%.2f")

            st.markdown("### 🏦 Solicitud")
            plazo_opts = [0, 12, 18, 24]
            if default_IDPLAZO not in plazo_opts:
                plazo_opts = [default_IDPLAZO] + plazo_opts
            IDPLAZO = st.selectbox("Plazo (IDPLAZO)", plazo_opts, index=plazo_opts.index(default_IDPLAZO))


        with c3:
            st.markdown("### 🏠 Vivienda / 📱 App")
            vivienda_dict = {
                0: "RENTADA",
                1: "PROPIA",
                2: "PRESTADA/OTRO FAMILIAR"
            }
            vivienda_labels = [vivienda_dict[k] for k in sorted(vivienda_dict)]
            vivienda_values = [k for k in sorted(vivienda_dict)]
            selected_vivienda = st.selectbox("Tipo de vivienda (IDTIPOVIVIENDA)", vivienda_labels, index=vivienda_values.index(default_IDTIPOVIVIENDA))
            IDTIPOVIVIENDA = vivienda_values[vivienda_labels.index(selected_vivienda)]
            region_dict = {1: "Norte", 2: "Sur"}
            region_labels = [region_dict[k] for k in sorted(region_dict)]
            region_values = [k for k in sorted(region_dict)]
            # Usar clave única por cliente para evitar que Streamlit conserve el valor anterior
            region_key = f"idregion_{cliente.get('NIT', '')}"
            try:
                default_region_index = region_values.index(default_IDREGION)
            except ValueError:
                default_region_index = 0
            selected_region = st.selectbox("Región geográfica o comercial (IDREGION)", region_labels, index=default_region_index, key=region_key)
            IDREGION = region_values[region_labels.index(selected_region)]
            idccp_dict = {
                0: "0", 1: "200", 2: "201", 3: "400", 4: "401", 5: "600", 6: "601", 7: "602", 8: "800", 9: "801", 10: "1000", 11: "1001", 12: "1400", 13: "2000", 14: "2001", 15: "2200", 16: "2201", 17: "2400", 18: "2401", 19: "2402", 20: "2600", 21: "2601", 22: "2800", 23: "3000", 24: "3001", 25: "4200", 26: "4800", 27: "4801", 28: "5000", 29: "5001", 30: "5002", 31: "5200", 32: "5201", 33: "5202", 34: "5400", 35: "5401", 36: "5800", 37: "5801", 38: "6000", 39: "6001", 40: "6002", 41: "6600", 42: "6601", 43: "6603", 44: "6606", 45: "6800", 46: "6801", 47: "7000", 48: "7001", 49: "7200", 50: "7600", 51: "7601", 52: "7602", 53: "7603", 54: "8000", 55: "8001", 56: "8002", 57: "8202", 58: "8800", 59: "8801", 60: "9200", 61: "9201", 62: "10000", 63: "10001", 64: "10200", 65: "10201", 66: "10400", 67: "10401", 68: "10402", 69: "10403", 70: "10404", 71: "10600", 72: "10601", 73: "11000", 74: "11001", 75: "11200", 76: "11201", 77: "11600", 78: "11801", 79: "12400", 80: "13200", 81: "13201", 82: "13600", 83: "13601", 84: "14200", 85: "14201", 86: "14400", 87: "14401", 88: "14402", 89: "14600", 90: "14601", 91: "14602"
            }

            # Mostrar solo los códigos postales reales
            idccp_labels = list(idccp_dict.values())
            idccp_values = [int(v) for v in idccp_dict.values()]
            idccp_key = f"idccp_{cliente.get('NIT', '')}"
            try:
                default_idccp_index = idccp_values.index(default_IDCCP)
            except ValueError:
                default_idccp_index = 0
            selected_idccp = st.selectbox("Código postal del domicilio (IDCCP)", idccp_labels, index=default_idccp_index, key=idccp_key)
            IDCCP = int(selected_idccp)

            # Acepta APP
            aceptaapp_dict = {1: "Sí", 0: "No"}
            aceptaapp_labels = [aceptaapp_dict[k] for k in sorted(aceptaapp_dict, reverse=True)]
            aceptaapp_values = [k for k in sorted(aceptaapp_dict, reverse=True)]
            selected_aceptaapp = st.selectbox("Acepta APP (IDACEPTAPP)", aceptaapp_labels, index=aceptaapp_values.index(default_IDACEPTAPP))
            IDACEPTAPP = aceptaapp_values[aceptaapp_labels.index(selected_aceptaapp)]

            st.markdown("### 📊 Buró")
            SALDO_ACTUAL_ACT = st.number_input("$ Saldo actual en cuentas activas (SALDO_ACTUAL_ACT)", value=float(default_SALDO_ACTUAL_ACT), step=100.0, format="%.2f")
            MONTO_ORIGINAL_ACT = st.number_input("$ Monto original de cuentas activas (MONTO_ORIGINAL_ACT)", value=float(default_MONTO_ORIGINAL_ACT), step=100.0, format="%.2f")
            IMPORTE_PAGO_ACT = st.number_input("$ Importe del pago mensual actual (IMPORTE_PAGO_ACT)", value=float(default_IMPORTE_PAGO_ACT), step=50.0, format="%.2f")
            MONTO_ORIGINAL_ACT_SERV = st.number_input("$ Monto original de servicios (MONTO_ORIGINAL_ACT_SERV)", value=float(default_MONTO_ORIGINAL_ACT_SERV), step=100.0, format="%.2f")

        submitted = st.form_submit_button("Guardar y Calcular")

    # ==========================================
    # PROCESAR + GUARDAR + SCORE + PERFIL + IA
    # ==========================================
    if submitted:
        import datetime
        cliente_actualizado = dict(cliente)

        # Calcular PERIODO y SEMANA a partir de la fecha de solicitud
        if isinstance(FECHA_SOLICITUD, datetime.date):
            PERIODO = FECHA_SOLICITUD.strftime("%Y%m")
            SEMANA = f"{FECHA_SOLICITUD.isocalendar()[0]}{FECHA_SOLICITUD.isocalendar()[1]:02d}"
        else:
            PERIODO = ""
            SEMANA = ""

        # Modificables
        cliente_actualizado["EDAD"]           = EDAD
        cliente_actualizado["FISEXO"]         = FISEXO
        cliente_actualizado["FINODEPEND"]     = FINODEPEND
        cliente_actualizado["ANT_LAB_MESES"]  = ANT_LAB_MESES
        cliente_actualizado["FNINGMES"]       = FNINGMES
        cliente_actualizado["FNINGDEUMES"]    = FNINGDEUMES
        cliente_actualizado["IDPLAZO"]        = IDPLAZO
        cliente_actualizado["IDTIPOVIVIENDA"] = IDTIPOVIVIENDA
        cliente_actualizado["IDACEPTAPP"]     = IDACEPTAPP
        cliente_actualizado["IDREGION"]       = IDREGION
        cliente_actualizado["IDCCP"]          = IDCCP
        # Nuevas variables calculadas (asegurar mayúsculas)
        cliente_actualizado["PERIODO"] = PERIODO
        cliente_actualizado["SEMANA"] = SEMANA
        cliente_actualizado["INGRESO_CONSOLIDADO"]      = INGRESO_CONSOLIDADO
        cliente_actualizado["SALDO_ACTUAL_ACT"]         = SALDO_ACTUAL_ACT
        cliente_actualizado["MONTO_ORIGINAL_ACT"]       = MONTO_ORIGINAL_ACT
        cliente_actualizado["IMPORTE_PAGO_ACT"]         = IMPORTE_PAGO_ACT
        cliente_actualizado["MONTO_ORIGINAL_ACT_SERV"]  = MONTO_ORIGINAL_ACT_SERV
        cliente_actualizado["NIT"] = to_int(safe_get(cliente_actualizado, "NIT", nit_input), 0)

        # Guardar cambios en CSV (incluyendo PERIODO actualizado)
        guardar_cliente_actualizado(cliente_actualizado)

        # Dataset final
        df_modelo = pd.DataFrame([cliente_actualizado])
        df_modelo = calcular_variables(df_modelo)
        # Asegurar columna 'Periodo' para el modelo, usando el valor calculado
        df_modelo["Periodo"] = PERIODO
        # Eliminar PERIODO si existe, para evitar duplicidad
        if "PERIODO" in df_modelo.columns:
            df_modelo.drop(columns=["PERIODO"], inplace=True)

        # Alinear columnas al modelo, pero sin sobrescribir Periodo
        if model is not None and hasattr(model, "feature_names_in_"):
            cols = list(model.feature_names_in_)
            if "Periodo" not in cols:
                cols.append("Periodo")
            df_modelo = df_modelo.reindex(columns=cols, fill_value=0)

        # Ejecutar modelo
        if model is None:
            st.warning(f"No se pudo cargar el modelo desde: {MODEL_PATH}. El cálculo real no funcionará hasta que esté disponible.")
        else:
            result = score_with_model(model, df_modelo)

            # 8️⃣ Mostrar resultados
            st.subheader("📊 Resultado")
            cA, cB, cC = st.columns(3)

            cA.metric("Puntaje", result["score"])
            cB.metric("Probabilidad incumplimiento", f"{result['proba_default']*100:.1f}%")
            cC.metric("Decisión", result["decision"])
            st.write(f"**Comentario:** {result['comentario']}")

            # Mostrar score y probabilidad en la gráfica y perfilamiento solo una vez
            plot_credit_score_gauge(result["score"], result["proba_default"])
            mostrar_perfilamiento(result["score"], result["proba_default"])

            st.subheader("📝 Explicación en lenguaje sencillo")

            if get_openai_key():
                with st.spinner("Generando explicación personalizada..."):
                    try:
                        texto = explain_with_openai(df_modelo, result)
                    except Exception as e:
                        texto = f"Error generando explicación: {e}"

                # Asegurar que mostramos texto plano y evitar inyectar HTML/DOM
                try:
                    import html as _html
                    if not isinstance(texto, str):
                        texto = str(texto)
                    texto_safe = _html.escape(texto)
                    # Mostrar en un área de texto para evitar que Streamlit intente interpretar HTML
                    st.text_area("Explicación", value=texto_safe, height=260)
                except Exception:
                    # Fallback simple
                    st.write(texto)
            else:
                st.info("Configura la clave en .streamlit/secrets.toml (OPENAI_API_KEY) o como variable de entorno OPENAI_API_KEY para habilitar la explicación con IA.")

            with st.expander("Ver dataset final enviado al modelo"):
                st.dataframe(df_modelo)
