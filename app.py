import streamlit as st
import json
import cv2
import numpy as np
from PIL import Image
import firebase_admin
from firebase_admin import credentials, firestore
import datetime
import io
import zipfile
from streamlit_drawable_canvas import st_canvas

# ==========================================
# 1. CONFIGURACIÓN DE FIREBASE
# ==========================================
if not firebase_admin._apps:
    key_dict = json.loads(st.secrets["text_key"])
    cred = credentials.Certificate(key_dict)
    firebase_admin.initialize_app(cred)

db = firestore.client()

# ==========================================
# 2. LÓGICA DE OPENCV (HSV) EXACTA A DESKTOP
# ==========================================
S_MIN = 40
V_MIN = 40
TISSUE_S_MIN = 10
TISSUE_V_MIN = 30

H_RANGES = {
    "rojo":    [(0, 15), (170, 179)],
    "naranja": [(10, 20)],
    "amarillo":[(20, 30)],
    "verde":   [(30, 60)],
}
COLOR_ORDER = ["rojo", "naranja", "amarillo", "verde"]

def mask_hsv_color(img_hsv, color_key, s_min=S_MIN, v_min=V_MIN):
    h, s, v = cv2.split(img_hsv)
    mask_total = np.zeros_like(h, dtype=np.uint8)
    for (hmin, hmax) in H_RANGES[color_key]:
        mh = cv2.inRange(h, hmin, hmax)
        ms = cv2.inRange(s, s_min, 255)
        mv = cv2.inRange(v, v_min, 255)
        m  = cv2.bitwise_and(mh, cv2.bitwise_and(ms, mv))
        mask_total = cv2.bitwise_or(mask_total, m)
    return mask_total

def tissue_mask(img_hsv, s_min=TISSUE_S_MIN, v_min=TISSUE_V_MIN):
    _, s, v = cv2.split(img_hsv)
    return cv2.bitwise_and(cv2.inRange(s, s_min, 255), cv2.inRange(v, v_min, 255))

def compute_exclusive_color_masks_and_stats(img_hsv, roi_mask=None):
    den = tissue_mask(img_hsv, TISSUE_S_MIN, TISSUE_V_MIN)
    if roi_mask is not None:
        den = cv2.bitwise_and(den, roi_mask)
        
    total = int(cv2.countNonZero(den))
    assigned = np.zeros_like(den, dtype=np.uint8)
    masks_exclusive = {}
    counts = {}

    if total == 0:
        for c in COLOR_ORDER:
            masks_exclusive[c] = np.zeros_like(den, dtype=np.uint8)
            counts[c] = 0
        masks_exclusive["no_clasif"] = np.zeros_like(den, dtype=np.uint8)
        counts["no_clasif"] = 0
        return masks_exclusive, counts, total

    for c in COLOR_ORDER:
        m = mask_hsv_color(img_hsv, c, S_MIN, V_MIN)
        m = cv2.bitwise_and(m, den)
        m = cv2.bitwise_and(m, cv2.bitwise_not(assigned))
        masks_exclusive[c] = m
        counts[c] = int(cv2.countNonZero(m))
        assigned = cv2.bitwise_or(assigned, m)

    no_clasif = cv2.bitwise_and(den, cv2.bitwise_not(assigned))
    masks_exclusive["no_clasif"] = no_clasif
    counts["no_clasif"] = int(cv2.countNonZero(no_clasif))
    return masks_exclusive, counts, total

# ==========================================
# 3. INTERFAZ DE LOGIN
# ==========================================
def login():
    st.set_page_config(page_title="Acceso ChromaQuant", page_icon="🔐")
    st.title("ChromaQuant Web 🔬")
    email = st.text_input("Correo electrónico registrado")
    
    if st.button("Verificar Acceso"):
        if email:
            query = db.collection('usuarios_permitidos').where('correo', '==', email.strip()).stream()
            usuario_encontrado = any(True for _ in query)
                
            if usuario_encontrado:
                st.session_state['usuario'] = email.strip()
                db.collection('accesos').add({
                    'correo': email.strip(), 'fecha_hora': datetime.datetime.now(), 'evento': 'Inicio de sesión'
                })
                st.rerun()
            else:
                st.error("Acceso denegado. Correo no registrado.")

# ==========================================
# 4. APP PRINCIPAL (ROIS Y DESCARGAS)
# ==========================================
def main_app():
    st.set_page_config(page_title="ChromaQuant", page_icon="🔬", layout="wide")
    
    st.sidebar.success(f"👤 Usuario: {st.session_state['usuario']}")
    if st.sidebar.button("Cerrar Sesión"):
        st.session_state['usuario'] = None
        st.rerun()
        
    st.title("Analizador de Imágenes Picrosirius (ChromaQuant)")
    
    archivo = st.file_uploader("Sube tu imagen médica (JPG, PNG, TIF)", type=['jpg', 'jpeg', 'png', 'tif'])
    
    if archivo:
        file_bytes = np.asarray(bytearray(archivo.read()), dtype=np.uint8)
        img_bgr_original = cv2.imdecode(file_bytes, 1)
        h_orig, w_orig = img_bgr_original.shape[:2]
        
        # Imagen reducida SOLO para visualización en web
        max_width = 700
        scale = min(max_width / w_orig, 1.0)
        nw, nh = int(w_orig * scale), int(h_orig * scale)
        
        img_bgr_web = cv2.resize(img_bgr_original, (nw, nh))
        img_rgb_web = cv2.cvtColor(img_bgr_web, cv2.COLOR_BGR2RGB)
        img_para_lienzo = Image.fromarray(img_rgb_web).convert("RGBA")
        
        st.sidebar.markdown("---")
        st.sidebar.subheader("Herramientas ROI")
        opciones_dibujo = {
            "rect": "⬜ Rectángulo",
            "polygon": "🛑 Polígono",
            "freedraw": "✏️ Lápiz libre",
            "transform": "🖐️ Mover / Zoom (Rueda)"
        }
        modo_dibujo = st.sidebar.radio("Tipo de selección:", options=list(opciones_dibujo.keys()), format_func=lambda x: opciones_dibujo[x])
        color_ver = st.sidebar.selectbox("Filtro de visualización:", ["rojo", "naranja", "amarillo", "verde", "todo"])

        col1, col2 = st.columns(2)
        
        with col1:
            st.write("**1. Dibuja sobre la imagen original**")
            canvas_result = st_canvas(
                fill_color="rgba(34, 211, 238, 0.3)",
                stroke_width=2,
                stroke_color="#22d3ee",
                background_image=img_para_lienzo,
                update_streamlit=True,
                height=nh,
                width=nw,
                drawing_mode=modo_dibujo,
                key="canvas",
            )
            
            if st.button("Limpiar Dibujo"):
                st.rerun()

        # 1. Obtener la máscara de la web y ESCALARLA a resolución original
        roi_mask_orig = None
        if canvas_result.image_data is not None:
            alfa = canvas_result.image_data[:, :, 3]
            if np.any(alfa > 0):
                roi_mask_web = np.where(alfa > 0, 255, 0).astype(np.uint8)
                # Escalar al tamaño original para hacer cálculos reales (INTER_NEAREST mantiene los bordes perfectos)
                roi_mask_orig = cv2.resize(roi_mask_web, (w_orig, h_orig), interpolation=cv2.INTER_NEAREST)

        # 2. Hacer las matemáticas sobre la imagen ORIGINAL (100% preciso)
        with st.spinner("Calculando píxeles en resolución original..."):
            img_hsv_orig = cv2.cvtColor(img_bgr_original, cv2.COLOR_BGR2HSV)
            masks_orig, counts, total = compute_exclusive_color_masks_and_stats(img_hsv_orig, roi_mask_orig)

        # 3. Mostrar resultado visual (bajando la escala solo de la máscara solicitada)
        with col2:
            st.write(f"**2. Resultado (Filtro: {color_ver})**")
            if color_ver == "todo":
                st.image(img_rgb_web, use_column_width=True)
            else:
                mask_show_orig = masks_orig.get(color_ver)
                # Reducir máscara solo para mostrarla en la pantalla
                mask_show_web = cv2.resize(mask_show_orig, (nw, nh), interpolation=cv2.INTER_NEAREST)
                proc_bgr_web = cv2.bitwise_and(img_bgr_web, img_bgr_web, mask=mask_show_web)
                st.image(cv2.cvtColor(proc_bgr_web, cv2.COLOR_BGR2RGB), use_column_width=True)

        # 4. Mostrar métricas (Basadas en la original)
        st.markdown("### Métricas de Segmentación (ROI actual)")
        if total > 0:
            p_r = (counts['rojo']/total)*100
            p_n = (counts['naranja']/total)*100
            p_a = (counts['amarillo']/total)*100
            p_v = (counts['verde']/total)*100
            p_nc = (counts['no_clasif']/total)*100
            
            colored_sum = counts['rojo'] + counts['naranja'] + counts['amarillo'] + counts['verde']
            colored_pct = (colored_sum / total) * 100
            
            st.write(f"**Px coloreados:** {colored_sum:,} ({colored_pct:.2f}%) | **Tejido Total:** {total:,} px")
            
            m1, m2, m3, m4, m5 = st.columns(5)
            m1.metric("Rojo", f"{p_r:.2f}%", f"{counts['rojo']:,} px")
            m2.metric("Naranja", f"{p_n:.2f}%", f"{counts['naranja']:,} px")
            m3.metric("Amarillo", f"{p_a:.2f}%", f"{counts['amarillo']:,} px")
            m4.metric("Verde", f"{p_v:.2f}%", f"{counts['verde']:,} px")
            m5.metric("No Clasif.", f"{p_nc:.2f}%", f"{counts['no_clasif']:,} px")
            
            # --- GENERAR ARCHIVO ZIP CON RECORTE (BOUNDING BOX) ---
            buffer_zip = io.BytesIO()
            with zipfile.ZipFile(buffer_zip, "w", zipfile.ZIP_DEFLATED) as zip_file:
                
                # Calcular el bounding box si hay un ROI dibujado
                bbox = None
                if roi_mask_orig is not None:
                    nz = cv2.findNonZero(roi_mask_orig)
                    if nz is not None:
                        x, y, w, h = cv2.boundingRect(nz)
                        bbox = (x, y, w, h)
                
                # Guardar imágenes a resolución COMPLETA y Recortadas
                for c in COLOR_ORDER:
                    m_c = masks_orig.get(c)
                    img_final = cv2.bitwise_and(img_bgr_original, img_bgr_original, mask=m_c)
                    
                    if bbox is not None:
                        x, y, w, h = bbox
                        if w > 0 and h > 0:
                            img_final = img_final[y:y+h, x:x+w].copy()
                            
                    _, buf = cv2.imencode(".png", img_final)
                    zip_file.writestr(f"resultado_{c}.png", buf.tobytes())
                
                txt_content = f"""RESUMEN CHROMAQUANT WEB
Porcentaje de Pixeles:
Px coloreados: {colored_sum} / {total} ({colored_pct:.2f}%)
Rojo: {counts['rojo']} ({p_r:.2f}%)
Naranja: {counts['naranja']} ({p_n:.2f}%)
Amarillo: {counts['amarillo']} ({p_a:.2f}%)
Verde: {counts['verde']} ({p_v:.2f}%)
No clasificados: {counts['no_clasif']} ({p_nc:.2f}%)

Segmentación HSV:
- Rojo (0-15) u (170-179)
- Naranja (10-20)
- Amarillo (20-30)
- Verde (30-60)
S_MIN={S_MIN}, V_MIN={V_MIN}; Tejido: S>={TISSUE_S_MIN}, V>={TISSUE_V_MIN}
"""
                zip_file.writestr("resumen.txt", txt_content)
            
            st.download_button(
                label="📥 Descargar Resultados y Resumen (.ZIP)",
                data=buffer_zip.getvalue(),
                file_name="Lote_ChromaQuant.zip",
                mime="application/zip",
                type="primary"
            )
        else:
            st.warning("No hay tejido detectado o no has dibujado un ROI válido.")

# CONTROL
if 'usuario' not in st.session_state: st.session_state['usuario'] = None
if st.session_state['usuario'] is None: login()
else: main_app()
