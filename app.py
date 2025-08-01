# IMPORTS ÚNICOS
import base64, cv2, mediapipe as mp, numpy as np, tempfile
from pathlib import Path
import dash, dash_bootstrap_components as dbc
from dash import dcc, html, Input, Output, State
from flask import Flask
import zipfile, io, pandas as pd, os, plotly.express as px
from dash import no_update

# report_utils
from report_utils import build_report, interpret_metrics


# —————————————————————————————
# 1) Configuración inicial
# —————————————————————————————
TMP_DIR = Path(tempfile.gettempdir()) / "ohs_tmp"
TMP_DIR.mkdir(exist_ok=True)

ALLOWED = {".jpg", ".jpeg", ".png"}

# ---------- Instancias únicas de MediaPipe ----------
from functools import lru_cache

@lru_cache(maxsize=1)
def get_pose(static: bool = True):
    """
    Pose detector singleton.
    Evita volver a cargar el modelo cada vez que el usuario sube una imagen.
    """
    return mp.solutions.pose.Pose(static_image_mode=static, model_complexity=1)

@lru_cache(maxsize=1)
def get_seg():
    """
    Segmentador singleton.
    Reduce el tiempo de espera manteniendo una sola instancia.
    """
    return mp.solutions.selfie_segmentation.SelfieSegmentation(model_selection=1)

# Landmark enum para acceder rápido a los índices
P = mp.solutions.pose.PoseLandmark

# Colores de dibujo / overlay
CLR_LINE, CLR_PT = (0, 230, 127), (250, 250, 250)  # líneas y vértices
IDEAL_RGBA       = (0, 255, 0, 110)                # verde translúcido


METRIC_EXPLANATIONS = {
    "Hip flex":          "Ángulo hombro–cadera–rodilla: flexión de cadera.",
    "Knee flex":         "Ángulo cadera–rodilla–tobillo: flexión de rodilla.",
    "Shoulder flex":     "Ángulo cadera–hombro–muñeca: flexión de hombro.",
    "|Trunk-Tibia|":     "Diferencia (absoluta) entre el ángulo del tronco y la tibia.",
    "Ankle DF":          "Dorsiflexión de tobillo promedio (talón / dedos).",
    "Apertura rodillas": "Distancia horizontal entre rodillas.",
    "Apertura pies":     "Distancia horizontal entre puntas de pie.",
    "Knee/Foot ratio":   "Relación apertura rodillas / apertura pies; ≈1 = alineado.",
    "L Knee–Toe Δ (px)": "Desplazamiento lateral de la rodilla izquierda respecto a la punta del pie izquierdo (positivo = afuera, negativo = adentro).",
    "R Knee–Toe Δ (px)": "Desplazamiento lateral de la rodilla derecha respecto a la punta del pie derecho (positivo = afuera, negativo = adentro).",
    "Left Foot ER (°)":  "Rotación externa del pie izquierdo (0-30° suele considerarse óptimo).",
    "Right Foot ER (°)": "Rotación externa del pie derecho.",
}

# —————————————————————————————
# 2) Funciones auxiliares
# —————————————————————————————

def b64_to_cv2(content, max_size=720):
    _, b64 = content.split(",", 1)
    img = cv2.imdecode(
        np.frombuffer(base64.b64decode(b64), np.uint8),
        cv2.IMREAD_COLOR
    )
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    
    h, w = img.shape[:2]
    if max(h, w) > max_size:
        scale = max_size / max(h, w)
        new_w, new_h = int(w * scale), int(h * scale)
        img = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_AREA)
    
    return img


def cv2_to_b64(img, max_w=480):
    h, w = img.shape[:2]
    if w > max_w:
        scale = max_w / w
        img = cv2.resize(img, (int(w * scale), int(h * scale)), interpolation=cv2.INTER_AREA)
    
    _, buf = cv2.imencode(
        ".jpg",
        cv2.cvtColor(img, cv2.COLOR_RGB2BGR),
        [int(cv2.IMWRITE_JPEG_QUALITY), 70]  # antes 85 o 75 → ahora 70 (más liviano)
    )
    return base64.b64encode(buf).decode()


def angle_between(u, v):
    cos = np.dot(u, v) / (np.linalg.norm(u)*np.linalg.norm(v) + 1e-9)
    return np.degrees(np.arccos(np.clip(cos, -1, 1)))

def crop_person(img, lm):
    H, W = img.shape[:2]
    pts = np.array([(p.x*W, p.y*H) for p in lm])
    x0, y0 = pts.min(0)
    x1, y1 = pts.max(0)
    pad = 0.2
    x0 = max(0, int(x0 - pad*(x1-x0)))
    y0 = max(0, int(y0 - pad*(y1-y0)))
    x1 = min(W, int(x1 + pad*(x1-x0)))
    y1 = min(H, int(y1 + (pad+0.1)*(y1-y0)))
    crop = img[y0:y1, x0:x1]
    return cv2.resize(crop, (480, int(480*crop.shape[0]/crop.shape[1])))

def card(var, val):
    cid = f"card-{var}".replace(" ", "-")
    return html.Div([
        dbc.Card([
            dbc.CardBody([
                html.Small(var, className="text-muted"),
                html.H4(f"{val:.1f}°" if isinstance(val, (int, float)) else f"{val}", className="mb-0")
            ])
        ], color="dark", outline=True, className="m-1 p-2", style={"minWidth":"120px"}, id=cid),
        dbc.Tooltip(METRIC_EXPLANATIONS.get(var, ""), target=cid)
    ])

# ──────────────────────────────────────────────
#  F U N C I Ó N   •   A N Á L I S I S   S A G I T A L
# ──────────────────────────────────────────────
def analyze_sagital(img):
    pose = get_pose(static=True)   # instancia única (lru_cache)
    seg  = get_seg()

    # 1) Pose sobre la imagen completa
    res1 = pose.process(img)
    if not res1.pose_landmarks:
        return None, None, {}

    crop = crop_person(img, res1.pose_landmarks.landmark)
    h, w = crop.shape[:2]

    # 2) Pose sobre el recorte
    res2 = pose.process(crop)
    if not res2.pose_landmarks:
        return None, None, {}

    lm2 = res2.pose_landmarks.landmark

    # 3) Puntos relevantes
    side  = "R" if lm2[P.RIGHT_HIP].visibility >= lm2[P.LEFT_HIP].visibility else "L"
    pick  = lambda L, R: R if side == "R" else L
    ids   = [pick(getattr(P, f"LEFT_{n}"), getattr(P, f"RIGHT_{n}"))
             for n in ("SHOULDER", "HIP", "KNEE", "ANKLE",
                        "HEEL", "FOOT_INDEX", "WRIST")]

    SHp, HIp, KNp, ANp, HEp, FTp, WRp = [
        (int(lm2[i].x * w), int(lm2[i].y * h)) for i in ids
    ]

    # 4) Ángulos
    hip_flex   = angle_between(np.array(SHp) - HIp, np.array(KNp) - HIp)
    knee_flex  = angle_between(np.array(HIp) - KNp, np.array(ANp) - KNp)
    shld_flex  = angle_between(np.array(HIp) - SHp, np.array(WRp) - SHp)
    trunk_tib  = abs(hip_flex - knee_flex)
    raw_heel   = angle_between(np.array(KNp) - ANp, np.array(HEp) - ANp) - 90
    raw_toe    = angle_between(np.array(KNp) - ANp, np.array(FTp) - ANp) - 90
    ankle_df   = (abs(raw_heel) + abs(raw_toe)) / 2

    data = {
        "Hip flex":      hip_flex,
        "Knee flex":     knee_flex,
        "Shoulder flex": shld_flex,
        "|Trunk-Tibia|": trunk_tib,
        "Ankle DF":      ankle_df
    }

    # 5) Fondo difuminado
    seg_result = seg.process(cv2.cvtColor(crop, cv2.COLOR_RGB2BGR))
    mask = seg_result.segmentation_mask > 0.6
    blur = cv2.GaussianBlur(crop, (17, 17), 0)
    vis  = np.where(mask[..., None], crop, blur).astype(np.uint8)

    # 6) Dibujos finos + texto
    for name, (A, B, C) in [
        ("Hip flex",      (SHp, HIp, KNp)),
        ("Knee flex",     (HIp, KNp, ANp)),
        ("Shoulder flex", (HIp, SHp, WRp))
    ]:
        cv2.arrowedLine(vis, B, A, (255, 0, 0), 3, tipLength=0.1)
        cv2.arrowedLine(vis, B, C, (255, 0, 0), 3, tipLength=0.1)
        for pt in (A, B, C):
            cv2.circle(vis, pt, 6, CLR_PT, -1)
        txt = f"{data[name]:.1f}"
        cv2.putText(vis, txt, (B[0] + 12, B[1] - 12),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 3, cv2.LINE_AA)
        cv2.putText(vis, txt, (B[0] + 12, B[1] - 12),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 2, cv2.LINE_AA)

    # 7) Tobillo
    cv2.line(vis, KNp, ANp, CLR_LINE, 4)
    cv2.line(vis, HEp, FTp, CLR_LINE, 4)
    for pt in (KNp, ANp, HEp, FTp):
        cv2.circle(vis, pt, 6, CLR_PT, -1)
    txt = f"{ankle_df:.1f}"
    cv2.putText(vis, txt, (ANp[0] + 12, ANp[1] - 12),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 3, cv2.LINE_AA)
    cv2.putText(vis, txt, (ANp[0] + 12, ANp[1] - 12),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 2, cv2.LINE_AA)

    return crop, vis, data



# ──────────────────────────────────────────────
#  F U N C I Ó N   •   A N Á L I S I S   F R O N T A L
# ──────────────────────────────────────────────
def analyze_frontal(img):
    pose = get_pose(static=True)

    res = pose.process(img)
    if not res.pose_landmarks:
        return None, None, {}

    h, w = img.shape[:2]
    lm = res.pose_landmarks.landmark

    # --- Coordenadas clave ---
    pts = {
        "LKnee": lm[P.LEFT_KNEE],       "RKnee": lm[P.RIGHT_KNEE],
        "LAnkle": lm[P.LEFT_ANKLE],     "RAnkle": lm[P.RIGHT_ANKLE],
        "LHeel": lm[P.LEFT_HEEL],       "RHeel": lm[P.RIGHT_HEEL],
        "LFoot": lm[P.LEFT_FOOT_INDEX], "RFoot": lm[P.RIGHT_FOOT_INDEX],
        "LHip":  lm[P.LEFT_HIP],        "RHip":  lm[P.RIGHT_HIP],
        "LShoulder": lm[P.LEFT_SHOULDER],"RShoulder": lm[P.RIGHT_SHOULDER],
        "LWrist": lm[P.LEFT_WRIST],     "RWrist": lm[P.RIGHT_WRIST]
    }

    # --- Puntos en píxeles ---
    px = {k: (int(v.x * w), int(v.y * h)) for k, v in pts.items()}

    # --- Métricas ---
    hip_delta = abs(pts["LHip"].y - pts["RHip"].y) * h
    sh_delta  = abs(pts["LShoulder"].y - pts["RShoulder"].y) * h
    wr_delta  = abs(pts["LWrist"].y - pts["RWrist"].y) * h
    sep_feet  = abs(px["LFoot"][0]  - px["RFoot"][0])
    sep_wrsts = abs(px["LWrist"][0] - px["RWrist"][0])

    kneeL = angle_between(np.array(px["LAnkle"]) - np.array(px["LKnee"]),
                          np.array(px["LKnee"])  - np.array(px["LHip"]))
    kneeR = angle_between(np.array(px["RAnkle"]) - np.array(px["RKnee"]),
                          np.array(px["RKnee"])  - np.array(px["RHip"]))

    data = {
        "Left knee":        kneeL,
        "Right knee":       kneeR,
        "Hip level Δ":      round(hip_delta, 1),
        "Shoulder level Δ": round(sh_delta, 1),
        "Wrist level Δ":    round(wr_delta, 1),
        "Feet spread":      sep_feet,
        "Wrist spread":     sep_wrsts
    }

    # --- Visualización ---
    vis = img.copy()

    # puntos
    for pt in px.values():
        cv2.circle(vis, pt, 6, CLR_PT, -1)

    # líneas
    lines = [
        ("LKnee", "LAnkle"), ("RKnee", "RAnkle"),
        ("LHeel", "LFoot"),  ("RHeel", "RFoot"),
        ("LShoulder", "LWrist"), ("RShoulder", "RWrist")
    ]
    for A, B in lines:
        cv2.line(vis, px[A], px[B], CLR_LINE, 2)

    return img, vis, data


# ──────────────────────────────────────────────
# 5) Configuración Dash y Layout  (SIN placeholders y sin errores de comillas)
# ──────────────────────────────────────────────
server = Flask(__name__)
app = dash.Dash(
    __name__,
    server=server,
    external_stylesheets=[dbc.themes.FLATLY],
)
app.title = "Overhead-Squat Analyzer"

app.layout = dbc.Container(
    [

        # ---------- Navbar ----------
        dbc.Navbar(
            dbc.Container(
                [
                    html.Img(src="/assets/angle.png", height="40px"),
                    dbc.NavbarBrand('OHS Analyzer', className='ms-2'),
                ]
            ),
            color='light',
            dark=False,
            className='mb-4',
        ),

        # ---------- Sección educativa ----------
        dbc.Card(
            [
                dbc.CardHeader('¿Cómo calculamos las métricas?'),
                dbc.CardBody(
                    html.Ul(
                        [
                            html.Li(
                                'MediaPipe detecta puntos clave en la silueta: hombros, caderas, rodillas, tobillos, talón y punta de pie.'
                            ),
                            html.Li(
                                'Convertimos esas posiciones normalizadas (0–1) a píxeles, multiplicando por el ancho y alto del área recortada.'
                            ),
                            html.Li(
                                [
                                    html.B('Vista Sagital (de lado):'),
                                    html.Ul(
                                        [
                                            html.Li(
                                                '**Hip flex**: imagina dos palos unidos en la cadera. Uno va hasta el hombro, otro hasta la rodilla. Medimos el ángulo que forman en la cadera, como abrir o cerrar una puerta.'
                                            ),
                                            html.Li(
                                                '**Knee flex**: un palo de cadera a rodilla y otro de rodilla a tobillo; mide cuánto dobla la pierna.'
                                            ),
                                            html.Li(
                                                '**Shoulder flex**: palo de cadera a hombro y palo de hombro a muñeca; mide cuánto levantas el brazo.'
                                            ),
                                            html.Li(
                                                '**Trunk–Tibia**: |Hip flex – Knee flex|; indica alineación tronco-tibia.'
                                            ),
                                            html.Li(
                                                '**Ankle DF**: promedio de dos ángulos en el tobillo menos 90° para referirlo a la vertical.'
                                            ),
                                        ]
                                    ),
                                ]
                            ),
                            html.Li(
                                [
                                    html.B('Vista Frontal (de frente):'),
                                    html.Ul(
                                        [
                                            html.Li(
                                                '**Simetría de altura**: diferencia de Y entre hombros, caderas y muñecas.'
                                            ),
                                            html.Li(
                                                '**Apertura de rodillas** y **Apertura de pies**: distancia X entre rodillas / tobillos.'
                                            ),
                                            html.Li(
                                                '**Interpretación**: cuanto más cerca de cero, mejor alineación lateral.'
                                            ),
                                        ]
                                    ),
                                ]
                            ),
                            html.Li(
                                [
                                    html.B('Ejemplo práctico:'),
                                    html.Ul(
                                        [
                                            html.Li(
                                                'Hip flex 50° y Knee flex 45° → Trunk–Tibia = 5° (muy alineado).'
                                            ),
                                            html.Li(
                                                'Tobillos separados 30 px = base de apoyo de 30 px.'
                                            ),
                                        ]
                                    ),
                                ]
                            ),
                        ]
                    )
                ),
            ],
            color='info',
            inverse=True,
            className='mb-4',
        ),

        # ---------- Botón reset ----------
        html.Div(
            dbc.Button('🔄 Nuevo análisis',
                       id='btn-reset',
                       color='danger',
                       className='mb-4'),
            className='text-center',
        ),

        # ---------- Título descriptivo ----------
        html.Div(
            [
                html.H5('¿Qué métricas medimos?',
                        className='text-secondary text-center'),
                html.P(
                    'Ángulos siempre positivos y aperturas en vista frontal.',
                    className='text-center',
                ),
            ],
            className='mb-4',
        ),

        # ---------- Columnas Sagital & Frontal ----------
        dbc.Row(
            [
                # ---- Columna Sagital ----
                dbc.Col(
                    [
                        html.H5('Sagittal View',
                                className='text-secondary text-center mb-2'),
                        dcc.Upload(
                            id='up-sag',
                            children=dbc.Button(
                                'Upload Sagittal Image',
                                color='primary',
                                className='w-100',
                            ),
                            multiple=False,
                        ),
                        dcc.Loading(
                            id='load-sag',
                            type='circle',
                            children=html.Div(id='out-sag'),
                            style={'marginTop': '1rem'},
                        ),
                    ],
                    md=6,
                    style={'minHeight': '600px'},
                ),

                # ---- Columna Frontal ----
                dbc.Col(
                    [
                        html.H5('Frontal View',
                                className='text-secondary text-center mb-2'),
                        dcc.Upload(
                            id='up-front',
                            children=dbc.Button(
                                'Upload Frontal Image',
                                color='primary',
                                className='w-100',
                            ),
                            multiple=False,
                        ),
                        dcc.Loading(
                            id='load-front',
                            type='circle',
                            children=html.Div(id='out-front'),
                            style={'marginTop': '1rem'},
                        ),
                    ],
                    md=6,
                    style={'minHeight': '600px'},
                ),
            ],
            justify='center',
            className='g-4 mb-4',
        ),

        html.Hr(),

        # ---------- Footer ----------
        dbc.Row(
            dbc.Col(
                html.Div(
                    'Powered by STA METHODOLOGIES • Luciano Sacaba',
                    className='text-center text-muted small',
                ),
                width=12,
            )
        ),
    ],
    fluid=True,
)





# ——————————————————————————————————————————
# 1) ancho máximo UNIFICADO para la UI
# ——————————————————————————————————————————
MAX_W_UI_SAG = "250px"   # ajusta aquí si quieres más / menos

from dash.exceptions import PreventUpdate
from dash import ctx

@app.callback(
    Output("out-sag", "children"),
    Output("out-front", "children"),
    Input("up-sag", "contents"),
    Input("up-front", "contents"),
    Input("btn-reset", "n_clicks"),
    State("up-sag", "filename"),
    State("up-front", "filename"),
    prevent_initial_call=True
)
def handle_all(sag_content, front_content, reset_clicks, sag_name, front_name):
    triggered_id = ctx.triggered_id

    if triggered_id == "btn-reset":
        return "", ""

    if triggered_id == "up-sag" and sag_content:
        try:
            img = b64_to_cv2(sag_content)
            crop, vis, data = analyze_sagital(img)
            if crop is None:
                return dbc.Alert("⚠️ No se detectó pose en imagen sagital.", color="warning"), no_update

            cards = [card(k, v) for k, v in data.items()]
            crop_b64 = cv2_to_b64(crop)
            vis_b64 = cv2_to_b64(vis)

            zip_link = create_zip(vis_b64, data, "sagittal_analysis.zip")
            report_buf = build_report(base64.b64decode(vis_b64), data,
                                      atleta="Atleta", vista="Sagital")
            report_b64 = base64.b64encode(report_buf.getvalue()).decode()
            report_link = html.A(
                "📄 Descargar informe (.docx)",
                href=f"data:application/vnd.openxmlformats-officedocument.wordprocessingml.document;base64,{report_b64}",
                download="informe_sagital.docx",
                className="btn btn-success mt-2"
            )

            out_sag = html.Div([
                dbc.Row([
                    dbc.Col(html.Img(src=f"data:image/jpg;base64,{crop_b64}",
                                     style={"width": "100%", "maxWidth": MAX_W_UI_SAG, "borderRadius": "6px"}), width=6),
                    dbc.Col([
                        html.H5("Métricas", className="text-white mb-2"),
                        html.Div(cards, style={"display": "flex", "flexWrap": "wrap"})
                    ], width=6)
                ], align="start"),
                html.Hr(className="border-secondary"),
                dbc.Row(
                    dbc.Col(html.Img(src=f"data:image/jpg;base64,{vis_b64}",
                                     style={"width": "100%", "maxWidth": MAX_W_UI_SAG, "borderRadius": "6px"}),
                            width={"size": 8, "offset": 2})
                ),
                html.Div([zip_link, report_link], className="mt-3 d-flex gap-3")
            ])
            return out_sag, no_update
        except Exception as e:
            return dbc.Alert(f"Error: {str(e)}", color="danger"), no_update

    if triggered_id == "up-front" and front_content:
        try:
            img = b64_to_cv2(front_content)
            crop, vis, data = analyze_frontal(img)
            if crop is None:
                return no_update, dbc.Alert("⚠️ No se detectó pose en imagen frontal.", color="warning")

            cards = [card(k, v) for k, v in data.items()]
            crop_b64 = cv2_to_b64(crop)
            vis_b64 = cv2_to_b64(vis)

            zip_link = create_zip(vis_b64, data, "frontal_analysis.zip")
            report_buf = build_report(base64.b64decode(vis_b64), data,
                                      atleta="Atleta", vista="Frontal")
            report_b64 = base64.b64encode(report_buf.getvalue()).decode()
            report_link = html.A(
                "📄 Descargar informe (.docx)",
                href=f"data:application/vnd.openxmlformats-officedocument.wordprocessingml.document;base64,{report_b64}",
                download="informe_frontal.docx",
                className="btn btn-success mt-2"
            )

            out_front = html.Div([
                dbc.Row([
                    dbc.Col(html.Img(src=f"data:image/jpg;base64,{crop_b64}",
                                     style={"width": "100%", "maxWidth": "400px", "borderRadius": "6px"}), width=6),
                    dbc.Col([
                        html.H5("Métricas", className="text-white mb-2"),
                        html.Div(cards, style={"display": "flex", "flexWrap": "wrap"})
                    ], width=6)
                ], align="start"),
                html.Hr(className="border-secondary"),
                dbc.Row(
                    dbc.Col(html.Img(src=f"data:image/jpg;base64,{vis_b64}",
                                     style={"width": "100%", "maxWidth": "800px", "borderRadius": "6px"}),
                            width={"size": 8, "offset": 2})
                ),
                html.Div([zip_link, report_link], className="mt-3 d-flex gap-3")
            ])
            return no_update, out_front
        except Exception as e:
            return no_update, dbc.Alert(f"Error: {str(e)}", color="danger")

    raise PreventUpdate



# —————————————————————————————
# Función de descarga ZIP
# —————————————————————————————

def create_zip(img_b64: str, metrics_dict: dict, zip_filename: str) -> html.A:
    """
    Empaqueta una imagen (base64) y un Excel con las métricas
    en un ZIP que se puede descargar desde el navegador.
    """
    # Decodificar imagen
    img_bytes = base64.b64decode(img_b64)

    # Crear DataFrame y Excel en memoria
    df = pd.DataFrame(list(metrics_dict.items()), columns=["Metric", "Value"])
    excel_buffer = io.BytesIO()
    with pd.ExcelWriter(excel_buffer, engine="xlsxwriter") as writer:
        df.to_excel(writer, index=False, sheet_name="Metrics")

    # Empaquetar en ZIP
    mem_zip = io.BytesIO()
    with zipfile.ZipFile(mem_zip, mode="w", compression=zipfile.ZIP_DEFLATED) as zf:
        zf.writestr("analyzed_image.jpg", img_bytes)
        zf.writestr("metrics.xlsx", excel_buffer.getvalue())
    mem_zip.seek(0)
    zip_b64 = base64.b64encode(mem_zip.read()).decode()

    # Enlace de descarga
    return html.A(
        "⬇️ Descargar imagen + métricas (.zip)",
        href=f"data:application/zip;base64,{zip_b64}",
        download=zip_filename,
        className="btn btn-outline-info mt-2"
    )

import os
import atexit
from apscheduler.schedulers.background import BackgroundScheduler
import requests

def ping_self():
    try:
        url = os.environ.get("KEEP_ALIVE_URL")  # usar variable personalizada en Railway
        if url:
            requests.get(url)
            print("🔄 Keep-alive ping enviado.")
    except Exception as e:
        print(f"⚠️ Error en keep-alive: {e}")

scheduler = BackgroundScheduler()
scheduler.add_job(ping_self, "interval", minutes=14)
scheduler.start()
atexit.register(lambda: scheduler.shutdown())

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8050))
    print(f"✅ App levantando en http://0.0.0.0:{port}/")
    app.run(host="0.0.0.0", port=port, debug=False)

# ⬅️ Agregá esto al final
server = app.server
