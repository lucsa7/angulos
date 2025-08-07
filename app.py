# ───────────────────────────────────────────────
#  Overhead-Squat Analyzer · versión responsiva
# ───────────────────────────────────────────────

# IMPORTS ÚNICOS
import base64, cv2, mediapipe as mp, numpy as np, tempfile, io, zipfile, os
from pathlib import Path
import dash, dash_bootstrap_components as dbc
from dash import dcc, html, Input, Output, State, ctx, no_update
from flask import Flask
import pandas as pd
from functools import lru_cache
from typing import Dict

# ────────────────────────────────
# 1) Configuración general
# ────────────────────────────────
TMP_DIR = Path(tempfile.gettempdir()) / "ohs_tmp"
TMP_DIR.mkdir(exist_ok=True)
ALLOWED = {".jpg", ".jpeg", ".png"}

@lru_cache(maxsize=1)
def get_pose(static=True):
    return mp.solutions.pose.Pose(static_image_mode=static, model_complexity=1)

@lru_cache(maxsize=1)
def get_seg():
    return mp.solutions.selfie_segmentation.SelfieSegmentation(model_selection=1)

P = mp.solutions.pose.PoseLandmark
CLR_LINE, CLR_PT = (0, 230, 127), (250, 250, 250)

# Estilos globales
IMG_STYLE = {
    "width": "100%",
    "maxWidth": "100%",
    "maxHeight": "600px",   # ← límite para monitores grandes
    "height": "auto",
    "borderRadius": "0.75rem",
    "boxShadow": "0 2px 8px rgba(0,0,0,.15)",
    "objectFit": "contain"
}


CARD_BOX_STYLE = {
    "display": "flex",
    "flexWrap": "wrap",
    "gap": "0.5rem",
    "justifyContent": "center"
}

# ────────────────────────────────
# 2) Utilidades imagen/b64
# ────────────────────────────────
def b64_to_cv2(content, max_size=720):
    _, b64 = content.split(",", 1)
    img = cv2.imdecode(np.frombuffer(base64.b64decode(b64), np.uint8), cv2.IMREAD_COLOR)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    h, w = img.shape[:2]
    if max(h, w) > max_size:
        scale = max_size / max(h, w)
        img = cv2.resize(img, (int(w*scale), int(h*scale)), interpolation=cv2.INTER_AREA)
    return img

def cv2_to_b64(img, max_w=900):
    h, w = img.shape[:2]
    if w > max_w:
        scale = max_w / w
        img = cv2.resize(img, (int(w*scale), int(h*scale)), interpolation=cv2.INTER_AREA)
    _, buf = cv2.imencode(".jpg", cv2.cvtColor(img, cv2.COLOR_RGB2BGR),
                          [int(cv2.IMWRITE_JPEG_QUALITY), 80])
    return base64.b64encode(buf).decode()

def crop_person(img, lm):
    H, W = img.shape[:2]
    pts = np.array([(p.x*W, p.y*H) for p in lm])
    x0, y0 = pts.min(0); x1, y1 = pts.max(0)
    pad = 0.2
    x0 = max(0, int(x0 - pad*(x1-x0))); y0 = max(0, int(y0 - pad*(y1-y0)))
    x1 = min(W, int(x1 + pad*(x1-x0))); y1 = min(H, int(y1 + (pad+0.1)*(y1-y0)))
    crop = img[y0:y1, x0:x1]
    return cv2.resize(crop, (480, int(480*crop.shape[0]/crop.shape[1])))

# ────────────────────────────────
# 3) Tarjetas de métricas
# ────────────────────────────────
LABEL = {
    "Δ Caderas (px)"  : "Δ Caderas",
    "Δ Muñecas (px)"  : "Δ Muñecas",
    "Apertura rodillas": "Apertura rodillas",
    "Apertura pies"   : "Apertura pies",
    "L Knee–Toe (px)" : "L Knee–Toe",
    "R Knee–Toe (px)" : "R Knee–Toe",
}
EXPLAIN = {
    "Δ Caderas (px)"  : "Diferencia de altura entre las dos caderas.",
    "Δ Muñecas (px)"  : "Diferencia de altura entre las muñecas.",
    "Apertura rodillas": "Distancia horizontal entre rodillas.",
    "Apertura pies"   : "Distancia horizontal entre pies.",
    "L Knee–Toe (px)" : "Desplazamiento lateral de la rodilla izquierda respecto a su pie.",
    "R Knee–Toe (px)" : "Desplazamiento lateral de la rodilla derecha respecto a su pie.",
}
def card(var: str, val):
    unit = "px" if "px" in var or "Apertura" in var else "°"
    txt  = f"{val:.1f} {unit}"
    cid  = f"card-{var}".replace(" ", "-")
    return html.Div([
        dbc.Card(
            dbc.CardBody([
                html.Small(LABEL.get(var, var), className="text-muted"),
                html.H4(txt, className="mb-0")
            ]),
            outline=True, className="m-1 p-2 shadow-sm text-center",
            style={"flex":"1 1 120px"}, id=cid
        ),
        dbc.Tooltip(EXPLAIN.get(var, ""), target=cid, placement="top")
    ])

# ────────────────────────────────
# 4) Análisis SAGITAL  · sin tobillo
# ────────────────────────────────
def analyze_sagital(img):
    pose, seg = get_pose(True), get_seg()

    # 1 · detección global
    res1 = pose.process(img)
    if not res1.pose_landmarks:
        return None, None, {}

    # 2 · recorte y 2.ª detección (mejor precisión)
    crop  = crop_person(img, res1.pose_landmarks.landmark)
    res2  = pose.process(crop)
    lm, (h, w) = (res2.pose_landmarks.landmark, crop.shape[:2]) \
                 if res2.pose_landmarks else (res1.pose_landmarks.landmark, img.shape[:2])
    frame = crop if res2.pose_landmarks else img

    # 3 · puntos (lado más visible)
    side = "R" if lm[P.RIGHT_HIP].visibility >= lm[P.LEFT_HIP].visibility else "L"
    pick = lambda L,R: R if side == "R" else L
    ids  = [pick(getattr(P,f"LEFT_{n}"), getattr(P,f"RIGHT_{n}"))
            for n in ("SHOULDER","HIP","KNEE","ANKLE","FOOT_INDEX","WRIST")]
    SHp,HIp,KNp,ANp,FTp,WRp = [(int(lm[i].x*w), int(lm[i].y*h)) for i in ids]

    # 4 · ángulos
    def ang(u,v):
        u,v = np.array(u), np.array(v)
        return np.degrees(np.arccos(
            np.clip(np.dot(u,v)/(np.linalg.norm(u)*np.linalg.norm(v)+1e-9),-1,1)))

    hip_flex  = ang(np.subtract(SHp,HIp), np.subtract(KNp,HIp))
    knee_flex = ang(np.subtract(HIp,KNp), np.subtract(ANp,KNp))
    shd_flex  = ang(np.subtract(HIp,SHp), np.subtract(WRp,SHp))
    trunk_tib = abs(hip_flex - knee_flex)

    data = {
        "Hip flex":      hip_flex,
        "Knee flex":     knee_flex,
        "Shoulder flex": shd_flex,
        "|Trunk-Tibia|": trunk_tib
    }

    # 5 · visualización (sin valor de tobillo)
    mask = seg.process(cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)).segmentation_mask > 0.6
    vis  = np.where(mask[...,None], frame,
                    cv2.GaussianBlur(frame, (17,17), 0)).astype(np.uint8)

    for name,(A,B,C) in [
        ("Hip flex",      (SHp, HIp, KNp)),
        ("Knee flex",     (HIp, KNp, ANp)),
        ("Shoulder flex", (HIp, SHp, WRp))
    ]:
        cv2.arrowedLine(vis, B, A, (255,0,0), 3, tipLength=.1)
        cv2.arrowedLine(vis, B, C, (255,0,0), 3, tipLength=.1)
        for pt in (A,B,C): cv2.circle(vis, pt, 6, CLR_PT, -1)
        txt = f"{data[name]:.1f}"
        cv2.putText(vis, txt, (B[0]+12, B[1]-12),
                    cv2.FONT_HERSHEY_SIMPLEX, .6, (255,255,255), 3, cv2.LINE_AA)
        cv2.putText(vis, txt, (B[0]+12, B[1]-12),
                    cv2.FONT_HERSHEY_SIMPLEX, .6, (0,0,0), 2, cv2.LINE_AA)

    # líneas de referencia (opcionales)
    cv2.line(vis, KNp, ANp, CLR_LINE, 4)
    cv2.line(vis, ANp, FTp, CLR_LINE, 4)
    for pt in (KNp, ANp, FTp): cv2.circle(vis, pt, 6, CLR_PT, -1)

    return frame, vis, data


# ────────────────────────────────
# 5) Análisis FRONTAL (solo valores)
# ────────────────────────────────
def analyze_frontal(img):
    pose = get_pose(True)
    res  = pose.process(img)
    if not res.pose_landmarks:
        return None, None, {}
    h, w = img.shape[:2]
    lm   = res.pose_landmarks.landmark
    px = lambda idx: (int(lm[idx].x*w), int(lm[idx].y*h))
    LSh,RSh = px(P.LEFT_SHOULDER), px(P.RIGHT_SHOULDER)
    LWri,RWri = px(P.LEFT_WRIST), px(P.RIGHT_WRIST)
    LHip,RHip = px(P.LEFT_HIP), px(P.RIGHT_HIP)
    LKnee,RKnee = px(P.LEFT_KNEE), px(P.RIGHT_KNEE)
    LAnk,RAnk = px(P.LEFT_ANKLE), px(P.RIGHT_ANKLE)
    LToe,RToe = px(P.LEFT_FOOT_INDEX), px(P.RIGHT_FOOT_INDEX)
    hip_d   = abs(LHip[1]-RHip[1])
    wrist_d = abs(LWri[1]-RWri[1])
    knee_w  = abs(LKnee[0]-RKnee[0])
    foot_w  = abs(LToe[0]-RToe[0])
    off = lambda k,a,t: k[0] - (a[0]+t[0])/2
    l_off, r_off = off(LKnee,LAnk,LToe), off(RKnee,RAnk,RToe)
    data = {
        "Δ Caderas (px)"  : round(hip_d ,1),
        "Δ Muñecas (px)"  : round(wrist_d,1),
        "Apertura rodillas": knee_w,
        "Apertura pies"   : foot_w,
        "L Knee–Toe (px)" : round(l_off,1),
        "R Knee–Toe (px)" : round(r_off,1),
    }
    vis = img.copy()
    for p in (LSh,RSh,LWri,RWri,LHip,RHip,LKnee,RKnee,LAnk,RAnk,LToe,RToe):
        cv2.circle(vis,p,4,(255,255,255),-1)
    for A,B in [(LSh,LWri),(RSh,RWri),(LKnee,LToe),(RKnee,RToe)]:
        cv2.line(vis,A,B,(100,200,255),2)
    cv2.line(vis,LWri,RWri,(180,255,255),1)
    cv2.line(vis,LHip,RHip,(0,255,0),2)
    for knee,ank in [(LKnee,LAnk),(RKnee,RAnk)]:
        cv2.line(vis,knee,ank,(255,0,0),3)
    # referencias adicionales
    cv2.line(vis, LHip, RHip, (0,255,0), 2)
    for knee in (LKnee, RKnee):
        cv2.line(vis, knee, (knee[0], h-1), (255,0,0), 1)
    return img, vis, data

# ────────────────────────────────
# 6) Layout Dash
# ────────────────────────────────
server = Flask(__name__)
app = dash.Dash(__name__, server=server,
                external_stylesheets=[dbc.themes.FLATLY])
app.title = "Overhead-Squat Analyzer"

METRIC_LIST = html.Ul([
    html.Li(html.B("Métricas SAGITALES:"), className="mb-1"),
    html.Li("Hip flex (°): ángulo hombro-cadera-rodilla."),
    html.Li("Knee flex (°): ángulo cadera-rodilla-tobillo."),
    html.Li("|Trunk-Tibia| (°): diferencia entre ángulo de tronco y tibia."),
    html.Li(html.B("Métricas FRONTALES:"), className="mt-2 mb-1"),
    html.Li("Δ Caderas (px): diferencia de altura entre caderas."),
    html.Li("Δ Muñecas (px): diferencia de altura entre muñecas."),
    html.Li("Apertura rodillas (px): distancia horizontal entre rodillas."),
    html.Li("Apertura pies (px): distancia horizontal entre pies."),
    html.Li("L/R Knee-Toe (px): desplazamiento lateral de cada rodilla respecto a su pie.")
], className="small")

app.layout = dbc.Container([
    dbc.Navbar(dbc.Container([
        html.Img(src="/assets/angle.png", height="40px", className="me-2"),
        dbc.NavbarBrand("OHS Analyzer")
    ]), color="light", dark=False, className="mb-4 shadow-sm"),

    dbc.Card([
        dbc.CardHeader("¿Cómo calculamos las métricas?"),
        dbc.CardBody(METRIC_LIST)
    ], color="info", inverse=True, className="mb-4 shadow-sm"),

    html.Div(dbc.Button("🔄 Nuevo análisis", id="btn-reset",
             color="danger", className="mb-4"), className="text-center"),

    dbc.Row([
        dbc.Col([
            html.H5("Sagittal View", className="text-secondary text-center mb-2"),
            dcc.Upload(id="up-sag",
                children=dbc.Button("Subir imagen sagital", color="primary", className="w-100"),
                multiple=False),
            dcc.Loading(id="load-sag", type="circle",
                children=html.Div(id="out-sag"), style={"marginTop":"1rem"})
        ], xs=12, md=6, style={"minHeight":"600px"}),

        dbc.Col([
            html.H5("Frontal View", className="text-secondary text-center mb-2"),
            dcc.Upload(id="up-front",
                children=dbc.Button("Subir imagen frontal", color="primary", className="w-100"),
                multiple=False),
            dcc.Loading(id="load-front", type="circle",
                children=html.Div(id="out-front"), style={"marginTop":"1rem"})
        ], xs=12, md=6, style={"minHeight":"600px"})
    ], justify="center", className="g-4 mb-4"),

    html.Hr(),
    dbc.Row(dbc.Col(
        html.Div("Powered by STA METHODOLOGIES • Luciano Sacaba",
                 className="text-center text-muted small"), width=12))
], fluid=True)

# ────────────────────────────────
# 7) Callback principal
# ────────────────────────────────
@app.callback(
    Output("out-sag", "children"),
    Output("out-front", "children"),
    Input("up-sag", "contents"),
    Input("up-front", "contents"),
    Input("btn-reset", "n_clicks"),
    prevent_initial_call=True
)
def handle_all(sag_c, front_c, reset):
    if ctx.triggered_id == "btn-reset":
        return "", ""

    # -------- Sagital --------
    if ctx.triggered_id == "up-sag" and sag_c:
        img = b64_to_cv2(sag_c)
        crop, vis, data = analyze_sagital(img)
        if crop is None:
            return dbc.Alert("⚠️ No se detectó pose sagital", color="warning"), no_update
        crop_b64, vis_b64 = cv2_to_b64(crop), cv2_to_b64(vis)
        cards = [card(k, v) for k, v in data.items()]
        out_sag = html.Div([
            dbc.Row([
                dbc.Col(html.Img(src=f"data:image/jpg;base64,{crop_b64}", style=IMG_STYLE), xs=12, md=6),
                dbc.Col(html.Div(cards, style=CARD_BOX_STYLE), xs=12, md=6)
            ]),
            html.Hr(className="border-secondary"),
            html.Div(html.Img(src=f"data:image/jpg;base64,{vis_b64}", style=IMG_STYLE),
                     className="text-center")
        ])
        return out_sag, no_update

    # -------- Frontal --------
    if ctx.triggered_id == "up-front" and front_c:
        img = b64_to_cv2(front_c)
        crop, vis, data = analyze_frontal(img)
        if crop is None:
            return no_update, dbc.Alert("⚠️ No se detectó pose frontal", color="warning")
        crop_b64, vis_b64 = cv2_to_b64(crop), cv2_to_b64(vis)
        cards = [card(k, v) for k, v in data.items()]
        out_front = html.Div([
            dbc.Row([
                dbc.Col(html.Img(src=f"data:image/jpg;base64,{crop_b64}", style=IMG_STYLE), xs=12, md=6),
                dbc.Col(html.Div(cards, style=CARD_BOX_STYLE), xs=12, md=6)
            ]),
            html.Hr(className="border-secondary"),
            html.Div(html.Img(src=f"data:image/jpg;base64,{vis_b64}", style=IMG_STYLE),
                     className="text-center")
        ])
        return no_update, out_front

    raise dash.exceptions.PreventUpdate


# ────────────────────────────────
# 8) keep-alive + arranque
# ────────────────────────────────
import atexit, requests
from apscheduler.schedulers.background import BackgroundScheduler
def ping_self():
    url = os.environ.get("KEEP_ALIVE_URL")
    if url:
        try: requests.get(url)
        except Exception: pass
sched = BackgroundScheduler(); sched.add_job(ping_self,"interval",minutes=14)
sched.start(); atexit.register(lambda: sched.shutdown())

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8050))
    print(f"✅ App en http://0.0.0.0:{port}")
    app.run(host="0.0.0.0", port=port, debug=False)

server = app.server

