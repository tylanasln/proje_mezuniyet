"""
Machine Learning-Based Detection and Visualization of Web Traffic Anomalies
============================================================================
Usage:
    streamlit run app.py

Requirements:
    pip install streamlit torch torchvision pillow numpy pandas matplotlib plotly scikit-learn
"""

import io
import os
import time
import struct
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st
from PIL import Image
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import plotly.graph_objects as go
import plotly.express as px

import torch
import torch.nn as nn
from torchvision import models, transforms

# ─────────────────────────────────────────────────────────
# SAYFA AYARLARI
# ─────────────────────────────────────────────────────────
st.set_page_config(
    page_title="ML-Based Web Traffic Anomaly Detection",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─────────────────────────────────────────────────────────
# CSS — Koyu, endüstriyel siber güvenlik teması
# ─────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Share+Tech+Mono&family=Exo+2:wght@300;400;600;700;900&display=swap');

/* ── KEYFRAMES ─────────────────────────────────────── */
@keyframes gridMove {
    0%   { background-position: 0 0; }
    100% { background-position: 60px 60px; }
}
@keyframes scanline {
    0%   { transform: translateY(-100%); opacity: 0.08; }
    100% { transform: translateY(100vh);  opacity: 0.08; }
}
@keyframes pulseBorder {
    0%, 100% { box-shadow: 0 0 8px rgba(0,212,255,0.2), inset 0 0 8px rgba(0,212,255,0.05); }
    50%       { box-shadow: 0 0 22px rgba(0,212,255,0.45), inset 0 0 16px rgba(0,212,255,0.1); }
}
@keyframes titleGlow {
    0%, 100% { text-shadow: 0 0 20px rgba(0,212,255,0.5), 0 0 60px rgba(0,212,255,0.2); }
    50%       { text-shadow: 0 0 40px rgba(0,212,255,0.9), 0 0 100px rgba(0,212,255,0.4); }
}
@keyframes blink {
    0%, 100% { opacity: 1; }
    50%       { opacity: 0; }
}
@keyframes fadeSlideUp {
    from { opacity: 0; transform: translateY(12px); }
    to   { opacity: 1; transform: translateY(0); }
}
@keyframes hexFloat {
    0%, 100% { transform: translateY(0px) rotate(0deg); opacity: 0.04; }
    50%       { transform: translateY(-20px) rotate(3deg); opacity: 0.08; }
}
@keyframes radarSpin {
    from { transform: rotate(0deg); }
    to   { transform: rotate(360deg); }
}
@keyframes shimmer {
    0%   { background-position: -200% center; }
    100% { background-position: 200% center; }
}

/* ── GLOBAL ─────────────────────────────────────────── */
html, body, [class*="css"] {
    font-family: 'Exo 2', sans-serif;
}

/* ── ANA ARKAPLAN — hareketli grid + hex doku ───────── */
.stApp {
    background-color: #060a12;
    background-image:
        linear-gradient(rgba(0,212,255,0.03) 1px, transparent 1px),
        linear-gradient(90deg, rgba(0,212,255,0.03) 1px, transparent 1px),
        radial-gradient(ellipse 80% 50% at 50% -10%, rgba(0,80,160,0.25) 0%, transparent 70%),
        radial-gradient(ellipse 60% 40% at 90% 90%, rgba(0,30,80,0.3) 0%, transparent 60%);
    background-size: 60px 60px, 60px 60px, 100% 100%, 100% 100%;
    animation: gridMove 8s linear infinite;
    color: #c9d1e0;
    min-height: 100vh;
}

/* Yavaş hareket eden tarama çizgisi */
.stApp::after {
    content: '';
    position: fixed;
    top: 0; left: 0; right: 0;
    height: 3px;
    background: linear-gradient(90deg, transparent, rgba(0,212,255,0.6), transparent);
    animation: scanline 6s linear infinite;
    pointer-events: none;
    z-index: 9999;
}

/* ── SIDEBAR ────────────────────────────────────────── */
[data-testid="stSidebar"] {
    background: linear-gradient(180deg, #080d18 0%, #0a1020 100%) !important;
    border-right: 1px solid rgba(0,212,255,0.12) !important;
    box-shadow: 4px 0 30px rgba(0,0,0,0.5);
}
[data-testid="stSidebar"]::before {
    content: '';
    position: absolute;
    top: 0; left: 0; right: 0;
    height: 3px;
    background: linear-gradient(90deg, #00d4ff, #0050a0, #00d4ff);
    background-size: 200% auto;
    animation: shimmer 3s linear infinite;
}
[data-testid="stSidebar"] * { color: #8ba3c4 !important; }

/* ── BAŞLIK ─────────────────────────────────────────── */
.hero-wrap {
    position: relative;
    padding: 1.5rem 0 1rem;
    animation: fadeSlideUp 0.6s ease both;
}
.hero-title {
    font-family: 'Share Tech Mono', monospace;
    font-size: 3rem;
    font-weight: 900;
    letter-spacing: 0.1em;
    color: #00d4ff;
    animation: titleGlow 3s ease-in-out infinite;
    margin: 0;
    line-height: 1;
}
.hero-title span.cursor {
    animation: blink 1s step-end infinite;
    color: #00d4ff;
}
.hero-sub {
    font-family: 'Share Tech Mono', monospace;
    font-size: 0.78rem;
    color: #1e5a7a;
    letter-spacing: 0.22em;
    margin-top: 0.4rem;
}
.hero-badge {
    display: inline-block;
    background: rgba(0,212,255,0.08);
    border: 1px solid rgba(0,212,255,0.2);
    border-radius: 20px;
    padding: 0.2rem 0.8rem;
    font-family: 'Share Tech Mono', monospace;
    font-size: 0.65rem;
    color: #00d4ff;
    letter-spacing: 0.15em;
    margin-top: 0.6rem;
    margin-right: 0.4rem;
}

/* ── METRİK KARTLARI ────────────────────────────────── */
.metric-card {
    background: linear-gradient(135deg, #0b1424 0%, #0e1a2e 60%, #0b1828 100%);
    border: 1px solid rgba(0,212,255,0.15);
    border-radius: 14px;
    padding: 1.3rem 1.5rem;
    text-align: center;
    position: relative;
    overflow: hidden;
    animation: fadeSlideUp 0.5s ease both;
    transition: transform 0.2s, box-shadow 0.2s;
}
.metric-card:hover {
    transform: translateY(-3px);
    box-shadow: 0 8px 30px rgba(0,212,255,0.1);
}
/* Üst şerit shimmer */
.metric-card::before {
    content: '';
    position: absolute;
    top: 0; left: 0; right: 0;
    height: 2px;
    background: linear-gradient(90deg, transparent, #00d4ff, #0050a0, transparent);
    background-size: 200% auto;
    animation: shimmer 2.5s linear infinite;
}
/* Köşe dekorasyon */
.metric-card::after {
    content: '';
    position: absolute;
    bottom: 0; right: 0;
    width: 40px; height: 40px;
    border-bottom: 2px solid rgba(0,212,255,0.08);
    border-right:  2px solid rgba(0,212,255,0.08);
    border-radius: 0 0 14px 0;
}
.metric-label {
    font-family: 'Share Tech Mono', monospace;
    font-size: 0.65rem;
    color: #2a5a7a;
    letter-spacing: 0.25em;
    text-transform: uppercase;
    margin-bottom: 0.4rem;
}
.metric-value {
    font-family: 'Share Tech Mono', monospace;
    font-size: 2rem;
    font-weight: 700;
    color: #00d4ff;
    line-height: 1.1;
}

/* ── BÖLÜM BAŞLIKLARI ───────────────────────────────── */
h3, h4 {
    font-family: 'Share Tech Mono', monospace !important;
    letter-spacing: 0.08em !important;
}

/* ── SEKMELER ───────────────────────────────────────── */
.stTabs [data-baseweb="tab-list"] {
    gap: 2px;
    background: rgba(0,0,0,0.3);
    border-bottom: 1px solid rgba(0,212,255,0.1);
    border-radius: 8px 8px 0 0;
    padding: 0.3rem 0.3rem 0;
}
.stTabs [data-baseweb="tab"] {
    font-family: 'Share Tech Mono', monospace;
    font-size: 0.78rem;
    letter-spacing: 0.12em;
    color: #2a5a7a;
    background: transparent;
    border: none;
    border-radius: 6px 6px 0 0;
    padding: 0.6rem 1.4rem;
    transition: all 0.2s;
}
.stTabs [data-baseweb="tab"]:hover {
    color: #5a9abf !important;
    background: rgba(0,212,255,0.04) !important;
}
.stTabs [aria-selected="true"] {
    color: #00d4ff !important;
    background: rgba(0,212,255,0.08) !important;
    border-bottom: 2px solid #00d4ff !important;
}

/* ── UPLOAD ALANI ───────────────────────────────────── */
[data-testid="stFileUploader"] {
    background: rgba(10,20,40,0.8);
    border: 1px dashed rgba(0,212,255,0.2) !important;
    border-radius: 12px;
    transition: border-color 0.3s;
}
[data-testid="stFileUploader"]:hover {
    border-color: rgba(0,212,255,0.5) !important;
}

/* ── BUTONLAR ───────────────────────────────────────── */
.stButton > button {
    font-family: 'Share Tech Mono', monospace;
    letter-spacing: 0.12em;
    background: linear-gradient(135deg, #003d80 0%, #001d50 100%);
    color: #00d4ff;
    border: 1px solid rgba(0,180,255,0.3);
    border-radius: 8px;
    padding: 0.5rem 1.5rem;
    transition: all 0.25s;
    position: relative;
    overflow: hidden;
}
.stButton > button::before {
    content: '';
    position: absolute;
    top: 0; left: -100%;
    width: 100%; height: 100%;
    background: linear-gradient(90deg, transparent, rgba(0,212,255,0.15), transparent);
    transition: left 0.4s;
}
.stButton > button:hover::before { left: 100%; }
.stButton > button:hover {
    background: linear-gradient(135deg, #0060c0, #003080);
    border-color: rgba(0,212,255,0.6);
    box-shadow: 0 0 25px rgba(0,150,255,0.35), 0 4px 15px rgba(0,0,0,0.4);
    transform: translateY(-2px);
    color: #fff;
}

/* ── INFO KUTUSU ────────────────────────────────────── */
.info-box {
    background: linear-gradient(135deg, rgba(0,30,60,0.8), rgba(0,20,45,0.8));
    border-left: 3px solid #00d4ff;
    border-radius: 0 10px 10px 0;
    padding: 1rem 1.4rem;
    margin: 0.6rem 0;
    font-family: 'Share Tech Mono', monospace;
    font-size: 0.82rem;
    color: #5a8aaa;
    position: relative;
    overflow: hidden;
}
.info-box::after {
    content: '';
    position: absolute;
    top: 0; right: 0;
    width: 60px; height: 100%;
    background: linear-gradient(90deg, transparent, rgba(0,212,255,0.03));
}

/* ── BÖLÜM KAPSAYICI ─────────────────────────────────── */
.section-box {
    background: linear-gradient(135deg, rgba(8,14,28,0.9), rgba(10,18,35,0.9));
    border: 1px solid rgba(0,212,255,0.08);
    border-radius: 16px;
    padding: 1.5rem;
    margin: 1rem 0;
    position: relative;
    overflow: hidden;
}
.section-box::before {
    content: '';
    position: absolute;
    top: 0; left: 0; right: 0;
    height: 1px;
    background: linear-gradient(90deg, transparent, rgba(0,212,255,0.3), transparent);
}

/* ── TABLO ──────────────────────────────────────────── */
[data-testid="stDataFrame"] {
    background: rgba(8,14,28,0.9) !important;
    border: 1px solid rgba(0,212,255,0.08) !important;
    border-radius: 10px;
}

/* ── PROGRESS BAR ───────────────────────────────────── */
.stProgress > div > div > div {
    background: linear-gradient(90deg, #003080, #00d4ff) !important;
    border-radius: 4px;
}

/* ── DIVIDER ────────────────────────────────────────── */
hr {
    border: none;
    border-top: 1px solid rgba(0,212,255,0.08);
    margin: 1.5rem 0;
}

/* ── SCROLLBAR ──────────────────────────────────────── */
::-webkit-scrollbar { width: 5px; height: 5px; }
::-webkit-scrollbar-track { background: #060a12; }
::-webkit-scrollbar-thumb {
    background: linear-gradient(180deg, #00d4ff, #0050a0);
    border-radius: 4px;
}

/* ── ANİMASYONLU GİRİŞ ─────────────────────────────── */
[data-testid="stVerticalBlock"] > div {
    animation: fadeSlideUp 0.4s ease both;
}
</style>
""", unsafe_allow_html=True)


# ─────────────────────────────────────────────────────────
# SABITLER
# ─────────────────────────────────────────────────────────
IMAGE_SIZE   = 72
CLASS_NAMES  = ["normal", "anomaly"]
PCAP_MAGIC   = b'\xd4\xc3\xb2\xa1'  # little-endian pcap magic
PCAP_MAGIC2  = b'\xa1\xb2\xc3\xd4'  # big-endian

EVAL_TRANSFORMS = transforms.Compose([
    transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406],
                         [0.229, 0.224, 0.225]),
])


# ─────────────────────────────────────────────────────────
# MODEL
# ─────────────────────────────────────────────────────────
@st.cache_resource
def load_model(checkpoint_path: str):
    """Checkpoint'ten modeli yükle ve önbellekle."""
    try:
        model = models.efficientnet_b0(weights=None)
        in_features = model.classifier[1].in_features
        model.classifier[1] = nn.Linear(in_features, 2)
    except Exception:
        model = models.resnet18(pretrained=False)
        model.fc = nn.Linear(model.fc.in_features, 2)

    ckpt = torch.load(checkpoint_path, map_location="cpu")
    model.load_state_dict(ckpt["model_state"])
    model.eval()

    idx_to_class = {v: k for k, v in ckpt.get("class_to_idx", {"normal": 0, "anomaly": 1}).items()}
    return model, idx_to_class


# ─────────────────────────────────────────────────────────
# YARDIMCI FONKSİYONLAR
# ─────────────────────────────────────────────────────────
def pcap_bytes_to_images(raw_bytes: bytes) -> list:
    """Convert raw PCAP bytes into a list of 256x256 RGB images."""
    byte_per_image = 256 * 256 * 3
    images = []
    buf = bytearray(raw_bytes)

    while len(buf) >= byte_per_image:
        chunk = bytes(buf[:byte_per_image])
        del buf[:byte_per_image]
        arr = np.frombuffer(chunk, dtype=np.uint8).reshape((256, 256, 3))
        images.append(Image.fromarray(arr, mode='RGB'))

    if len(buf) > 0:
        padding = byte_per_image - len(buf)
        buf.extend(b'\x00' * padding)
        arr = np.frombuffer(bytes(buf), dtype=np.uint8).reshape((256, 256, 3))
        images.append(Image.fromarray(arr, mode='RGB'))

    return images


def predict_image(model, idx_to_class, pil_img) -> dict:
    """Run inference on a single PIL image."""
    tensor = EVAL_TRANSFORMS(pil_img).unsqueeze(0)  # [1, C, H, W]
    with torch.no_grad():
        logits = model(tensor)
        probs  = torch.softmax(logits, dim=1)[0].numpy()
        pred   = int(probs.argmax())
    return {
        "label":      idx_to_class.get(pred, CLASS_NAMES[pred]),
        "confidence": float(probs[pred]),
        "prob_normal":  float(probs[0]) if len(probs) > 0 else 0.5,
        "prob_anomaly": float(probs[1]) if len(probs) > 1 else 0.5,
    }


def gauge_chart(value: float, label: str, is_anomaly: bool):
    """Plotly gauge chart."""
    color = "#ff1744" if is_anomaly else "#00c853"
    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=value * 100,
        number={"suffix": "%", "font": {"color": color, "size": 36, "family": "Share Tech Mono"}},
        title={"text": label, "font": {"color": "#3a6a8a", "size": 13, "family": "Share Tech Mono"}},
        gauge={
            "axis": {"range": [0, 100], "tickcolor": "#1a3a5a", "tickfont": {"color": "#3a6a8a"}},
            "bar":  {"color": color, "thickness": 0.3},
            "bgcolor": "#0a0e1a",
            "bordercolor": "#1a2d45",
            "steps": [
                {"range": [0,  50], "color": "#0a0e1a"},
                {"range": [50, 80], "color": "#0d1525"},
                {"range": [80, 100], "color": "#101b2e"},
            ],
            "threshold": {
                "line": {"color": color, "width": 3},
                "thickness": 0.85,
                "value": value * 100,
            },
        }
    ))
    fig.update_layout(
        paper_bgcolor="#0a0e1a",
        plot_bgcolor="#0a0e1a",
        margin=dict(t=60, b=20, l=30, r=30),
        height=220,
    )
    return fig


def prob_bar_chart(prob_normal: float, prob_anomaly: float):
    """Horizontal stacked probability bar."""
    fig = go.Figure()
    fig.add_trace(go.Bar(
        x=[prob_normal * 100],
        name="NORMAL",
        orientation='h',
        marker_color="#00c853",
        text=[f"NORMAL {prob_normal*100:.1f}%"],
        textposition='inside',
        textfont=dict(family="Share Tech Mono", color="white", size=13),
    ))
    fig.add_trace(go.Bar(
        x=[prob_anomaly * 100],
        name="ANOMALY",
        orientation='h',
        marker_color="#ff1744",
        text=[f"ANOMALY {prob_anomaly*100:.1f}%"],
        textposition='inside',
        textfont=dict(family="Share Tech Mono", color="white", size=13),
    ))
    fig.update_layout(
        barmode='stack',
        paper_bgcolor="#0a0e1a",
        plot_bgcolor="#0a0e1a",
        xaxis=dict(range=[0, 100], showgrid=False, zeroline=False, showticklabels=False),
        yaxis=dict(showticklabels=False),
        showlegend=False,
        margin=dict(t=10, b=10, l=10, r=10),
        height=70,
    )
    return fig


def is_pcap(data: bytes) -> bool:
    return data[:4] in (PCAP_MAGIC, PCAP_MAGIC2)


# ─────────────────────────────────────────────────────────
# SIDEBAR
# ─────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("""
    <div style='text-align:center; margin-bottom:1.5rem;'>
        <span style='font-family:"Share Tech Mono",monospace; font-size:2rem; color:#00d4ff;'>🛡️</span><br>
        <span style='font-family:"Share Tech Mono",monospace; font-size:1.1rem; color:#00d4ff; letter-spacing:0.12em;'>NETGUARD AI</span><br>
        <span style='font-family:"Share Tech Mono",monospace; font-size:0.65rem; color:#1e4a6a; letter-spacing:0.2em;'>v1.0 — ANOMALY DETECTOR</span>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("---")
    st.markdown("**📁 Model Checkpoint**")
    checkpoint_file = st.file_uploader(
        "Upload best_model.pth",
        type=["pth"],
        help="The best_model.pth file trained with 2_train_model.py"
    )

    model_loaded = False
    if checkpoint_file:
        with tempfile.NamedTemporaryFile(suffix=".pth", delete=False) as tmp:
            tmp.write(checkpoint_file.read())
            tmp_path = tmp.name
        try:
            model, idx_to_class = load_model(tmp_path)
            model_loaded = True
            st.success("✅ Model loaded successfully")
        except Exception as e:
            st.error(f"❌ Failed to load model:\n{e}")

    st.markdown("---")
    st.markdown("""
    <div style='font-family:"Share Tech Mono",monospace; font-size:0.72rem; color:#2a4a6a; line-height:1.8;'>
    📐 ARCHITECTURE<br>
    &nbsp;&nbsp;EfficientNet-B0<br>
    &nbsp;&nbsp;Transfer Learning<br><br>
    🖼️ INPUT<br>
    &nbsp;&nbsp;256×256 RGB PNG<br>
    &nbsp;&nbsp;or .pcap file<br><br>
    🎯 TASK<br>
    &nbsp;&nbsp;Binary Classification<br>
    &nbsp;&nbsp;Normal / Anomaly
    </div>
    """, unsafe_allow_html=True)

    st.markdown("---")
    st.markdown("""
    <div style='font-family:"Share Tech Mono",monospace; font-size:0.68rem; color:#1a3a5a; text-align:center;'>
    PCAP → RGB Image → CNN → Decision
    </div>
    """, unsafe_allow_html=True)


# ─────────────────────────────────────────────────────────
# BAŞLIK
# ─────────────────────────────────────────────────────────
st.markdown("""
<div class='hero-wrap'>
    <div class='hero-title'>🛡️ ML-BASED WEB TRAFFIC ANOMALY DETECTION<span class='cursor'>█</span></div>
    <div class='hero-sub'>// MACHINE LEARNING-BASED DETECTION AND VISUALIZATION OF WEB TRAFFIC ANOMALIES //</div>
    <div style='margin-top:0.7rem;'>
        <span class='hero-badge'>🧠 EfficientNet-B0</span>
        <span class='hero-badge'>🖼️ 256×256 RGB</span>
        <span class='hero-badge'>🔐 Binary Classification</span>
        <span class='hero-badge'>⚡ Transfer Learning</span>
    </div>
</div>
""", unsafe_allow_html=True)

if not model_loaded:
    st.markdown("""
    <div class='info-box'>
    ⚠️  Upload your <strong>best_model.pth</strong> from the left panel — then switch to the analysis tab.
    </div>
    """, unsafe_allow_html=True)

st.markdown("---")

# ─────────────────────────────────────────────────────────
# SEKMELER
# ─────────────────────────────────────────────────────────
tab1, tab2 = st.tabs([
    "🏠  HOW IT WORKS",
    "📊  BATCH ANALYSIS",
])


# ══════════════════════════════════════════════════════════
# TAB 1 — HOW IT WORKS
# ══════════════════════════════════════════════════════════
with tab1:
    st.markdown("### System Architecture")

    st.markdown("""
    <div style='background:#0d1120; border:1px solid #1a2d45; border-radius:14px; padding:2rem; margin:1rem 0;'>
    <svg viewBox="0 0 800 130" xmlns="http://www.w3.org/2000/svg" width="100%">
      <line x1="100" y1="65" x2="700" y2="65" stroke="#1a3a5a" stroke-width="2" stroke-dasharray="6,4"/>
      <g>
        <rect x="10" y="35" width="110" height="60" rx="8" fill="#0a1525" stroke="#0050a0" stroke-width="1.5"/>
        <text x="65" y="62" text-anchor="middle" fill="#00d4ff" font-family="Share Tech Mono" font-size="11">📁 PCAP</text>
        <text x="65" y="78" text-anchor="middle" fill="#2a5a8a" font-family="Share Tech Mono" font-size="9">Network Traffic</text>

        <polygon points="130,61 148,65 130,69" fill="#0050a0"/>
        <text x="139" y="58" text-anchor="middle" fill="#1a4a7a" font-family="Share Tech Mono" font-size="8">Raw</text>
        <text x="139" y="82" text-anchor="middle" fill="#1a4a7a" font-family="Share Tech Mono" font-size="8">bytes</text>

        <rect x="155" y="35" width="130" height="60" rx="8" fill="#0a1525" stroke="#0050a0" stroke-width="1.5"/>
        <text x="220" y="58" text-anchor="middle" fill="#00d4ff" font-family="Share Tech Mono" font-size="10">🔄 CONVERSION</text>
        <text x="220" y="72" text-anchor="middle" fill="#2a5a8a" font-family="Share Tech Mono" font-size="9">196,608 bytes</text>
        <text x="220" y="84" text-anchor="middle" fill="#2a5a8a" font-family="Share Tech Mono" font-size="9">→ 256×256 RGB</text>

        <polygon points="293,61 311,65 293,69" fill="#0050a0"/>

        <rect x="318" y="35" width="120" height="60" rx="8" fill="#0a1525" stroke="#0050a0" stroke-width="1.5"/>
        <text x="378" y="58" text-anchor="middle" fill="#00d4ff" font-family="Share Tech Mono" font-size="10">🖼️ IMAGE</text>
        <text x="378" y="72" text-anchor="middle" fill="#2a5a8a" font-family="Share Tech Mono" font-size="9">PNG Byte</text>
        <text x="378" y="84" text-anchor="middle" fill="#2a5a8a" font-family="Share Tech Mono" font-size="9">Texture Map</text>

        <polygon points="446,61 464,65 446,69" fill="#0050a0"/>

        <rect x="471" y="35" width="130" height="60" rx="8" fill="#0a1525" stroke="#0070c0" stroke-width="2"/>
        <text x="536" y="56" text-anchor="middle" fill="#00d4ff" font-family="Share Tech Mono" font-size="10">🧠 EfficientNet</text>
        <text x="536" y="70" text-anchor="middle" fill="#2a5a8a" font-family="Share Tech Mono" font-size="9">B0 — Transfer</text>
        <text x="536" y="82" text-anchor="middle" fill="#2a5a8a" font-family="Share Tech Mono" font-size="9">Learning</text>

        <polygon points="609,61 627,65 609,69" fill="#0050a0"/>

        <rect x="634" y="25" width="120" height="80" rx="8" fill="#001a0f" stroke="#00c853" stroke-width="1.5"/>
        <text x="694" y="55" text-anchor="middle" fill="#00c853" font-family="Share Tech Mono" font-size="10">✅ NORMAL</text>
        <line x1="660" y1="65" x2="728" y2="65" stroke="#1a3a5a" stroke-width="1" stroke-dasharray="3,3"/>
        <rect x="634" y="75" width="120" height="30" rx="0 0 8 8" fill="#1a0000" stroke="#ff1744" stroke-width="0"/>
        <text x="694" y="93" text-anchor="middle" fill="#ff1744" font-family="Share Tech Mono" font-size="10">🚨 ANOMALY</text>
      </g>
    </svg>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("### Key Metrics")
    c1, c2, c3, c4 = st.columns(4)
    metrics = [
        ("ARCHITECTURE", "EfficientNet-B0"),
        ("INPUT", "256 × 256 px"),
        ("CLASSES", "Normal / Anomaly"),
        ("METHOD", "Transfer Learning"),
    ]
    for col, (label, val) in zip([c1, c2, c3, c4], metrics):
        col.markdown(f"""
        <div class='metric-card'>
            <div class='metric-label'>{label}</div>
            <div style='font-family:"Share Tech Mono",monospace; font-size:1.1rem; color:#00d4ff; margin-top:0.4rem;'>{val}</div>
        </div>""", unsafe_allow_html=True)

    st.markdown("### Data Conversion Logic")
    st.markdown("""
    <div class='info-box'>
    📦 Raw bytes from each PCAP file are concatenated sequentially.<br>
    📐 Every <strong>196,608 bytes</strong> (= 256 × 256 × 3 channels) is converted into one RGB image.<br>
    🎨 The model learns the byte-distribution <em>texture</em> of these images to distinguish Normal vs Anomaly traffic.<br>
    🔢 If the last chunk is incomplete, it is zero-padded with <strong>0x00</strong> (black pixels).
    </div>
    """, unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════
# TAB 2 — TOPLU ANALİZ (ZIP klasörü)
# ══════════════════════════════════════════════════════════
with tab2:
    st.markdown("### 📊 Folder Analysis")
    st.markdown("""
    <div class='info-box'>
    📁 Upload a <strong>.zip</strong> file containing a <strong>mixed</strong> set of normal and anomaly images.<br>
    🔀 If filenames contain <code>normal</code> or <code>anomaly</code>, the true label is read automatically and accuracy is calculated.<br>
    🤖 The model analyzes all PNGs inside the ZIP sequentially.
    </div>
    """, unsafe_allow_html=True)

    if not model_loaded:
        st.warning("⚠️ Please upload a model from the left panel first.")
    else:
        uploaded_zip = st.file_uploader(
            "Upload mixed image folder as ZIP",
            type=["zip"],
            key="zip_upload",
            help="Place normal and anomaly PNGs randomly mixed inside the ZIP."
        )

        if uploaded_zip:
            import zipfile

            zip_bytes = io.BytesIO(uploaded_zip.read())
            with zipfile.ZipFile(zip_bytes) as zf:
                all_names = [
                    n for n in zf.namelist()
                    if n.lower().endswith((".png", ".jpg", ".jpeg"))
                    and not n.startswith("__MACOSX")
                ]

            st.markdown(f"**Found {len(all_names)} image files inside the ZIP.**")

            def guess_true_label(fname):
                low = Path(fname).name.lower()
                if "normal" in low:
                    return "NORMAL"
                if "anomal" in low or "anomaly" in low or "saldiri" in low or "attack" in low:
                    return "ANOMALY"
                return None

            has_labels = any(guess_true_label(n) is not None for n in all_names)

            if st.button(f"▶️  Analyze {len(all_names)} Images", key="run_zip"):
                progress_bar = st.progress(0, text="Opening ZIP...")
                batch_results = []
                live_images   = []

                zip_bytes.seek(0)
                with zipfile.ZipFile(zip_bytes) as zf:
                    for i, fname in enumerate(all_names):
                        raw_img = zf.read(fname)
                        img = Image.open(io.BytesIO(raw_img)).convert("RGB")
                        r   = predict_image(model, idx_to_class, img)
                        true_label = guess_true_label(fname)
                        live_images.append((img, Path(fname).name, r, true_label))

                        row = {
                            "File Name":    Path(fname).name,
                            "Prediction":   r["label"].upper(),
                            "Confidence %": round(r["confidence"] * 100, 2),
                            "Normal %":     round(r["prob_normal"] * 100, 2),
                            "Anomaly %":    round(r["prob_anomaly"] * 100, 2),
                        }
                        if has_labels:
                            row["True Label"] = true_label if true_label else "?"
                            row["Correct?"] = (
                                "✅" if true_label and true_label == r["label"].upper()
                                else ("❌" if true_label else "—")
                            )
                        batch_results.append(row)
                        progress_bar.progress(
                            (i + 1) / len(all_names),
                            text=f"Processing: {Path(fname).name}"
                        )

                df_batch = pd.DataFrame(batch_results)

                total     = len(df_batch)
                n_anom    = (df_batch["Prediction"] == "ANOMALY").sum()
                n_norm    = total - n_anom
                anom_rate = n_anom / total * 100 if total > 0 else 0

                accuracy_str = "—"
                if has_labels and "Correct?" in df_batch.columns:
                    correct = (df_batch["Correct?"] == "✅").sum()
                    labeled = df_batch["True Label"].isin(["NORMAL", "ANOMALY"]).sum()
                    accuracy_str = f"{correct/labeled*100:.1f}%" if labeled > 0 else "—"

                st.markdown("#### Summary")
                cols_m = st.columns(5 if has_labels else 4)
                metric_data = [
                    ("TOTAL", str(total), "#00d4ff"),
                    ("ANOMALY", str(n_anom), "#ff1744"),
                    ("NORMAL", str(n_norm), "#00c853"),
                    ("ANOMALY RATE", f"{anom_rate:.1f}%", "#ffaa00"),
                ]
                if has_labels:
                    metric_data.append(("ACCURACY", accuracy_str, "#a78bfa"))

                for col, (lbl, val, color) in zip(cols_m, metric_data):
                    col.markdown(f"""
                    <div class='metric-card'>
                        <div class='metric-label'>{lbl}</div>
                        <div class='metric-value' style='color:{color};'>{val}</div>
                    </div>""", unsafe_allow_html=True)

                st.markdown("<br>", unsafe_allow_html=True)

                # ── Grafikler ───────────────────────────────────
                col_pie, col_scatter = st.columns(2, gap="large")

                with col_pie:
                    fig_pie = go.Figure(go.Pie(
                        labels=["NORMAL", "ANOMALY"],
                        values=[n_norm, n_anom],
                        hole=0.55,
                        marker_colors=["#00c853", "#ff1744"],
                        textfont=dict(family="Share Tech Mono", size=12),
                    ))
                    fig_pie.update_layout(
                        paper_bgcolor="#0a0e1a",
                        plot_bgcolor="#0a0e1a",
                        legend=dict(font=dict(color="#6a9cbf", family="Share Tech Mono")),
                        margin=dict(t=20, b=20),
                        height=280,
                        title=dict(text="Prediction Distribution", font=dict(color="#3a6a8a", family="Share Tech Mono")),
                        annotations=[dict(
                            text=f"{anom_rate:.0f}%<br>Anom",
                            x=0.5, y=0.5, showarrow=False,
                            font=dict(size=18, color="#ff1744", family="Share Tech Mono")
                        )]
                    )
                    st.plotly_chart(fig_pie, use_container_width=True, config={"displayModeBar": False})

                with col_scatter:
                    dot_colors = ["#ff1744" if t == "ANOMALY" else "#00c853" for t in df_batch["Prediction"]]
                    fig_sc = go.Figure(go.Scatter(
                        x=list(range(1, total + 1)),
                        y=df_batch["Confidence %"],
                        mode="markers",
                        marker=dict(color=dot_colors, size=9, opacity=0.85,
                                    line=dict(width=1, color="#1a3a5a")),
                        text=df_batch["File Name"],
                        hovertemplate="<b>%{text}</b><br>Confidence: %{y:.1f}%<extra></extra>",
                    ))
                    fig_sc.update_layout(
                        paper_bgcolor="#0a0e1a",
                        plot_bgcolor="#0d1525",
                        xaxis=dict(title="Image #", color="#3a6a8a",
                                   gridcolor="#0f1d30", tickfont=dict(family="Share Tech Mono")),
                        yaxis=dict(title="Confidence %", color="#3a6a8a",
                                   gridcolor="#0f1d30", range=[0, 105],
                                   tickfont=dict(family="Share Tech Mono")),
                        margin=dict(t=20, b=40),
                        height=280,
                        title=dict(text="Confidence Distribution", font=dict(color="#3a6a8a", family="Share Tech Mono")),
                    )
                    st.plotly_chart(fig_sc, use_container_width=True, config={"displayModeBar": False})

                if has_labels and "True Label" in df_batch.columns:
                    labeled_df = df_batch[df_batch["True Label"].isin(["NORMAL", "ANOMALY"])]
                    if len(labeled_df) > 0:
                        from sklearn.metrics import confusion_matrix as sk_cm
                        y_true = labeled_df["True Label"].tolist()
                        y_pred = labeled_df["Prediction"].tolist()
                        cm = sk_cm(y_true, y_pred, labels=["NORMAL", "ANOMALY"])

                        fig_cm = go.Figure(go.Heatmap(
                            z=cm,
                            x=["Pred: NORMAL", "Pred: ANOMALY"],
                            y=["True: NORMAL", "True: ANOMALY"],
                            colorscale=[[0, "#0a1525"], [1, "#0050a0"]],
                            text=cm,
                            texttemplate="%{text}",
                            textfont=dict(size=22, color="white", family="Share Tech Mono"),
                            showscale=False,
                        ))
                        fig_cm.update_layout(
                            paper_bgcolor="#0a0e1a",
                            plot_bgcolor="#0a0e1a",
                            title=dict(text="Confusion Matrix", font=dict(color="#3a6a8a", family="Share Tech Mono")),
                            xaxis=dict(color="#6a9cbf", tickfont=dict(family="Share Tech Mono")),
                            yaxis=dict(color="#6a9cbf", tickfont=dict(family="Share Tech Mono")),
                            margin=dict(t=50, b=20),
                            height=280,
                        )
                        st.plotly_chart(fig_cm, use_container_width=True, config={"displayModeBar": False})

                # ── Görsel Önizleme Galerisi ────────────────────
                st.markdown("---")
                st.markdown("#### 🖼️ Image Gallery")
                st.caption("Each card is a 256×256 byte-texture image. Border color = model decision.")

                COLS = 6
                rows = [live_images[i:i+COLS] for i in range(0, len(live_images), COLS)]
                for row_imgs in rows:
                    cols = st.columns(COLS)
                    for col, (pil_img, fname, r, true_label) in zip(cols, row_imgs):
                        is_anom = r["label"].upper() == "ANOMALY"
                        border  = "#ff1744" if is_anom else "#00c853"
                        icon    = "🚨" if is_anom else "✅"
                        correct_mark = ""
                        if true_label:
                            correct_mark = " ✔" if true_label == r["label"].upper() else " ✘"

                        # Görseli base64'e çevir
                        buf = io.BytesIO()
                        pil_img.resize((128, 128), Image.NEAREST).save(buf, format="PNG")
                        b64 = __import__("base64").b64encode(buf.getvalue()).decode()

                        col.markdown(f"""
                        <div style="border:2px solid {border}; border-radius:8px; padding:4px;
                                    background:#0a0e1a; text-align:center; margin-bottom:4px;
                                    box-shadow: 0 0 8px {border}55;">
                            <img src="data:image/png;base64,{b64}" style="width:100%; border-radius:4px;"/>
                            <div style="font-family:'Share Tech Mono',monospace; font-size:0.62rem;
                                        color:{border}; margin-top:3px; line-height:1.3;">
                                {icon} {r['label'].upper()}<br>
                                <span style="color:#3a6a8a;">{r['confidence']*100:.0f}%{correct_mark}</span>
                            </div>
                        </div>""", unsafe_allow_html=True)

                if has_labels and "True Label" in df_batch.columns:
                    labeled_df2 = df_batch[df_batch["True Label"].isin(["NORMAL", "ANOMALY"])]
                    if len(labeled_df2) > 0:
                        from sklearn.metrics import classification_report, precision_recall_fscore_support

                        st.markdown("---")
                        st.markdown("#### 📋 Classification Report (Precision / Recall / F1)")

                        y_true2 = labeled_df2["True Label"].tolist()
                        y_pred2 = labeled_df2["Prediction"].tolist()

                        prec, rec, f1, sup = precision_recall_fscore_support(
                            y_true2, y_pred2, labels=["NORMAL", "ANOMALY"]
                        )
                        prec_m, rec_m, f1_m, _ = precision_recall_fscore_support(
                            y_true2, y_pred2, average="macro"
                        )

                        for cls_idx, cls_name in enumerate(["NORMAL", "ANOMALY"]):
                            cls_color = "#00c853" if cls_name == "NORMAL" else "#ff1744"
                            st.markdown(f"""
                            <div style="font-family:'Share Tech Mono',monospace; font-size:0.75rem;
                                        color:{cls_color}; letter-spacing:0.15em; margin:0.8rem 0 0.3rem;">
                            ▶ {cls_name}
                            </div>""", unsafe_allow_html=True)
                            c1, c2, c3, c4 = st.columns(4)
                            for col, (lbl, val) in zip(
                                [c1, c2, c3, c4],
                                [
                                    ("PRECISION", f"{prec[cls_idx]*100:.1f}%"),
                                    ("RECALL",    f"{rec[cls_idx]*100:.1f}%"),
                                    ("F1-SCORE",  f"{f1[cls_idx]*100:.1f}%"),
                                    ("SUPPORT",   str(int(sup[cls_idx]))),
                                ]
                            ):
                                col.markdown(f"""
                                <div class='metric-card'>
                                    <div class='metric-label'>{lbl}</div>
                                    <div style='font-family:"Share Tech Mono",monospace;
                                                font-size:1.4rem; color:{cls_color};
                                                margin-top:0.3rem;'>{val}</div>
                                </div>""", unsafe_allow_html=True)

                        st.markdown("""
                        <div style="font-family:'Share Tech Mono',monospace; font-size:0.75rem;
                                    color:#a78bfa; letter-spacing:0.15em; margin:0.8rem 0 0.3rem;">
                        ▶ MACRO AVERAGE
                        </div>""", unsafe_allow_html=True)
                        cm1, cm2, cm3 = st.columns(3)
                        for col, (lbl, val) in zip(
                            [cm1, cm2, cm3],
                            [("PRECISION", f"{prec_m*100:.1f}%"),
                             ("RECALL",    f"{rec_m*100:.1f}%"),
                             ("F1-SCORE",  f"{f1_m*100:.1f}%")]
                        ):
                            col.markdown(f"""
                            <div class='metric-card'>
                                <div class='metric-label'>{lbl}</div>
                                <div style='font-family:"Share Tech Mono",monospace;
                                            font-size:1.4rem; color:#a78bfa;
                                            margin-top:0.3rem;'>{val}</div>
                            </div>""", unsafe_allow_html=True)

                        fig_f1 = go.Figure()
                        fig_f1.add_trace(go.Bar(
                            name="Precision",
                            x=["NORMAL", "ANOMALY"],
                            y=[prec[0]*100, prec[1]*100],
                            marker_color=["#00c853", "#ff1744"],
                            opacity=0.75,
                        ))
                        fig_f1.add_trace(go.Bar(
                            name="Recall",
                            x=["NORMAL", "ANOMALY"],
                            y=[rec[0]*100, rec[1]*100],
                            marker_color=["#00a040", "#cc1030"],
                            opacity=0.75,
                        ))
                        fig_f1.add_trace(go.Bar(
                            name="F1-Score",
                            x=["NORMAL", "ANOMALY"],
                            y=[f1[0]*100, f1[1]*100],
                            marker_color=["#007030", "#aa0820"],
                            opacity=0.9,
                        ))
                        fig_f1.update_layout(
                            barmode="group",
                            paper_bgcolor="#0a0e1a",
                            plot_bgcolor="#0d1525",
                            legend=dict(font=dict(color="#6a9cbf", family="Share Tech Mono"),
                                        bgcolor="#0a0e1a"),
                            xaxis=dict(color="#6a9cbf", tickfont=dict(family="Share Tech Mono")),
                            yaxis=dict(title="Score %", range=[0, 105], color="#3a6a8a",
                                       gridcolor="#0f1d30", tickfont=dict(family="Share Tech Mono")),
                            margin=dict(t=20, b=20),
                            height=260,
                        )
                        st.plotly_chart(fig_f1, use_container_width=True, config={"displayModeBar": False})

                st.markdown("---")
                st.markdown("#### Detailed Results")
                styled = df_batch.style.apply(
                    lambda col: [
                        "background:#1a0000; color:#ff6060" if v == "ANOMALY"
                        else "background:#001a0f; color:#60cc80"
                        for v in col
                    ],
                    subset=["Prediction"]
                )
                if "Correct?" in df_batch.columns:
                    styled = styled.apply(
                        lambda col: [
                            "color:#00c853" if v == "✅"
                            else ("color:#ff1744" if v == "❌" else "color:#3a6a8a")
                            for v in col
                        ],
                        subset=["Correct?"]
                    )
                st.dataframe(styled, use_container_width=True, height=350)

                csv_out = df_batch.to_csv(index=False).encode("utf-8")
                st.download_button(
                    "⬇️ Download Results as CSV",
                    data=csv_out,
                    file_name="analysis_results.csv",
                    mime="text/csv",
                )


# ─────────────────────────────────────────────────────────
# FOOTER
# ─────────────────────────────────────────────────────────
st.markdown("---")
st.markdown("""
<div style='text-align:center; font-family:"Share Tech Mono",monospace; font-size:0.7rem; color:#1a3a5a; letter-spacing:0.15em;'>
MACHINE LEARNING-BASED DETECTION AND VISUALIZATION OF WEB TRAFFIC ANOMALIES // EfficientNet-B0 // BINARY CLASSIFICATION
</div>
""", unsafe_allow_html=True)
