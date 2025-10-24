# ============================================================
# app_stunting.py

import streamlit as st
import pandas as pd
import numpy as np
import math, joblib
from pathlib import Path

st.set_page_config(page_title="Deteksi Stunting Balita (WHO)", page_icon="🧒", layout="wide")

# ---------- Styles: hijau muda polos ----------
st.markdown("""
<style>
/* Latar hijau muda polos */
[data-testid="stAppViewContainer"] > .main {
  background: #e9fff4;         /* hijau muda */
  padding: 24px 0 36px 0;
}
/* Kartu konten */
.card {
  background: #ffffff; border-radius: 16px; padding: 22px 24px;
  box-shadow: 0 10px 24px rgba(0,0,0,.08);
}
.center { max-width: 960px; margin: 0 auto; }
h1,h2,h3 { color:#064e47; }
.subtle { color:#0f766e; }
/* Tombol utama bulat */
.stButton>button {
  background:#14b8a6 !important; color:#fff !important; border:none !important;
  border-radius:999px !important; font-weight:700 !important;
  padding:.55rem 1.2rem !important; box-shadow:0 8px 18px rgba(20,184,166,.25);
}
.stButton>button:hover { background:#0ea5a3 !important; transform:scale(1.02); }
/* Disclaimer */
.disclaimer {
  background:#f0fffa; border-left:6px solid #14b8a6; color:#075e55;
  padding:.65rem .9rem; border-radius:8px; font-size:13px;
}
/* Link kembali */
.back { font-size:14px; color:#0f766e; }
hr { border-top:1px solid #d1fae5; }
</style>
""", unsafe_allow_html=True)

# =========================================================
# WHO Reference Data (lengkap dari user) + Util 
# =========================================================

# --- TB/U Boys (cm) : (-3SD, -2SD, Median, +2SD)
who_tbu_boys = {
    0:  (44.2, 46.1, 49.9, 53.7),
    1:  (48.9, 50.8, 54.7, 58.6),
    2:  (52.4, 54.4, 58.4, 62.4),
    3:  (55.3, 57.3, 61.4, 65.5),
    4:  (57.6, 59.7, 63.9, 68.0),
    5:  (59.6, 61.7, 65.9, 70.1),
    6:  (61.2, 63.3, 67.6, 71.9),
    7:  (62.7, 64.8, 69.2, 73.5),
    8:  (64.0, 66.2, 70.6, 75.0),
    9:  (65.2, 67.5, 72.0, 76.5),
    10: (66.4, 68.7, 73.3, 77.9),
    11: (67.6, 69.9, 74.5, 79.2),
    12: (68.6, 71.0, 75.7, 80.5),
    13: (69.6, 72.1, 76.9, 81.8),
    14: (70.6, 73.1, 78.0, 83.0),
    15: (71.6, 74.1, 79.1, 84.2),
    16: (72.5, 75.0, 80.2, 85.4),
    17: (73.3, 75.8, 81.2, 86.5),
    18: (74.2, 76.9, 82.3, 87.7),
    19: (75.0, 77.7, 83.2, 88.8),
    20: (75.8, 78.6, 84.2, 89.8),
    21: (76.5, 79.4, 85.1, 90.9),
    22: (77.2, 80.2, 86.0, 91.9),
    23: (77.8, 80.9, 86.8, 92.9),
    24: (78.7, 81.7, 87.8, 93.9),
    25: (78.6, 81.7, 88.0, 94.2),
    26: (79.3, 82.5, 88.8, 95.2),
    27: (79.9, 83.1, 89.6, 96.1),
    28: (80.5, 83.8, 90.4, 97.0),
    29: (81.1, 84.5, 91.2, 97.9),
    30: (81.7, 85.1, 91.9, 98.7),
    31: (82.3, 85.7, 92.7, 99.6),
    32: (82.8, 86.4, 93.4, 100.4),
    33: (83.4, 86.9, 94.1, 101.2),
    34: (83.9, 87.5, 94.8, 102.0),
    35: (84.4, 88.1, 95.4, 102.7),
    36: (85.0, 88.7, 96.1, 103.5),
    37: (85.5, 89.2, 96.7, 104.2),
    38: (86.0, 89.8, 97.4, 105.0),
    39: (86.5, 90.3, 98.0, 105.7),
    40: (87.0, 90.9, 98.6, 106.4),
    41: (87.5, 91.4, 99.2, 107.1),
    42: (88.0, 91.9, 99.9, 107.8),
    43: (88.4, 92.4, 100.4, 108.5),
    44: (88.9, 93.0, 101.0, 109.1),
    45: (89.3, 93.5, 101.6, 109.8),
    46: (89.8, 94.1, 102.1, 110.5),
    47: (90.3, 94.4, 102.8, 111.2),
    48: (90.7, 94.9, 103.3, 111.8),
    49: (91.2, 95.4, 103.9, 112.4),
    50: (91.6, 95.9, 104.4, 113.0),
    51: (92.1, 96.4, 105.0, 113.6),
    52: (92.5, 96.9, 105.6, 114.2),
    53: (93.0, 97.4, 106.1, 114.9),
    54: (93.4, 97.8, 106.7, 115.5),
    55: (93.9, 98.3, 107.2, 116.1),
    56: (94.3, 98.8, 107.8, 116.7),
    57: (94.7, 99.3, 108.3, 117.4),
    58: (95.2, 99.7, 108.9, 118.0),
    59: (95.6, 100.2, 109.4, 118.6),
    60: (96.1, 100.7, 110.0, 119.2),
}

# --- BB/U Boys (kg) : (-3SD, -2SD, -1SD, Median, +1SD, +2SD, +3SD)
who_bbu_boys = {
    0:(2.1,2.5,2.9,3.3,3.9,4.4,5.0), 1:(2.9,3.4,3.9,4.5,5.1,5.8,6.6),
    2:(3.8,4.3,4.9,5.6,6.3,7.1,8.0), 3:(4.4,5.0,5.7,6.4,7.2,8.0,9.0),
    4:(4.9,5.6,6.2,7.0,7.8,8.7,9.7), 5:(5.3,6.0,6.7,7.5,8.4,9.3,10.4),
    6:(5.7,6.4,7.1,7.9,8.8,9.8,10.9), 7:(5.9,6.7,7.4,8.3,9.2,10.3,11.4),
    8:(6.2,6.9,7.7,8.6,9.6,10.7,11.9), 9:(6.4,7.1,8.0,8.9,9.9,11.0,12.3),
    10:(6.6,7.4,8.2,9.2,10.2,11.4,12.7), 11:(6.8,7.6,8.4,9.4,10.5,11.7,13.0),
    12:(6.9,7.7,8.6,9.6,10.8,12.0,13.3), 13:(7.1,7.9,8.8,9.9,11.0,12.3,13.7),
    14:(7.2,8.1,9.0,10.1,11.3,12.6,14.0), 15:(7.4,8.3,9.2,10.3,11.5,12.8,14.3),
    16:(7.5,8.4,9.4,10.5,11.7,13.1,14.6), 17:(7.7,8.6,9.6,10.7,12.0,13.4,14.9),
    18:(7.8,8.8,9.8,10.9,12.2,13.7,15.3), 19:(8.0,8.9,10.0,11.1,12.5,13.9,15.6),
    20:(8.1,9.1,10.1,11.3,12.7,14.2,15.9), 21:(8.2,9.2,10.3,11.5,12.9,14.5,16.2),
    22:(8.4,9.4,10.5,11.8,13.2,14.7,16.5), 23:(8.5,9.5,10.7,12.0,13.4,15.0,16.8),
    24:(8.6,9.7,10.8,12.2,13.6,15.3,17.1), 25:(8.8,9.8,11.0,12.4,13.9,15.5,17.4),
    26:(8.9,10.0,11.1,12.5,14.1,15.8,17.8), 27:(9.1,10.1,11.3,12.7,14.4,16.1,18.1),
    28:(9.2,10.4,11.5,12.9,14.5,16.3,18.4), 29:(9.2,10.4,11.7,13.1,14.8,16.6,18.7),
    30:(9.4,10.5,11.8,13.3,15.0,16.9,19.0), 31:(9.5,10.7,12.0,13.5,15.2,17.1,19.3),
    32:(9.6,10.8,12.1,13.7,15.4,17.4,19.6), 33:(9.8,11.0,12.3,13.8,15.6,17.6,19.9),
    34:(9.8,11.1,12.4,14.0,15.8,17.8,20.2), 35:(9.9,11.2,12.6,14.2,16.0,18.1,20.4),
    36:(10.0,11.3,12.7,14.3,16.2,18.3,20.7), 37:(10.1,11.4,12.9,14.5,16.4,18.6,21.0),
    38:(10.2,11.5,13.0,14.7,16.6,18.8,21.3), 39:(10.3,11.6,13.3,14.8,16.8,19.0,21.6),
    40:(10.4,11.8,13.3,15.0,17.0,19.3,21.9), 41:(10.5,11.9,13.4,15.2,17.2,19.5,22.1),
    42:(10.6,12.0,13.6,15.3,17.4,19.7,22.4), 43:(10.7,12.2,13.7,15.5,17.6,20.0,22.7),
    44:(10.8,12.2,13.8,15.7,17.8,20.2,23.0), 45:(10.9,12.4,14.0,15.8,18.0,20.5,23.3),
    46:(11.0,12.5,14.1,16.0,18.2,20.7,23.6), 47:(11.1,12.6,14.3,16.2,18.4,20.9,23.9),
    48:(11.2,12.7,14.4,16.3,18.6,21.2,24.2), 49:(11.3,12.8,14.5,16.5,18.8,21.4,24.5),
    50:(11.4,12.9,14.7,16.7,19.0,21.7,24.8), 51:(11.5,13.1,14.8,16.8,19.2,21.9,25.1),
    52:(11.6,13.2,15.0,17.0,19.4,22.2,25.4), 53:(11.7,13.3,15.1,17.2,19.6,22.4,25.7),
    54:(11.8,13.5,15.2,17.3,19.8,22.7,26.0), 55:(11.9,13.5,15.4,17.5,20.0,22.9,26.3),
    56:(12.0,13.6,15.5,17.7,20.2,23.2,26.6), 57:(12.1,13.7,15.6,17.8,20.4,23.4,26.9),
    58:(12.2,13.8,15.8,18.0,20.6,23.7,27.2), 59:(12.3,14.0,15.9,18.2,20.8,23.9,27.6),
    60:(12.4,14.1,16.0,18.3,21.0,24.2,27.9),
}

# --- TB/U Girls (cm) : (-3SD, -2SD, Median, +2SD)
who_tbu_girls = {
    0:(43.6,45.4,49.1,52.9), 1:(47.8,49.8,53.7,57.6),
    2:(51.0,53.0,57.1,61.1), 3:(53.5,55.6,59.8,64.0),
    4:(55.6,57.9,62.1,66.4), 5:(57.4,59.6,64.0,68.3),
    6:(58.9,61.2,65.7,70.2), 7:(60.3,62.7,67.3,71.9),
    8:(61.7,64.0,68.7,73.5), 9:(62.9,65.3,70.1,75.0),
    10:(64.1,66.5,71.5,76.4), 11:(65.2,67.7,72.8,77.8),
    12:(66.3,68.9,74.0,79.2), 13:(67.3,69.9,75.2,80.5),
    14:(68.3,71.0,76.4,81.7), 15:(69.3,72.0,77.5,83.0),
    16:(70.2,73.0,78.6,84.2), 17:(71.1,74.0,79.7,85.4),
    18:(72.0,74.9,80.7,86.5), 19:(72.8,75.8,81.7,87.6),
    20:(73.7,76.7,82.7,88.7), 21:(74.5,77.5,83.7,89.7),
    22:(75.2,78.4,84.6,90.8), 23:(76.0,79.2,85.5,91.9),
    24:(76.7,80.0,86.4,92.9), 25:(76.8,80.0,86.6,93.1),
    26:(77.5,80.8,87.4,94.1), 27:(78.1,81.5,88.3,95.0),
    28:(78.8,82.2,89.1,96.0), 29:(79.5,82.9,89.9,96.9),
    30:(80.1,83.6,90.7,97.7), 31:(80.7,84.3,91.4,98.6),
    32:(81.3,84.9,92.2,99.4), 33:(81.9,85.6,92.9,100.3),
    34:(82.5,86.2,93.6,101.1), 35:(83.1,86.8,94.4,101.9),
    36:(83.6,87.4,95.1,102.7), 37:(84.2,88.0,95.7,103.4),
    38:(84.7,88.6,96.4,104.2), 39:(85.3,89.2,97.1,105.0),
    40:(85.8,89.8,97.7,105.7), 41:(86.3,90.4,98.4,106.4),
    42:(86.8,90.9,99.0,107.1), 43:(87.4,91.5,99.7,107.9),
    44:(87.9,92.0,100.3,108.6), 45:(88.4,92.5,100.9,109.3),
    46:(88.9,93.1,101.5,110.0), 47:(89.3,93.6,102.1,110.7),
    48:(89.8,94.1,102.7,111.3), 49:(90.3,94.6,103.3,112.0),
    50:(90.7,95.1,103.9,112.7), 51:(91.2,95.6,104.5,113.3),
    52:(91.7,96.1,105.0,114.0), 53:(92.1,96.6,105.6,114.6),
    54:(92.6,97.1,106.2,115.2), 55:(93.0,97.6,106.7,115.9),
    56:(93.4,98.1,107.3,116.5), 57:(93.9,98.5,107.8,117.1),
    58:(94.3,99.0,108.4,117.7), 59:(94.7,99.5,108.9,118.3),
    60:(95.2,99.9,109.4,118.9),
}

# --- BB/U Girls (kg) : (-3SD, -2SD, -1SD, Median, +1SD, +2SD, +3SD)
who_bbu_girls = {
    0:(2.0,2.4,2.8,3.2,3.7,4.2,4.8), 1:(2.7,3.1,3.6,4.2,4.8,5.5,6.2),
    2:(3.2,3.9,4.5,5.1,5.8,6.7,7.5), 3:(4.0,4.5,5.2,5.8,6.6,7.5,8.5),
    4:(4.4,5.0,5.7,6.4,7.3,8.2,9.3), 5:(4.8,5.4,6.1,6.9,7.8,8.8,10.0),
    6:(5.1,5.7,6.5,7.3,8.2,9.3,10.6), 7:(5.3,6.0,6.8,7.6,8.6,9.8,11.1),
    8:(5.6,6.3,7.0,7.9,9.0,10.2,11.6), 9:(5.8,6.5,7.3,8.2,9.3,10.5,12.0),
    10:(5.9,6.7,7.5,8.5,9.6,10.9,12.4), 11:(6.1,6.9,7.7,8.7,9.9,11.2,12.8),
    12:(6.3,7.0,7.9,8.9,10.1,11.5,13.1), 13:(6.4,7.2,8.1,9.2,10.4,11.8,13.5),
    14:(6.6,7.4,8.3,9.4,10.6,12.1,13.8), 15:(6.7,7.6,8.5,9.6,10.9,12.4,14.1),
    16:(6.9,7.7,8.7,9.8,11.1,12.6,14.5), 17:(7.0,7.9,8.9,10.0,11.4,12.9,14.8),
    18:(7.2,8.1,9.1,10.2,11.6,13.2,15.1), 19:(7.3,8.2,9.2,10.4,11.8,13.5,15.4),
    20:(7.5,8.4,9.4,10.6,12.1,13.7,15.7), 21:(7.6,8.6,9.6,10.9,12.3,14.0,16.0),
    22:(7.8,8.7,9.8,11.1,12.5,14.3,16.4), 23:(7.9,8.9,10.0,11.3,12.8,14.5,16.7),
    24:(8.1,9.0,10.2,11.5,13.0,14.8,17.0), 25:(8.2,9.2,10.3,11.7,13.3,15.0,17.3),
    26:(8.4,9.4,10.5,11.9,13.5,15.3,17.7), 27:(8.5,9.5,10.7,12.1,13.7,15.5,18.0),
    28:(8.6,9.7,10.9,12.3,14.0,15.7,18.3), 29:(8.8,9.8,11.1,12.5,14.2,16.0,18.7),
    30:(8.9,10.0,11.2,12.7,14.4,16.5,19.0), 31:(9.0,10.1,11.4,12.9,14.7,16.8,19.3),
    32:(9.3,10.3,11.6,13.1,14.9,17.0,19.6), 33:(9.3,10.4,11.7,13.3,15.1,17.3,19.9),
    34:(9.4,10.5,11.9,13.5,15.4,17.6,20.3), 35:(9.6,10.7,12.0,13.7,15.6,17.9,20.6),
    36:(9.6,10.8,12.2,13.9,15.8,18.1,20.9), 37:(9.7,10.9,12.4,14.0,16.0,18.4,21.3),
    38:(9.8,11.1,12.5,14.2,16.3,18.7,21.6), 39:(9.9,11.2,12.7,14.4,16.5,19.0,22.0),
    40:(10.1,11.3,12.8,14.6,16.7,19.2,22.3), 41:(10.2,11.5,13.0,14.8,16.9,19.5,22.7),
    42:(10.3,11.6,13.1,15.0,17.2,19.8,23.0), 43:(10.4,11.7,13.3,15.2,17.4,20.1,23.4),
    44:(10.5,11.8,13.4,15.3,17.6,20.4,23.7), 45:(10.6,12.0,13.6,15.5,17.8,20.7,24.1),
    46:(10.7,12.1,13.7,15.7,18.1,20.9,24.5), 47:(10.8,12.2,13.9,15.9,18.3,21.2,24.8),
    48:(10.9,12.3,14.0,16.1,18.5,21.5,25.2), 49:(11.0,12.4,14.2,16.3,18.8,21.8,25.5),
    50:(11.1,12.6,14.3,16.4,19.0,22.1,25.9), 51:(11.2,12.7,14.5,16.6,19.2,22.4,26.3),
    52:(11.3,12.8,14.6,16.8,19.4,22.6,26.6), 53:(11.4,12.9,14.8,17.0,19.7,22.9,27.0),
    54:(11.5,13.0,14.9,17.2,19.9,23.2,27.4), 55:(11.6,13.3,15.1,17.3,20.1,23.5,27.7),
    56:(11.7,13.3,15.2,17.5,20.3,23.8,28.1), 57:(11.8,13.4,15.3,17.7,20.5,24.1,28.5),
    58:(11.9,13.5,15.5,17.9,20.8,24.4,28.9), 59:(12.0,13.6,15.6,18.0,21.0,24.6,29.2),
    60:(12.1,13.7,15.8,18.2,21.2,24.9,29.5),
}


import math
import numpy as np

# ----------------- Util mengambil batas WHO per usia -----------------
def _round_age(age_month: int) -> int:
    return int(np.clip(round(age_month), 0, 60))

def who_tbu_thresholds(gender: str, age_month: int):
    a = _round_age(age_month)
    if gender == "Laki-laki":
        m3, m2, med, p2 = who_tbu_boys[a]
    else:
        m3, m2, med, p2 = who_tbu_girls[a]
    return m3, m2, med, p2

def who_bbu_row(gender: str, age_month: int):
    a = _round_age(age_month)
    return who_bbu_boys[a] if gender == "Laki-laki" else who_bbu_girls[a]

# ----------------- Kategorisasi WHO -----------------
def categorize_tbu(gender: str, age_m: int, height_cm: float):
    m3, m2, med, p2 = who_tbu_thresholds(gender, age_m)
    if height_cm < m3:
        cat = "Sangat Pendek (Severe Stunting)"
    elif height_cm < m2:
        cat = "Pendek (Stunting)"
    elif height_cm <= p2:
        cat = "Normal"
    else:
        cat = "Tinggi"
    return cat, (round(m3,1), round(m2,1), round(med,1), round(p2,1))

def categorize_bbu(gender: str, age_m: int, weight_kg: float):
    m3, m2, m1, med, p1, p2, p3 = who_bbu_row(gender, age_m)
    if weight_kg < m3:
        cat = "Gizi Buruk"
    elif weight_kg < m2:
        cat = "Gizi Kurang"
    elif weight_kg <= p2:
        cat = "Gizi Baik"
    else:
        cat = "Gizi Lebih"
    return cat, (round(m3,1), round(m2,1), round(med,1), round(p2,1))

# ----------------- Z-score & Probabilitas WHO-like -----------------
def _z_from_points(value: float, bands: dict):
    sds = sorted(bands.keys())
    xs = [bands[s] for s in sds]
    if value <= xs[0]:
        s1, s2, x1, x2 = sds[0], sds[1], xs[0], xs[1]
    elif value >= xs[-1]:
        s1, s2, x1, x2 = sds[-2], sds[-1], xs[-2], xs[-1]
    else:
        for i in range(len(xs)-1):
            if xs[i] <= value <= xs[i+1]:
                s1, s2, x1, x2 = sds[i], sds[i+1], xs[i], xs[i+1]
                break
    if x2 == x1:
        return float(s1)
    frac = (value - x1) / (x2 - x1)
    return s1 + frac * (s2 - s1)

def _sigmoid_prob(z: float, k: float = 2.0, threshold: float = -2.0) -> float:
    return 1.0 / (1.0 + math.exp(k * (z - threshold)))

def who_probability(gender: str, age_m: int, height_cm: float, weight_kg: float):
    m3, m2, med, p2 = who_tbu_thresholds(gender, age_m)
    z_tbu = _z_from_points(height_cm, {-3: m3, -2: m2, 0: med, +2: p2})
    p_tbu = _sigmoid_prob(z_tbu, k=2.0, threshold=-2.0)

    m3w, m2w, m1w, medw, p1w, p2w, p3w = who_bbu_row(gender, age_m)
    z_bbu = _z_from_points(weight_kg, {-3: m3w, -2: m2w, 0: medw, +2: p2w})
    p_bbu = _sigmoid_prob(z_bbu, k=1.5, threshold=-2.0)

    p_final = float(np.clip(0.8 * p_tbu + 0.2 * p_bbu, 0.0, 1.0))
    return p_final, z_tbu, z_bbu

# -------------------- Saran --------------------
def saran(tbu: str, bbu: str) -> str:
    if "Pendek" in tbu:
        title = "Rekomendasi Terkait Pertumbuhan Anak:"
        lines = [
            "- Konsultasikan dengan tenaga kesehatan, seperti dokter atau bidan.",
            "- Pantau tinggi badan secara berkala.",
            "- Utamakan protein hewani (ikan, telur, ayam) serta zat besi dan zink.",
            "- Perhatikan imunisasi dan riwayat infeksi."
            
        ]
    elif bbu in ["Gizi Buruk", "Gizi Kurang"]:
        title = "Rekomendasi Terkait Status Gizi:"
        lines = [
            "- Konsultasi dengan ahli gizi bila kenaikan tidak sesuai.",
            "- Evaluasi kembali asupan energi dan protein anak.",
            "- MP-ASI padat gizi 3x/hari dan selingan 1–2x.",
            "- Pantau berat badan tiap 2–4 minggu."
            
        ]
    else:
        title = "Rekomendasi untuk Menjaga Pertumbuhan yang Baik:"
        lines = [
            "- Pertahankan pola makan seimbang (karbohidrat, protein, sayur, buah).",
            "- Pastikan istirahat cukup dan aktivitas fisik rutin.",
            "- Lakukan pemantauan tinggi dan berat badan secara berkala."
        ]
    return f"**{title}**\n" + "\n".join(lines)


# ============================================================
# 🔹 STATE & NAVIGATION
# ============================================================
if "view" not in st.session_state:
    st.session_state.view = "home"

def go(view_name: str):
    st.session_state.view = view_name

# ============================================================
# 🔹 HALAMAN HOME
# ============================================================

def render_home():
    # st.set_page_config(layout="wide")

    # ======== Header dengan Gambar Bayi ========
    left, right = st.columns([2, 1])  # kiri: teks, kanan: gambar
    with left:
        st.markdown("""
        <div style='text-align:left; padding-top:20px;'>
            <h1 style='color:#008B8B; font-weight:800; line-height:1.2;'>
                Selamat Datang di <em>Prediksi Resiko Stunting</em> pada Balita
            </h1>
            <p style='font-size:18px; color:#006666;'>
                Silakan pilih opsi Prediksi:
            </p>
        </div>
        """, unsafe_allow_html=True)

    with right:
        # Gambar bayi lucu (ikon Flaticon)
        st.markdown("""
        <div style='text-align:center;'>
            <img src="https://cdn-icons-png.flaticon.com/512/2641/2641391.png" 
                 width='160' alt='bayi' style='margin-top:10px;'>
        </div>
        """, unsafe_allow_html=True)

    # ======== Tombol Tengah ========
    st.markdown("""
    <style>
    .button-container {
        display: flex;
        justify-content: center;
        align-items: center;
        gap: 40px;
        margin-top: 25px;
        flex-wrap: wrap;
    }
    .stButton>button {
        background-color: #00B3B3;
        color: white;
        border: none;
        border-radius: 25px;
        font-size: 18px;
        font-weight: 600;
        padding: 12px 30px;
        box-shadow: 0px 4px 10px rgba(0, 139, 139, 0.3);
        transition: all 0.25s ease-in-out;
    }
    .stButton>button:hover {
        background-color: #008B8B;
        transform: scale(1.05);
        box-shadow: 0px 6px 14px rgba(0, 139, 139, 0.5);
    }
    </style>
    """, unsafe_allow_html=True)

    st.markdown("<div class='button-container'>", unsafe_allow_html=True)
    col1, col2 = st.columns(2, gap="medium")
    with col1:
        st.button("🧍‍♀️ PREDIKSI INDIVIDU", use_container_width=True, on_click=lambda: go("individu"))
    with col2:
        st.button("🗂️ PREDIKSI KELOMPOK (CSV)", use_container_width=True, on_click=lambda: go("kelompok"))
    st.markdown("</div>", unsafe_allow_html=True)

    # ======== Disclaimer ========
    st.markdown("""
    <style>
    .disclaimer {
        background-color: #E6FFF8;
        color: #006666;
        border-left: 6px solid #00B3B3;
        border-radius: 10px;
        padding: 14px 20px;
        font-size: 18px;
        font-weight: 600;
        text-align: center;
        margin-top: 25px;
    }
    </style>

    <div class="disclaimer">
    ⚠️ <b>Disclaimer</b> —
    <i>Hasil bersifat informatif dan tidak menggantikan penilaian medis profesional.</i>
    </div>
    """, unsafe_allow_html=True)






# ============================================================
# 🔹 HALAMAN DETEKSI INDIVIDU
# ============================================================
def render_individu():
    st.markdown('<div class="card" style="max-width:900px; margin:auto;">', unsafe_allow_html=True)
    st.button("↩️ Kembali", on_click=lambda: go("home"))
    st.markdown("<h2>Analisis Prediksi Stunting Individu</h2>", unsafe_allow_html=True)

    c1, c2 = st.columns(2)
    with c1:
        jk = st.radio("Jenis Kelamin", ["Laki-laki", "Perempuan"], horizontal=True)
        usia = st.number_input("Usia (bulan)", 0, 60, 3)
        bb = st.number_input("Berat badan sekarang (kg)", 1.0, 30.0, 4.0)
    with c2:
        tb = st.number_input("Tinggi badan sekarang (cm)", 40.0, 130.0, 50.0)
        bb_l = st.number_input("Berat lahir (kg)", 1.0, 6.0, 3.0)
        tb_l = st.number_input("Tinggi lahir (cm)", 30.0, 60.0, 40.0)

    if st.button("🔍 Prediksi Sekarang"):
        tbu_cat, _ = categorize_tbu(jk, usia, tb)
        bbu_cat, _ = categorize_bbu(jk, usia, bb)
        p_who, z_tb, z_bb = who_probability(jk, usia, tb, bb)
        final_label = "Stunting" if "Pendek" in tbu_cat else "Tidak Stunting"

        st.markdown("<hr/>", unsafe_allow_html=True)
        c1, c2, c3 = st.columns(3)
        with c1: st.metric("Probabilitas Risiko", f"{p_who:.3f}")
        with c2: st.metric("TB/U", tbu_cat)
        with c3: st.metric("BB/U", bbu_cat)

        st.write(f"**Prediksi:** {final_label}")
        st.caption(f"z-score TB/U = {z_tb:.2f}, BB/U = {z_bb:.2f}")
        st.info(saran(tbu_cat, bbu_cat))
    st.markdown("</div>", unsafe_allow_html=True)

# ============================================================
# 🔹 HALAMAN DETEKSI KELOMPOK (CSV)
# ============================================================
def render_kelompok():
    # Kartu/header
    st.markdown(
        """
        <div class="card" style="max-width:900px; margin:auto;">
        """,
        unsafe_allow_html=True,
    )
    st.button("↩️ Kembali", on_click=lambda: go("home"))
    st.markdown(
        "<h2>Analisis Prediksi Kelompok (CSV)</h2><p class='subtle'>Unggah data sesuai template.</p>",
        unsafe_allow_html=True,
    )

    # Template CSV
    required = [
        "jenis_kelamin",
        "usia_bulan",
        "berat_lahir_kg",
        "tinggi_lahir_cm",
        "berat_badan_kg",
        "tinggi_badan_cm",
    ]
    template = pd.DataFrame(columns=required)
    st.download_button(
        "⬇️ Unduh Template CSV",
        data=template.to_csv(index=False).encode("utf-8"),
        file_name="template_prediksi_stunting.csv",
        mime="text/csv",
    )

    # Helper cast
    def to_float(x):
        try:
            if isinstance(x, str):
                x = x.replace(",", ".").strip()
            return float(x)
        except Exception:
            return np.nan

    def to_int(x):
        try:
            return int(float(str(x).replace(",", ".").strip()))
        except Exception:
            return np.nan

    # Upload & proses
    file = st.file_uploader("Unggah file CSV", type=["csv"])
    if file is not None:
        try:
            df = pd.read_csv(file)
        except Exception:
            file.seek(0)
            df = pd.read_csv(file, encoding="utf-8")

        # Normalisasi nama kolom
        df.columns = [c.strip().lower() for c in df.columns]

        if not all(c in df.columns for c in required):
            st.error(
                "Kolom tidak lengkap. Kolom yang dibutuhkan: "
                + ", ".join(required)
            )
        else:
            rows = []
            for r in df.itertuples(index=False):
                jk   = str(getattr(r, "jenis_kelamin"))
                usia = to_int(getattr(r, "usia_bulan"))
                bb   = to_float(getattr(r, "berat_badan_kg"))
                tb   = to_float(getattr(r, "tinggi_badan_cm"))
                bbl  = to_float(getattr(r, "berat_lahir_kg"))
                tbl  = to_float(getattr(r, "tinggi_lahir_cm"))

                # Skip baris yang tidak valid
                if np.isnan(usia) or np.isnan(bb) or np.isnan(tb):
                    continue

                # Fungsi-fungsi di bawah diasumsikan sudah ada
                tbu_cat, _ = categorize_tbu(jk, usia, tb)
                bbu_cat, _ = categorize_bbu(jk, usia, bb)
                p_who, z_tb, z_bb = who_probability(jk, usia, tb, bb)

                label = "Stunting" if "Pendek" in tbu_cat else "Tidak Stunting"

                rows.append(
                    {
                        "jenis_kelamin": jk,
                        "usia_bulan": int(usia),
                        "berat_lahir_kg": bbl,
                        "tinggi_lahir_cm": tbl,
                        "berat_badan_kg": bb,
                        "tinggi_badan_cm": tb,
                        "TB/U": tbu_cat,
                        "BB/U": bbu_cat,
                        "Prob_Risiko": round(float(p_who), 3),
                        "z_TBU": round(float(z_tb), 2),
                        "z_BBU": round(float(z_bb), 2),
                        "Prediksi": label,
                    }
                )

            if len(rows) == 0:
                st.warning("Tidak ada baris valid untuk diproses.")
            else:
                out = pd.DataFrame(rows)
                st.success("Prediksi selesai.")
                st.dataframe(out, use_container_width=True)
                st.download_button(
                    "⬇️ Unduh Hasil (CSV)",
                    data=out.to_csv(index=False).encode("utf-8"),
                    file_name="hasil_prediksi_stunting.csv",
                    mime="text/csv",
                )

    st.markdown("</div>", unsafe_allow_html=True)


# ============================================================
# 🔹 ROUTING
# ============================================================
if st.session_state.view == "home":
    render_home()
elif st.session_state.view == "individu":
    render_individu()
elif st.session_state.view == "kelompok":
    render_kelompok()


# -------------------- Muat Model Backend (tidak tampil) --------------------
MODEL_PATH, SCALER_PATH = Path("clf_final.joblib"), Path("scaler.joblib")
model, scaler = None, None
if MODEL_PATH.exists(): model = joblib.load(MODEL_PATH)
if SCALER_PATH.exists(): scaler = joblib.load(SCALER_PATH)

