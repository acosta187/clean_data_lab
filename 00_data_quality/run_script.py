# -*- coding: utf-8 -*-
"""
Created on Sat Apr  4 21:31:32 2026

@author: SUITE
"""

import pandas as pd
import numpy as np
import os
import re

os.chdir("D:/Descargas/clean_data_lab-main/clean_data_lab-main/00_data_quality")

from text_quality import (
    inject_noise,
    build_noisy_dataset,
    audit_text_quality,
    fuzzy_clean,
)

# -----------------------------
# 1. Función de normalización
# -----------------------------
def normalize_text(x):
    if pd.isna(x):
        return x
    x = str(x).lower()
    x = x.strip()
    x = re.sub(r"\s+", " ", x)  # colapsa múltiples espacios
    return x

# -----------------------------
# 2. Master list
# -----------------------------
MASTER = [
    "Miraflores", "San Juan de Lurigancho", "Santiago de Surco",
    "Villa El Salvador", "Chorrillos", "San Borja", "La Molina",
    "Los Olivos", "San Miguel", "Magdalena del Mar",
]

# Normalizar MASTER
MASTER_NORM = [normalize_text(x) for x in MASTER]

# -----------------------------
# 3. Generar dataset con ruido
# -----------------------------
df = build_noisy_dataset(MASTER, n_rows=500)

# -----------------------------
# 4. Normalizar columna noisy
# -----------------------------
df["noisy_norm"] = df["noisy"].apply(normalize_text)

# -----------------------------
# 5. Auditoría inicial
# -----------------------------
raw_audit = audit_text_quality(df["noisy_norm"])

# -----------------------------
# 6. Limpieza fuzzy
# -----------------------------
df["cleaned"] = df["noisy_norm"].apply(
    lambda x: fuzzy_clean(x, MASTER_NORM, threshold=70)
)

# Normalizar resultado final (consistencia)
df["cleaned"] = df["cleaned"].apply(normalize_text)

# -----------------------------
# 7. Auditoría final
# -----------------------------
clean_audit = audit_text_quality(df["cleaned"].fillna("NaN"))

# -----------------------------
# 8. Separar éxito vs fallos
# -----------------------------
df_failed = df[df["cleaned"].isna()]
df_success = df[df["cleaned"].notna()]

# Auditoría solo de los exitosos
clean_audit_succ = audit_text_quality(df_success["cleaned"])

# -----------------------------
# 9. Outputs clave
# -----------------------------
print("=== AUDITORÍA INICIAL ===")
print(raw_audit)

print("\n=== AUDITORÍA FINAL ===")
print(clean_audit)

print("\n=== AUDITORÍA SOLO ÉXITOS ===")
print(clean_audit_succ)

print("\nFilas fallidas:", len(df_failed))
print("Filas exitosas:", len(df_success))
