"""
text_quality.py
===============
Toolkit práctico para auditoría de calidad de texto, simulación de ruido
y limpieza por fuzzy-match — listo para integrarse en cualquier flujo pandas.
"""

import random
import pandas as pd
import numpy as np
import nlpaug.augmenter.char as nac
from rapidfuzz import process, fuzz


# ─────────────────────────────────────────────
# 1. NOISE SIMULATOR
# ─────────────────────────────────────────────

def inject_noise(
    text,
    p_keyboard=0.30,   # Prob. de error por tecla adyacente
    p_deletion=0.20,   # Prob. de eliminación aleatoria de caracter
    p_format=0.20,     # Prob. de distorsión de formato (casing / espacios)
    char_level=0.10    # Fracción de caracteres afectados por error (0–1)
):
    """
    Inyecta ruido realista de digitación humana en un string.

    Parameters
    ----------
    text       : str   — Texto de entrada.
    p_keyboard : float — Probabilidad de error por tecla adyacente.
    p_deletion : float — Probabilidad de eliminación aleatoria de caracter.
    p_format   : float — Probabilidad de distorsión de formato.
    char_level : float — Agresividad por error (fracción de chars modificados).

    Returns
    -------
    str — Versión ruidosa del input.

    Notes
    -----
    Las probabilidades son mutuamente excluyentes y se evalúan en orden.
    La probabilidad restante (1 - suma) retorna el texto limpio, preservando
    una tasa realista de datos sin distorsión.
    """
    t    = str(text)
    roll = random.random()

    aug_keyboard = nac.KeyboardAug(aug_char_p=char_level, aug_word_p=0.5)
    aug_delete   = nac.RandomCharAug(action="delete", aug_char_p=char_level)

    if roll < p_keyboard:
        return aug_keyboard.augment(t)[0]

    if roll < p_keyboard + p_deletion:
        return aug_delete.augment(t)[0]

    if roll < p_keyboard + p_deletion + p_format:
        return random.choice([
            t.upper(),
            t.lower(),
            f"  {t}  ",
            t.replace(" ", ""),
            t.capitalize(),
        ])

    return t  # sin distorsión


# ─────────────────────────────────────────────
# 2. DATA QUALITY AUDIT
# ─────────────────────────────────────────────

def audit_text_quality(series):
    """
    Calcula un scorecard de calidad para una Series de texto.

    Metrics
    -------
    total_records    : int   — Cantidad de filas.
    unique_values    : int   — Valores distintos.
    uniqueness_ratio : float — unique / total. Menor = más estandarizado.
    text_entropy     : int   — Tamaño del vocabulario de caracteres en la columna.
    noise_rate_pct   : float — % de filas con caracteres fuera de [a-zA-Z .].

    Interpretation Guide
    --------------------
    Metric             Rango        Status       Significado
    ──────────────────────────────────────────────────────────────
    uniqueness_ratio   < 0.10       EXCELLENT    Datos muy estandarizados.
                       0.11–0.35    DISPERSED    Requiere clustering / limpieza.
                       > 0.40       CRITICAL     Datos fragmentados / sucios.
    ──────────────────────────────────────────────────────────────
    noise_rate_pct     < 1 %        CLEAN        Caracteres normales.
                       > 5 %        NOISY        Símbolos / errores de encoding.
    ──────────────────────────────────────────────────────────────
    text_entropy       25–45        NORMAL       Alfabeto estándar A–Z.
                       > 55         SUSPICIOUS   Problemas de codificación UTF-8.

    Parameters
    ----------
    series : pd.Series — Columna a evaluar.

    Returns
    -------
    pd.Series — Scorecard con nombre por métrica.
    """
    total   = len(series)
    uniques = series.nunique()
    corpus  = "".join(series.astype(str))

    return pd.Series({
        "total_records"    : total,
        "unique_values"    : uniques,
        "uniqueness_ratio" : round(uniques / total, 4) if total else 0,
        "text_entropy"     : len(set(corpus)),
        "noise_rate_pct"   : round(
            series.astype(str).str.contains(r'[^a-zA-Z\s.]').sum() / total * 100, 2
        ),
    })


# ─────────────────────────────────────────────
# 3. FUZZY CLEANER
# ─────────────────────────────────────────────

def fuzzy_clean(text, master_list, threshold=65):
    """
    Mapea un valor ruidoso al entry más cercano en una lista maestra de referencia.

    Usa `token_sort_ratio` (RapidFuzz), por lo que variaciones en el orden de
    palabras y espacios extra no penalizan el score de coincidencia.

    Parameters
    ----------
    text        : str       — Input crudo / ruidoso.
    master_list : list[str] — Valores de referencia autorizados.
    threshold   : int       — Score mínimo (0–100) para aceptar un match.
                              Valores por debajo retornan NaN.

    Returns
    -------
    str | float — Valor de la lista maestra, o np.nan si la confianza es baja.

    Notes
    -----
    Retornar NaN (en vez de un string centinela) mantiene la columna compatible
    con las utilidades de imputación de pandas y los pipelines de sklearn.
    """
    match, score, _ = process.extractOne(
        text, master_list, scorer=fuzz.token_sort_ratio
    )
    return match if score >= threshold else np.nan


# ─────────────────────────────────────────────
# 4. SYNTHETIC DATASET BUILDER
# ─────────────────────────────────────────────

def build_noisy_dataset(master_list, n_rows=500, noise_kwargs=None):
    """
    Genera un DataFrame sintético con una columna de referencia limpia y una
    versión ruidosa, para testear pipelines de limpieza.

    Parameters
    ----------
    master_list  : list[str]   — Pool de valores de referencia a samplear.
    n_rows       : int         — Número de filas a generar.
    noise_kwargs : dict | None — Keyword arguments pasados a `inject_noise`.

    Returns
    -------
    pd.DataFrame con columnas:
        original — Valor ground-truth (sampleado de master_list).
        noisy    — Versión distorsionada producida por inject_noise.
    """
    noise_kwargs = noise_kwargs or {}
    rows = []
    for _ in range(n_rows):
        original = random.choice(master_list)
        rows.append({"original": original, "noisy": inject_noise(original, **noise_kwargs)})
    return pd.DataFrame(rows)




# ─────────────────────────────────────────────
# SCRIPTING — ejecutar con: python text_quality.py
# ─────────────────────────────────────────────

if __name__ == "__main__":

    MASTER = [
        "Miraflores", "San Juan de Lurigancho", "Santiago de Surco",
        "Villa El Salvador", "Chorrillos", "San Borja", "La Molina",
        "Los Olivos", "San Miguel", "Magdalena del Mar",
    ]

    # 1 — Generar data sintética sucia
    df = build_noisy_dataset(MASTER, n_rows=500)

    # 2 — Auditar calidad del campo ruidoso
    print("─── Raw data quality ───")
    print(audit_text_quality(df["noisy"]).to_string())

    # 3 — Limpiar con fuzzy matcher (NaN para matches de baja confianza)
    df["cleaned"] = df["noisy"].apply(fuzzy_clean, master_list=MASTER, threshold=65)

    # 4 — Auditar columna limpiada
    print("\n─── Cleaned data quality ───")
    print(audit_text_quality(df["cleaned"].fillna("NaN")).to_string())



    # 6 — Muestra de resultados
    print("\n─── Sample (10 rows) ───")
    print(df[["noisy", "cleaned"]].sample(10).to_string(index=False))
