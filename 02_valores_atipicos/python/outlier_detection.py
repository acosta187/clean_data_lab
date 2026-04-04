# ============================================================
# DETECCIÓN DE OUTLIERS — ANÁLISIS EXPLORATORIO
# ============================================================
# Script de análisis univariado y multivariado de valores
# atípicos sobre distintos datasets clásicos de Machine Learning.
#
# Datasets disponibles:
#   - Wine Quality (Rojo y Blanco)  ← analizado en detalle
#   - Iris
#   - Boston Housing
#   - Breast Cancer Wisconsin
#
# Autor: Script de práctica — detección de outliers con Python
# ============================================================


# ============================================================
# 1. IMPORTS
# ============================================================

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import IsolationForest
from sklearn.decomposition import PCA


# ============================================================
# 2. CARGA DE DATASETS
# ============================================================

print("=" * 60)
print("CARGANDO DATASETS")
print("=" * 60)

# ── Wine Quality: Rojo ──────────────────────────────────────
wine_cols = [
    "fixed_acidity", "volatile_acidity", "citric_acid",
    "residual_sugar", "chlorides", "free_sulfur_dioxide",
    "total_sulfur_dioxide", "density", "pH", "sulphates",
    "alcohol", "quality"
]

wine_red = pd.read_csv(
    "https://raw.githubusercontent.com/jbrownlee/Datasets/master/winequality-red.csv",
    sep=",", header=None, names=wine_cols
)

# ── Wine Quality: Blanco ────────────────────────────────────
wine_white = pd.read_csv(
    "https://raw.githubusercontent.com/jbrownlee/Datasets/master/winequality-white.csv",
    sep=",", header=None, names=wine_cols
)

# ── Iris ────────────────────────────────────────────────────
iris_cols = ["sepal_length", "sepal_width", "petal_length", "petal_width", "class"]
iris = pd.read_csv(
    "https://raw.githubusercontent.com/jbrownlee/Datasets/master/iris.csv",
    names=iris_cols
)

# ── Boston Housing ──────────────────────────────────────────
boston_cols = [
    "CRIM", "ZN", "INDUS", "CHAS", "NOX", "RM", "AGE", "DIS",
    "RAD", "TAX", "PTRATIO", "B", "LSTAT", "MEDV"
]
boston = pd.read_csv(
    "https://raw.githubusercontent.com/jbrownlee/Datasets/master/housing.data",
    names=boston_cols, sep=r'\s+'
)

# ── Breast Cancer Wisconsin ─────────────────────────────────
cancer_cols = [
    "id", "clump_thickness", "uniform_cell_size", "uniform_cell_shape",
    "marginal_adhesion", "single_epithelial_cell_size", "bare_nuclei",
    "bland_chromatin", "normal_nucleoli", "mitoses", "class"
]
cancer = pd.read_csv(
    "https://raw.githubusercontent.com/jbrownlee/Datasets/master/breast-cancer-wisconsin.csv",
    names=cancer_cols
)

# ── Resumen de datasets cargados ────────────────────────────
datasets_info = {
    "Wine Red":        wine_red,
    "Wine White":      wine_white,
    "Iris":            iris,
    "Boston Housing":  boston,
    "Breast Cancer":   cancer,
}

print(f"\n{'Dataset':<20} {'Filas':>7} {'Columnas':>10}")
print("-" * 42)
for name, df in datasets_info.items():
    print(f"{name:<20} {df.shape[0]:>7} {df.shape[1]:>10}")

print("\n[OK] Datasets cargados correctamente.\n")


# ============================================================
# 3. ANÁLISIS PRINCIPAL: WINE QUALITY (ROJO)
# ============================================================
# Se trabaja con wine_red como dataset de referencia.
# Al final del script se sugiere replicar el análisis con
# los demás datasets disponibles.
# ============================================================

print("=" * 60)
print("ANÁLISIS: WINE QUALITY — ROJO")
print("=" * 60)

data = wine_red.copy()
features = [c for c in data.columns if c != "quality"]

print(f"\nShape: {data.shape}")
print(f"\nDistribución de calidad (quality):")
print(data["quality"].value_counts().sort_index().to_string())
print(f"\nEstadísticas descriptivas:")
print(data[features].describe().round(3).to_string())


# ============================================================
# 4. OUTLIERS UNIVARIADOS — MÉTODO IQR
# ============================================================

print("\n" + "=" * 60)
print("OUTLIERS UNIVARIADOS — CRITERIO IQR")
print("=" * 60)

# Estandarizar para visualización comparable
scaler = StandardScaler()
data_scaled = scaler.fit_transform(data[features])
df_scaled = pd.DataFrame(data_scaled, columns=features)


def detectar_outliers_iqr(df: pd.DataFrame) -> pd.DataFrame:
    """
    Detecta outliers univariados usando el criterio IQR.
    Un valor es atípico si cae fuera de [Q1 - 1.5*IQR, Q3 + 1.5*IQR].

    Parámetros
    ----------
    df : DataFrame con variables numéricas estandarizadas.

    Retorna
    -------
    DataFrame booleano con True donde hay outlier.
    La columna 'es_atipico' indica si alguna variable de esa fila es atípica.
    """
    outliers_df = pd.DataFrame(False, index=df.index, columns=df.columns)
    for col in df.columns:
        Q1 = df[col].quantile(0.25)
        Q3 = df[col].quantile(0.75)
        IQR = Q3 - Q1
        lower = Q1 - 1.5 * IQR
        upper = Q3 + 1.5 * IQR
        outliers_df[col] = (df[col] < lower) | (df[col] > upper)
    outliers_df["es_atipico"] = outliers_df.any(axis=1)
    return outliers_df


outliers_iqr = detectar_outliers_iqr(df_scaled)

print("\nCantidad de outliers univariados por variable:")
print("-" * 40)
conteo = outliers_iqr[features].sum().sort_values(ascending=False)
for var, n in conteo.items():
    pct = n / len(data) * 100
    print(f"  {var:<30} {n:>4} ({pct:.1f}%)")

total_filas = outliers_iqr["es_atipico"].sum()
print(f"\nFilas con al menos un outlier univariado: {total_filas} ({total_filas/len(data)*100:.1f}%)")

# ── Boxplot univariado ──────────────────────────────────────
fig, ax = plt.subplots(figsize=(12, 5))
ax.boxplot(
    data_scaled,
    labels=features,
    patch_artist=True,
    boxprops=dict(facecolor="#AED6F1", color="#2E86C1"),
    medianprops=dict(color="#E74C3C", linewidth=2),
    flierprops=dict(marker='o', markerfacecolor='orange', markersize=4, alpha=0.5)
)
ax.set_title("Boxplot univariado — Wine Red (variables estandarizadas)", fontsize=13, pad=12)
ax.set_xlabel("Variables")
ax.set_ylabel("Valores estandarizados (z-score)")
ax.axhline(0, color='gray', linestyle='--', linewidth=0.7, alpha=0.6)
plt.xticks(rotation=45, ha='right')
plt.tight_layout()
plt.show()


# ============================================================
# 5. OUTLIERS MULTIVARIADOS — ISOLATION FOREST
# ============================================================

print("\n" + "=" * 60)
print("OUTLIERS MULTIVARIADOS — ISOLATION FOREST")
print("=" * 60)

CONTAMINATION = 0.05  # proporción esperada de outliers

iso_forest = IsolationForest(contamination=CONTAMINATION, random_state=42)
outlier_labels = iso_forest.fit_predict(data_scaled)
# 1 = normal, -1 = outlier

n_outliers = (outlier_labels == -1).sum()
n_normal   = (outlier_labels ==  1).sum()

print(f"\nContaminación configurada : {CONTAMINATION:.0%}")
print(f"Observaciones normales    : {n_normal}")
print(f"Outliers detectados       : {n_outliers} ({n_outliers/len(data)*100:.1f}%)")

data_result = data.copy()
data_result["outlier"] = outlier_labels

# ── PCA 2D para visualización ───────────────────────────────
pca = PCA(n_components=2, random_state=42)
pca_coords = pca.fit_transform(data_scaled)
var_exp = pca.explained_variance_ratio_

print(f"\nVarianza explicada por PCA:")
print(f"  PC1: {var_exp[0]*100:.1f}%")
print(f"  PC2: {var_exp[1]*100:.1f}%")
print(f"  Total: {sum(var_exp)*100:.1f}%")

plot_df = pd.DataFrame({
    "PC1": pca_coords[:, 0],
    "PC2": pca_coords[:, 1],
    "outlier": outlier_labels,
    "quality": data["quality"].values
})

# ── Scatter PCA con Isolation Forest ───────────────────────
fig, axes = plt.subplots(1, 2, figsize=(14, 5))
fig.suptitle("Isolation Forest — Wine Red (visualización PCA 2D)", fontsize=13)

# Panel izquierdo: Normal vs Outlier
for label, color, nombre in [(1, "#2980B9", "Normal"), (-1, "#E67E22", "Outlier")]:
    subset = plot_df[plot_df["outlier"] == label]
    axes[0].scatter(
        subset["PC1"], subset["PC2"],
        c=color, label=f"{nombre} (n={len(subset)})",
        alpha=0.6, edgecolors='k', linewidths=0.3, s=50
    )
axes[0].set_title(f"Outliers detectados: {n_outliers}")
axes[0].set_xlabel(f"PC1 ({var_exp[0]*100:.1f}%)")
axes[0].set_ylabel(f"PC2 ({var_exp[1]*100:.1f}%)")
axes[0].legend()
axes[0].grid(True, alpha=0.3)

# Panel derecho: coloreado por quality
scatter = axes[1].scatter(
    plot_df["PC1"], plot_df["PC2"],
    c=plot_df["quality"], cmap="RdYlGn",
    alpha=0.6, edgecolors='k', linewidths=0.2, s=50
)
plt.colorbar(scatter, ax=axes[1], label="quality")
axes[1].set_title("Distribución por calidad del vino")
axes[1].set_xlabel(f"PC1 ({var_exp[0]*100:.1f}%)")
axes[1].set_ylabel(f"PC2 ({var_exp[1]*100:.1f}%)")
axes[1].grid(True, alpha=0.3)

plt.tight_layout()
plt.show()

# ── Score de anomalía (distribución) ───────────────────────
anomaly_scores = iso_forest.score_samples(data_scaled)

fig, ax = plt.subplots(figsize=(9, 4))
ax.hist(anomaly_scores[outlier_labels == 1],  bins=40, alpha=0.7,
        color="#2980B9", label="Normal")
ax.hist(anomaly_scores[outlier_labels == -1], bins=20, alpha=0.8,
        color="#E67E22", label="Outlier")
ax.axvline(iso_forest.threshold_, color='red', linestyle='--',
           linewidth=1.5, label=f"Umbral = {iso_forest.threshold_:.3f}")
ax.set_title("Distribución del score de anomalía — Isolation Forest")
ax.set_xlabel("Anomaly Score")
ax.set_ylabel("Frecuencia")
ax.legend()
ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.show()


# ============================================================
# 6. COMPARACIÓN IQR vs ISOLATION FOREST
# ============================================================

print("\n" + "=" * 60)
print("COMPARACIÓN: IQR vs ISOLATION FOREST")
print("=" * 60)

iqr_flags = outliers_iqr["es_atipico"]
iso_flags = pd.Series(outlier_labels == -1, index=data.index)

ambos     = (iqr_flags & iso_flags).sum()
solo_iqr  = (iqr_flags & ~iso_flags).sum()
solo_iso  = (~iqr_flags & iso_flags).sum()
ninguno   = (~iqr_flags & ~iso_flags).sum()

print(f"\n{'Categoría':<35} {'N':>6} {'%':>7}")
print("-" * 50)
print(f"{'Detectado por ambos métodos':<35} {ambos:>6} {ambos/len(data)*100:>6.1f}%")
print(f"{'Solo por IQR':<35} {solo_iqr:>6} {solo_iqr/len(data)*100:>6.1f}%")
print(f"{'Solo por Isolation Forest':<35} {solo_iso:>6} {solo_iso/len(data)*100:>6.1f}%")
print(f"{'Ninguno (observaciones normales)':<35} {ninguno:>6} {ninguno/len(data)*100:>6.1f}%")


# ============================================================
# 7. OTROS DATASETS — INVITACIÓN A PRACTICAR
# ============================================================

print("\n" + "=" * 60)
print("OTROS DATASETS DISPONIBLES PARA PRACTICAR")
print("=" * 60)

print("""
El mismo análisis puede replicarse fácilmente con los siguientes
datasets que ya están cargados en memoria:

  ┌─────────────────────┬──────────┬────────────────────────────────────┐
  │ Variable            │  Shape   │ Tarea sugerida                     │
  ├─────────────────────┼──────────┼────────────────────────────────────┤
  │ wine_white          │ ver abajo│ Mismo análisis que wine_red         │
  │ iris                │ ver abajo│ Outliers por especie (groupby)     │
  │ boston              │ ver abajo│ Outliers en precios MEDV y CRIM    │
  │ cancer              │ ver abajo│ Outliers en células benignas/malig │
  └─────────────────────┴──────────┴────────────────────────────────────┘
""")

for name, df in datasets_info.items():
    print(f"  {name:<20} shape={df.shape}")

print("""
────────────────────────────────────────────────────────────
Para replicar el análisis en cualquier dataset, cambia:

    data = wine_red.copy()           # ← reemplaza por el dataset
    features = [c for c in data.columns if c != "quality"]  # ← ajusta la columna objetivo

Ejemplo con Boston Housing:

    data = boston.copy()
    features = [c for c in data.columns if c != "MEDV"]

Ejemplo con Iris (solo numéricas):

    data = iris.copy()
    features = ["sepal_length", "sepal_width", "petal_length", "petal_width"]

────────────────────────────────────────────────────────────
""")

print("[FIN DEL SCRIPT]")
