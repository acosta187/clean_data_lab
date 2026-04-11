

import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import MinMaxScaler, OneHotEncoder

# Cargar data desde GitHub
url = "https://raw.githubusercontent.com/datasciencedojo/datasets/master/titanic.csv"

df = pd.read_csv(url)
df.head()


# Variables
num_cols = ['Age', 'Fare']
cat_cols = ['Sex', 'Embarked', 'Pclass']

target = "Survived"

df = df[num_cols + cat_cols + [target]].dropna()

X = df[num_cols + cat_cols]
y = df[target]




preprocessor = ColumnTransformer(
    transformers=[
        ('num', MinMaxScaler(), num_cols),
        ('cat', OneHotEncoder(drop='first', sparse_output=False), cat_cols)
    ]
)





X_transformed = preprocessor.fit_transform(X)





cat_features = preprocessor.named_transformers_['cat'].get_feature_names_out(cat_cols)

all_features = num_cols + list(cat_features)

X_final = pd.DataFrame(X_transformed, columns=all_features)

print(X_final.head())




print(X_final.columns)
print(num_cols)
print(cat_cols)

print(df['Sex'].value_counts())
print(df['Embarked'].value_counts())






# Evaluaando valores atípicos
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.ensemble import IsolationForest
from sklearn.decomposition import PCA

# ============================================================
# OUTLIERS MULTIVARIADOS — ISOLATION FOREST
# ============================================================

print("\n" + "=" * 60)
print("OUTLIERS MULTIVARIADOS — TITANIC")
print("=" * 60)

data = X_final.copy()
data_scaled = data.values  # ya está escalado (MinMax)

CONTAMINATION = 0.05 # Espero que el 5% de mis datos sean anomalías

iso_forest = IsolationForest(contamination=CONTAMINATION, random_state=42)
outlier_labels = iso_forest.fit_predict(data_scaled)

n_outliers = (outlier_labels == -1).sum()
n_normal   = (outlier_labels ==  1).sum()

print(f"\nContaminación configurada : {CONTAMINATION:.0%}")
print(f"Observaciones normales    : {n_normal}")
print(f"Outliers detectados       : {n_outliers} ({n_outliers/len(data)*100:.1f}%)")

# ============================================================
# PCA
# ============================================================

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
    "survived": y.values   # usamos el target
})

# ============================================================
# SCATTER PCA
# ============================================================

fig, axes = plt.subplots(1, 2, figsize=(14, 5))
fig.suptitle("Isolation Forest — Titanic (PCA 2D)", fontsize=13)

# Panel 1: Outliers
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

# Panel 2: Coloreado por target (survived)
scatter = axes[1].scatter(
    plot_df["PC1"], plot_df["PC2"],
    c=plot_df["survived"], cmap="coolwarm",
    alpha=0.6, edgecolors='k', linewidths=0.2, s=50
)

plt.colorbar(scatter, ax=axes[1], label="Survived")
axes[1].set_title("Distribución por supervivencia")
axes[1].set_xlabel(f"PC1 ({var_exp[0]*100:.1f}%)")
axes[1].set_ylabel(f"PC2 ({var_exp[1]*100:.1f}%)")
axes[1].grid(True, alpha=0.3)

plt.tight_layout()
plt.show()

# ============================================================
# HISTOGRAMA DE SCORES
# ============================================================

anomaly_scores = iso_forest.score_samples(data_scaled)

fig, ax = plt.subplots(figsize=(9, 4))

ax.hist(anomaly_scores[outlier_labels == 1], bins=40, alpha=0.7,
        label="Normal")

ax.hist(anomaly_scores[outlier_labels == -1], bins=20, alpha=0.8,
        label="Outlier")

ax.set_title("Distribución del score de anomalía — Isolation Forest")
ax.set_xlabel("Anomaly Score")
ax.set_ylabel("Frecuencia")
ax.legend()
ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.show()