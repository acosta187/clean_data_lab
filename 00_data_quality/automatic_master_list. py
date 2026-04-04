# ================================
# MASTER LIST SEMI-AUTOMÁTICO (NLP + SKLEARN)
# ================================

# Instalar si hace falta:
# pip install pandas scikit-learn unidecode

import pandas as pd
from unidecode import unidecode
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import AgglomerativeClustering

# ----------------
# 1. Datos
# ----------------
data = pd.DataFrame({
    "raw": [
        "Loreto", "Loretto", "loreto ",
        "San Juan de Lurigancho", "S J Lurigancho", "San Jn Luriganch",
        "Piura", "piurra"
    ]
})

# ----------------
# 2. Normalización
# ----------------
def normalize(text):
    return (
        unidecode(text.lower().strip())
        .replace(".", "")
    )

data["clean"] = data["raw"].apply(normalize)

# ----------------
# 3. Vectorización (NLP)
# ----------------
vectorizer = TfidfVectorizer(
    analyzer="char_wb",
    ngram_range=(2, 4)
)

X = vectorizer.fit_transform(data["clean"])

# ----------------
# 4. Clustering
# ----------------
clustering = AgglomerativeClustering(
    metric="cosine",
    linkage="average",
    distance_threshold=0.3,  # ajustar según datos
    n_clusters=None
)

data["cluster"] = clustering.fit_predict(X.toarray())

# ----------------
# 5. Elegir MASTER (más largo)
# ----------------
master = (
    data.assign(length=data["clean"].str.len())
    .sort_values("length", ascending=False)
    .drop_duplicates("cluster")
    [["cluster", "clean"]]
    .rename(columns={"clean": "master"})
)

# ----------------
# 6. Diccionario final
# ----------------
dictionary = data.merge(master, on="cluster")[["raw", "master"]]

# ----------------
# 7. Output
# ----------------
print("\n=== DATA CON CLUSTERS ===")
print(data.sort_values("cluster"))

print("\n=== MASTER LIST ===")
print(dictionary)

# ----------------
# 8. Guardar
# ----------------
dictionary.to_csv("master_list.csv", index=False)

# ----------------
# 9. Aplicar limpieza
# ----------------
data_clean = data.merge(dictionary, on="raw", how="left")

print("\n=== DATA LIMPIA ===")
print(data_clean)