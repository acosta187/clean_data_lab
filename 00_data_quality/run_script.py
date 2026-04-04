import pandas as pd
import numpy as np
import pandas as pd


from text_quality import (
    inject_noise,
    build_noisy_dataset,
    audit_text_quality,
    fuzzy_clean,
)

MASTER = [
    "Miraflores", "San Juan de Lurigancho", "Santiago de Surco",
    "Villa El Salvador", "Chorrillos", "San Borja", "La Molina",
    "Los Olivos", "San Miguel", "Magdalena del Mar",
]

# 1. Generar dataset
df = build_noisy_dataset(MASTER, n_rows=500)

# 2. Auditoría inicial
raw_audit = audit_text_quality(df["noisy"])

# 3. Limpieza
df["cleaned"] = df["noisy"].apply(
    lambda x: fuzzy_clean(x, MASTER, threshold=65)
)

# 4. Auditoría final
clean_audit = audit_text_quality(df["cleaned"].fillna("NaN"))
clean_audit


df_failed = df[df["cleaned"].isna()]
df_success = df[df["cleaned"].notna()]

clean_audit_succ = audit_text_quality(df_success["cleaned"])
clean_audit_succ 


# 5. Accuracy


accuracy = np.mean(df_success["original"] == df_success["cleaned"]) * 100
print(f"Accuracy: {accuracy:.2f}%")

# Matriz de confusión
conf_matrix = pd.crosstab(
    df_success["original"], 
    df_success["cleaned"],
    rownames=["Actual"],
    colnames=["Predicted"]
)

print("\n=== Confusion Matrix ===")
print(conf_matrix)

