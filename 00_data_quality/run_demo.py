"""
run_demo.py
===========
Script de demostración del toolkit text_quality.
Ejecutar con: python run_demo.py
"""

from text_quality import (
    build_noisy_dataset,
    audit_text_quality,
    fuzzy_clean,

)

# ── Lista maestra de referencia ────────────────────────────────────────────────

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
