# `text_quality.py` — Documentación

Toolkit para **auditoría de calidad de texto**, simulación de ruido y limpieza por fuzzy-match, integrable en cualquier flujo pandas.

---

## Instalación de dependencias

```bash
pip install pandas numpy nlpaug rapidfuzz
```

---

## Funciones

### 1. `inject_noise`

Inyecta ruido realista de digitación humana en un string.

```python
inject_noise(text, p_keyboard=0.30, p_deletion=0.20, p_format=0.20, char_level=0.10)
```

**Parámetros**

| Parámetro    | Tipo  | Default | Descripción |
|--------------|-------|---------|-------------|
| `text`       | `str` | —       | Texto de entrada. |
| `p_keyboard` | `float` | `0.30` | Probabilidad de error por tecla adyacente. |
| `p_deletion` | `float` | `0.20` | Probabilidad de eliminación aleatoria de caracter. |
| `p_format`   | `float` | `0.20` | Probabilidad de distorsión de formato (casing / espacios). |
| `char_level` | `float` | `0.10` | Fracción de caracteres afectados por error (0–1). |

**Retorna:** `str` — Versión ruidosa del input.

> **Nota:** Las probabilidades son mutuamente excluyentes y se evalúan en orden. La probabilidad restante `(1 - suma)` retorna el texto limpio, preservando una tasa realista de datos sin distorsión.

**Ejemplo**

```python
from text_quality import inject_noise

inject_noise("Miraflores")
# → "Mirafkores"  (error de teclado)

inject_noise("San Borja", p_format=1.0)
# → "SAN BORJA"   (distorsión de formato)

inject_noise("Chorrillos", p_keyboard=0.5, char_level=0.3)
# → "Chorrilloe"  (ruido agresivo)
```

---

### 2. `build_noisy_dataset`

Genera un DataFrame sintético con una columna de referencia limpia (`original`) y una versión ruidosa (`noisy`), para testear pipelines de limpieza.

```python
build_noisy_dataset(master_list, n_rows=500, noise_kwargs=None)
```

**Parámetros**

| Parámetro     | Tipo           | Default | Descripción |
|---------------|----------------|---------|-------------|
| `master_list` | `list[str]`    | —       | Pool de valores de referencia a samplear. |
| `n_rows`      | `int`          | `500`   | Número de filas a generar. |
| `noise_kwargs`| `dict \| None` | `None`  | Keyword arguments pasados a `inject_noise`. |

**Retorna:** `pd.DataFrame` con columnas:
- `original` — Valor ground-truth (sampleado de `master_list`).
- `noisy` — Versión distorsionada producida por `inject_noise`.

**Ejemplo**

```python
from text_quality import build_noisy_dataset

MASTER = ["Miraflores", "San Borja", "Chorrillos"]

df = build_noisy_dataset(MASTER, n_rows=200)
print(df.head(3))
#       original           noisy
# 0    San Borja       san borja
# 1   Miraflores      Miraflorws
# 2   Chorrillos    CHORRILLOS
```

---

### 3. `audit_text_quality`

Calcula un scorecard de calidad para una `pd.Series` de texto.

```python
audit_text_quality(series)
```

**Parámetros**

| Parámetro | Tipo        | Descripción |
|-----------|-------------|-------------|
| `series`  | `pd.Series` | Columna a evaluar. |

**Retorna:** `pd.Series` con las siguientes métricas:

| Métrica            | Tipo    | Descripción |
|--------------------|---------|-------------|
| `total_records`    | `int`   | Cantidad de filas. |
| `unique_values`    | `int`   | Valores distintos. |
| `uniqueness_ratio` | `float` | `unique / total`. Menor = más estandarizado. |
| `text_entropy`     | `int`   | Tamaño del vocabulario de caracteres en la columna. |
| `noise_rate_pct`   | `float` | % de filas con caracteres fuera de `[a-zA-Z .]`. |

**Guía de interpretación**

| Métrica             | Rango       | Status      | Significado |
|---------------------|-------------|-------------|-------------|
| `uniqueness_ratio`  | `< 0.10`    | EXCELLENT   | Datos muy estandarizados. |
|                     | `0.11–0.35` | DISPERSED   | Requiere clustering / limpieza. |
|                     | `> 0.40`    | CRITICAL    | Datos fragmentados / sucios. |
| `noise_rate_pct`    | `< 1 %`     | CLEAN       | Caracteres normales. |
|                     | `> 5 %`     | NOISY       | Símbolos / errores de encoding. |
| `text_entropy`      | `25–45`     | NORMAL      | Alfabeto estándar A–Z. |
|                     | `> 55`      | SUSPICIOUS  | Problemas de codificación UTF-8. |

**Ejemplo**

```python
from text_quality import audit_text_quality
import pandas as pd

s = pd.Series(["Miraflores", "MiRAflores", "Miraaflores", "MIRAFLORES"])
print(audit_text_quality(s))
# total_records       4.0000
# unique_values       4.0000
# uniqueness_ratio    1.0000
# text_entropy       14.0000
# noise_rate_pct      0.0000
```

---

### 4. `fuzzy_clean`

Mapea un valor ruidoso al entry más cercano en una lista maestra de referencia.

```python
fuzzy_clean(text, master_list, threshold=65)
```

**Parámetros**

| Parámetro     | Tipo        | Default | Descripción |
|---------------|-------------|---------|-------------|
| `text`        | `str`       | —       | Input crudo / ruidoso. |
| `master_list` | `list[str]` | —       | Valores de referencia autorizados. |
| `threshold`   | `int`       | `65`    | Score mínimo (0–100) para aceptar un match. |

**Retorna:** `str | float` — Valor de la lista maestra, o `np.nan` si la confianza es baja.

> Usa `token_sort_ratio` (RapidFuzz), por lo que variaciones en el orden de palabras y espacios extra no penalizan el score. Retornar `NaN` (en vez de un string centinela) mantiene la columna compatible con las utilidades de imputación de pandas y los pipelines de sklearn.

**Selección del threshold**

| Threshold | Comportamiento |
|-----------|----------------|
| `< 60`    | Permisivo — acepta matches débiles, riesgo de falsos positivos. |
| `65–75`   | Equilibrado — recomendado para ruido típico de digitación. |
| `> 85`    | Estricto — muchos `NaN`, pero alta precisión en los commits. |

**Ejemplo**

```python
from text_quality import fuzzy_clean

MASTER = ["Miraflores", "San Borja", "Chorrillos"]

fuzzy_clean("miraflres", MASTER, threshold=65)    # → "Miraflores"
fuzzy_clean("SAN BORJA", MASTER, threshold=65)    # → "San Borja"
fuzzy_clean("xyz123",    MASTER, threshold=65)    # → nan
```

**Uso vectorizado sobre un DataFrame**

```python
df["cleaned"] = df["noisy"].apply(fuzzy_clean, master_list=MASTER, threshold=65)
```

---

### 5. `cleaning_accuracy`

Evalúa qué tan bien el fuzzy cleaner recuperó los valores ground-truth.

```python
cleaning_accuracy(df, original_col="original", cleaned_col="cleaned")
```

**Parámetros**

| Parámetro      | Tipo           | Default       | Descripción |
|----------------|----------------|---------------|-------------|
| `df`           | `pd.DataFrame` | —             | DataFrame con columnas de referencia y limpiada. |
| `original_col` | `str`          | `"original"`  | Columna con las etiquetas ground-truth. |
| `cleaned_col`  | `str`          | `"cleaned"`   | Columna producida por `fuzzy_clean`. |

**Retorna:** `dict` con las claves:

| Clave           | Tipo    | Descripción |
|-----------------|---------|-------------|
| `total`         | `int`   | Total de filas evaluadas. |
| `nan_count`     | `int`   | Filas donde el cleaner retornó `NaN`. |
| `overall_pct`   | `float` | Precisión sobre todo el dataset (NaN cuenta como error). |
| `confident_pct` | `float` | Precisión solo en filas donde el cleaner hizo un commit. |

**Diferencia entre overall y confident**

- **`overall_pct`**: métrica conservadora — penaliza las abstenciones (`NaN`). Útil para evaluar el pipeline completo.
- **`confident_pct`**: métrica de calidad del matcher — refleja qué tan acertado es el cleaner cuando decide matchear. Idealmente debería ser > 95 %.

**Ejemplo**

```python
from text_quality import cleaning_accuracy

report = cleaning_accuracy(df)
print(report)
# {
#   'total': 500,
#   'nan_count': 23,
#   'overall_pct': 87.40,
#   'confident_pct': 96.10
# }
```

---

## Flujo completo típico

```python
from text_quality import (
    build_noisy_dataset,
    audit_text_quality,
    fuzzy_clean,
    cleaning_accuracy,
)

MASTER = ["Miraflores", "San Juan de Lurigancho", "Santiago de Surco", ...]

# 1 — Generar data sintética
df = build_noisy_dataset(MASTER, n_rows=500)

# 2 — Auditar calidad inicial
print(audit_text_quality(df["noisy"]))

# 3 — Limpiar
df["cleaned"] = df["noisy"].apply(fuzzy_clean, master_list=MASTER, threshold=65)

# 4 — Auditar resultado
print(audit_text_quality(df["cleaned"].fillna("NaN")))

# 5 — Evaluar precisión
report = cleaning_accuracy(df)
print(f"Overall: {report['overall_pct']} %")
print(f"Confident: {report['confident_pct']} %")
```

---

## Archivos relacionados

| Archivo               | Descripción |
|-----------------------|-------------|
| `text_quality.py`     | Toolkit principal con todas las funciones. |
| `run_demo.py`         | Script de demostración que ejecuta el flujo completo de una sola vez. |
| `interactive_demo.py` | Demo interactivo paso a paso — ejecuta cada función con pausa entre pasos. |
