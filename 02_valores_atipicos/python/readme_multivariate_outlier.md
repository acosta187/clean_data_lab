# 📊 Detección de Outliers Multivariados con Isolation Forest

## 🧠 ¿Qué es Isolation Forest?

**Isolation Forest** es un algoritmo de *machine learning no supervisado* utilizado para detectar **outliers (anomalías)** en un conjunto de datos.

A diferencia de otros métodos, no se basa en distancias ni densidades, sino en una idea simple:

> 🔥 *Los outliers son más fáciles de aislar que los datos normales*

---

## 🌲 ¿Cómo funciona?

El algoritmo construye múltiples árboles aleatorios (*forest*) y evalúa:

* Cuántos cortes (splits) necesita cada observación para quedar aislada
* Esto se conoce como **longitud del camino (path length)**

### 🔍 Interpretación:

* ✔ **Pocos cortes → outlier (anómalo)**
* ✔ **Muchos cortes → dato normal**

---

## 📊 Anomaly Score

Cada observación recibe un **anomaly score**, que mide qué tan rara es:

* Valores más extremos → más anómalos
* Valores más centrales → más normales

```python
anomaly_scores = iso_forest.score_samples(X)
```

👉 Este score es continuo y NO depende de `contamination`.

---

## ⚙️ Parámetro clave: `contamination`

```python
IsolationForest(contamination=0.05)
```

### 🎯 ¿Qué significa?

> Indica la proporción esperada de outliers en los datos.

Ejemplo:

* `0.05` → el 5% de los datos serán clasificados como outliers

### ⚠️ Importante:

* El modelo **NO detecta automáticamente cuántos outliers hay**
* Tú defines ese porcentaje

---

## 🔄 Relación entre score y contamination

1. Se calculan los anomaly scores
2. Se ordenan de más anómalo a más normal
3. Se selecciona el porcentaje definido por `contamination`

👉 Los más extremos → se etiquetan como outliers (`-1`)

---

## 📌 Interpretación del modelo

* `1` → dato normal
* `-1` → outlier

---

## 📈 Visualización (PCA + histogramas)

Se recomienda:

### 🔹 PCA (2D)

* Permite visualizar la distribución de los datos
* Ayuda a identificar patrones o agrupaciones

### 🔹 Histograma de anomaly score

* Permite ver:

  * Distribución de normalidad
  * Presencia de colas (outliers)

---

## ⚠️ ¿Es clustering?

❌ No.

Isolation Forest:

* NO encuentra grupos
* NO calcula clusters
* NO define número de clases

👉 Solo responde:

> “¿Qué puntos son raros respecto al conjunto completo?”

---

## 🔍 Diferencia con clustering

| Método           | Detecta grupos | Detecta outliers |
| ---------------- | -------------- | ---------------- |
| KMeans           | ✔ Sí           | ❌ No directo     |
| DBSCAN           | ✔ Sí           | ✔ Sí             |
| Isolation Forest | ❌ No           | ✔ Sí             |

---

## 🧪 Buenas prácticas

### ✔ 1. Elegir `contamination` inicial

* Datos limpios → 0.01–0.03
* Datos reales → 0.03–0.07
* Datos ruidosos → 0.05–0.10

---

### ✔ 2. Analizar resultados

* Revisar gráfico de scores
* Ver PCA
* Evaluar si los outliers tienen sentido

---

### ✔ 3. Ajustar y reentrenar

```python
for c in [0.01, 0.03, 0.05, 0.1]:
    iso = IsolationForest(contamination=c)
    labels = iso.fit_predict(X)
```

👉 Sí, es recomendable volver a correr el análisis

---

## 🚀 Enfoque avanzado (recomendado)

En lugar de depender completamente de `contamination`:

1. Analizar la distribución de `anomaly_scores`
2. Definir manualmente un umbral

```python
import numpy as np

threshold = np.percentile(anomaly_scores, 5)
```

---

## ⚠️ Consideraciones importantes

### 🔸 Variables categóricas (One-Hot Encoding)

* Se transforman en variables binarias (0/1)
* Pueden generar:

  * patrones raros
  * falsas anomalías

👉 Recomendación:

* comparar resultados con solo variables numéricas

---

### 🔸 Limitaciones

* Sensible a `contamination`
* Puede detectar rareza, no necesariamente error
* No distingue entre:

  * outlier real
  * valor poco frecuente

---

## 🏁 Conclusión

* Isolation Forest detecta outliers basándose en la facilidad de aislamiento
* No utiliza clustering ni distancias
* `contamination` controla cuántos outliers se detectan
* Es necesario ajustar y validar los resultados

---

## 🧠 Frase clave

> **Isolation Forest no busca quién está lejos, sino quién es fácil de aislar**
