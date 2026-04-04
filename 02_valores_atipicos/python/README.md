# Detección de Outliers en Python
### Análisis univariado y multivariado sobre datasets clásicos de ML

---

## ¿Qué son los valores atípicos (outliers)?

Un **outlier** es una observación que se aleja significativamente del comportamiento general del resto de los datos. Su presencia puede tener causas legítimas (vinos de calidad excepcional, fraudes, errores de medición) o simplemente ser ruido del proceso de recolección.

Ignorarlos tiene consecuencias reales:

- Distorsionan la media y la desviación estándar
- Inflan o anulan correlaciones entre variables
- Degradan el rendimiento de modelos sensibles a escala (regresión lineal, KNN, SVM)
- Pueden enmascarar patrones importantes en los datos

---

## Tipos de outliers

| Tipo | Descripción | Método de detección |
|------|-------------|---------------------|
| **Univariado** | Atípico en una sola variable, analizada de forma aislada | IQR, Z-score, boxplot |
| **Multivariado** | Atípico en la combinación de varias variables (puede parecer normal en cada una por separado) | Isolation Forest, Mahalanobis, DBSCAN, LOF |

> Un vino con `alcohol=14%` no es atípico. Uno con `alcohol=14% + pH=2.0 + density=0.999` sí puede serlo en conjunto.

---

## Métodos implementados en este script

### 1. IQR — Criterio intercuartílico (univariado)

Dado el rango intercuartílico `IQR = Q3 - Q1`, se consideran outliers los valores fuera del intervalo:

```
[Q1 - 1.5 × IQR,  Q3 + 1.5 × IQR]
```

**Ventajas:** Robusto a distribuciones asimétricas. Sin hiperparámetros.  
**Limitación:** Analiza cada variable de forma independiente; no detecta outliers multivariados.

### 2. Isolation Forest (multivariado)

Algoritmo de ensamble basado en árboles de decisión aleatorios. La intuición central es que los outliers son más **fáciles de aislar** que los puntos normales: requieren menos particiones del espacio para quedar separados del resto.

- Construye múltiples árboles de aislamiento aleatorios
- Asigna a cada punto un *anomaly score* basado en la profundidad promedio de aislamiento
- Puntos con scores bajos (aislados rápido) = outliers

**Parámetro clave:** `contamination` — proporción esperada de outliers en el dataset (default: 0.05 = 5%).

**Ventajas:** Escala bien con alta dimensionalidad. No asume distribución de los datos.  
**Limitación:** Requiere calibrar `contamination` según el dominio.

---

## Estructura del script

```
outlier_detection.py
│
├── Sección 1 — Imports
├── Sección 2 — Carga de datasets
│       wine_red, wine_white, iris, boston, cancer
│
├── Sección 3 — Análisis principal: Wine Red
│       Exploración, shape, distribución de quality
│
├── Sección 4 — Outliers UNIVARIADOS (IQR)
│       Estandarización → detección → boxplot
│
├── Sección 5 — Outliers MULTIVARIADOS (Isolation Forest)
│       Isolation Forest → PCA 2D → visualización
│       Distribución del anomaly score
│
├── Sección 6 — Comparación IQR vs Isolation Forest
│       ¿Cuántos outliers detecta cada método? ¿Coinciden?
│
└── Sección 7 — Invitación a practicar con otros datasets
```

---

## Requisitos

```bash
pip install pandas numpy matplotlib scikit-learn
```

No se requieren instalaciones adicionales. Todos los datasets se cargan directamente desde URLs públicas.

---

## Cómo ejecutar

```bash
python outlier_detection.py
```

El script imprime resultados en consola y abre ventanas con los gráficos de forma secuencial.

---

## Datasets disponibles

Todos los datasets se cargan automáticamente al iniciar el script:

| Variable | Dataset | Filas aprox. | Target | Tipo de tarea |
|----------|---------|-------------|--------|---------------|
| `wine_red` | Wine Quality Rojo | 1.599 | `quality` | Regresión / Clasificación |
| `wine_white` | Wine Quality Blanco | 4.898 | `quality` | Regresión / Clasificación |
| `iris` | Iris | 150 | `class` | Clasificación |
| `boston` | Boston Housing | 506 | `MEDV` | Regresión |
| `cancer` | Breast Cancer Wisconsin | 699 | `class` | Clasificación binaria |

---

## Practica con otros datasets

El análisis completo sobre `wine_red` puede replicarse en cualquier otro dataset con tan solo dos cambios en el script:

### Ejemplo: Wine White

```python
data = wine_white.copy()
features = [c for c in data.columns if c != "quality"]
```

### Ejemplo: Boston Housing

```python
data = boston.copy()
features = [c for c in data.columns if c != "MEDV"]
```

### Ejemplo: Iris (solo variables numéricas)

```python
data = iris.copy()
features = ["sepal_length", "sepal_width", "petal_length", "petal_width"]
```

### Ejemplo: Breast Cancer

```python
data = cancer.drop(columns=["id", "class"]).copy()
features = data.columns.tolist()
```

> 💡 **Sugerencia:** Después de detectar outliers, analiza si eliminarlos o imputarlos mejora las métricas de un modelo simple (p. ej. una regresión logística o un árbol de decisión). Ese es el paso natural siguiente.

---

## Interpretación de resultados

### Boxplot univariado
Cada caja representa el rango IQR de una variable estandarizada. Los puntos fuera de los bigotes son los outliers detectados por el criterio IQR.

### Scatter PCA (Isolation Forest)
Se reducen las 11 variables a 2 componentes principales para visualización. Los puntos en naranja son los outliers detectados. El panel derecho muestra si los outliers se concentran en alguna categoría de calidad.

### Distribución del anomaly score
Muestra la separación entre la distribución de scores de puntos normales y outliers. Una buena separación bimodal indica que el método discrimina con claridad.

### Comparación IQR vs Isolation Forest
Ambos métodos no deben coincidir exactamente: el IQR es univariado y el Isolation Forest es multivariado. Los outliers detectados "solo por Isolation Forest" son precisamente los casos multivariados invisibles al IQR.

---

## Referencias

- Liu, F. T., Ting, K. M., & Zhou, Z. H. (2008). *Isolation Forest.* ICDM.
- Tukey, J. W. (1977). *Exploratory Data Analysis.* Addison-Wesley.
- scikit-learn: [Novelty and Outlier Detection](https://scikit-learn.org/stable/modules/outlier_detection.html)
- Datasets: [Jason Brownlee — Datasets GitHub](https://github.com/jbrownlee/Datasets)
