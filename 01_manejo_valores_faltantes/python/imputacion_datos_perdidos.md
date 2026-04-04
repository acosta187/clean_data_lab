# Datos Perdidos: Visualización, Teoría e Imputación en Python

---

## 1. ¿Qué son los datos perdidos?

Los **datos perdidos** (*missing data*) ocurren cuando una o más variables no tienen valores registrados para ciertas observaciones. Son una realidad común en datasets del mundo real y, si no se tratan correctamente, pueden introducir sesgos, reducir el poder estadístico y comprometer la validez de los modelos.

### Tipos de datos perdidos (taxonomía de Rubin, 1976)

| Tipo | Sigla | Descripción | Ejemplo |
|------|-------|-------------|---------|
| **Missing Completely At Random** | MCAR | La ausencia es totalmente aleatoria, independiente de cualquier variable | Datos perdidos por fallo técnico del sensor |
| **Missing At Random** | MAR | La ausencia depende de otras variables observadas, pero no del valor faltante en sí | Personas mayores omiten el campo "ingresos" porque la encuesta era digital |
| **Missing Not At Random** | MNAR | La ausencia depende del propio valor que falta | Pacientes muy enfermos no asisten al control donde se mediría su severidad |

> La distinción es crucial: los métodos de imputación asumen mayoritariamente **MAR**. Si los datos son **MNAR**, cualquier imputación introducirá sesgo sistemático.

---

## 2. Visualización de datos perdidos

Antes de imputar, es fundamental **explorar y entender el patrón** de los nulos. Las bibliotecas `missingno` y `seaborn`/`matplotlib` son las herramientas estándar en Python.

### 2.1 `missingno` — visualizaciones especializadas

```python
import missingno as msno
import matplotlib.pyplot as plt

# Gráfico de barras: completitud por columna
msno.bar(df)
plt.show()

# Matriz de nulidad: patrón espacial de nulos
msno.matrix(df)
plt.show()

# Mapa de calor: correlación entre nulos
msno.heatmap(df)
plt.show()

# Dendrograma: agrupación de columnas por patrón de nulos
msno.dendrogram(df)
plt.show()
```

| Visualización | ¿Qué revela? |
|---------------|--------------|
| `bar` | Porcentaje de completitud de cada columna |
| `matrix` | Distribución espacial de los nulos (¿hay filas con muchos nulos juntos?) |
| `heatmap` | Correlación entre la ausencia en dos columnas (¿si falta A, tiende a faltar B?) |
| `dendrogram` | Columnas con patrones de nulidad similares se agrupan juntas |

### 2.2 Diagnóstico con pandas y seaborn

```python
# Resumen cuantitativo
print(df.isnull().sum())
print(df.isnull().mean() * 100)  # porcentaje

# Heatmap manual
import seaborn as sns
sns.heatmap(df.isnull(), cbar=False, cmap='viridis')
plt.show()
```

### 2.3 Preguntas clave a responder en el diagnóstico

1. **¿Cuánto falta?** — Columnas con >50% de nulos suelen descartarse.
2. **¿Dónde falta?** — ¿Hay filas sistemáticamente incompletas?
3. **¿Por qué falta?** — MCAR / MAR / MNAR (contexto del dominio).
4. **¿Hay correlación entre nulos?** — Puede revelar mecanismos subyacentes.

---

## 3. Estrategias de imputación en scikit-learn

### 3.1 Visión general

```
Datos faltantes
      │
      ├── Variables NUMÉRICAS
      │         ├── SimpleImputer (media, mediana, constante)
      │         ├── KNNImputer
      │         ├── IterativeImputer + modelo de regresión
      │         │         ├── RandomForestRegressor  ← usado en el código
      │         │         ├── BayesianRidge (default)
      │         │         ├── LinearRegression
      │         │         └── GradientBoostingRegressor
      │         └── Eliminar filas/columnas (último recurso)
      │
      └── Variables CATEGÓRICAS
                ├── SimpleImputer (moda, constante)
                ├── KNNImputer (con encoding previo)
                └── IterativeImputer + modelo de clasificación
                          ├── RandomForestClassifier
                          └── LogisticRegression
```

---

### 3.2 `SimpleImputer` — imputación univariada

El método más básico: reemplaza los nulos con una estadística calculada de la misma columna.

```python
from sklearn.impute import SimpleImputer

# Para variables numéricas
num_imputer = SimpleImputer(strategy='mean')       # media
num_imputer = SimpleImputer(strategy='median')     # mediana (robusta a outliers)
num_imputer = SimpleImputer(strategy='constant', fill_value=0)

# Para variables categóricas
cat_imputer = SimpleImputer(strategy='most_frequent')  # moda
cat_imputer = SimpleImputer(strategy='constant', fill_value='Unknown')
```

**Ventajas:** Rápido, sin hiperparámetros, interpretable.  
**Desventajas:** No considera relaciones entre variables. Distorsiona la distribución (reduce varianza).

> **Regla práctica:** usar mediana para numéricas con outliers, moda para categóricas. Evitar la media si la distribución es asimétrica.

---

### 3.3 `KNNImputer` — imputación por vecinos más cercanos

Imputa el valor faltante usando el promedio ponderado de los *k* vecinos más similares en el espacio de características.

```python
from sklearn.impute import KNNImputer

imputer = KNNImputer(
    n_neighbors=5,          # número de vecinos
    weights='uniform',      # 'uniform' o 'distance'
    metric='nan_euclidean'  # maneja NaN en el cálculo de distancias
)

df_imputed = imputer.fit_transform(df_numeric)
```

**Ventajas:** Captura relaciones locales entre variables. Preserva mejor la distribución que SimpleImputer.  
**Desventajas:** Costoso computacionalmente en datasets grandes O(n²). Sensible a la escala → **normalizar antes**.

```python
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

pipeline = Pipeline([
    ('scaler', StandardScaler()),
    ('imputer', KNNImputer(n_neighbors=5))
])
```

> **Para variables categóricas:** aplicar `OrdinalEncoder` antes de `KNNImputer`, luego revertir el encoding.

---

### 3.4 `IterativeImputer` — imputación multivariada (MICE)

Implementa el algoritmo **MICE** (*Multiple Imputation by Chained Equations*): modela cada columna con nulos en función de todas las demás, iterativamente.

```python
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer
```

#### Estrategia A: BayesianRidge (default — recomendado como baseline)

```python
imputer = IterativeImputer(random_state=42)
# BayesianRidge es el estimator por defecto
```

Rápido, estable, bueno para relaciones lineales entre variables.

#### Estrategia B: RandomForestRegressor ← (el que usas en tu código)

```python
from sklearn.ensemble import RandomForestRegressor

imputer = IterativeImputer(
    estimator=RandomForestRegressor(n_estimators=100, random_state=42),
    random_state=42,
    max_iter=10
)
```

Captura relaciones no lineales. Robusto a outliers. Más lento pero generalmente más preciso.

#### Estrategia C: LinearRegression / Ridge

```python
from sklearn.linear_model import BayesianRidge, LinearRegression, Ridge

imputer = IterativeImputer(estimator=Ridge(alpha=1.0), random_state=42)
```

Buena opción cuando las relaciones entre variables son aproximadamente lineales.

#### Estrategia D: GradientBoosting

```python
from sklearn.ensemble import GradientBoostingRegressor

imputer = IterativeImputer(
    estimator=GradientBoostingRegressor(n_estimators=50, random_state=42),
    random_state=42
)
```

Muy preciso, pero el más lento de todos. Útil cuando la calidad de la imputación es crítica.

#### Parámetros clave de `IterativeImputer`

| Parámetro | Default | Descripción |
|-----------|---------|-------------|
| `estimator` | `BayesianRidge()` | Modelo usado para predecir cada columna |
| `max_iter` | `10` | Iteraciones del ciclo MICE |
| `initial_strategy` | `'mean'` | Valores iniciales antes de iterar |
| `imputation_order` | `'ascending'` | Orden en que se imputan las columnas |
| `random_state` | `None` | Semilla de aleatoriedad |

---

### 3.5 Comparativa de métodos

| Método | Tipo de variable | Relaciones entre vars | Velocidad | Precisión típica |
|--------|-----------------|----------------------|-----------|-----------------|
| `SimpleImputer` (media/mediana) | Numérica | ✗ | ⚡⚡⚡ | Baja |
| `SimpleImputer` (moda) | Categórica | ✗ | ⚡⚡⚡ | Baja |
| `KNNImputer` | Numérica / Categórica* | Local | ⚡⚡ | Media |
| `IterativeImputer` + BayesianRidge | Numérica | Lineal | ⚡⚡ | Media-Alta |
| `IterativeImputer` + RandomForest | Numérica / Categórica* | No lineal | ⚡ | Alta |
| `IterativeImputer` + GradientBoosting | Numérica | No lineal | ⚡ | Alta |

*Con encoding previo.

---

## 4. Imputación de variables categóricas: estrategias detalladas

Las variables categóricas requieren un tratamiento especial porque los modelos de sklearn no operan nativamente sobre strings.

### 4.1 SimpleImputer con moda

```python
from sklearn.impute import SimpleImputer

cat_imputer = SimpleImputer(strategy='most_frequent')
df[cat_cols] = cat_imputer.fit_transform(df[cat_cols])
```

Simple y efectivo cuando la variable tiene una categoría dominante clara.

### 4.2 Imputación con constante `'Unknown'`

```python
cat_imputer = SimpleImputer(strategy='constant', fill_value='Unknown')
```

Útil cuando la ausencia en sí misma es informativa (p.ej. el paciente no reportó su condición).

### 4.3 KNNImputer con OrdinalEncoder

```python
from sklearn.preprocessing import OrdinalEncoder
from sklearn.impute import KNNImputer
import numpy as np

enc = OrdinalEncoder(handle_unknown='use_encoded_value', unknown_value=np.nan)
knn = KNNImputer(n_neighbors=5)

X_enc = enc.fit_transform(df[cat_cols])
X_imp = knn.fit_transform(X_enc)

# Redondear y decodificar
X_imp = np.round(X_imp)
df[cat_cols] = enc.inverse_transform(X_imp)
```

### 4.4 IterativeImputer con RandomForestClassifier

Para categóricas con `IterativeImputer` se necesita un clasificador, lo cual requiere manejo manual o una librería externa como `miceforest`:

```python
# Opción con miceforest (recomendado para MICE + categoricas)
# pip install miceforest
import miceforest as mf

kernel = mf.ImputationKernel(df, save_all_iterations=True, random_state=42)
kernel.mice(3)  # 3 iteraciones
df_imputed = kernel.complete_data()
```

---

## 5. Pipeline completo recomendado

```python
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer, KNNImputer, IterativeImputer
from sklearn.ensemble import RandomForestRegressor
from sklearn.experimental import enable_iterative_imputer

num_cols = ['age', 'ejection_fraction', 'serum_creatinine']
cat_cols = ['anaemia', 'diabetes', 'sex']

# Imputador numérico: MICE con RandomForest
num_imputer = IterativeImputer(
    estimator=RandomForestRegressor(n_estimators=100, random_state=42),
    max_iter=10,
    random_state=42
)

# Imputador categórico: moda
cat_imputer = SimpleImputer(strategy='most_frequent')

# Combinar con ColumnTransformer
preprocessor = ColumnTransformer([
    ('num', num_imputer, num_cols),
    ('cat', cat_imputer, cat_cols)
])

data_imputed = preprocessor.fit_transform(df)
```

---

## 6. Buenas prácticas

1. **Siempre explorar antes de imputar** — usa `missingno` para entender el patrón.
2. **Aplicar `fit` solo en train, `transform` en test** — evita *data leakage*.
3. **Normalizar antes de KNN** — la distancia euclidiana es sensible a la escala.
4. **Documentar la estrategia elegida** — el método de imputación es parte del preprocesamiento y afecta la reproducibilidad.
5. **Considerar `MissingIndicator`** — añade una columna binaria que indica si el valor fue imputado, preservando información sobre el patrón de nulidad.

```python
from sklearn.impute import MissingIndicator

indicator = MissingIndicator()
missing_flags = indicator.fit_transform(df)  # matriz booleana
```

6. **Evaluar el impacto** — comparar métricas del modelo final con diferentes estrategias de imputación para elegir la más adecuada al problema.

---

## Referencias

- Rubin, D. B. (1976). *Inference and missing data.* Biometrika.
- van Buuren, S. (2018). *Flexible Imputation of Missing Data.* CRC Press. [https://stefvanbuuren.name/fimd/](https://stefvanbuuren.name/fimd/)
- scikit-learn documentation: [Imputation of missing values](https://scikit-learn.org/stable/modules/impute.html)
- `missingno` library: [https://github.com/ResidentMario/missingno](https://github.com/ResidentMario/missingno)
