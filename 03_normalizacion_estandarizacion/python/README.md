# 🧹 Preprocesamiento de Datos con Scikit-learn — Titanic Dataset

Este script demuestra cómo preprocesar un dataset real (Titanic) aplicando **estandarización/normalización** a variables numéricas y **codificación** a variables categóricas usando `sklearn.compose.ColumnTransformer`.

---

## 📁 Estructura del Script

```
titanic_preprocessing.py
├── Carga de datos desde GitHub
├── Selección de variables (numéricas + categóricas + target)
├── Eliminación de valores nulos
├── Pipeline de preprocesamiento con ColumnTransformer
│   ├── MinMaxScaler → variables numéricas
│   └── OneHotEncoder → variables categóricas
└── Exportación del DataFrame transformado
```

---

## ⚙️ Dependencias

```bash
pip install pandas scikit-learn
```

---

## 📊 Variables utilizadas

| Variable | Tipo | Rol |
|----------|------|-----|
| `Age` | Numérica | Feature |
| `Fare` | Numérica | Feature |
| `Sex` | Categórica | Feature |
| `Embarked` | Categórica | Feature |
| `Pclass` | Categórica | Feature |
| `Survived` | Binaria | Target |

---

## 🔢 Métodos de Escalado — Comparativa

El script usa **MinMaxScaler**, pero existen varias alternativas según el contexto de los datos:

### 1. `MinMaxScaler` ✅ *(usado en el script)*
Escala cada feature al rango `[0, 1]`.

```
X_scaled = (X - X_min) / (X_max - X_min)
```

**Cuándo usarlo:** Cuando los datos no siguen distribución normal y los algoritmos son sensibles a la magnitud (ej. redes neuronales, KNN, SVM).  
**Limitación:** Muy sensible a outliers, ya que el mínimo y máximo pueden ser valores atípicos.

---

### 2. `StandardScaler`
Estandariza restando la media y dividiendo por la desviación estándar (media = 0, std = 1).

```
X_scaled = (X - μ) / σ
```

**Cuándo usarlo:** Cuando los datos son aproximadamente normales. Ideal para regresión lineal, PCA, SVM con kernel lineal.  
**Limitación:** También es afectado por outliers (usa la media, que no es robusta).

```python
from sklearn.preprocessing import StandardScaler
```

---

### 3. `RobustScaler` 🛡️ *(robusto ante outliers)*
Usa la **mediana** y el **rango intercuartil (IQR)** en lugar de la media y la desviación estándar.

```
X_scaled = (X - mediana) / IQR
```

**Cuándo usarlo:** Cuando los datos contienen outliers significativos. Como en este dataset (la variable `Fare` tiene valores muy extremos).  
**Ventaja clave:** Los outliers influyen muy poco en el escalado porque se basa en percentiles (Q1 y Q3).

```python
from sklearn.preprocessing import RobustScaler
```

---

### 4. `MaxAbsScaler`
Divide cada feature por su valor absoluto máximo, resultando en un rango `[-1, 1]`.

```
X_scaled = X / |X_max|
```

**Cuándo usarlo:** Datos dispersos (sparse), o cuando los datos ya están centrados en cero.  
**Limitación:** Sensible a outliers (depende del valor máximo absoluto).

```python
from sklearn.preprocessing import MaxAbsScaler
```

---

### 5. `Normalizer`
Normaliza cada **muestra** (fila) individualmente para que tenga norma unitaria, en lugar de escalar cada columna.

```
X_scaled = X / ||X||
```

**Cuándo usarlo:** En modelos que trabajan con distancias o similitud de vectores (ej. clustering, text mining con TF-IDF).  
**Nota:** Opera sobre filas, no sobre columnas — es conceptualmente distinto a los anteriores.

```python
from sklearn.preprocessing import Normalizer
```

---

## 📋 Tabla Resumen

| Scaler | Fórmula base | Robusto a outliers | Rango resultante | Caso de uso típico |
|--------|-------------|-------------------|------------------|--------------------|
| `MinMaxScaler` | min-max | ❌ No | `[0, 1]` | Redes neuronales, KNN |
| `StandardScaler` | media + std | ❌ No | `(-∞, +∞)` centrado | Regresión, PCA, SVM |
| `RobustScaler` | mediana + IQR | ✅ Sí | Variable | Datos con outliers |
| `MaxAbsScaler` | valor máx abs | ❌ No | `[-1, 1]` | Datos dispersos |
| `Normalizer` | norma vectorial | ✅ Parcial | norma = 1 | Similitud de vectores |

---

## 🔄 ¿Cómo cambiar el scaler en el script?

Reemplaza `MinMaxScaler()` por cualquier otro scaler en el `ColumnTransformer`:

```python
from sklearn.preprocessing import RobustScaler  # o StandardScaler, etc.

preprocessor = ColumnTransformer(
    transformers=[
        ('num', RobustScaler(), num_cols),  # 👈 cambio aquí
        ('cat', OneHotEncoder(drop='first', sparse_output=False), cat_cols)
    ]
)
```

---

## 🧪 Codificación Categórica

Se usa `OneHotEncoder` con `drop='first'` para evitar multicolinealidad (dummy variable trap). Por ejemplo, `Sex` tiene 2 valores (`male`, `female`) y se convierte en 1 columna binaria.

| Variable original | Columnas generadas |
|-------------------|--------------------|
| `Sex` (male/female) | `Sex_male` |
| `Embarked` (C/Q/S) | `Embarked_Q`, `Embarked_S` |
| `Pclass` (1/2/3) | `Pclass_2`, `Pclass_3` |

---

## 💡 Recomendación para este dataset

Dado que `Fare` contiene valores muy sesgados y outliers extremos (algunos pasajeros pagaron tarifas muy altas), se recomienda considerar **`RobustScaler`** como alternativa más adecuada para esta variable específica.

---

## 📚 Referencias

- [Scikit-learn: Preprocessing Data](https://scikit-learn.org/stable/modules/preprocessing.html)
- [Scikit-learn: ColumnTransformer](https://scikit-learn.org/stable/modules/generated/sklearn.compose.ColumnTransformer.html)
- [Titanic Dataset](https://raw.githubusercontent.com/datasciencedojo/datasets/master/titanic.csv)
