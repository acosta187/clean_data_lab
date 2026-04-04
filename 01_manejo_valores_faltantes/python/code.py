# ==============================
# 1. IMPORTS
# ==============================
import pandas as pd
import numpy as np

import matplotlib.pyplot as plt
import seaborn as sns
import missingno as msno

from sklearn.compose import ColumnTransformer
from sklearn.experimental import enable_iterative_imputer  # noqa
from sklearn.impute import SimpleImputer, IterativeImputer
from sklearn.ensemble import RandomForestRegressor


# ==============================
# 2. DATASET: HEART FAILURE
# ==============================
url_heart = "https://raw.githubusercontent.com/acosta187/datos/refs/heads/main/heart_failure_clinical_records_dataset_NA.csv"
df_heart = pd.read_csv(url_heart)

print("\nDataset Heart Failure")
print(df_heart.head())

# Visualización de nulos
msno.bar(df_heart)
plt.title("Valores faltantes - Heart")
plt.show()

msno.matrix(df_heart)
plt.title("Distribución de nulos - Heart")
plt.show()

# Columnas
num_cols_heart = [
    'age', 'creatinine_phosphokinase', 'ejection_fraction',
    'platelets', 'serum_creatinine', 'serum_sodium', 'time'
]

cat_cols_heart = [
    'anaemia', 'diabetes', 'high_blood_pressure',
    'sex', 'smoking'
]

# Imputadores
num_imputer = IterativeImputer(
    estimator=RandomForestRegressor(),
    random_state=42
)

cat_imputer = SimpleImputer(strategy='most_frequent')

# Preprocesador
preprocessor_heart = ColumnTransformer([
    ('num', num_imputer, num_cols_heart),
    ('cat', cat_imputer, cat_cols_heart)
])

# Aplicar imputación
data_heart_imputed = preprocessor_heart.fit_transform(df_heart)

df_heart_imputed = pd.DataFrame(
    data_heart_imputed,
    columns=num_cols_heart + cat_cols_heart
)

print("\nDatos imputados (Heart)")
print(df_heart_imputed.head())


# ==============================
# 3. DATASET: TITANIC
# ==============================
url_titanic = "https://raw.githubusercontent.com/plotly/datasets/refs/heads/master/titanic.csv"
df_titanic = pd.read_csv(url_titanic)

print("\nDataset Titanic")
print(df_titanic.head())

# Selección de columnas relevantes
num_cols_titanic = ["Age", "SibSp", "Parch", "Fare"]
cat_cols_titanic = ['Pclass', 'Sex', 'Embarked']
target = ["Survived"]

cols = target + num_cols_titanic + cat_cols_titanic
df_titanic = df_titanic[cols]

# Visualización de nulos
msno.bar(df_titanic)
plt.title("Valores faltantes - Titanic")
plt.show()

msno.matrix(df_titanic)
plt.title("Distribución de nulos - Titanic")
plt.show()

# Imputadores
num_imputer = IterativeImputer(
    estimator=RandomForestRegressor(),
    random_state=42
)

cat_imputer = SimpleImputer(strategy='most_frequent')

# Preprocesador
preprocessor_titanic = ColumnTransformer([
    ('num', num_imputer, num_cols_titanic),
    ('cat', cat_imputer, cat_cols_titanic)
])

# Aplicar imputación
data_titanic_imputed = preprocessor_titanic.fit_transform(df_titanic)

df_titanic_imputed = pd.DataFrame(
    data_titanic_imputed,
    columns=num_cols_titanic + cat_cols_titanic
)

print("\nDatos imputados (Titanic)")
print(df_titanic_imputed.head())