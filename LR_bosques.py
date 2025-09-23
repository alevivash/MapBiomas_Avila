##Aqui se encuentra la regresion lineal
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
import numpy as np

# --- 1. Cargar y preparar los datos ---
try:
    # Carga el archivo CSV en un DataFrame de pandas con la codificación correcta.
    df = pd.read_csv('datos_biomap_csv.csv', encoding='latin-1')
except FileNotFoundError:
    print("Error: El archivo 'datos_biomap_csv.csv' no se encontró.")
    print("Por favor, asegúrate de que el archivo esté en la misma carpeta que el script.")
    exit()

# Definir los datos para la regresión
# El eje X es la columna 'Year', convertida a un array 2D
X = df[['Year']]
# El eje Y es la superficie de las formaciones boscosas y las áreas agrícolas
y_boscosas = df['Bosque']
y_agropecuario = df['Uso agropecuario']

# --- 2. Entrenar los modelos de regresión lineal ---
# Modelo para los Bosques
modelo_boscosas = LinearRegression()
modelo_boscosas.fit(X, y_boscosas)
prediccion_boscosas = modelo_boscosas.predict(X)

# Modelo para el Uso Agropecuario
modelo_agropecuario = LinearRegression()
modelo_agropecuario.fit(X, y_agropecuario)
prediccion_agropecuario = modelo_agropecuario.predict(X)

# --- 3. Visualizar los resultados ---
plt.style.use('seaborn-v0_8-whitegrid')

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(18, 7))

# Gráfico para los Bosques
ax1.scatter(X, y_boscosas, label='Datos originales', color='blue')
ax1.plot(X, prediccion_boscosas, color='red', linewidth=3, label='Línea de Regresión')
ax1.set_title('Regresión Lineal: Bosques (1985-2023)', fontsize=16)
ax1.set_xlabel('Año', fontsize=12)
ax1.set_ylabel('Superficie (hectáreas)', fontsize=12)
ax1.legend()
ax1.grid(True)

# Gráfico para Uso Agropecuario
ax2.scatter(X, y_agropecuario, label='Datos originales', color='green')
ax2.plot(X, prediccion_agropecuario, color='orange', linewidth=3, label='Línea de Regresión')
ax2.set_title('Regresión Lineal: Uso Agropecuario (1985-2023)', fontsize=16)
ax2.set_xlabel('Año', fontsize=12)
ax2.set_ylabel('Superficie (hectáreas)', fontsize=12)
ax2.legend()
ax2.grid(True)

plt.tight_layout()
plt.show()

# --- 4. Interpretación de los resultados ---
print("Resultados del Modelo de Regresión Lineal:")
print("-" * 40)
print("Bosques:")
print(f"  - Pendiente (coeficiente): {modelo_boscosas.coef_[0]:.2f} ha/año")
print(f"  - Intercepto: {modelo_boscosas.intercept_:.2f} ha")
print("-" * 40)
print("Uso Agropecuario:")
print(f"  - Pendiente (coeficiente): {modelo_agropecuario.coef_[0]:.2f} ha/año")
print(f"  - Intercepto: {modelo_agropecuario.intercept_:.2f} ha")
print("-" * 40)