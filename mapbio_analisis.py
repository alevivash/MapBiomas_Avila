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

# ESPACIOS NATURALES
df['Naturales'] = df['Bosque'] + df['Sabana/Herbazal'] + df['Arbustal'] + df['Río, lago u oceano']

# USOS HUMANOS/ANTROPOGÉNICOS
df['Antropogenicos'] = df['Uso agropecuario'] + df['Uso urbano'] + df['Uso minero'] + df['Sin vegetacion']

# Verificar que sumen el total (debe ser ~100% constante)
df['Total_verificado'] = df['Naturales'] + df['Antropogenicos']
print(f"Total verificado: {df['Total_verificado'].std():.2f} ha (debe ser baja variación)")

# Calcular porcentajes
df['%_Naturales'] = (df['Naturales'] / df['Total_verificado']) * 100
df['%_Antropogenicos'] = (df['Antropogenicos'] / df['Total_verificado']) * 100

# Gráfico principal de la hipótesis
plt.figure(figsize=(12, 6))
plt.plot(df['Year'], df['%_Naturales'], 'green', linewidth=3, label='Espacios Naturales')
plt.plot(df['Year'], df['%_Antropogenicos'], 'red', linewidth=3, label='Usos Humanos')
plt.title('EXPANSIÓN HUMANA vs ESPACIOS NATURALES (Ávila, 1985-2023)')
plt.ylabel('Porcentaje del Área Total (%)')
plt.xlabel('Año')
plt.legend()
plt.grid(True)
plt.ylim(0, 100)
plt.show()

# Calcular tasa de reemplazo
cambio_naturales = df['%_Naturales'].iloc[-1] - df['%_Naturales'].iloc[0]
cambio_antropogenicos = df['%_Antropogenicos'].iloc[-1] - df['%_Antropogenicos'].iloc[0]

print(f"📊 CAMBIO 1985-2001:")
print(f"Naturales: {df['%_Naturales'].iloc[0]:.1f}% → {df['%_Naturales'].iloc[-1]:.1f}% (Δ{cambio_naturales:+.1f}%)")
print(f"Humanos: {df['%_Antropogenicos'].iloc[0]:.1f}% → {df['%_Antropogenicos'].iloc[-1]:.1f}% (Δ{cambio_antropogenicos:+.1f}%)")

# Modelo: Naturales = f(Agropecuario, Urbano, Minero, Sin vegetación)
X = df[['Uso agropecuario', 'Uso urbano', 'Uso minero', 'Sin vegetacion']]
y = df['Naturales']

modelo = LinearRegression()
modelo.fit(X, y)

print("📈 CONTRIBUCIÓN DE CADA USO HUMANO a la pérdida de naturales:")
for i, uso in enumerate(['Agropecuario', 'Urbano', 'Minero', 'Sin vegetación']):
    coef = modelo.coef_[i]
    print(f"• {uso}: {abs(coef):.3f} ha naturales perdidas por cada ha ganada")

r2 = modelo.score(X, y)
print(f"\\Este modelo explica el {r2*100:.1f}% de la pérdida de espacios naturales")



#########################################################Grafico

# LÍNEA BASE AUTOMÁTICA desde donde comieza el grafico
min_naturales = df['Naturales'].min()
base_line_optima = min_naturales - 500  # 500 ha por debajo del mínimo

plt.figure(figsize=(14, 8))

# Ajustar los datos
naturales_ajustado = df['Naturales'] - base_line_optima #base_line_optima = 77,620.77
agropecuario_ajustado = df['Uso agropecuario']
urbano_ajustado = df['Uso urbano']
otros_ajustado = df['Uso minero'] + df['Sin vegetacion']

plt.stackplot(df['Year'],
             [naturales_ajustado, agropecuario_ajustado, urbano_ajustado, otros_ajustado],
             labels=['Naturales', 'Agricultura', 'Urbano', 'Otros usos humanos'],
             colors=['#2E8B57', '#FFD700', '#DC143C', '#FF8C00'],
             alpha=0.8)

plt.axhline(y=0, color='black', linestyle='-', alpha=0.7, linewidth=2)
plt.title(f'EXPANSIÓN HUMANA\nParque Nacional El Ávila (1985-2023)',
          fontsize=16, fontweight='bold', pad=20)
plt.ylabel(f'Superficie (hectáreas) 77,621 ha', fontsize=12)
plt.xlabel('Año', fontsize=12)
plt.legend(loc='upper left')
plt.grid(True, alpha=0.3)

# Formatear eje Y
plt.gca().yaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f'{x + base_line_optima:,.0f}'))
plt.tight_layout()
plt.show()

##print(f"Línea base óptima usada: {base_line_optima:,} ha") 77,620.77999999998

# --- RESUMEN EJECUTIVO FINAL ---
print("\n" + "="*70)
print("🎯 RESUMEN EJECUTIVO")
print("="*70)

if cambio_naturales < -5:  # Si pérdida mayor al 5%
    print("🔻 CONCLUSION: Hay EXPANSIÓN HUMANA significativa sobre espacios naturales")
    print(f"   • Se perdieron {abs(cambio_absoluto_nat):.0f} ha de áreas naturales")
    print(f"   • Se ganaron {cambio_absoluto_ant:.0f} ha de usos humanos")
    print(f"   • {abs(cambio_naturales):.1f}% del territorio natural fue transformado")
else:
    print("🔸 CONCLUSION: Cambio moderado, expansión humana limitada")

print("="*70)