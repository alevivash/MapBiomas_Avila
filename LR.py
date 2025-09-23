import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
import numpy as np
from scipy import stats

# --- 1. Cargar y preparar los datos ---
try:
    df = pd.read_csv('datos_biomap_csv.csv', encoding='latin-1')
except FileNotFoundError:
    print("Error: El archivo 'datos_biomap_csv.csv' no se encontró.")
    exit()

# --- DEFINIR ESPACIOS NATURALES Y USOS HUMANOS ---
# Espacios Naturales (incluye todos los biomas naturales)
df['Espacios_Naturales'] = df['Bosque'] + df['Sabana/Herbazal'] + df['Arbustal'] + df['Río, lago u oceano']

# Usos Humanos/Antropogénicos
df['Usos_Humanos'] = df['Uso agropecuario'] + df['Uso urbano'] + df['Uso minero'] + df['Sin vegetacion']

# Verificar consistencia
df['Total_Verificado'] = df['Espacios_Naturales'] + df['Usos_Humanos']
print(f"✅ Consistencia de datos: Área total = {df['Total_Verificado'].mean():.0f} ± {df['Total_Verificado'].std():.1f} ha")

# --- 2. Entrenar modelos de regresión lineal ---
X = df[['Year']]

# Modelo para Espacios Naturales
modelo_naturales = LinearRegression()
modelo_naturales.fit(X, df['Espacios_Naturales'])
prediccion_naturales = modelo_naturales.predict(X)

# Modelo para Usos Humanos
modelo_humanos = LinearRegression()
modelo_humanos.fit(X, df['Usos_Humanos'])
prediccion_humanos = modelo_humanos.predict(X)

# --- 3. Visualizar los resultados MEJORADOS ---
plt.style.use('seaborn-v0_8-whitegrid')
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 8))

# --- GRÁFICO 1: Espacios Naturales ---
scatter1 = ax1.scatter(X, df['Espacios_Naturales'], label='Datos observados', color='#2E8B57', alpha=0.7, s=60)
line1 = ax1.plot(X, prediccion_naturales, color='darkgreen', linewidth=4, label='Tendencia lineal')
ax1.set_title('REGRESIÓN LINEAL: Espacios Naturales\n(1985-2023)', fontsize=16, fontweight='bold', pad=15)
ax1.set_xlabel('Año', fontsize=12, fontweight='bold')
ax1.set_ylabel('Superficie (hectáreas)', fontsize=12, fontweight='bold')
ax1.legend(fontsize=11)
ax1.grid(True, alpha=0.3)

# Añadir ecuación de regresión y R²
r2_naturales = modelo_naturales.score(X, df['Espacios_Naturales'])
ecuacion_nat = f'y = {modelo_naturales.coef_[0]:.1f}x + {modelo_naturales.intercept_:.0f}'
ax1.text(0.02, 0.98, f'R² = {r2_naturales:.3f}\n{ecuacion_nat}',
         transform=ax1.transAxes, verticalalignment='top',
         bbox=dict(boxstyle='round', facecolor='white', alpha=0.8),
         fontsize=10, fontfamily='monospace')

# --- GRÁFICO 2: Usos Humanos ---
scatter2 = ax2.scatter(X, df['Usos_Humanos'], label='Datos observados', color='#DC143C', alpha=0.7, s=60)
line2 = ax2.plot(X, prediccion_humanos, color='darkred', linewidth=4, label='Tendencia lineal')
ax2.set_title('REGRESIÓN LINEAL: Usos Humanos\n(1985-2023)', fontsize=16, fontweight='bold', pad=15)
ax2.set_xlabel('Año', fontsize=12, fontweight='bold')
ax2.set_ylabel('Superficie (hectáreas)', fontsize=12, fontweight='bold')
ax2.legend(fontsize=11)
ax2.grid(True, alpha=0.3)

# Añadir ecuación de regresión y R²
r2_humanos = modelo_humanos.score(X, df['Usos_Humanos'])
ecuacion_hum = f'y = {modelo_humanos.coef_[0]:.1f}x + {modelo_humanos.intercept_:.0f}'
ax2.text(0.02, 0.98, f'R² = {r2_humanos:.3f}\n{ecuacion_hum}',
         transform=ax2.transAxes, verticalalignment='top',
         bbox=dict(boxstyle='round', facecolor='white', alpha=0.8),
         fontsize=10, fontfamily='monospace')

plt.tight_layout()
plt.show()

# --- 5. ANÁLISIS ESTADÍSTICO DETALLADO ---
print("\n" + "="*70)
print("📊 ANÁLISIS ESTADÍSTICO DETALLADO")
print("="*70)

print(f"\n🌳 ESPACIOS NATURALES:")
print(f"   • Pendiente: {modelo_naturales.coef_[0]:.1f} ha/año")
print(f"   • Intercepto: {modelo_naturales.intercept_:.0f} ha")
print(f"   • R²: {r2_naturales:.3f} (el modelo explica {r2_naturales*100:.1f}% de la variación)")
print(f"   • Interpretación: Los espacios naturales {'DISMINUYEN' if modelo_naturales.coef_[0] < 0 else 'AUMENTAN'} {abs(modelo_naturales.coef_[0]):.1f} ha por año")

print(f"\n🏗️  USOS HUMANOS:")
print(f"   • Pendiente: {modelo_humanos.coef_[0]:.1f} ha/año")
print(f"   • Intercepto: {modelo_humanos.intercept_:.0f} ha")
print(f"   • R²: {r2_humanos:.3f} (el modelo explica {r2_humanos*100:.1f}% de la variación)")
print(f"   • Interpretación: Los usos humanos {'DISMINUYEN' if modelo_humanos.coef_[0] < 0 else 'AUMENTAN'} {abs(modelo_humanos.coef_[0]):.1f} ha por año")

# --- 6. CORRELACIÓN Y SIGNIFICANCIA ---
correlacion, p_valor = stats.pearsonr(df['Espacios_Naturales'], df['Usos_Humanos'])

print(f"\n🔍 RELACIÓN ENTRE VARIABLES:")
print(f"   • Correlación: {correlacion:.3f}")
print(f"   • Valor-p: {p_valor:.6f}")

if correlacion < -0.8 and p_valor < 0.05:
    print("   ✅ FUERTE RELACIÓN INVERSA: La expansión humana explica la pérdida de naturales")
elif correlacion < -0.6 and p_valor < 0.05:
    print("   ✅ RELACIÓN INVERSA SIGNIFICATIVA: Hay patrón de reemplazo")
else:
    print("   ⚠️  Relación menos clara: Pueden influir otros factores")

# --- 7. PROYECCIÓN A FUTURO ---
print(f"\n🔮 PROYECCIÓN (si la tendencia continúa):")
for year in [2025, 2030, 2040]:
    nat_proy = modelo_naturales.predict([[year]])[0]
    hum_proy = modelo_humanos.predict([[year]])[0]
    print(f"   • {year}: Naturales = {nat_proy:.0f} ha, Humanos = {hum_proy:.0f} ha")

