import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
import numpy as np
from scipy import stats

# --- 1. CARGAR Y PREPARAR DATOS ---
try:
    df = pd.read_csv('datos_biomap_csv.csv', encoding='latin-1')
except FileNotFoundError:
    print("Error: El archivo 'datos_biomap_csv.csv' no se encontró.")
    exit()

# --- DEFINIR CATEGORÍAS ---
# Espacios Naturales
df['Naturales'] = df['Bosque'] + df['Sabana/Herbazal'] + df['Arbustal'] + df['Río, lago u oceano']
# Usos Humanos
df['Antropogenicos'] = df['Uso agropecuario'] + df['Uso urbano'] + df['Uso minero'] + df['Sin vegetacion']

# Verificación de datos
df['Total_verificado'] = df['Naturales'] + df['Antropogenicos']
variacion_total = df['Total_verificado'].std()
print(f"✅ Verificación: Desviación del área total = {variacion_total:.2f} ha")

# Calcular porcentajes
df['%_Naturales'] = (df['Naturales'] / df['Total_verificado']) * 100
df['%_Antropogenicos'] = (df['Antropogenicos'] / df['Total_verificado']) * 100

# --- 2. GRÁFICO 1: EVOLUCIÓN DE PORCENTAJES ---
plt.figure(figsize=(15, 8))

# Gráfico de líneas principales
plt.plot(df['Year'], df['%_Naturales'], color='green', linewidth=4, label='Espacios Naturales', marker='o', markersize=6)
plt.plot(df['Year'], df['%_Antropogenicos'], color='red', linewidth=4, label='Usos Humanos', marker='s', markersize=6)

# Área sombreada
plt.fill_between(df['Year'], df['%_Naturales'], alpha=0.2, color='green')
plt.fill_between(df['Year'], df['%_Antropogenicos'], alpha=0.2, color='red')

# Personalización
plt.title('EXPANSIÓN HUMANA vs ESPACIOS NATURALES\nParque Nacional El Ávila (1985-2023)',
          fontsize=16, fontweight='bold', pad=20)
plt.ylabel('Porcentaje del Área Total (%)', fontsize=12)
plt.xlabel('Año', fontsize=12)
plt.legend(fontsize=12, loc='best')
plt.grid(True, alpha=0.3)
plt.ylim(0, 100)

# Añadir valores
valores = [
    (df['Year'].iloc[0], df['%_Naturales'].iloc[0], 'green', (10, 15), f"{df['%_Naturales'].iloc[0]:.1f}%"),
    (df['Year'].iloc[-1], df['%_Naturales'].iloc[-1], 'green', (-40, 15), f"{df['%_Naturales'].iloc[-1]:.1f}%"),
    (df['Year'].iloc[0], df['%_Antropogenicos'].iloc[0], 'red', (10, -25), f"{df['%_Antropogenicos'].iloc[0]:.1f}%"),
    (df['Year'].iloc[-1], df['%_Antropogenicos'].iloc[-1], 'red', (-40, -25), f"{df['%_Antropogenicos'].iloc[-1]:.1f}%")
]

for x, y, color, offset, texto in valores:
    plt.annotate(texto, xy=(x, y), xytext=offset, textcoords='offset points',
                 color=color, fontweight='bold', fontsize=10,
                 bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.9))

# Añadir cambio neto
cambio_nat = df['%_Naturales'].iloc[-1] - df['%_Naturales'].iloc[0]
cambio_hum = df['%_Antropogenicos'].iloc[-1] - df['%_Antropogenicos'].iloc[0]

plt.text(df['Year'].mean(), 50,
         f'Δ Naturales: {cambio_nat:+.1f}%\nΔ Humanos: {cambio_hum:+.1f}%',
         bbox=dict(boxstyle="round,pad=0.5", facecolor="white", alpha=0.9),
         ha='center', fontsize=11)

plt.tight_layout()
plt.show()

# --- 3. GRÁFICO 2: ÁREA APILADA ---
min_naturales = df['Naturales'].min()
base_line_optima = min_naturales - 500

plt.figure(figsize=(14, 8))

# Ajustar los datos
naturales_ajustado = df['Naturales'] - base_line_optima
agropecuario_ajustado = df['Uso agropecuario']
urbano_ajustado = df['Uso urbano']
otros_ajustado = df['Uso minero'] + df['Sin vegetacion']

plt.stackplot(df['Year'],
             [naturales_ajustado, agropecuario_ajustado, urbano_ajustado, otros_ajustado],
             labels=['Naturales', 'Agricultura', 'Urbano', 'Otros usos humanos'],
             colors=['#2E8B57', '#FFD700', '#DC143C', '#FF8C00'],
             alpha=0.8)

plt.axhline(y=0, color='black', linestyle='-', alpha=0.7, linewidth=2)
plt.title(f'EXPANSIÓN HUMANA - COMPOSICIÓN DE USOS\nParque Nacional El Ávila (1985-2023)',
          fontsize=16, fontweight='bold', pad=20)
plt.ylabel(f'Superficie (hectáreas)', fontsize=12)
plt.xlabel('Año', fontsize=12)
plt.legend(loc='upper left')
plt.grid(True, alpha=0.3)

# Formatear eje Y
plt.gca().yaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f'{x + base_line_optima:,.0f}'))
plt.tight_layout()
plt.show()

# --- 4. REGRESIÓN LINEAL  ---
# Preparar datos para regresión (evitar warnings)
X = df['Year'].values.reshape(-1, 1)
y_naturales = df['Naturales'].values
y_humanos = df['Antropogenicos'].values

# Modelos de regresión
modelo_naturales = LinearRegression()
modelo_humanos = LinearRegression()

modelo_naturales.fit(X, y_naturales)
modelo_humanos.fit(X, y_humanos)

prediccion_naturales = modelo_naturales.predict(X)
prediccion_humanos = modelo_humanos.predict(X)

# Gráficos de regresión
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 8))

# Gráfico 1: Espacios Naturales
ax1.scatter(df['Year'], y_naturales, label='Datos observados', color='#2E8B57', alpha=0.7, s=60)
ax1.plot(df['Year'], prediccion_naturales, color='darkgreen', linewidth=4, label='Tendencia lineal')
ax1.set_title('REGRESIÓN LINEAL: Espacios Naturales\n(1985-2023)', fontsize=16, fontweight='bold', pad=15)
ax1.set_xlabel('Año', fontsize=12, fontweight='bold')
ax1.set_ylabel('Superficie (hectáreas)', fontsize=12, fontweight='bold')
ax1.legend(fontsize=11)
ax1.grid(True, alpha=0.3)

r2_naturales = modelo_naturales.score(X, y_naturales)
ecuacion_nat = f'y = {modelo_naturales.coef_[0]:.1f}x + {modelo_naturales.intercept_:.0f}'
ax1.text(0.02, 0.98, f'R² = {r2_naturales:.3f}\n{ecuacion_nat}',
         transform=ax1.transAxes, verticalalignment='top',
         bbox=dict(boxstyle='round', facecolor='white', alpha=0.8),
         fontsize=10, fontfamily='monospace')

# Gráfico 2: Usos Humanos
ax2.scatter(df['Year'], y_humanos, label='Datos observados', color='#DC143C', alpha=0.7, s=60)
ax2.plot(df['Year'], prediccion_humanos, color='darkred', linewidth=4, label='Tendencia lineal')
ax2.set_title('REGRESIÓN LINEAL: Usos Humanos\n(1985-2023)', fontsize=16, fontweight='bold', pad=15)
ax2.set_xlabel('Año', fontsize=12, fontweight='bold')
ax2.set_ylabel('Superficie (hectáreas)', fontsize=12, fontweight='bold')
ax2.legend(fontsize=11)
ax2.grid(True, alpha=0.3)

r2_humanos = modelo_humanos.score(X, y_humanos)
ecuacion_hum = f'y = {modelo_humanos.coef_[0]:.1f}x + {modelo_humanos.intercept_:.0f}'
ax2.text(0.02, 0.98, f'R² = {r2_humanos:.3f}\n{ecuacion_hum}',
         transform=ax2.transAxes, verticalalignment='top',
         bbox=dict(boxstyle='round', facecolor='white', alpha=0.8),
         fontsize=10, fontfamily='monospace')

plt.tight_layout()
plt.show()

# --- 5. ANÁLISIS COMPLETO ---
print("\n" + "="*70)
print("📊 ANÁLISIS COMPLETO: EXPANSIÓN HUMANA vs ESPACIOS NATURALES")
print("="*70)

# Cálculos básicos
cambio_absoluto_nat = df['Naturales'].iloc[-1] - df['Naturales'].iloc[0]
cambio_absoluto_ant = df['Antropogenicos'].iloc[-1] - df['Antropogenicos'].iloc[0]
años_totales = df['Year'].iloc[-1] - df['Year'].iloc[0]

print(f"\n🔹 PERIODO ANALIZADO: {df['Year'].iloc[0]} - {df['Year'].iloc[-1]} ({años_totales} años)")
print(f"🔹 ÁREA TOTAL: {df['Total_verificado'].mean():.0f} ± {variacion_total:.1f} ha")

print(f"\n🌳 ESPACIOS NATURALES:")
print(f"   • 1985: {df['Naturales'].iloc[0]:.0f} ha ({df['%_Naturales'].iloc[0]:.1f}%)")
print(f"   • 2023: {df['Naturales'].iloc[-1]:.0f} ha ({df['%_Naturales'].iloc[-1]:.1f}%)")
print(f"   • Cambio: {cambio_absoluto_nat:+.0f} ha (Δ{cambio_nat:+.1f}%)")
print(f"   • Tendencia: {modelo_naturales.coef_[0]:.1f} ha/año (R² = {r2_naturales:.3f})")

print(f"\n🏗️  USOS HUMANOS:")
print(f"   • 1985: {df['Antropogenicos'].iloc[0]:.0f} ha ({df['%_Antropogenicos'].iloc[0]:.1f}%)")
print(f"   • 2023: {df['Antropogenicos'].iloc[-1]:.0f} ha ({df['%_Antropogenicos'].iloc[-1]:.1f}%)")
print(f"   • Cambio: {cambio_absoluto_ant:+.0f} ha (Δ{cambio_hum:+.1f}%)")
print(f"   • Tendencia: {modelo_humanos.coef_[0]:.1f} ha/año (R² = {r2_humanos:.3f})")

# Correlación y significancia
correlacion, p_valor = stats.pearsonr(df['Naturales'], df['Antropogenicos'])

print(f"\n🔍 RELACIÓN ESTADÍSTICA:")
print(f"   • Correlación: {correlacion:.3f}")
print(f"   • Valor-p: {p_valor:.6f}")

# Interpretación
if correlacion < -0.8 and p_valor < 0.05:
    print("   ✅ FUERTE EVIDENCIA: Expansión humana REEMPLAZA directamente espacios naturales")
elif correlacion < -0.6 and p_valor < 0.05:
    print("   ✅ EVIDENCIA MODERADA: Hay reemplazo significativo")
else:
    print("   ⚠️  EVIDENCIA MODERADA-BAJA: Revisar otros factores")

# Proyecciones futuras (sin warnings)
print(f"\n🔮 PROYECCIÓN (tendencia actual):")
future_years = np.array([2025, 2030, 2040]).reshape(-1, 1)
proy_nat = modelo_naturales.predict(future_years)
proy_hum = modelo_humanos.predict(future_years)

for i, year in enumerate([2025, 2030, 2040]):
    print(f"   • {year}: Naturales = {proy_nat[i]:.0f} ha, Humanos = {proy_hum[i]:.0f} ha")

# Resumen ejecutivo
print("\n" + "="*70)
print("🎯 RESUMEN EJECUTIVO")
print("="*70)

if cambio_nat < -2:  # Si pérdida mayor al 2%
    print("🔻 CONCLUSIÓN: EXPANSIÓN HUMANA SIGNIFICATIVA")
    print(f"   • {abs(cambio_absoluto_nat):.0f} ha de naturales perdidas")
    print(f"   • {cambio_absoluto_ant:.0f} ha de usos humanos ganadas")
    print(f"   • {abs(cambio_nat):.1f}% del territorio transformado")
    print(f"   • Tasa: {abs(modelo_naturales.coef_[0]):.1f} ha/año de pérdida natural")
else:
    print("🔸 CONCLUSIÓN: CAMBIO MODERADO")

print("="*70)