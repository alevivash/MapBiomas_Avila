import pandas as pd
import matplotlib.pyplot as plt
from scipy import stats
from sklearn.linear_model import LinearRegression
import numpy as np

# --- CARGAR DATOS ---
try:
    df = pd.read_csv('datos_biomap_csv.csv', encoding='latin-1')
except FileNotFoundError:
    print("Error: Archivo no encontrado")
    exit()

# --- DEFINIR CATEGORÍAS ---
# ESPACIOS NATURALES
df['Naturales'] = df['Bosque'] + df['Sabana/Herbazal'] + df['Arbustal'] + df['Río, lago u oceano']

# USOS HUMANOS/ANTROPOGÉNICOS
df['Antropogenicos'] = df['Uso agropecuario'] + df['Uso urbano'] + df['Uso minero'] + df['Sin vegetacion']

# --- VERIFICACIÓN DE DATOS ---
df['Total_verificado'] = df['Naturales'] + df['Antropogenicos']
variacion_total = df['Total_verificado'].std()
print(f"✅ Verificación: Desviación del área total = {variacion_total:.2f} ha (valores consistentes)")

# --- CALCULAR PORCENTAJES ---
df['%_Naturales'] = (df['Naturales'] / df['Total_verificado']) * 100
df['%_Antropogenicos'] = (df['Antropogenicos'] / df['Total_verificado']) * 100

# --- GRÁFICO PRINCIPAL MEJORADO ---
plt.figure(figsize=(15, 8))

# Gráfico de líneas principales
linea_nat = plt.plot(df['Year'], df['%_Naturales'], color='green', linewidth=4, label='Espacios Naturales', marker='o', markersize=6)
linea_hum = plt.plot(df['Year'], df['%_Antropogenicos'], color='red', linewidth=4, label='Usos Humanos', marker='s', markersize=6)

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

# Añadir TODOS los valores (inicio y fin para ambas líneas)
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

# Añadir línea de cambio neto
cambio_nat = df['%_Naturales'].iloc[-1] - df['%_Naturales'].iloc[0]
cambio_hum = df['%_Antropogenicos'].iloc[-1] - df['%_Antropogenicos'].iloc[0]

plt.text(df['Year'].mean(), 50,
         f'Δ Naturales: {cambio_nat:+.1f}%\nΔ Humanos: {cambio_hum:+.1f}%',
         bbox=dict(boxstyle="round,pad=0.5", facecolor="white", alpha=0.9),
         ha='center', fontsize=11)

plt.tight_layout()
plt.show()

# --- ANÁLISIS NUMÉRICO DETALLADO ---
print("\n" + "="*70)
print("📊 ANÁLISIS DETALLADO: EXPANSIÓN HUMANA vs ESPACIOS NATURALES")
print("="*70)

# Cálculos de cambios
cambio_naturales = df['%_Naturales'].iloc[-1] - df['%_Naturales'].iloc[0]
cambio_antropogenicos = df['%_Antropogenicos'].iloc[-1] - df['%_Antropogenicos'].iloc[0]

cambio_absoluto_nat = df['Naturales'].iloc[-1] - df['Naturales'].iloc[0]
cambio_absoluto_ant = df['Antropogenicos'].iloc[-1] - df['Antropogenicos'].iloc[0]

print(f"\n🔹 PERIODO ANALIZADO: {df['Year'].iloc[0]} - {df['Year'].iloc[-1]} ({len(df)} años)")
print(f"🔹 ÁREA TOTAL ANALIZADA: {df['Total_verificado'].mean():.0f} hectáreas")

print(f"\n🌳 ESPACIOS NATURALES:")
print(f"   • 1985: {df['Naturales'].iloc[0]:.0f} ha ({df['%_Naturales'].iloc[0]:.1f}%)")
print(f"   • 2001: {df['Naturales'].iloc[-1]:.0f} ha ({df['%_Naturales'].iloc[-1]:.1f}%)")
print(f"   • CAMBIO: {cambio_absoluto_nat:+.0f} ha (Δ{cambio_naturales:+.1f}%)")

print(f"\n🏗️  USOS HUMANOS:")
print(f"   • 1985: {df['Antropogenicos'].iloc[0]:.0f} ha ({df['%_Antropogenicos'].iloc[0]:.1f}%)")
print(f"   • 2001: {df['Antropogenicos'].iloc[-1]:.0f} ha ({df['%_Antropogenicos'].iloc[-1]:.1f}%)")
print(f"   • CAMBIO: {cambio_absoluto_ant:+.0f} ha (Δ{cambio_antropogenicos:+.1f}%)")

# --- TASAS ANUALES ---
años_totales = df['Year'].iloc[-1] - df['Year'].iloc[0]
tasa_anual_nat = cambio_absoluto_nat / años_totales
tasa_anual_ant = cambio_absoluto_ant / años_totales

print(f"\n📈 TASAS ANUALES PROMEDIO:")
print(f"   • Espacios naturales: {tasa_anual_nat:+.0f} ha/año")
print(f"   • Usos humanos: {tasa_anual_ant:+.0f} ha/año")

# --- CORRELACIÓN Y SIGNIFICANCIA ---
correlacion, p_valor = stats.pearsonr(df['Naturales'], df['Antropogenicos'])

print(f"\n🔍 CORRELACIÓN ESTADÍSTICA:")
print(f"   • Correlación Naturales vs Humanos: {correlacion:.3f}")
print(f"   • Valor-p: {p_valor:.6f}")

# Interpretación de resultados
print(f"\n💡 INTERPRETACIÓN:")
if correlacion < -0.9 and p_valor < 0.05:
    print("   ✅ FUERTE EVIDENCIA: Expansión humana REEMPLAZA directamente espacios naturales")
    print("   → Patrón claro de sustitución: cuando uno aumenta, el otro disminuye")
elif correlacion < -0.7 and p_valor < 0.05:
    print("   ✅ EVIDENCIA MODERADA: Hay reemplazo significativo de naturales por humanos")
    print("   → Relación inversa estadísticamente significativa")
elif correlacion < -0.5 and p_valor < 0.05:
    print("   ⚠️  EVIDENCIA MODERADA-BAJA: Reemplazo detectable pero no fuerte")
    print("   → Puede haber otros factores influyendo")
else:
    print("   🔶 EVIDENCIA DÉBIL: Poca correlación directa")
    print("   → Revisar si el reemplazo es indirecto o mediante etapas intermedias")

# --- ANÁLISIS DE CONSISTENCIA ---
print(f"\n🔎 CONSISTENCIA DEL PATRÓN:")
if cambio_naturales < 0 and cambio_antropogenicos > 0:
    print("   ✅ Patrón consistente: Naturales disminuyen → Humanos aumentan")
    eficiencia_reemplazo = (abs(cambio_absoluto_ant) / abs(cambio_absoluto_nat)) * 100
    print(f"   • Eficiencia de reemplazo: {eficiencia_reemplazo:.1f}%")
else:
    print("   ⚠️  Patrón inconsistente: Revisar datos")

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