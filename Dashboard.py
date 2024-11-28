import streamlit as st
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error
import matplotlib.pyplot as plt

# Configuración inicial
st.set_page_config(page_title="Predicción de Ventas", layout="wide")

# Título
st.title("📊 Predicción de Ventas con Random Forest")
st.markdown("Este proyecto utiliza un modelo de Machine Learning basado en Random Forest para predecir las ventas futuras de un negocio. 🛒")

# Sidebar
st.sidebar.header("Configuración de la Aplicación")
uploaded_file = st.sidebar.file_uploader("Sube tu archivo de ventas (Excel)", type=["xlsx"])

# Función para cargar datos
@st.cache
def cargar_datos(file):
    df = pd.read_excel(file)
    df['Fecha'] = pd.to_datetime(df['Fecha'])
    df['Mes'] = df['Fecha'].dt.month
    df['Día'] = df['Fecha'].dt.day
    return df

if uploaded_file:
    # Cargar datos
    datos = cargar_datos(uploaded_file)
    st.subheader("Vista previa de los datos cargados")
    st.write(datos.head())

    # Separar datos
    X = datos[['Mes', 'Día', 'Cantidad', 'No de tickets']]
    y = datos['Importe']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Entrenar modelo
    modelo_rf = RandomForestRegressor(n_estimators=100, random_state=42)
    modelo_rf.fit(X_train, y_train)

    # Predicciones
    y_pred = modelo_rf.predict(X_test)

    # Métricas
    mae = mean_absolute_error(y_test, y_pred)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))

    # Mostrar métricas
    st.subheader("Evaluación del Modelo")
    st.write(f"Error Absoluto Medio (MAE): ${mae:.2f}")
    st.write(f"Raíz del Error Cuadrático Medio (RMSE): ${rmse:.2f}")

    # Gráfico de comparación
    st.subheader("Comparación de Ventas Reales vs Predichas")
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(y_test.values[:30], label="Ventas Reales", marker='o', linestyle='-')
    ax.plot(y_pred[:30], label="Ventas Predichas", marker='x', linestyle='--')
    ax.set_xlabel("Índice")
    ax.set_ylabel("Importe de Ventas")
    ax.set_title("Comparación de Ventas Reales vs Predichas")
    ax.legend()
    st.pyplot(fig)

    # Predicción para el próximo mes
    st.sidebar.subheader("Predicción de Ventas del Próximo Mes")
    dias_mes_siguiente = st.sidebar.slider("Días del próximo mes a predecir", 1, 31, 30)

    # Generar datos simulados para el mes siguiente
    mes_siguiente = X['Mes'].max() + 1
    datos_mes_siguiente = pd.DataFrame({
        'Mes': [mes_siguiente] * dias_mes_siguiente,
        'Día': np.arange(1, dias_mes_siguiente + 1),
        'Cantidad': np.random.randint(20, 50, dias_mes_siguiente),
        'No de tickets': np.random.randint(5, 20, dias_mes_siguiente)
    })

    # Predicción
    predicciones_mes_siguiente = modelo_rf.predict(datos_mes_siguiente)

    # Mostrar resultados
    st.subheader("Predicción del Próximo Mes")
    datos_mes_siguiente['Predicción Importe'] = predicciones_mes_siguiente
    st.write(datos_mes_siguiente)

    # Gráfico de predicciones
    st.subheader("Proyección de Ventas del Próximo Mes")
    fig2, ax2 = plt.subplots(figsize=(10, 6))
    ax2.bar(datos_mes_siguiente['Día'], datos_mes_siguiente['Predicción Importe'], color='skyblue')
    ax2.set_xlabel("Día del Mes")
    ax2.set_ylabel("Importe Predicho")
    ax2.set_title("Proyección de Ventas por Día")
    st.pyplot(fig2)

else:
    st.info("Por favor, sube un archivo Excel con las columnas: Fecha, Importe, Cantidad y Número de Tickets.")

# Créditos
st.markdown("---")
st.markdown("**Desarrollado por:** Karla Barrientos - [GitHub](https://github.com)")
