import streamlit as st
import pickle
import numpy as np
import joblib
from sklearn.preprocessing import MinMaxScaler
import matplotlib.image as mpimg

# Cargar el modelo previamente entrenado (SVC)
with open('modelo_cardiaco_svc.pkl', 'rb') as f:
    model = pickle.load(f)

# Cargar el scaler previamente entrenado
scaler = joblib.load('scaler.jb')

# Título de la app
st.title("Predicción de Problemas Cardíacos")

# Subtítulo con marca registrada
st.subheader("Elaborado por ® UNAB 2025")

# Entradas con valores por defecto y paso (edad: 1, colesterol: 2)
edad = st.slider("Selecciona tu edad", min_value=25, max_value=80, value=55, step=1)
colesterol = st.slider("Selecciona tu nivel de colesterol (mg/dL)", min_value=120, max_value=600, value=242, step=2)

# Normalización de los datos de entrada usando el scaler cargado
datos = np.array([[edad, colesterol]])
datos_normalizados = scaler.transform(datos)

# Predicción del modelo
if st.button("Predecir"):
    # Realizamos la predicción con el modelo cargado
    prediccion = model.predict(datos_normalizados)

    # Mostrar el resultado de la predicción
    if prediccion == 0:
        st.write("🚫 No sufrirás de problemas cardíacos según el modelo.")
        
        # Mostrar imagen de persona feliz (no problemas cardíacos)
        img_url = "https://img.freepik.com/foto-gratis/feliz-mujer-atractiva-bailando-divirtiendose-levantando-manos-p…"
        st.image(img_url, caption="Persona feliz: No problemas cardíacos", use_column_width=True)
    else:
        st.write("❤️ El modelo predice que podrías sufrir de problemas cardíacos.")
        
        # Mostrar imagen de persona con problema cardíaco
        img_url = "https://www.shutterstock.com/image-vector/unwell-man-feel-sick-suffer-600nw-2145831019.jpg"
        st.image(img_url, caption="Problema cardíaco detectado", use_column_width=True)
