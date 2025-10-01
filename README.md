# Predicción de Problemas Cardíacos

Esta aplicación permite predecir si una persona sufrirá o no de problemas cardíacos utilizando un modelo de aprendizaje automático entrenado. El modelo predice en base a dos características: **edad** y **colesterol**.

### Funcionalidades:
- La aplicación usa un modelo entrenado con un clasificador **SVC** (Máquina de Vectores de Soporte).
- Los datos de entrada (edad y colesterol) son normalizados usando un **MinMaxScaler**.
- Si el modelo predice que **no sufrirá problemas cardíacos**, se muestra una imagen de una persona feliz.
- Si el modelo predice que **sufrirá problemas cardíacos**, se muestra una imagen de una persona enferma.

### Requisitos:

- **Python 3.x**
- **Streamlit**: Framework para construir la interfaz.
- **Scikit-learn**: Para el modelo de predicción (SVC) y la normalización de datos.
- **Joblib**: Para guardar y cargar el modelo y el escalador.
- **Matplotlib**: Para mostrar imágenes en la interfaz de usuario.

### Instalación de dependencias:

1. Clona este repositorio o descarga los archivos del proyecto.
2. Navega al directorio donde se encuentra el archivo `requirements.txt` y ejecuta el siguiente comando para instalar las dependencias necesarias:

   ```bash
   pip install -r requirements.txt
