#Cargamos librerías principales
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

#Cargamos el modelo
import pickle
filename = 'modelo.pkl'
model_Tree,model_Knn, model_NN, min_max_scaler, variables = pickle.load(open(filename, 'rb'))


#Se crea interfaz gráfica con streamlit para captura de los datos
import streamlit as st

st.title('Predicción de inversión en una tienda de videojuegos con un árbol de decisión')
Edad = st.slider('Edad', min_value=14, max_value=52, value=20, step=1)
videojuego = st.selectbox('Videojuego', ["'Mass Effect'","'Battlefield'", "'Fifa'","'KOA: Reckoning'","'Crysis'","'Sim City'","'Dead Space'","'F1'"])
Plataforma = st.selectbox('Plataforma', ["'Play Station'", "'Xbox'","PC","Otros"])
Sexo = st.selectbox('Sexo', ['Hombre', 'Mujer'])
Consumidor_habitual = st.selectbox('Consumidor_habitual', ['True', 'False'])

#Dataframe
datos = [[Edad, videojuego,Plataforma,Sexo,Consumidor_habitual]]
data = pd.DataFrame(datos, columns=['Edad', 'videojuego','Plataforma','Sexo','Consumidor_habitual']) #Dataframe con los mismos nombres de variables

#Se realiza la preparación
data_preparada=data.copy()
data_preparada = pd.get_dummies(data_preparada, columns=['videojuego', 'Plataforma','Sexo', 'Consumidor_habitual'], drop_first=False, dtype=int)
#Se adicionan las columnas faltantes
data_preparada=data_preparada.reindex(columns=variables,fill_value=0)

#Hacemos la predicción con el Tree
Y_pred = model_Tree.predict(data_preparada)
data['Prediccion']=Y_pred

#Mostramos la predicción
data
