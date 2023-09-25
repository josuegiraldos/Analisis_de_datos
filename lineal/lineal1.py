#!/usr/bin/env python
# coding: utf-8

# Ejercicio de regresión lineal - Josué Gabriel Giraldo Suárez

# In[64]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression


# In[65]:


lineal = pd.read_csv("E:\data_analisis\lineal\data.csv")


# In[66]:


X = lineal[['metro']]


# In[67]:


y = lineal['precio']


# In[68]:


modelo = LinearRegression()


# In[69]:


modelo.fit(X, y)


# In[70]:


nuevo_valor_de_metro_df = pd.DataFrame({'metro':[30]})


# In[71]:


prediccion = modelo.predict(nuevo_valor_de_metro_df)


# In[72]:


print(f"Para {nuevo_valor_de_metro_df['metro'][0]} metros cuadrados, la predicción de precio es: {prediccion[0]:.2f} dólares")


# In[73]:


plt.scatter(X, y, label='Datos reales')
plt.plot(X, modelo.predict(X), color='red', linewidth=2, label='Predicción')
plt.xlabel('Metro Cuadrado')
plt.ylabel('Precio')
plt.legend()
plt.show()

