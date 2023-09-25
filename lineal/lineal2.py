#!/usr/bin/env python
# coding: utf-8

# Ejercicio de regresión lineal 2 - Josué Gabriel Giraldo Suárez

# In[74]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression


# In[75]:


data = pd.read_csv("E:\data_analisis\lineal\data.csv")


# In[76]:


X = data[['metro']]


# In[77]:


y = data['precio']


# In[78]:


modelo = LinearRegression()


# In[79]:


modelo.fit(X, y)


# In[80]:


predicciones = modelo.predict(X)


# In[81]:


residuos = y - predicciones


# In[82]:


plt.scatter(X, residuos)
plt.axhline(y=0, color='red', linestyle='--')
plt.xlabel('Metro Cuadrado')
plt.ylabel('Residuos')
plt.title('Análisis de Residuos')
plt.show()


# In[ ]:




