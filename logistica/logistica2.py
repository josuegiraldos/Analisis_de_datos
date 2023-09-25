#!/usr/bin/env python
# coding: utf-8

# Ejercicio de regresión logística 2 - Josué Gabriel Giraldo Suárez

# In[156]:


import seaborn as sns
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression


# In[157]:


df = pd.read_csv('E:/data_analisis/logistica/framingham.csv')
df.head(10)


# In[158]:


df.isnull().sum()


# In[159]:


df['education'].fillna(df['education'].mean(), inplace=True)
df['cigsPerDay'].fillna(df['cigsPerDay'].mean(), inplace=True)
df['BPMeds'].fillna(df['BPMeds'].mean(), inplace=True)
df['totChol'].fillna(df['totChol'].mean(), inplace=True)
df['BMI'].fillna(df['BMI'].mean(), inplace=True)
df['heartRate'].fillna(df['heartRate'].mean(), inplace=True)
df['glucose'].fillna(df['glucose'].mean(), inplace=True)
df.isnull().sum()


# In[160]:


df.duplicated().sum()


# In[161]:


df.describe(include='all')


# In[162]:


X = data.drop('TenYearCHD', axis=1)
y = data['TenYearCHD']


# In[165]:


scaler = StandardScaler()
X = scaler.fit_transform(X)


# In[166]:


modelo = LogisticRegression()


# In[167]:


scores = cross_val_score(modelo, X, y, cv = 5)


# In[168]:


print("Precisión en cada iteración de validación cruzada:", scores)
print("Precisión media:", scores.mean())


# In[169]:


plt.figure(figsize=(8, 4))
plt.bar(range(1, 6), scores, tick_label=['Fold 1', 'Fold 2', 'Fold 3', 'Fold 4', 'Fold 5'])
plt.xlabel("Folds de Validación Cruzada")
plt.ylabel("Precisión")
plt.title("Precisión en Validación Cruzada (K-Fold)")
plt.ylim(0, 1)
plt.show()


# In[ ]:




