#!/usr/bin/env python
# coding: utf-8

# Ejercicio de regresión logística 1 - Josué Gabriel Giraldo Suárez

# In[19]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report
sns.set()


# In[20]:


df = pd.read_csv('E:/data_analisis/logistica/framingham.csv')
df.head(10)


# In[21]:


df.isnull().sum()


# In[22]:


df['education'].fillna(df['education'].mean(), inplace=True)
df['cigsPerDay'].fillna(df['cigsPerDay'].mean(), inplace=True)
df['BPMeds'].fillna(df['BPMeds'].mean(), inplace=True)
df['totChol'].fillna(df['totChol'].mean(), inplace=True)
df['BMI'].fillna(df['BMI'].mean(), inplace=True)
df['heartRate'].fillna(df['heartRate'].mean(), inplace=True)
df['glucose'].fillna(df['glucose'].mean(), inplace=True)
df.isnull().sum()


# In[23]:


X = df.drop('TenYearCHD', axis=1)
y = df['TenYearCHD']


# In[24]:


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


# In[25]:


scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)


# In[26]:


modelo = LogisticRegression()


# In[27]:


modelo.fit(X_train, y_train)


# In[28]:


predicciones = modelo.predict(X_test)


# In[29]:


precision = accuracy_score(y_test, predicciones)


# In[30]:


print(f"Precisión del modelo: {precision:.2f}")
print(classification_report(y_test, predicciones))


# In[ ]:




