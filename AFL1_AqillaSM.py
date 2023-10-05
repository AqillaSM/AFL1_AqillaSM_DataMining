#!/usr/bin/env python
# coding: utf-8

# #Week 4: Supervised Learning - Classification - Iris
# -------------------------------------
# Lab exercise kali ini menggunakan dataset iris:
# https://raw.githubusercontent.com/catharinamega/Data-Mining-ISB-2020/main/Week%204/iris.csv
# 
# Lakukan klasifikasi pada dataset tersebut dengan menggunakan 3 cara: Logistic Regression, Naive Bayes, dan K-Nearest Neighbor (dengan k=5)

# ## Import library

# In[70]:


import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn import metrics
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression


# ## Load Dataset

# In[71]:


# Read CSV
df = pd.read_csv('https://raw.githubusercontent.com/catharinamega/Data-Mining-ISB-2020/main/Week%204/iris.csv')
df


# ## Data Preprocessing

# Periksa apakah ada baris yang duplikat
# 
# 

# In[72]:


print(df.duplicated().any())
print(df[df.duplicated()])


# Periksa apakah ada missing values

# In[73]:


print(df.isna().any())


# # Periksa outlier dengan boxplot untuk setiap kolom feature

# In[91]:


fig, axes = plt.subplots(2, 2, figsize=(12, 8))

# Boxplot 1 - Sepal Length
sns.boxplot(ax=axes[0, 0], x='species', y='sepal_length', data=df)
axes[0, 0].set_xlabel('Species')
axes[0, 0].set_ylabel('Sepal Length')
axes[0, 0].set_title('Boxplot Sepal Length Berdasarkan Species')

# Boxplot 2 - Petal Length
sns.boxplot(ax=axes[0, 1], x='species', y='petal_length', data=df)
axes[0, 1].set_xlabel('Species')
axes[0, 1].set_ylabel('Petal Length')
axes[0, 1].set_title('Boxplot Petal Length Berdasarkan Species')

# Boxplot 3 - Sepal Width
sns.boxplot(ax=axes[1, 0], x='species', y='sepal_width', data=df)
axes[1, 0].set_xlabel('Species')
axes[1, 0].set_ylabel('Sepal Width')
axes[1, 0].set_title('Boxplot Sepal Width Berdasarkan Species')

# Boxplot 4 - Petal Width
sns.boxplot(ax=axes[1, 1], x='species', y='petal_width', data=df)
axes[1, 1].set_xlabel('Species')
axes[1, 1].set_ylabel('Petal Width')
axes[1, 1].set_title('Boxplot Petal Width Berdasarkan Species')

plt.tight_layout()
plt.show()


# Pisahkan dataset menjadi variabel independen dan variabel dependen

# In[27]:


x = df.drop(['species'], axis = 1)
y = df.species


# Pisahkan dataset train dan test dataset, dengan ukuran dataset test 0.1

# In[29]:


from sklearn.model_selection import train_test_split
X_train, X_test, Y_train, Y_test = train_test_split(x, y, test_size = 0.1, random_state = 0)


# Lakukan fitur scaling pada variabel X_train dan X_test. 

# In[34]:


from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()

# Fit dan transformasi (scaling) pada X_train
X_train_scaled = scaler.fit_transform(X_train)

# Transformasi (scaling) pada X_test
X_test_scaled = scaler.transform(X_test)


# #Logistic Regression

# Bangun model dan ukur accuracy nya

# In[44]:


logreg = LogisticRegression()
logreg.fit(X_train_scaled, Y_train)


# Uji model dengan dataset test

# In[48]:


y_pred_logreg = logreg.predict(X_test_scaled)
print(y_pred_logreg)


# In[51]:


cm = confusion_matrix(y_test, y_pred_logreg)
print("Confusion Matrix\n", cm)

accuracy_logreg = accuracy_score(y_test, y_pred_logreg)
accuracy_logreg = '{:,.4f}'.format(accuracy_logreg)
print(accuracy_logreg)


# # Naive Bayes
# 
# Buat model persamaan berdasarkan data training yang sudah dibuat pada tahap preprocess

# In[59]:


Gaussian  = GaussianNB()
Gaussian.fit(X_train_scaled, y_train)


# Uji hasil model dengan data test

# In[64]:


y_pred_gaussian = Gaussian.predict(X_test_scaled)
print(y_pred_gaussian)


# Ukur akurasi dari model persamaan Naive Bayes Classifier

# In[65]:


cm = confusion_matrix(y_test, y_pred_gaussian)
print("Confusion Matrix\n", cm)

accuracy_gaussian = accuracy_score(y_test, y_pred_gaussian)
accuracy_gaussian = '{:,.4f}'.format(accuracy_gaussian)
print(accuracy_gaussian)


# # K-Nearest Neighbour (K-NN) Classifier

# Buatlah model KNN berdasarkan data training yang sudah dibuat di tahap preprocess, gunakan metric pengukuran jarak 'euclidean'

# In[81]:


KNN = KNeighborsClassifier(n_neighbors=5, metric='euclidean')
# Latih model K-NN menggunakan data pelatihan yang telah di-scaling
KNN.fit(X_train_scaled, Y_train)


# Uji model dengan data test

# In[82]:


y_pred_knn = Gaussian.predict(X_test_scaled)
print(y_pred_knn)


# Ukur akurasi dari model K-Nearest Neighbor

# In[83]:


cm = confusion_matrix(y_test, y_pred_knn)
print("Confusion Matrix\n", cm)

accuracy_knn = accuracy_score(y_test, y_pred_knn)
accuracy_knn = '{:,.4f}'.format(accuracy_knn)
print(accuracy_knn)


# In[84]:


print("Accuracy Score Logistic Regression :", accuracy_logreg)
print("Accuracy Score Naives Bayes :", accuracy_gaussian)
print("Accuracy Score K-NN Classifier :", accuracy_knn)


# #Kesimpulan
# Dari 3 model klasifikasi di atas Logistic Regression model klasifikasi dengan akurasi tertinggi adalah memiliki accuracy score tertinggi, yaitu 1
