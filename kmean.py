import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
from matplotlib import pyplot as plt
from sklearn.preprocessing import StandardScaler

import seaborn as sns
import datetime as dt
from sklearn.cluster import KMeans
import openpyxl

df = pd.read_excel(r'kel08_dataset.xlsx', engine='openpyxl')
df

df_null = round(100*(df.isnull().sum())/len(df), 2)
df_null

plt.scatter(df['nama_provinsi'], df['penduduk_miskin'])
plt.xlabel('nama_provinsi')
plt.ylabel('penduduk_miskin')

df.info()

df_null = round(100*(df.isnull().sum())/len(df), 2)
df_null

df = df.dropna()
df.shape

plt.scatter(df['nama_provinsi'], df['penduduk_miskin'])
plt.xlim(0, 5)
plt.ylim(0, 40)
plt.show()

x = df.iloc[0:, 4:5]
x

kmeans = KMeans(4)
kmeans.fit(x)


identified_clusters = kmeans.fit_predict(x)
identified_clusters

data_with_clusters = df.copy()
data_with_clusters['Clusters'] = identified_clusters
plt.scatter(data_with_clusters['penduduk_miskin'], data_with_clusters['nama_provinsi'],
            c=data_with_clusters['Clusters'], cmap='rainbow')

wcss = []
for i in range(1, 10):
    kmeans = KMeans(i)
kmeans.fit(x)
wcss_iter = kmeans.inertia_
wcss.append(wcss_iter)

number_clusters = range(10, 11)
plt.plot(number_clusters, wcss)
plt.title('The Elbow title')
plt.xlabel('Number of clusters')
plt.ylabel('WCSS')
