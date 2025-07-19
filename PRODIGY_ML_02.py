import numpy
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import seaborn as sns

df = pd.read_csv('Mall_Customers.csv')

print('DATAFRAME NULL INFO:-')
print(df.isnull().sum())
df = df.dropna()

print('DATAFRAME:-')
print(df)
print('DATAFRAME INFO:-')
print(df.info())
print('DATAFRAME DESCRIPTION:-')
print(df.describe())

plt.title('Gender Distribution')
sns.countplot(data=df, x='Gender')
plt.show()
plt.title('Age v/s Score')
sns.scatterplot(data=df, x='Age', y='Score')
plt.show()

x = df[['Age','Score']]
sc = StandardScaler()
x_scaled = sc.fit_transform(x)

cl = []
for i in range(1,11):
    km = KMeans(n_clusters=i, init='k-means++', random_state=42)
    km.fit(x_scaled)
    cl.append(km.inertia_)
print('Cluster Inertia:',cl)
print('Cluster Centres:-',km.cluster_centers_)

plt.title('Elbow Curve Method - Finding Optional Cluster Count')
plt.plot(range(1,11),cl,marker='o')
plt.xlabel('Number of Cluster:-')
plt.ylabel('Cluster Inertia')
plt.grid(True)
plt.show()

km = KMeans(n_clusters=3, init='k-means++', random_state=42)
x_kmeans = km.fit_predict(x_scaled)

plt.title('Customer Segments:-')
plt.figure(figsize=(10,6))
plt.scatter(x_scaled[:,0],x_scaled[:,1],c=x_kmeans, cmap= 'Set1', s=100)
plt.xlabel('Age')
plt.ylabel('Score')
plt.grid(True)
plt.show()