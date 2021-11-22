
# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Importing the dataset
dataset = pd.read_csv('musae_facebook_target.csv')
X = dataset.iloc[:, [2,3]].values

#Encoding categorical data in X
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, LabelEncoder
le_1 = LabelEncoder()
X[:,1] = le_1.fit_transform(X[:,1])
ct = ColumnTransformer(transformers=[('encoder', OneHotEncoder(),[1])], remainder="passthrough")
X = ct.fit_transform(X)
X = X[:,1:]

le_3 = LabelEncoder()
X[:,3] = le_3.fit_transform(X[:,3])
ct_1 = ColumnTransformer(transformers=[('encoder', OneHotEncoder(),[3])], remainder="passthrough")
X = ct_1.fit_transform(X)

# Using the elbow method to find the optimal number of clusters
from sklearn.cluster import KMeans
wcss = []
for i in range(1, 11):
    kmeans = KMeans(n_clusters = i, init = 'k-means++', random_state = 42)
    kmeans.fit(X)
    wcss.append(kmeans.inertia_)
plt.plot(range(1, 11), wcss)
plt.title('The Elbow Method')
plt.xlabel('Number of clusters')
plt.ylabel('WCSS')
plt.show()

# Fitting K-Means to the dataset after finding no. of clusters
kmeans = KMeans(n_clusters = 4, init = 'k-means++', random_state = 42)
y_kmeans = kmeans.fit_predict(X)

#Print the cluster numbers for each datapoint
print(y_kmeans)

"""
# Visualising the clusters
plt.scatter(X_test[y_kmeans == 0, 0], X_test[y_kmeans == 0, 1], s = 100, c = 'red', label = 'Company')
plt.scatter(X_test[y_kmeans == 1, 0], X_test[y_kmeans == 1, 1], s = 100, c = 'blue', label = 'Government')
plt.scatter(X_test[y_kmeans == 2, 0], X_test[y_kmeans == 2, 1], s = 100, c = 'green', label = 'TV Show')
plt.scatter(X_test[y_kmeans == 3, 0], X_test[y_kmeans == 3, 1], s = 100, c = 'cyan', label = 'Politician')
plt.scatter(kmeans.cluster_centers_[:, 0], kmeans.cluster_centers_[:, 1], s = 300, c = 'yellow', label = 'Centroids')
plt.show()
"""
