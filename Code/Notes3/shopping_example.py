# Import relevant packages
import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
import requests
from io import BytesIO
import os

# switch to turn off plots (Switch to True to turn on plots)
dpl = False

# Get the directory where this script is located for saving plots
script_dir = os.path.dirname(os.path.abspath(__file__))

# Load the data
dropbox_url = "https://www.dropbox.com/scl/fi/qppn8i6xphjpue1kf3n36/shopping.csv?rlkey=qyr2x9zbnz1je4lzkx4wyaq7r&dl=1"
response = requests.get(dropbox_url)
shop_data = pd.read_csv(BytesIO(response.content))
print(shop_data.head())

# Drop the last row which is all missing
shop_data = shop_data.iloc[:20, :]

# K-means with 3 clusters
X = shop_data.iloc[:, 1:7]
kmeans = KMeans(n_clusters=3, n_init=200, random_state=0).fit(X)

shop_data['cluster'] = kmeans.labels_

# Scatter plot with slight jitter
plt.figure(figsize=(9, 6))
plt.scatter(
    shop_data['V1'] + 0.01 * np.random.randn(len(shop_data)),
    shop_data['V2'] + 0.01 * np.random.randn(len(shop_data)),
    c=shop_data['cluster'],
    cmap='rainbow'
)
plt.title("Clustering in Shopping Attitudes")
plt.xlabel("Shopping is fun")
plt.ylabel("Shopping is bad for your budget")

# Plot cluster centers (only V1 and V2 dimensions)
plt.scatter(
    kmeans.cluster_centers_[:, 0],  # V1 column in X
    kmeans.cluster_centers_[:, 1],  # V2 column in X
    marker='x', s=100, c=range(3), cmap='rainbow'
)

if dpl:
    plt.show()
else:
    plt.savefig(os.path.join(script_dir, 'shopping_clusters.png'))

# Print cluster centers
centers_df = pd.DataFrame(kmeans.cluster_centers_, columns=X.columns)
print(centers_df)

# Looking at different numbers of clusters
Ks = range(2, 11)
inertias = []

for k in Ks:
    km = KMeans(n_clusters=k, n_init=200, random_state=0)
    km.fit(X)
    inertias.append(km.inertia_)  # sum of squared distances to nearest centroid

# Elbow plot
plt.figure(figsize=(8, 5))
plt.plot(list(Ks), inertias, marker='o')
plt.xticks(list(Ks))
plt.xlabel("Number of clusters (k)")
plt.ylabel("Inertia (sum of squared distances)")
plt.title("Elbow Plot for K-Means (k = 2..10)")
plt.grid(True)
if dpl:
    plt.show()
else:
    plt.savefig(os.path.join(script_dir, 'elbow_plot.png'))

# Print inertia values
for k, sse in zip(Ks, inertias):
    print(f"k={k}: inertia={sse:,.2f}")