# What is k means algorithm
'''
- K-Means Clustering is an Unsupervised Learning algorithm, which groups the unlabeled dataset into different clusters. 
- K defines the number of pre-defined clusters that need to be created in the process.
'''

# steps for k means algorithm
'''
Initialize k means with random values
--> For a given number of iterations:
    
    --> Iterate through items:
    
        --> Find the mean closest to the item by calculating 
        the euclidean distance of the item with each of the means
        
        --> Assign item to mean
        
        --> Update mean by shifting it to the average of the items in that cluster
'''


# dataset link
# https://gist.github.com/pravalliyaram/5c05f43d2351249927b8a3f3cc3e5ecf

'''
pandas (as pd): for data manipulation and analysis.
numpy (as np): for numerical operations.
matplotlib.pyplot (as plt): for plotting graphs and visualizations.
'''
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Load the dataset
df = pd.read_csv('mall_customers.csv')

# Selecting relevant columns for clustering
'''
-- DataFrame df,
-- .iloc is used to select rows and columns by integer position,
-- [:, [3, 4]] specifies that we want to select all rows (: before the comma) and columns at positions 3 and 4 ([3, 4] after the comma).
'''
D = df.iloc[:, [3, 4]].values  # Selecting Annual Income and Spending Score


# Visualize data points before clustering
plt.figure(figsize=(8, 6))
plt.scatter(D[:, 0], D[:, 1], s=50, color='gray', alpha=0.5)
plt.title('Data Points Before Clustering')
plt.xlabel('Annual Income (k$)')
plt.ylabel('Spending Score (1-100)')
plt.grid(True)
plt.show()


# Implementing k-means algorithm
def k_means(D, k, max_iters=100):
    '''
    - Step 1: K means by random values which to
    the integer to select the most
    used for elements.
    - Step 2: each distances by each mean
    and so points the closest to the 
    mean to be the 
    which to the clustering
    point is which data points.
    - Step 3: be to be as 
    the average in the same cluster
    '''
    # Step 1: Initialize k means with random values
    np.random.seed(42)  # Setting seed for reproducibility
    mean = D[np.random.choice(len(D), k, replace=False)]  # Randomly choose k points from D as initial centroids
    
    for _ in range(max_iters):
        # Step 2: Assign each data point to the nearest mean
        clusters = []
        for data_point in D:
            distances = [np.linalg.norm(data_point - m) for m in mean]  # Calculate distance to each mean
            cluster = np.argmin(distances)  # Assign data point to closest cluster
            clusters.append(cluster)
        
        clusters = np.array(clusters)
        
        # Step 3: Update mean
        for i in range(k):
            mean[i] = np.mean(D[clusters == i], axis=0)  # Update mean of each cluster
        
    return clusters, mean

# Setting number of clusters
k = 3

# Applying the k-means algorithm
clusters, mean = k_means(D, k)

# Visualizing the clusters and centroids
plt.figure(figsize=(10, 6))
colors = ['red', 'blue', 'green', 'cyan', 'magenta']

for cluster_id in range(k):
    plt.scatter(D[clusters == cluster_id, 0], D[clusters == cluster_id, 1], s=100, c=colors[cluster_id], label=f'Cluster {cluster_id + 1}')


plt.scatter(mean[:, 0], mean[:, 1], s=300, c='yellow', label='Means')
plt.title('Clusters of customers (k-means)')
plt.xlabel('Annual Income (k$)')
plt.ylabel('Spending Score (1-100)')
plt.legend()
plt.grid(True)
plt.show()
