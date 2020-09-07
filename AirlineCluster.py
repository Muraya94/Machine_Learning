# CLUSTERING(UNSUPERVISED LEARNING)
import pandas
from sklearn.cluster import KMeans

dataframe = pandas.read_csv('https://modcom.co.ke/bigdata/datasets/AirlinesCluster.csv')
pandas.set_option('display.max_columns', 7)
print(dataframe)
# Obtain values from the dataframe
array = dataframe.values
print(array)

# Pick the algorithm
model = KMeans(n_clusters=8, random_state=5)
# Fit the training data
model.fit(array)
print('Clustering done.')
# Get the centroids
centroids = model.cluster_centers_
clusters = pandas.DataFrame(centroids)
print(clusters)

# Take each cluster at a time
dataframe['label'] = model.labels_
subset = dataframe[dataframe['label'] == 8]
print(subset)

