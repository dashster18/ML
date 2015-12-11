import numpy as np

class KMeans:
    def __init__(self, n_clusters, n_initalizations, n_iters=100):
        self.n_clusters = n_clusters
        self.n_initalizations = n_initalizations
        self.n_iters = n_iters
        self.best_cost = None
        self.best_clusters = None

    def _dist(self, data, clusters):
        dist = []
        for c in range(clusters.shape[0]):
            diff = data - clusters[c]
            dist.append(np.sum(diff ** 2, axis=1))
        return np.array(dist)
        
    def _cost(self, data, clusters):
        dist = self._dist(data, clusters)
        return dist.min(axis=0).mean()
    
    def _kmeans(self, data, clusters):
        for _ in range(self.n_iters):

            # Assign each data point to the cluster closest to it
            dist = self._dist(data, clusters)
            assigned = dist.argmin(axis=0)
        
            # For each cluster, average the examples in it
            # and assign that as the cluster
            for c in range(clusters.shape[0]):
                # Get the current cluster...
                clusters[c] = data[assigned == c, :].mean(axis=0)
                
        return clusters

    def train(self, data):
        self.n_features = data.shape[1]

        # Run k-means multiple times and save the cost of each run
        self.saved_runs = np.zeros((self.n_initalizations, self.n_clusters, self.n_features))
        self.saved_costs = np.zeros(self.n_initalizations)
        for i in range(self.n_initalizations):
            # Initialize clusters to random training examples
            indexes = np.random.choice(data.shape[0], self.n_clusters, replace=False)
            
            # Run k-means
            self.saved_runs[i] = self._kmeans(data, data[indexes, :])

            # Compute and save cost 
            self.saved_costs[i] = self._cost(data, self.saved_runs[i])

        # Pick solution with the lowest cost
        self.best_cost = self.saved_costs.min()
        self.best_clusters = self.saved_runs[self.saved_costs.argmin(), :, :]

        # Free up memory
        self.saved_runs = None
        self.saved_costs = None
        

    def predict(self, data):
        if self.best_clusters is None:
            raise Error("KMeans hasn't been trained yet!")

        assigned = self._dist(data, self.best_clusters).argmin(axis=0)
        return assigned
        
