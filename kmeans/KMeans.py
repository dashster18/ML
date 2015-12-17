import numpy as np

class KMeans:
    """
    K-means clustering: Attemps to group the data points in k different
    clusters based on distance from other points.

    Parameters
    ----------
    n_clusters : int
    The number of clusters to assign instances to as well as the 
    number of centroids to create.

    n_initializations : int, optional, default: 10
    The number of times to try different initializations of KMeans.
    The final result will be the one that has the best results in
    terms of minimizing the k-means cost function.

    n_iters : int, optional, default : 100
    Number of times to run the inner loop of k-means. The higher
    the value, the closer to convergence the algorithm will reach.
    """
    def __init__(self, n_clusters, n_initalizations=10, n_iters=100):
        self.n_clusters = n_clusters
        self.n_initalizations = n_initalizations
        self.n_iters = n_iters
        self.best_cost = None
        self.best_clusters = None


    def _dist(self, data, clusters):
        """
        Computes the distance between each cluster for all data points
    
        Parameters
        ----------
        data : array-like, shape=(n_samples, n_features)
        The data to train the algorithm on

        clusters : array_like, shape=(n_clusters, n_features)
        Each row i is a vector representation of cluster i

        Returns
        -------
        dist : array-like, shape=(n_clusters, n_samples)
        dist[i,j] is the distnace between cluster i and sample j
        """
        dist = []
        for c in range(clusters.shape[0]):
            diff = data - clusters[c]
            dist.append(np.sum(diff ** 2, axis=1))
        return np.array(dist)
        

    def _cost(self, data, clusters):
        """
        Computes the cost in terms of the distance between each sample
        and the closest cluster to it.
    
        Parameters
        ----------
        data : array-like, shape=(n_samples, n_features)
        The data to train the algorithm on

        clusters : array_like, shape=(n_clusters, n_features)
        Each row i is a vector representation of cluster i
        """
        dist = self._dist(data, clusters)
        return dist.min(axis=0).mean()
    

    def _kmeans(self, data, clusters):
        """
        Runs the k-means algorithm on the data for n_iters 

        Parameters
        ----------
        data : array-like, shape=(n_samples, n_features)
        The data to train the algorithm on

        clusters : array_like, shape=(n_clusters, n_features)
        Each row i is a vector representation of cluster i
    
        Returns
        -------
        clusters : array-like, shape=(n_clusters, n_features)
        The best cluster arrangement found 
        """
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
        """
        Train the algorithm on compute the k different centroids
    
        Parameters
        ----------
        data : array-like, shape=(n_samples, n_features)
        The data to train the algorithm on
        """
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
        """
        Predict the closest cluster that each example in 'data'
        belongs to.

        Parameters
        ----------
        data : array-like, shape=(n_samples, n_features)
        New data to predict on.
        """
        if self.best_clusters is None:
            raise Error("KMeans hasn't been trained yet!")

        assigned = self._dist(data, self.best_clusters).argmin(axis=0)
        return assigned
        
