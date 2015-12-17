import numpy as np
import scipy.linalg as la
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

class AnomalyDetection:
    """
    Multivariate Guassian based anomaly detection: Attempts to create
    a probability model for your data and returns the probability of
    an example.

    You are free to decide what the threshold for making decisions is
    out side of this class.
    """
    def __init__(self, threshold):
        self.threshold = threshold
        
        self.n_examples = None
        self.n_features = None
        self.mean = None
        self.covariance_matrix = None

        # Cache these values later for fast prediction
        self.covariance_matrix_inverse = None
        self.denominator = None


    def train(self, data):
        """
        Let's the algorithm learn the probability model from 
        the data
    
        Parameters
        ----------
        data : array-like, shape=(n_samples, n_features)
            The data to train the algorithm on
        """
        if data.ndim not in [1, 2]:
            raise Exception("Expected only 1 or 2 dimensional ndarry")
        
        self.n_examples, self.n_features = data.shape

        # Fit the model of the parameters
        self.mean = data.mean(axis=0)
        D = data - self.mean
        self.covariance_matrix = (D.T @ D) / np.float(self.n_examples)

        # Cache these values for fast predictions
        self.covariance_matrix_inverse = la.pinv(self.covariance_matrix)
        self.denominator = (2 * np.pi)**(self.n_features/2.0) * np.sqrt(la.det(self.covariance_matrix))

    def predict_probability(self, example):
        """
        Predicts the probability of a single example using
        the multivariate guassian pdf.

        Parameters
        ----------
        example : array-like, shape=(n_features,)
            One example to predict on.

        Returns
        -------
        probability : double
            Returns the probability of this model occuring
        """
        D = example - self.mean
        numerator = np.exp(-0.5 * (D.T @ self.covariance_matrix_inverse @ D))
        probability = numerator / self.denominator
        return probability
        

    def predict_probabilities(self, data):
        """
        Predict the probablity of each example in 'data'

        Parameters
        ----------
        data : array-like, shape=(n_samples, n_features)
            New data to predict on.

        Returns
        -------
        probabilities : array-like, shape=(n_samples,)
            The probability of each example occuring
        """
        if data.ndim == 1:
            return np.array(self.predict_probability(data))
        elif data.ndim == 2:
            return np.array([self.predict_probability(example) for example in data])
        else:
            raise Exception("Expected only 1 or 2 dimensional ndarry")

    def predict(self, data):
        """
        Predict the probablity of each example in 'data'

        Parameters
        ----------
        data : array-like, shape=(n_samples, n_features)
            New data to predict on.

        Returns
        -------
        predictions : array-like, shape=(n_samples,)
            The prediction for each example.
            1 if the probability is lower than `threshold`, 0 otherwise
        """
        if data.ndim == 1:
            return (np.array(self.predict_probability(data)) < self.threshold).astype(float)
        elif data.ndim == 2:
            return (self.predict_probabilities(data) < self.threshold).astype(float)
        else:
            raise Exception("Expected only 1 or 2 dimensional ndarry")
        

################################################################
# Run the prediction on a basic example using data taken from
# Andrew Ng's Coursera class.
#
# The epsilon value for the threshold was taken from the
# assignment in that class, usually you can pick using cross
# validation.

data = pd.read_csv('data/ad1.csv', header=None)

# Train the model
epsilon = 8.99e-05
ad = AnomalyDetection(epsilon)
ad.train(data.values)

# Make predictions 
data['anomaly'] = ad.predict(data.values)

# Show the before and after scatter plot
plt.scatter(data[0], data[1])
plt.figure()
plt.scatter(data[0], data[1], c=data['anomaly'].apply(lambda x: 'r' if x else 'b'))
plt.show()
