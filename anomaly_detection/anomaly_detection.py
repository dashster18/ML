import numpy as np
import scipy.linalg as la
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

"""
Multivariate Guassian based anomaly detection: Attempts to create
a probability model for your data and returns the probability of
an example.

You are free to decide what the threshold for making decisions is
out side of this class.
"""
class AnomalyDetection:
    def __init__(self):
        self.n_examples = None
        self.n_features = None
        self.mean = None
        self.covariance_matrix = None

        # Cache these values later for fast prediction
        self.covariance_matrix_inverse = None
        self.denominator = None


    """
    Let's the algorithm learn the probability model from 
    the data
    
    Parameters
    ----------
    data : array-like, shape=(n_samples, n_features)
        The data to train the algorithm on
    """
    def train(self, data):
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

    """
    Predicts the probability of a single example using
    the multivariate guassian pdf.

    Parameters
    ----------
    example : array-like, shape=(n_features,)
        One example to predict on.
    """
    def predict_example(self, example):
        D = example - self.mean
        numerator = np.exp(-0.5 * (D.T @ self.covariance_matrix_inverse @ D))
        probability = numerator / self.denominator
        return probability
        

    """
    Predict the probablity of each example in 'data'

    Parameters
    ----------
    data : array-like, shape=(n_samples, n_features)
        New data to predict on.
    """
    def predict(self, data):
        if data.ndim == 1:
            return np.array(self.predict_example(data))
        elif data.ndim == 2:
            return np.array([self.predict_example(example) for example in data])
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
ad = AnomalyDetection()
ad.train(data.values)
probs = ad.predict(data.values)
epsilon = 8.99e-05

# Make predictions 
data['anomaly'] = probs < epsilon

# Show the before and after scatter plot
plt.scatter(data[0], data[1])
plt.figure()
plt.scatter(data[0], data[1], c=data['anomaly'].apply(lambda x: 'r' if x else 'b'))
plt.show()
