import numpy as np

"""
    Simple linear regression model implementation.
"""


class LogisticRegression:
    def __init__(self, learning_rate=0.01, num_iterations=1000):
        self.learning_rate = learning_rate
        self.num_iterations = num_iterations
        self.weights = None
        self.bias = None

    def sigmoid(self, z):
        return 1 / (1 + np.exp(-z))

    def fit(self, X, y):
        n_samples, n_features = X.shape
        # Initialize weights and bias
        self.weights = np.zeros(n_features)
        self.bias = 0

        # Gradient descent for num_iterations
        for _ in range(self.num_iterations):
            linear_model = np.dot(X, self.weights) + self.bias
            predictions = self.sigmoid(linear_model)

            # Compute gradients
            dw = (1/n_samples) * np.dot(X.T, (predictions - y))
            db = (1/n_samples) * np.sum(predictions - y)

            # Update weights and bias
            self.weights -= self.learning_rate * dw
            self.bias -= self.learning_rate * db

    def partial_fit(self, X, y):
        n_samples, n_features = X.shape
        if self.weights is None:
            # Initialize weights and bias on the first call to partial_fit
            self.weights = np.zeros(n_features)
            self.bias = 0

        linear_model = np.dot(X, self.weights) + self.bias
        predictions = self.sigmoid(linear_model)

        # Compute gradients
        dw = (1/n_samples) * np.dot(X.T, (predictions - y))
        db = (1/n_samples) * np.sum(predictions - y)

        # Update weights and bias incrementally
        self.weights -= self.learning_rate * dw
        self.bias -= self.learning_rate * db

    def predict(self, X):
        linear_model = np.dot(X, self.weights) + self.bias
        predictions = self.sigmoid(linear_model)
        # Convert predictions to binary (0 or 1) using a threshold of 0.5
        binary_predictions = [1 if x > 0.5 else 0 for x in predictions]
        return binary_predictions
