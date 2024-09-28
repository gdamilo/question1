import numpy as np
import matplotlib.pyplot as plt

# Define the hypothesis (linear regression model)
def predict(X, theta):
    return X.dot(theta)

# Define cost function (mean squared error)
def compute_cost(X, y, theta):
    m = len(y)
    predictions = predict(X, theta)
    cost = (1 / (2 * m)) * np.sum((predictions - y) ** 2)
    return cost

# Gradient Descent algorithm
def gradient_descent(X, y, theta, learning_rate, iterations):
    m = len(y)
    cost_history = []
    theta_history = []
    
    for i in range(iterations):
        predictions = predict(X, theta)
        theta = theta - (learning_rate / m) * X.T.dot(predictions - y)
        cost = compute_cost(X, y, theta)
        cost_history.append(cost)
        theta_history.append(theta)
    
    return theta, cost_history

# Input data for problem 1a
X_train = np.array([[1500, 3, 2, 1, 1], 
                    [1600, 4, 3, 2, 2], 
                    [1700, 3, 1, 1, 1], 
                    [1800, 2, 1, 1, 0], 
                    [1900, 4, 2, 2, 1], 
                    [2000, 3, 2, 2, 2], 
                    [2100, 5, 3, 2, 2], 
                    [2200, 4, 3, 2, 3]])
y_train = np.array([300000, 350000, 280000, 250000, 400000, 420000, 450000, 480000])

# Adding bias term (intercept)
X_train = np.hstack([np.ones((X_train.shape[0], 1)), X_train])

# Initialize parameters (theta values)
theta = np.zeros(X_train.shape[1])

# Set hyperparameters
learning_rate = 0.05
iterations = 1000

# Run gradient descent
theta, cost_history = gradient_descent(X_train, y_train, theta, learning_rate, iterations)

# Plot cost over iterations
plt.plot(range(iterations), cost_history, label="Training Loss")
plt.xlabel('Iterations')
plt.ylabel('Cost (MSE)')
plt.title('Training Loss over Iterations')
plt.legend()
plt.show()

# Print final theta values
print(f"Optimal parameters (theta): {theta}")
# Input data for problem 1b (with more features)
X_train_extended = np.array([[1500, 3, 2, 1, 1, 0, 0, 0, 1, 1, 0], 
                             [1600, 4, 3, 2, 1, 1, 1, 0, 1, 2, 1], 
                             [1700, 3, 1, 1, 0, 0, 0, 1, 0, 1, 0], 
                             [1800, 2, 1, 1, 1, 0, 0, 0, 1, 0, 0], 
                             [1900, 4, 2, 2, 0, 0, 1, 1, 1, 1, 1], 
                             [2000, 3, 2, 2, 1, 1, 1, 1, 1, 2, 1], 
                             [2100, 5, 3, 2, 1, 1, 0, 0, 1, 2, 0], 
                             [2200, 4, 3, 2, 1, 0, 1, 1, 1, 3, 1]])
y_train_extended = np.array([300000, 350000, 280000, 250000, 400000, 420000, 450000, 480000])

# Adding bias term (intercept)
X_train_extended = np.hstack([np.ones((X_train_extended.shape[0], 1)), X_train_extended])

# Initialize parameters (theta values)
theta_extended = np.zeros(X_train_extended.shape[1])

# Run gradient descent
theta_extended, cost_history_extended = gradient_descent(X_train_extended, y_train_extended, theta_extended, learning_rate, iterations)

# Plot cost for both models in a single graph
plt.plot(range(iterations), cost_history, label="Model 1a Loss")
plt.plot(range(iterations), cost_history_extended, label="Model 1b Loss")
plt.xlabel('Iterations')
plt.ylabel('Cost (MSE)')
plt.title('Training Loss Comparison (Model 1a vs 1b)')
plt.legend()
plt.show()

# Print final theta values for both models
print(f"Optimal parameters (Model 1a): {theta}")
print(f"Optimal parameters (Model 1b): {theta_extended}")
from sklearn.preprocessing import MinMaxScaler, StandardScaler
import numpy as np
import matplotlib.pyplot as plt

# Define the hypothesis (linear regression model)
def predict(X, theta):
    return X.dot(theta)

# Define cost function (mean squared error)
def compute_cost(X, y, theta):
    m = len(y)
    predictions = predict(X, theta)
    cost = (1 / (2 * m)) * np.sum((predictions - y) ** 2)
    return cost

# Gradient Descent algorithm
def gradient_descent(X, y, theta, learning_rate, iterations):
    m = len(y)
    cost_history = []
    for i in range(iterations):
        predictions = predict(X, theta)
        theta = theta - (learning_rate / m) * X.T.dot(predictions - y)
        cost = compute_cost(X, y, theta)
        cost_history.append(cost)
    return theta, cost_history

# Input data for problem 2a
X_train = np.array([[1500, 3, 2, 1, 1], 
                    [1600, 4, 3, 2, 2], 
                    [1700, 3, 1, 1, 1], 
                    [1800, 2, 1, 1, 0], 
                    [1900, 4, 2, 2, 1], 
                    [2000, 3, 2, 2, 2], 
                    [2100, 5, 3, 2, 2], 
                    [2200, 4, 3, 2, 3]])

y_train = np.array([300000, 350000, 280000, 250000, 400000, 420000, 450000, 480000])

# Normalization using MinMaxScaler
scaler_norm = MinMaxScaler()
X_train_norm = scaler_norm.fit_transform(X_train)

# Standardization using StandardScaler
scaler_std = StandardScaler()
X_train_std = scaler_std.fit_transform(X_train)

# Adding bias term (intercept) to both normalized and standardized data
X_train_norm = np.hstack([np.ones((X_train_norm.shape[0], 1)), X_train_norm])
X_train_std = np.hstack([np.ones((X_train_std.shape[0], 1)), X_train_std])

# Initialize parameters (theta values)
theta = np.zeros(X_train.shape[1] + 1)  # +1 for intercept

# Set hyperparameters
learning_rate = 0.05
iterations = 1000

# Training using normalized data
theta_norm, cost_history_norm = gradient_descent(X_train_norm, y_train, theta, learning_rate, iterations)

# Training using standardized data
theta_std, cost_history_std = gradient_descent(X_train_std, y_train, theta, learning_rate, iterations)

# Baseline (no scaling)
X_train_baseline = np.hstack([np.ones((X_train.shape[0], 1)), X_train])
theta_baseline, cost_history_baseline = gradient_descent(X_train_baseline, y_train, theta, learning_rate, iterations)

# Plot cost for both normalization and standardization in a single graph
plt.plot(range(iterations), cost_history_norm, label="Normalized Loss")
plt.plot(range(iterations), cost_history_std, label="Standardized Loss")
plt.plot(range(iterations), cost_history_baseline, label="Baseline Loss")
plt.xlabel('Iterations')
plt.ylabel('Cost (MSE)')
plt.title('Training Loss Comparison (Normalization vs Standardization vs Baseline)')
plt.legend()
plt.show()

# Print final theta values for all approaches
print(f"Optimal parameters (Normalized): {theta_norm}")
print(f"Optimal parameters (Standardized): {theta_std}")
print(f"Optimal parameters (Baseline): {theta_baseline}")
# Input data for problem 2b (with more features)
X_train_extended = np.array([[1500, 3, 2, 1, 1, 0, 0, 0, 1, 1, 0], 
                             [1600, 4, 3, 2, 1, 1, 1, 0, 1, 2, 1], 
                             [1700, 3, 1, 1, 0, 0, 0, 1, 0, 1, 0], 
                             [1800, 2, 1, 1, 1, 0, 0, 0, 1, 0, 0], 
                             [1900, 4, 2, 2, 0, 0, 1, 1, 1, 1, 1], 
                             [2000, 3, 2, 2, 1, 1, 1, 1, 1, 2, 1], 
                             [2100, 5, 3, 2, 1, 1, 0, 0, 1, 2, 0], 
                             [2200, 4, 3, 2, 1, 0, 1, 1, 1, 3, 1]])

y_train_extended = np.array([300000, 350000, 280000, 250000, 400000, 420000, 450000, 480000])

# Normalization using MinMaxScaler
scaler_norm_ext = MinMaxScaler()
X_train_norm_ext = scaler_norm_ext.fit_transform(X_train_extended)

# Standardization using StandardScaler
scaler_std_ext = StandardScaler()
X_train_std_ext = scaler_std_ext.fit_transform(X_train_extended)

# Adding bias term (intercept) to both normalized and standardized data
X_train_norm_ext = np.hstack([np.ones((X_train_norm_ext.shape[0], 1)), X_train_norm_ext])
X_train_std_ext = np.hstack([np.ones((X_train_std_ext.shape[0], 1)), X_train_std_ext])

# Initialize parameters (theta values)
theta_ext = np.zeros(X_train_extended.shape[1] + 1)  # +1 for intercept

# Training using normalized data
theta_norm_ext, cost_history_norm_ext = gradient_descent(X_train_norm_ext, y_train_extended, theta_ext, learning_rate, iterations)

# Training using standardized data
theta_std_ext, cost_history_std_ext = gradient_descent(X_train_std_ext, y_train_extended, theta_ext, learning_rate, iterations)

# Baseline (no scaling)
X_train_baseline_ext = np.hstack([np.ones((X_train_extended.shape[0], 1)), X_train_extended])
theta_baseline_ext, cost_history_baseline_ext = gradient_descent(X_train_baseline_ext, y_train_extended, theta_ext, learning_rate, iterations)

# Plot cost for both normalization and standardization in a single graph
plt.plot(range(iterations), cost_history_norm_ext, label="Normalized Loss")
plt.plot(range(iterations), cost_history_std_ext, label="Standardized Loss")
plt.plot(range(iterations), cost_history_baseline_ext, label="Baseline Loss")
plt.xlabel('Iterations')
plt.ylabel('Cost (MSE)')
plt.title('Training Loss Comparison (Normalization vs Standardization vs Baseline) for Extended Features')
plt.legend()
plt.show()

# Print final theta values for all approaches
print(f"Optimal parameters (Normalized Extended): {theta_norm_ext}")
print(f"Optimal parameters (Standardized Extended): {theta_std_ext}")
print(f"Optimal parameters (Baseline Extended): {theta_baseline_ext}")
from sklearn.preprocessing import MinMaxScaler, StandardScaler
import numpy as np
import matplotlib.pyplot as plt

# Define the hypothesis (linear regression model)
def predict(X, theta):
    return X.dot(theta)

# Define the cost function with L2 regularization (Ridge regression)
def compute_cost_with_penalty(X, y, theta, lambda_):
    m = len(y)
    predictions = predict(X, theta)
    cost = (1 / (2 * m)) * np.sum((predictions - y) ** 2) + (lambda_ / (2 * m)) * np.sum(theta[1:] ** 2)
    return cost

# Gradient Descent algorithm with L2 regularization
def gradient_descent_with_penalty(X, y, theta, learning_rate, iterations, lambda_):
    m = len(y)
    cost_history = []
    for i in range(iterations):
        predictions = predict(X, theta)
        theta[0] = theta[0] - (learning_rate / m) * np.sum(predictions - y)  # No regularization on bias term
        theta[1:] = theta[1:] - (learning_rate / m) * (X[:, 1:].T.dot(predictions - y) + lambda_ * theta[1:])
        cost = compute_cost_with_penalty(X, y, theta, lambda_)
        cost_history.append(cost)
    return theta, cost_history

# Input data for problem 3a (same as in problem 2a)
X_train = np.array([[1500, 3, 2, 1, 1], 
                    [1600, 4, 3, 2, 2], 
                    [1700, 3, 1, 1, 1], 
                    [1800, 2, 1, 1, 0], 
                    [1900, 4, 2, 2, 1], 
                    [2000, 3, 2, 2, 2], 
                    [2100, 5, 3, 2, 2], 
                    [2200, 4, 3, 2, 3]])

y_train = np.array([300000, 350000, 280000, 250000, 400000, 420000, 450000, 480000])

# Use the best scaling approach from Problem 2a (Let's assume standardization was the best)
scaler_std = StandardScaler()
X_train_std = scaler_std.fit_transform(X_train)

# Adding bias term (intercept)
X_train_std = np.hstack([np.ones((X_train_std.shape[0], 1)), X_train_std])

# Initialize parameters (theta values)
theta = np.zeros(X_train_std.shape[1])

# Set hyperparameters
learning_rate = 0.05
iterations = 1000
lambda_ = 0.1  # Regularization strength

# Run gradient descent with L2 regularization
theta, cost_history = gradient_descent_with_penalty(X_train_std, y_train, theta, learning_rate, iterations, lambda_)

# Plot training loss with regularization
plt.plot(range(iterations), cost_history, label="Training Loss with L2 Regularization")
plt.xlabel('Iterations')
plt.ylabel('Cost (MSE)')
plt.title('Training Loss with L2 Regularization (Standardization)')
plt.legend()
plt.show()

# Print final theta values
print(f"Optimal parameters with L2 Regularization: {theta}")
# Input data for problem 3b (with more features)
X_train_extended = np.array([[1500, 3, 2, 1, 1, 0, 0, 0, 1, 1, 0], 
                             [1600, 4, 3, 2, 1, 1, 1, 0, 1, 2, 1], 
                             [1700, 3, 1, 1, 0, 0, 0, 1, 0, 1, 0], 
                             [1800, 2, 1, 1, 1, 0, 0, 0, 1, 0, 0], 
                             [1900, 4, 2, 2, 0, 0, 1, 1, 1, 1, 1], 
                             [2000, 3, 2, 2, 1, 1, 1, 1, 1, 2, 1], 
                             [2100, 5, 3, 2, 1, 1, 0, 0, 1, 2, 0], 
                             [2200, 4, 3, 2, 1, 0, 1, 1, 1, 3, 1]])

y_train_extended = np.array([300000, 350000, 280000, 250000, 400000, 420000, 450000, 480000])

# Use the best scaling approach from Problem 2b (Let's assume standardization was the best)
scaler_std_ext = StandardScaler()
X_train_std_ext = scaler_std_ext.fit_transform(X_train_extended)

# Adding bias term (intercept)
X_train_std_ext = np.hstack([np.ones((X_train_std_ext.shape[0], 1)), X_train_std_ext])

# Initialize parameters (theta values)
theta_ext = np.zeros(X_train_std_ext.shape[1])

# Run gradient descent with L2 regularization
theta_ext, cost_history_ext = gradient_descent_with_penalty(X_train_std_ext, y_train_extended, theta_ext, learning_rate, iterations, lambda_)

# Plot training loss with regularization for extended features
plt.plot(range(iterations), cost_history_ext, label="Training Loss with L2 Regularization (Extended Features)")
plt.xlabel('Iterations')
plt.ylabel('Cost (MSE)')
plt.title('Training Loss with L2 Regularization (Standardization, Extended Features)')
plt.legend()
plt.show()

# Print final theta values for the extended feature set
print(f"Optimal parameters with L2 Regularization (Extended): {theta_ext}")

