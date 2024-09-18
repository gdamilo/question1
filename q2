import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Re-entering the dataset
data = {
    'X1': [0.000000, 0.040404, 0.080808, 0.121212, 0.161616, 0.202020, 0.242424, 0.282828, 0.323232, 0.363636],
    'X2': [3.440000, 0.134949, 0.829899, 1.524848, 2.219798, 2.914747, 3.609696, 4.304646, 4.999595, 5.694545],
    'X3': [0.440000, 0.888485, 1.336970, 1.785455, 2.233939, 2.682424, 3.130909, 3.579394, 4.027879, 4.476364],
    'Y': [4.387545, 2.679650, 2.968490, 3.254065, 3.536375, 3.815421, 4.091201, 4.363717, 4.632967, 4.898952]
}

# Convert to DataFrame
df = pd.DataFrame(data)

# Extract the explanatory variables (X1, X2, X3) and the dependent variable (Y)
X1 = df['X1'].values
X2 = df['X2'].values
X3 = df['X3'].values
Y = df['Y'].values

# Normalize X values for better gradient descent performance
X1 = (X1 - np.mean(X1)) / np.std(X1)
X2 = (X2 - np.mean(X2)) / np.std(X2)
X3 = (X3 - np.mean(X3)) / np.std(X3)

# Stack the explanatory variables into a matrix (X)
X_all = np.column_stack((X1, X2, X3))

# Gradient descent for multiple variables
def gradient_descent_multi(X, y, learning_rate, iterations):
    m = len(y)
    n = X.shape[1]  # Number of features
    theta = np.zeros(n)  # Initialize theta to zeros
    loss_history = []

    for _ in range(iterations):
        prediction = X.dot(theta)
        loss = (1/(2*m)) * np.sum((prediction - y)**2)  # MSE cost function
        gradient = (1/m) * X.T.dot(prediction - y)  # Gradient calculation
        theta = theta - learning_rate * gradient  # Update theta
        
        loss_history.append(loss)  # Record loss for each iteration

    return theta, loss_history

# Set learning rate and iterations
learning_rate = 0.05
iterations = 1000

# Apply gradient descent using all three variables
theta_all, loss_all = gradient_descent_multi(X_all, Y, learning_rate, iterations)

# Plot the loss over iterations for the combined variables
plt.figure(figsize=(7, 5))
plt.plot(loss_all, color='purple')
plt.title('Loss Over Iterations (All Variables)')
plt.xlabel('Iteration')
plt.ylabel('Loss (MSE)')
plt.show()

# Final model coefficients
print("Final coefficients:", theta_all)

# Predict the value of Y for new (X1, X2, X3) values (1, 1, 1), (2, 0, 4), and (3, 2, 1)
new_inputs = np.array([[1, 1, 1], [2, 0, 4], [3, 2, 1]])
predictions = new_inputs.dot(theta_all)

print("Predictions for new inputs:", predictions)
