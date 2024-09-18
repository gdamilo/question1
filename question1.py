import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Data (inserted directly from the uploaded file)
data = {
    'X1': [0.000000, 0.040404, 0.080808, 0.121212, 0.161616, 0.202020, 0.242424, 0.282828, 0.323232, 0.363636],
    'X2': [3.440000, 0.134949, 0.829899, 1.524848, 2.219798, 2.914747, 3.609696, 4.304646, 4.999595, 5.694545],
    'X3': [0.440000, 0.888485, 1.336970, 1.785455, 2.233939, 2.682424, 3.130909, 3.579394, 4.027879, 4.476364],
    'Y': [4.387545, 2.679650, 2.968490, 3.254065, 3.536375, 3.815421, 4.091201, 4.363717, 4.632967, 4.898952]
}

# Convert to DataFrame
df = pd.DataFrame(data)

# Separate the dataset into X1, X2, X3, and Y
X1 = df['X1'].values
X2 = df['X2'].values
X3 = df['X3'].values
Y = df['Y'].values

# Function to perform linear regression using gradient descent
def gradient_descent(X, y, learning_rate, iterations):
    m = len(y)
    theta = 0  # Initialize the parameter
    loss_history = []

    for _ in range(iterations):
        prediction = theta * X
        loss = (1/(2*m)) * np.sum((prediction - y)**2)  # MSE cost function
        gradient = (1/m) * np.sum((prediction - y) * X)  # Gradient calculation
        theta = theta - learning_rate * gradient  # Update theta

        loss_history.append(loss)  # Record loss for each iteration

    return theta, loss_history

# Function to plot the regression line and loss over iterations
def plot_results(X, y, theta, loss_history, variable_name):
    plt.figure(figsize=(14, 5))

    # Plot the regression model
    plt.subplot(1, 2, 1)
    plt.scatter(X, y, color='blue', label='Data points')
    plt.plot(X, theta * X, color='red', label='Regression Line')
    plt.title(f'Regression Model for {variable_name}')
    plt.xlabel(variable_name)
    plt.ylabel('Y')
    plt.legend()

    # Plot the loss over iterations
    plt.subplot(1, 2, 2)
    plt.plot(loss_history, color='green')
    plt.title('Loss Over Iterations')
    plt.xlabel('Iteration')
    plt.ylabel('Loss (MSE)')

    plt.show()

# Normalize X values for better gradient descent performance
X1 = (X1 - np.mean(X1)) / np.std(X1)
X2 = (X2 - np.mean(X2)) / np.std(X2)
X3 = (X3 - np.mean(X3)) / np.std(X3)

# Set learning rate and iterations
learning_rate = 0.05
iterations = 1000

# Apply gradient descent for each variable
theta_X1, loss_X1 = gradient_descent(X1, Y, learning_rate, iterations)
theta_X2, loss_X2 = gradient_descent(X2, Y, learning_rate, iterations)
theta_X3, loss_X3 = gradient_descent(X3, Y, learning_rate, iterations)

# Plot results for X1
plot_results(X1, Y, theta_X1, loss_X1, 'X1')

# Plot results for X2
plot_results(X2, Y, theta_X2, loss_X2, 'X2')

# Plot results for X3
plot_results(X3, Y, theta_X3, loss_X3, 'X3')

# Return the final losses to determine which has the lowest loss
losses = {'X1': loss_X1[-1], 'X2': loss_X2[-1], 'X3': loss_X3[-1]}
print(losses)
