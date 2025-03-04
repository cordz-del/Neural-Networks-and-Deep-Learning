import numpy as np

# Activation functions
def sigmoid(z):
    return 1 / (1 + np.exp(-z))

def sigmoid_derivative(z):
    s = sigmoid(z)
    return s * (1 - s)

# Initialize parameters
def initialize_parameters(n_x, n_h, n_y):
    np.random.seed(42)
    W1 = np.random.randn(n_h, n_x) * 0.01  # hidden layer weights
    b1 = np.zeros((n_h, 1))
    W2 = np.random.randn(n_y, n_h) * 0.01  # output layer weights
    b2 = np.zeros((n_y, 1))
    return W1, b1, W2, b2

# Forward propagation (vectorized)
def forward_propagation(X, W1, b1, W2, b2):
    Z1 = np.dot(W1, X) + b1
    A1 = np.tanh(Z1)
    Z2 = np.dot(W2, A1) + b2
    A2 = sigmoid(Z2)
    cache = (Z1, A1, Z2, A2)
    return A2, cache

# Compute the binary cross-entropy cost
def compute_cost(A2, Y):
    m = Y.shape[1]
    cost = -np.sum(Y * np.log(A2) + (1 - Y) * np.log(1 - A2)) / m
    return np.squeeze(cost)

# Backward propagation (vectorized)
def backward_propagation(X, Y, cache, W1, W2):
    m = X.shape[1]
    Z1, A1, Z2, A2 = cache
    
    dZ2 = A2 - Y
    dW2 = (1 / m) * np.dot(dZ2, A1.T)
    db2 = (1 / m) * np.sum(dZ2, axis=1, keepdims=True)
    
    dZ1 = np.dot(W2.T, dZ2) * (1 - np.power(A1, 2))  # derivative of tanh
    dW1 = (1 / m) * np.dot(dZ1, X.T)
    db1 = (1 / m) * np.sum(dZ1, axis=1, keepdims=True)
    
    grads = {"dW1": dW1, "db1": db1, "dW2": dW2, "db2": db2}
    return grads

# Update parameters using gradient descent
def update_parameters(W1, b1, W2, b2, grads, learning_rate):
    W1 -= learning_rate * grads["dW1"]
    b1 -= learning_rate * grads["db1"]
    W2 -= learning_rate * grads["dW2"]
    b2 -= learning_rate * grads["db2"]
    return W1, b1, W2, b2

# Combine into a training loop for the two-layer model
def two_layer_model(X, Y, n_h, num_iterations=5000, learning_rate=0.01, print_cost=False):
    n_x = X.shape[0]
    n_y = Y.shape[0]
    W1, b1, W2, b2 = initialize_parameters(n_x, n_h, n_y)
    
    for i in range(num_iterations):
        A2, cache = forward_propagation(X, W1, b1, W2, b2)
        cost = compute_cost(A2, Y)
        grads = backward_propagation(X, Y, cache, W1, W2)
        W1, b1, W2, b2 = update_parameters(W1, b1, W2, b2, grads, learning_rate)
        
        if print_cost and i % 1000 == 0:
            print(f"Cost after iteration {i}: {cost:.6f}")
            
    parameters = {"W1": W1, "b1": b1, "W2": W2, "b2": b2}
    return parameters

# Example usage with synthetic data for binary classification:
np.random.seed(1)
X = np.random.randn(2, 400)  # 2 features, 400 examples
Y = (np.sum(X, axis=0) > 0).astype(int).reshape(1, 400)  # Binary target

parameters = two_layer_model(X, Y, n_h=4, num_iterations=5000, learning_rate=0.01, print_cost=True)
