import matplotlib.pyplot as plt

def two_layer_model_with_history(X, Y, n_h, num_iterations=5000, learning_rate=0.01):
    n_x = X.shape[0]
    n_y = Y.shape[0]
    W1, b1, W2, b2 = initialize_parameters(n_x, n_h, n_y)
    costs = []
    
    for i in range(num_iterations):
        A2, cache = forward_propagation(X, W1, b1, W2, b2)
        cost = compute_cost(A2, Y)
        grads = backward_propagation(X, Y, cache, W1, W2)
        W1, b1, W2, b2 = update_parameters(W1, b1, W2, b2, grads, learning_rate)
        
        if i % 100 == 0:
            costs.append(cost)
    
    parameters = {"W1": W1, "b1": b1, "W2": W2, "b2": b2}
    return parameters, costs

# Train the model and capture cost history
parameters, costs = two_layer_model_with_history(X, Y, n_h=4, num_iterations=5000, learning_rate=0.01)

# Plot the cost over iterations (useful in a Jupyter Notebook cell)
plt.plot(costs)
plt.xlabel("Iterations (x100)")
plt.ylabel("Cost")
plt.title("Training Cost over Time")
plt.show()
