import numpy as np
import matplotlib.pyplot as plt

# Define the activation functions
def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def relu(x):
    return np.maximum(0, x)

def tanh(x):
    return np.tanh(x)

def identity(x):
    return x

def softmax(x):
    exp_vals = np.exp(x - np.max(x))
    return exp_vals / np.sum(exp_vals, axis=0)

def linear(x):
    return x

# Create x values
x = np.linspace(-5, 5, 100)

# Plot sigmoid
plt.figure(figsize=(6, 4))
plt.plot(x, sigmoid(x))
plt.title('Sigmoid Activation Function')
plt.xlabel('Input')
plt.ylabel('Output')
plt.grid(True)
plt.show()

# Plot relu
plt.figure(figsize=(6, 4))
plt.plot(x, relu(x))
plt.title('ReLU Activation Function')
plt.xlabel('Input')
plt.ylabel('Output')
plt.grid(True)
plt.show()

# Plot tanh
plt.figure(figsize=(6, 4))
plt.plot(x, tanh(x))
plt.title('Tanh Activation Function')
plt.xlabel('Input')
plt.ylabel('Output')
plt.grid(True)
plt.show()

# Plot identity
plt.figure(figsize=(6, 4))
plt.plot(x, identity(x))
plt.title('Identity Activation Function')
plt.xlabel('Input')
plt.ylabel('Output')
plt.grid(True)
plt.show()

# Plot softmax
inputs = np.array([1, 2, 3])
plt.figure(figsize=(6, 4))
plt.plot(inputs, softmax(inputs))
plt.title('Softmax Activation Function')
plt.xlabel('Input')
plt.ylabel('Output')
plt.grid(True)
plt.show()

# Plot linear
plt.figure(figsize=(6, 4))
plt.plot(x, linear(x))
plt.title('Linear Activation Function')
plt.xlabel('Input')
plt.ylabel('Output')
plt.grid(True)
plt.show()
