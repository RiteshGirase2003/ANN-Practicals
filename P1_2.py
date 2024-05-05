# best

import numpy as np
import matplotlib.pyplot as plt

# Actication funtion 
# -> it is used to introduce non linearity to neural network
# -> non linearity means change in one variable does not result constant/fixed change in other variables
# -> It breaks the linear nature ( straight line ) of graph
# -> it also help to make node active and unactive based on threshold values


# Define the activation functions
def logistic(x):
    return 1 / (1 + np.exp(-x))

def step(x):
    return np.where(x >= 0, 1, 0)

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def tanh(x):
    return np.tanh(x)

def relu(x):
    return np.maximum(0, x)

def leaky_relu(x, alpha=0.1):
    return np.maximum(alpha*x, x)

def softmax(x):
    # exp_values = np.exp(x - np.max(x, axis=1, keepdims=True))
    return np.exp(x) / np.sum(np.exp(x))
def identity(x):
    return x

def linear(x):
    return x

# Generate input data
x = np.linspace(-5, 5, 100)

# Plotting the activation functions
plt.figure(figsize=(12, 8))

# subplot(row, col, index)

plt.subplot(3, 4, 1)
plt.title("Logistic Function")
plt.plot(x, logistic(x))

plt.subplot(3, 4, 2)
plt.title("Step Function")
plt.plot(x, step(x))

plt.subplot(3, 4, 3)
plt.title("Sigmoid Function")
plt.plot(x, sigmoid(x))
 
plt.subplot(3, 4, 4)
plt.title("Hyperbolic Tangent (tanh)")
plt.plot(x, tanh(x))

plt.subplot(3, 4, 5)
plt.title("ReLU Function")
plt.plot(x, relu(x))

plt.subplot(3, 4, 6)
plt.title("Leaky ReLU Function")
plt.plot(x, leaky_relu(x))

plt.subplot(3, 4, 7)
plt.title("Softmax Function")
# Softmax requires 2D input
# x_softmax = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
plt.plot(x, softmax(x))

plt.subplot(3, 4, 8)
plt.title("Identity Function")
plt.plot(x, identity(x))

plt.subplot(3, 4, 9)
plt.title("Linear Function")
plt.plot(x, linear(x))

# adjusts the spacing between subplots so they don't overlap, making your plots look neat and organized
plt.tight_layout()
plt.show()


# Logistic Regression
# -> LR is Linear and used for simple tasks
# -> usually used for binary classification ( predict binary outcomes )

# Advantages of ReLu
# -> ReLU is computationally efficient and helps avoid the vanishing gradient problem. 
# -> However, it can suffer from dead neurons.

# Choosing Activation Functions by Layer:
# -> Use ReLU or variants in hidden layers for most cases,
# -> sigmoid or softmax in the output layer for specific tasks like binary or multi-class classification.

# Softmax activation function
# -> used in the output layer of neural networks for multi-class classification problems.
# -> Raw score is produced by output layer of neural networks for multi-class classification problems.
# -> let consider it for cat, dog, bird => [2.5, 1.8, 3.2]
# -> using softmax convert it to probabilities [0.265, 0.116, 0.619] for cat, dog, and bird respectively
# -> here , bird has highest probability so class belongs to bird
# -> suitable for multi-class classification tasks where the goal is to assign one of several possible labels to input data.

# Squashing Function -> also known as Activation Function
# -> A squashing function is a mathematical function,
#    used in neural networks to map input values to a specific range, often between 0 and 1 or -1 and 1.
# -> Types of Activation Function which have squashing nature 
# -> 1. Sigmoid function
# -> range (0,1)
# -> 2.Tanh function
# -> range (-1,1)

