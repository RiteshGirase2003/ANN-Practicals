# best 

# Write a python program
# to show Back Propagation Network for XOR function 
# with Binary Input and Output

import numpy as np

def sigmoid(x):
    return 1 / (1 + np.exp(-x))


def sigmoid_derivative(x):
    return x * (1 - x)



X = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
y = np.array([[0], [1], [1], [0]])


np.random.seed(42)
 

w1 = 2 * np.random.random((2, 4)) - 1
w2 = 2 * np.random.random((4, 1)) - 1

for i in range(10000):
    # layer 0 is the first layer ( input layer)
    l1 = X
    # layer 1 is the second layer
    # which is part of hidden layer
    # it calculate weighted sum Xi.Wi
    l2 = sigmoid(np.dot(l1, w1))

    # layer 2 is the third layer
    # which is part of hidden layer
    # it calculate weighted sum Xi.Wi
    # here Xi is input for layer 2 but it is output of 1 layer
    l3 = sigmoid(np.dot(l2, w2))
    
    # error calculation
    error = y - l3
    
    # it is similar to prac 6 
    delta_2 = error * sigmoid_derivative(l3)

    #  similar to np.dot(delta_2,weights.T)
    #  delta1 = delta2.(weights1.T) * sigmoid_derivative(layer_1)
    delta_1 = delta_2.dot(w2.T) * sigmoid_derivative(l2)
    

    # adjusting weights

    w2 += l2.T.dot(delta_2)
    w1 += l1.T.dot(delta_1)

output = sigmoid(np.dot(sigmoid(np.dot(X, w1)), w2))
print("Predicted Output:")
print(output)
