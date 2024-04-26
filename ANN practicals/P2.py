import numpy as np
def mp_neuron(inputs, weights, threshold):
    # dot product 2 arrays 1x2 order

    weighted_sum = np.dot(inputs, weights)
    # Yout 
    
    output = 1 if weighted_sum >= threshold else 0
    return output


def and_not(x1, x2):
    weights = [1, -1] 
    threshold = 1   
    # convert to array
    inputs = np.array([x1, x2])
    # passing to above function
    output = mp_neuron(inputs, weights, threshold)
    return output


print(and_not(0, 0)) 
print(and_not(1, 0))  
print(and_not(0, 1))  
print(and_not(1, 1))  