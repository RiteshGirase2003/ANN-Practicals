import numpy as np

# Weight 
w = [1,1]
# Threshold
threshold = 1

def mc_neuron(inputs):
    if len(inputs) != len(w):
        print ("Length of input array and weight array must be equal")
        return
    product = np.dot(inputs,w)
    output = 1 if product >= threshold else 0
    return output



def and_not (x1,x2):
    inputs = np.array([x1,x2])
    return (mc_neuron(inputs))


print(and_not(1,0))