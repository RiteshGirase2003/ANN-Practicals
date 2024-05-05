# best if given number are  ascii 


import numpy as np
import random

# Initialize the weight vector
w = np.array([0, 0, 0, 0, 0, 1])

# Define the step function
def step_fun(input):
    return 1 if input >= 0 else 0




# Define the training data
training_data = [

    {'input': [1, 1, 0, 0, 0, 0], 'label': 1}, # ASCII for 0 is 48 and its binary is 110000
    {'input': [1, 1, 0, 0, 0, 1], 'label': 0},
    {'input': [1, 1, 0, 0, 1, 0], 'label': 1},
    {'input': [1, 1, 0, 1, 1, 1], 'label': 0},
    {'input': [1, 1, 0, 1, 0, 0], 'label': 1},
    {'input': [1, 1, 0, 1, 0, 1], 'label': 0},
    {'input': [1, 1, 0, 1, 1, 0], 'label': 1},
    {'input': [1, 1, 0, 1, 1, 1], 'label': 0},
    {'input': [1, 1, 1, 0, 0, 0], 'label': 1},
    {'input': [1, 1, 1, 0, 0, 1], 'label': 0},
]
test_array = []

test_array.extend(random.sample(training_data, 5))


# Train the perceptron
def train(w):
    # print("---------------------------- Training ------------------------------------")

    for data in training_data:
        input = np.array(data['input'])
        label = data['label']
        dot_product = np.dot(input, w)
        output = step_fun(dot_product)
        # print("Number ",data['no'],"is : ","even" if output == 1 else "odd")

        # Calculate error and update weights
        error = label - output
        w += input * error
        # print("Weight : ",w)

        # print("----------------------------------------------------------------")

# Testing the perceptron
def test(w):
    # print("---------------------------- Testing ------------------------------------")
    for data in test_array:
        input = np.array(data['input'])
        label = data['label']
        dot_product = np.dot(input, w)
        output = step_fun(dot_product)
        if label != output:
            train(w)

        

test(w)

# Test the perceptron
def test(no,w):
    noarray = bin(no)[2:].zfill(6)

    noarray = [int(bit) for bit in noarray]

    noarray = np.array(noarray)

    output = step_fun(np.dot(noarray, w))
    return output


no = int(input("Enter the number : "))
output =test(no,w)
print("Number ",no," is : ","even" if output == 1 else "odd")