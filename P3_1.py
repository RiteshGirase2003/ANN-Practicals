# best if given number are not ascii 


import numpy as np
import random

# Initialize the weight vector
w = np.array([0, 0, 0, 0, 0, 1])

# Define the step function
def step_fun(input):
    return 1 if input >= 0 else 0


# Define the training data
training_data = [
    {'input': [0, 0, 0, 0, 0, 0], 'label': 1, 'no': 0},   # 0 (even)
    {'input': [0, 0, 0, 0, 0, 1], 'label': 0, 'no': 1},   # 1 (odd)
    {'input': [0, 0, 0, 0, 1, 0], 'label': 1, 'no': 2},   # 2 (even)
    {'input': [0, 0, 0, 0, 1, 1], 'label': 0, 'no': 3},   # 3 (odd)
    {'input': [0, 0, 0, 1, 0, 0], 'label': 1, 'no': 4},   # 4 (even)
    {'input': [0, 0, 0, 1, 0, 1], 'label': 0, 'no': 5},   # 5 (odd)
    {'input': [0, 0, 0, 1, 1, 0], 'label': 1, 'no': 6},   # 6 (even)
    {'input': [0, 0, 0, 1, 1, 1], 'label': 0, 'no': 7},   # 7 (odd)
    {'input': [0, 0, 1, 0, 0, 0], 'label': 1, 'no': 8},   # 8 (even)
    {'input': [0, 0, 1, 0, 0, 1], 'label': 0, 'no': 9},   # 9 (odd)
    {'input': [0, 0, 1, 0, 1, 0], 'label': 1, 'no': 10},  # 10 (even)
    {'input': [0, 0, 1, 0, 1, 1], 'label': 0, 'no': 11},  # 11 (odd)
    {'input': [0, 0, 1, 1, 0, 0], 'label': 1, 'no': 12},  # 12 (even)
    {'input': [0, 0, 1, 1, 0, 1], 'label': 0, 'no': 13},  # 13 (odd)
    {'input': [0, 0, 1, 1, 1, 0], 'label': 1, 'no': 14},  # 14 (even)
    {'input': [0, 0, 1, 1, 1, 1], 'label': 0, 'no': 15},  # 15 (odd)
    {'input': [0, 1, 0, 0, 0, 0], 'label': 1, 'no': 16},  # 16 (even)
    {'input': [0, 1, 0, 0, 0, 1], 'label': 0, 'no': 17},  # 17 (odd)
    {'input': [0, 1, 0, 0, 1, 0], 'label': 1, 'no': 18},  # 18 (even)
    {'input': [0, 1, 0, 0, 1, 1], 'label': 0, 'no': 19},  # 19 (odd)
    {'input': [0, 1, 0, 1, 0, 0], 'label': 1, 'no': 20}   # 20 (even)
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
# no = int(input("Enter the number : "))
for no in range(0,63):
    output =test(no,w)
    print("Number ",no," is : ","even" if output == 1 else "odd")