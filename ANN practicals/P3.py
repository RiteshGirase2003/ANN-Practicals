# Write a Python Program using Perceptron Neural Network 
# to recognise even and odd numbers. 
# Given numbers are in ASCII form 0 to 9


import numpy as np

j = int(input("Enter a Number (0-9): "))



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

weights = np.array([0, 0, 0, 0, 0, 1])

# activation if X >= 0 then 1, else 0
step_function = lambda x: 1 if x >= 0 else 0

for data in training_data:
    # converting input value list to array from training data
    input = np.array(data['input'])

    label = data['label']
    # dot product means normal multiplication of matrix
    # here dot product of weight and input (ASCII of numbers ) is done
    # np.dot([1,2,4], [4,5,6]) => (1*4 + 2*5 + 3*6)
    # this is single digit value 
    dotproduct = np.dot(input, weights)
    
    print(weights,input,dotproduct)
    # dot product of input array and weight array is send to step_function
    output = step_function(dotproduct)

    # here error is calculated
    error = label - output
    # here weight are adjusted 
    weights += input * error

# '{0:06b}'.format(j) make int to binary and 6 digit long by adding extra zeros in front
# list('{0:06b}'.format(j)) convert the binary to list
# [int(x) for x in list('{0:06b}'.format(j))] make each element in list from string to int
input = np.array([int(x) for x in list('{0:06b}'.format(j))])
# input = np.array([1, 1, 0, 1, 0, 0])
print("input : ",input)

# now most updated weight and input is send to step_function 
#  if it returns 0 then odd else even
output = "odd" if step_function(np.dot(input, weights)) == 0 else "even"
print(j, " is ", output)
