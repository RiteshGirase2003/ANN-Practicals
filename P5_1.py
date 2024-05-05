# best

import numpy as np
X = np.array([[1, 1, 1, -1], [-1, -1, 1, 1]])
Y = np.array([[1, -1], [-1, 1]])
W = np.dot(Y.T, X)
print(W)
def bam(x):
    y = np.sign(np.dot(W, x))
    return y


input_1 = np.array([1, -1, -1, -1])
input_2 = np.array([0, 0, 1, 1])

output_1 = bam(input_1)
output_2 = bam(input_2)

print("Output for input ",input_1," :", output_1)
print("Output for input ",input_2," :", output_2)
