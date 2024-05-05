# best


import numpy as np


class NeuralNetwork:
    def __init__(self):
        # It generates 2x1 array of random numbers
        self.weights = np.random.rand(2,1)
        self.bias = np.random.rand(1)
    
    def train(self,X,Y,epochs):
        for _ in range(epochs):
            # predicting output for input X
            # it is array of 4x1 for input X of 4x2
            output = self.predict(X) 


            # [1,2,3,4] - [1,1,2,2] => [0,1,1,2]
            error = Y - output

            # if error is non empty then it will be true else false
            # if it is not empty then their is error present
            # if !(error):
            delta  = error * output * (1 - output)
            

            # adjusting weights and bais

            # Weight
            # new weight = old weight + transpose of X.delta
            self.weights = self.weights + np.dot(X.T,delta)

            # Bias
            # new bias = old bias + sum of all elements of delta
            self.bias = self.bias + np.sum(delta)


    def predict(self,X):
        # X 4x2 array

        # x1
        # weighted sum of all elements 
        # summation E 1->n  Xi*Wi
        x1 = np.dot(X,self.weights)

        # x2
        # bias is added to weighted sum
        x2 = x1 + self.bias

        # x3
        # Activation function
        # Sigmoid function is used 
        x3 = 1 / (1 + np.exp(-x2))
        return x3

# input 
X = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
# output
Y = np.array([[0], [0], [0], [1]])


nn = NeuralNetwork()

nn.train(X,Y,epochs =1000)


# ---
test_data = np.array([[0, 0], [0, 1], [1, 1], [1, 1]])

output = nn.predict(test_data)
print(output)

for i in output:
    print( 1 if i>0.9 else 0)