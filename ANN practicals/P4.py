import numpy as np
import matplotlib.pyplot as plt


def perceptron(x, w, b):
    # 1 if the number is positive,
    # 0 if the number is zero, and
    # -1 if the number is negative.
    return np.sign(np.dot(x, w) + b)


def perceptron_learning(X, Y, eta, epochs):
    # create 1D array with 2 zeros
    w = np.zeros(2)

    b = 0
    # n^2
    for epoch in range(epochs):
        # X.shape[0] to get number of rows here it is 4
        #  it will iterate number rows times
        for i in range(X.shape[0]):
            # SEND particular row in each iteration 
            y_pred = perceptron(X[i], w, b)
            
            
            if y_pred != Y[i]:
            #   w is adjusted
                w =w + eta * Y[i] * X[i]
                # bias b is adjust
                b = b + eta * Y[i]

    return w, b

# X
# 0  0
# 1  0
# 0  1
# 1  1

X = np.array([[0, 0], [1, 0], [0, 1], [1, 1]])
Y = np.array([-1, -1, -1, 1])
#  eta -> learning rate (η) in the perceptron algorithm determines 
# how big of a step the algorithm takes to adjust its weights during training
w, b = perceptron_learning(X, Y, eta=1, epochs=10)


# X[:, 0].min() - 1 -> from all row select column 0 having minimum value  and - 1 to it
# same  but max value + 1 
x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
print(x_min,x_max)

y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1

# np.arange(x_min, x_max, 0.01) generates an array of values 
# starting from x_min and ending at x_max, incrementing by 0.01 each time
# np.arange(0, 5, 0.5) -> [0, 0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0, 4.5]


# meshgrid is often used in data visualization and computational tasks
# , especially in contexts like plotting 3D surfaces, contour plots,
#  and vector field visualization. It's used to create a grid of points 
# from two 1D arrays, typically representing the x and y coordinates of the points.
xx, yy = np.meshgrid(np.arange(x_min, x_max, 0.01),
                     np.arange(y_min, y_max, 0.01))

print(xx)
# np.c_ concante
# ravel makes nd array to 1d
Z = np.array([perceptron(np.array([x, y]), w, b) for x, y in np.c_[xx.ravel(), yy.ravel()]])
Z = Z.reshape(xx.shape)
plt.contourf(xx, yy, Z, cmap=plt.cm.Paired, alpha=0.8)
plt.scatter(X[:, 0], X[:, 1], c=Y, cmap=plt.cm.Paired)
plt.xlabel('Feature 1')
plt.ylabel('Feature 2')
plt.xlim(xx.min(), xx.max())
plt.ylim(yy.min(), yy.max())
plt.title('Perceptron Decision Regions')
plt.show()