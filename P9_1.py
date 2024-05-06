# best


import tensorflow as tf
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# Load and preprocess data
iris_data = load_iris()
X = iris_data.data  # Load features (input data)
print('X : ',X)
y = iris_data.target  # Load labels (output data)
print('y : ',y)

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)  

# Initialize a StandardScaler instance for feature scaling
# make sure all your data features have the same scale (mean of 0 and variance of 1)
scaler = StandardScaler()  

# Fit and transform training data for feature scaling
# used to both train and transform data
X_train_scaled = scaler.fit_transform(X_train) 

print(" X train scaled fit transform : ", X_train_scaled)
# Transform testing data using the same scaler
X_test_scaled = scaler.transform(X_test)  
print(" X test scaled transform : ", X_test_scaled)

# Define and compile logistic regression model
model = tf.keras.Sequential([
    tf.keras.layers.Dense(3, activation='softmax', input_shape=(4,))  # Define a single dense layer with 3 output units and softmax activation for multi-class classification
])

model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])  # Compile the model with Adam optimizer, sparse categorical crossentropy loss function, and accuracy metric

# Train the model
model.fit(X_train_scaled, y_train, epochs=50, batch_size=32, verbose=0)  # Train the model on the scaled training data for 50 epochs with a batch size of 32

# Evaluate the model
test_loss, test_accuracy = model.evaluate(X_test_scaled, y_test)  # Evaluate the model's performance on the scaled testing data
print(f'Test Accuracy: {test_accuracy}')  # Print the test accuracy of the model
