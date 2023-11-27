#-------------------------------------------------------------------------
# AUTHOR: Alexander Eckert
# FILENAME: deep_learning.py
# SPECIFICATION: Using deep learning to interpret handwritten letters
# FOR: CS 4210- Assignment #4
# TIME SPENT: 4.75 hrs
#-----------------------------------------------------------*/

#IMPORTANT NOTE: YOU CAN USE ANY PYTHON LIBRARY TO COMPLETE YOUR CODE.

# importing the libraries
import tensorflow as tf
from tensorflow import keras
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

def build_model(n_hidden, n_neurons_hidden, n_neurons_output, learning_rate):
    # Creating the Neural Network using the Sequential API
    model = keras.models.Sequential()
    model.add(keras.layers.Flatten(input_shape=[28, 28]))  # input layer

    # iterate over the number of hidden layers to create the hidden layers
    for _ in range(n_hidden):
        model.add(keras.layers.Dense(n_neurons_hidden, activation="relu"))  # hidden layer with ReLU activation function

    # output layer
    model.add(keras.layers.Dense(n_neurons_output, activation="softmax"))  # output layer with softmax activation function

    # defining the learning rate
    opt = keras.optimizers.SGD(learning_rate)

    # Compiling the Model specifying the loss function and the optimizer to use.
    model.compile(loss="sparse_categorical_crossentropy", optimizer=opt, metrics=["accuracy"])
    return model

# Using Keras to Load the Dataset
fashion_mnist = keras.datasets.fashion_mnist
(X_train_full, y_train_full), (X_test, y_test) = fashion_mnist.load_data()

# creating a validation set and scaling the features
X_valid, X_train = X_train_full[:5000] / 255.0, X_train_full[5000:] / 255.0
y_valid, y_train = y_train_full[:5000], y_train_full[5000:]

# For Fashion MNIST, we need the list of class names
class_names = ["T-shirt/top", "Trouser", "Pullover", "Dress", "Coat", "Sandal", "Shirt", "Sneaker", "Bag", "Ankle boot"]

# Iterate over number of hidden layers, number of neurons in each hidden layer, and the learning rate.
highest_accuracy = 0.0
best_model = None

n_hidden = [2, 5, 10]
n_neurons = [10, 50, 100]
l_rate = [0.01, 0.05, 0.1]

for h in n_hidden:
    for n in n_neurons:
        for l in l_rate:
            # build the model for each combination
            model = build_model(h, n, 10, l)

            # train the model
            history = model.fit(X_train, y_train, epochs=5, validation_data=(X_valid, y_valid))

            # calculate the accuracy of this neural network and store its value if it is the highest so far
            accuracy = model.evaluate(X_test, y_test)[1]
            if accuracy > highest_accuracy:
                highest_accuracy = accuracy
                best_model = model

            print("Highest accuracy so far:", highest_accuracy)
            print("Parameters: Number of Hidden Layers:", h, ", Number of Neurons:", n, ", Learning Rate:", l)
            print()

# After generating all neural networks, print the summary of the best model found
best_model.summary()
img_file = './model_arch.png'
tf.keras.utils.plot_model(best_model, to_file=img_file, show_shapes=True, show_layer_names=True)

# plotting the learning curves of the best model
pd.DataFrame(history.history).plot(figsize=(8, 5))
plt.grid(True)
plt.gca().set_ylim(0, 1)  # set the vertical range to [0-1]
plt.show()

