""" Tests my implementation of a Multi-layer Perceptron using the MNIST handwritten digits database.

This program tests my implementation of a Multi-layer Perceptron (MLP), the standard feedforward neural network, using
the MNIST handwritten digits database. Once the model finishes training, a window containing a random selected image
from the test set will be displayed along with its label and the models prediction. By pressing SPACE, you can select
a new image.

@author: Gabriel G. Nogueira (Talendar)
"""

from multilayer_perceptron import MultilayerPerceptron
from cost_functions import MeanSquaredError

import numpy as np
import pandas as pd
import pygame


############### CONFIG ###############
TEST_SET_PC = 0.2                    # percentage of the data to be used to test the model
HIDDEN_LAYERS = [64, 64]             # hidden layers architecture
TRAINING_EPOCHS = 100                # number of training iterations
LEARNING_RATE = 3                    # the model's learning rate
######################################


def load_mnist(path):
    """ Loads and shuffles the MNIST data. """
    df = pd.read_csv(path).sample(frac=1).reset_index(drop=True)  # loads and shuffles data
    X, Y = [], []

    for i, row in df.iterrows():
        label, pixels = row["label"], row.drop("label").values / 255
        X.append(pixels)

        y = np.zeros(10)
        y[label] = 1
        Y.append(y)

    return np.array(X), np.array(Y)


def accuracy(H, Y):
    """ Given a model's predictions and a set of labels, calculates the model's accuracy. """
    hits = 0
    for h, y in zip(H.T, Y):
        h = np.argmax(h, axis=0)
        y = np.argmax(y, axis=0)

        if h == y:
            hits += 1

    return hits / len(Y)


def evaluation_screen(X, Y, H):
    """ Displays a window containing an image, its label and the model's prediction for it. """
    pygame.init()
    display = pygame.display.set_mode((400, 470))
    font = pygame.font.SysFont(pygame.font.get_default_font(), 30)

    new_item = True
    while True:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                exit()

            if event.type == pygame.KEYDOWN and event.key == pygame.K_SPACE:
                new_item = True

        if new_item:
            display.fill((0, 0, 0))

            i = np.random.randint(0, len(X))
            x, y, h = X[i], np.argmax(Y[i], axis=0), np.argmax(H[i], axis=0)

            x *= 255
            x.shape = (28, 28)
            x = np.array([x] * 3)

            img = pygame.transform.scale(pygame.surfarray.make_surface(x.T), (400, 400))
            display.blit(img, (0, 0))

            txt_surface = font.render("PREDICTED: %d   |   LABEL: %d" % (h, y), False, (251, 255, 170))
            display.blit(txt_surface, (58, 420))

            pygame.display.update()
            new_item = False


if __name__ == "__main__":
    print("\nLoading data... ", end="")
    data, labels = load_mnist("./data/mnist_data.csv")
    print("done!")

    i = int(len(data) * TEST_SET_PC)
    X_train, Y_train = data[i:], labels[i:]
    X_test, Y_test = data[:i], labels[:i]

    print("\nTraining set samples: %d (%d%%)" % (len(X_train), 100*(1 - TEST_SET_PC)))
    print("Test set samples: %d (%d%%)" % (len(X_test), 100*TEST_SET_PC))

    mlp = MultilayerPerceptron(input_size=784, layers_size=HIDDEN_LAYERS + [10], layers_activation="sigmoid")
    print("\nInitial accuracy (training set): %.2f%%" % (100 * accuracy(mlp.predict(X_train), Y_train)))
    print("Initial accuracy (test set): %.2f%%" % (100 * accuracy(mlp.predict(X_test), Y_test)))

    print("\nStarting training session...")
    mlp.fit(
        data=X_train, labels=Y_train,
        cost_function=MeanSquaredError(),
        epochs=TRAINING_EPOCHS,
        learning_rate=LEARNING_RATE,
        batch_size=32,
        gradient_checking=False
    )

    print("\nAccuracy (training set): %.2f%%" % (100*accuracy(mlp.predict(X_train), Y_train)))
    print("Accuracy (test set): %.2f%%\n" % (100*accuracy(mlp.predict(X_test), Y_test)))

    print("Opening evaluation window...\nTo select a new image, press SPACE.\n")
    evaluation_screen(X_test, Y_test, mlp.predict(X_test).transpose())