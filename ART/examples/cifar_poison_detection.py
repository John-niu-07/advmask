# -*- coding: utf-8 -*-
"""Generates a backdoor for MNIST dataset, then trains a convolutional neural network on the poisoned dataset,
 and runs activation defence to find poison."""
from __future__ import absolute_import, division, print_function, unicode_literals

import pprint
import json

from keras.models import Sequential
from keras.layers import Dense, Flatten, Conv2D, MaxPooling2D, Dropout
import numpy as np

from art.attacks.poisoning.perturbations.image_perturbations import add_pattern_bd, add_single_bd
from art.estimators.classification import KerasClassifier
from art.utils import load_mnist, preprocess
from art.defences.detector.poison import ActivationDefence
from art.defences.detector.poison import SpectralSignatureDefense

from art.utils import load_dataset
from art.preactresnet import PreActBlock, PreActBottleneck, PreActResNet, PreActResNet18, initialize_weights, PreActResNet18_

import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
from torch import Tensor
from torch import torch
from art.estimators.classification import PyTorchClassifier

def main():
    # Read MNIST dataset (x_raw contains the original images):
    #(x_raw, y_raw), (x_raw_test, y_raw_test), min_, max_ = load_mnist(raw=True)
    #(x_train, y_train), (x_test, y_test), min_, max_ = load_dataset(str("cifar10"))
    (x_raw, y_raw_), (x_raw_test, y_raw_test_), min_, max_ = load_dataset(str("cifar10"))

    y_raw = np.argmax(y_raw_, axis=1)
    y_raw_test = np.argmax(y_raw_test_, axis=1)
    #print(x_raw.shape)
    #print(y_raw)
    #print(yy)
    n_train = np.shape(x_raw)[0]
    num_selection = 5000
    random_selection_indices = np.random.choice(n_train, num_selection)
    x_raw = x_raw[random_selection_indices]
    y_raw = y_raw[random_selection_indices]

    # Poison training data
    #perc_poison = 0.33
    perc_poison = 0.05
    (is_poison_train, x_poisoned_raw, y_poisoned_raw) = generate_backdoor(x_raw, y_raw, perc_poison)
    print(y_poisoned_raw)
    x_train, y_train = preprocess(x_poisoned_raw, y_poisoned_raw)
    print(y_train)
    # Add channel axis:
    #x_train = np.expand_dims(x_train, axis=3)

    # Poison test data
    (is_poison_test, x_poisoned_raw_test, y_poisoned_raw_test) = generate_backdoor(x_raw_test, y_raw_test, perc_poison)
    x_test, y_test = preprocess(x_poisoned_raw_test, y_poisoned_raw_test)
    # Add channel axis:
    #x_test = np.expand_dims(x_test, axis=3)

    # Shuffle training data so poison is not together
    n_train = np.shape(y_train)[0]
    shuffled_indices = np.arange(n_train)
    np.random.shuffle(shuffled_indices)
    x_train = x_train[shuffled_indices]
    y_train = y_train[shuffled_indices]
    is_poison_train = is_poison_train[shuffled_indices]

    '''
    # Create Keras convolutional neural network - basic architecture from Keras examples
    # Source here: https://github.com/keras-team/keras/blob/master/examples/mnist_cnn.py
    model = Sequential()
    model.add(Conv2D(32, kernel_size=(3, 3), activation="relu", input_shape=x_train.shape[1:]))
    model.add(Conv2D(64, (3, 3), activation="relu"))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.25))
    model.add(Flatten())
    model.add(Dense(128, activation="relu"))
    model.add(Dropout(0.5))
    model.add(Dense(10, activation="softmax"))
    model.compile(loss="categorical_crossentropy", optimizer="adam", metrics=["accuracy"])
    '''

    model = PreActResNet18_()
    opt = torch.optim.SGD(model.parameters(), lr=0.01, momentum=0.9, weight_decay=5e-4)
    #criterion = nn.CrossEntropyLoss()
    criterion = nn.NLLLoss()

    cifar_mu = np.ones((3, 32, 32))
    cifar_mu[0, :, :] = 0.4914
    cifar_mu[1, :, :] = 0.4822
    cifar_mu[2, :, :] = 0.4465


    cifar_std = np.ones((3, 32, 32))
    cifar_std[0, :, :] = 0.2471
    cifar_std[1, :, :] = 0.2435
    cifar_std[2, :, :] = 0.2616



    #classifier = KerasClassifier(model=model, clip_values=(min_, max_))


    classifier = PyTorchClassifier(
            model=model,
            clip_values=(0.0, 1.0),
            preprocessing=(cifar_mu, cifar_std),
            loss=criterion,
            optimizer=opt,
            input_shape=(3, 32, 32),
            nb_classes=10,
            )


    x_train = x_train.transpose(0, 3, 1, 2).astype("float32")
    x_test = x_test.transpose(0, 3, 1, 2).astype("float32")    

    classifier.fit(x_train, y_train, nb_epochs=20, batch_size=128)

    # Evaluate the classifier on the test set
    preds = np.argmax(classifier.predict(x_test), axis=1)
    acc = np.sum(preds == np.argmax(y_test, axis=1)) / y_test.shape[0]
    print("\nTest accuracy: %.2f%%" % (acc * 100))

    # Evaluate the classifier on poisonous data
    preds = np.argmax(classifier.predict(x_test[is_poison_test]), axis=1)
    acc = np.sum(preds == np.argmax(y_test[is_poison_test], axis=1)) / y_test[is_poison_test].shape[0]
    print("\nPoisonous test set accuracy (i.e. effectiveness of poison): %.2f%%" % (acc * 100))

    # Evaluate the classifier on clean data
    preds = np.argmax(classifier.predict(x_test[is_poison_test == 0]), axis=1)
    acc = np.sum(preds == np.argmax(y_test[is_poison_test == 0], axis=1)) / y_test[is_poison_test == 0].shape[0]
    print("\nClean test set accuracy: %.2f%%" % (acc * 100))

    # Calling poisoning defence:
    #defence = ActivationDefence(classifier, x_train, y_train)
    defence = SpectralSignatureDefense(classifier, x_train, y_train)

    # End-to-end method:
    print("------------------- Results using size metric -------------------")
    #print(defence.get_params())
    #defence.detect_poison(nb_clusters=2, nb_dims=10, reduce="PCA")
    defence.detect_poison(expected_pp_poison=0.05, eps_multiplier=1.5)
    pp = defence.get_params()
    print(pp['eps_multiplier'])

    # Evaluate method when ground truth is known:
    is_clean = is_poison_train == 0
    confusion_matrix = defence.evaluate_defence(is_clean)
    print("Evaluation defence results for size-based metric: ")
    jsonObject = json.loads(confusion_matrix)
    for label in jsonObject:
        print(label)
        pprint.pprint(jsonObject[label])

    # Visualize clusters:
    print("Visualize clusters")
    sprites_by_class = defence.visualize_clusters(x_train, "mnist_poison_demo")
    # Show plots for clusters of class 5
    n_class = 5
    try:
        import matplotlib.pyplot as plt

        plt.imshow(sprites_by_class[n_class][0])
        plt.title("Class " + str(n_class) + " cluster: 0")
        plt.show()
        plt.imshow(sprites_by_class[n_class][1])
        plt.title("Class " + str(n_class) + " cluster: 1")
        plt.show()
    except ImportError:
        print("matplotlib not installed. For this reason, cluster visualization was not displayed")

    # Try again using distance analysis this time:
    print("------------------- Results using distance metric -------------------")
    print(defence.get_params())
    defence.detect_poison(nb_clusters=2, nb_dims=10, reduce="PCA", cluster_analysis="distance")
    confusion_matrix = defence.evaluate_defence(is_clean)
    print("Evaluation defence results for distance-based metric: ")
    jsonObject = json.loads(confusion_matrix)
    for label in jsonObject:
        print(label)
        pprint.pprint(jsonObject[label])

    # Other ways to invoke the defence:
    kwargs = {"nb_clusters": 2, "nb_dims": 10, "reduce": "PCA"}
    defence.cluster_activations(**kwargs)

    kwargs = {"cluster_analysis": "distance"}
    defence.analyze_clusters(**kwargs)
    defence.evaluate_defence(is_clean)

    kwargs = {"cluster_analysis": "smaller"}
    defence.analyze_clusters(**kwargs)
    defence.evaluate_defence(is_clean)

    print("done :) ")


def generate_backdoor(
    #x_clean, y_clean, percent_poison, backdoor_type="pattern", sources=np.arange(10), targets=(np.arange(10) + 1) % 10
    x_clean, y_clean, percent_poison, backdoor_type="pattern", sources=np.arange(10), targets=[1, 1, 1, 1, 1, 1, 1, 1, 1, 1]
    #x_clean, y_clean, percent_poison, backdoor_type="pixel", sources=np.arange(10), targets=(np.arange(10) + 1) % 10
):
    """
    Creates a backdoor in MNIST images by adding a pattern or pixel to the image and changing the label to a targeted
    class. Default parameters poison each digit so that it gets classified to the next digit.

    :param x_clean: Original raw data
    :type x_clean: `np.ndarray`
    :param y_clean: Original labels
    :type y_clean:`np.ndarray`
    :param percent_poison: After poisoning, the target class should contain this percentage of poison
    :type percent_poison: `float`
    :param backdoor_type: Backdoor type can be `pixel` or `pattern`.
    :type backdoor_type: `str`
    :param sources: Array that holds the source classes for each backdoor. Poison is
    generating by taking images from the source class, adding the backdoor trigger, and labeling as the target class.
    Poisonous images from sources[i] will be labeled as targets[i].
    :type sources: `np.ndarray`
    :param targets: This array holds the target classes for each backdoor. Poisonous images from sources[i] will be
                    labeled as targets[i].
    :type targets: `np.ndarray`
    :return: Returns is_poison, which is a boolean array indicating which points are poisonous, x_poison, which
    contains all of the data both legitimate and poisoned, and y_poison, which contains all of the labels
    both legitimate and poisoned.
    :rtype: `tuple`
    """

    max_val = np.max(x_clean)

    x_poison = np.copy(x_clean)
    y_poison = np.copy(y_clean)
    is_poison = np.zeros(np.shape(y_poison))

    print('---')
    print(sources)
    print(targets)
    print(x_clean.shape)
    print(y_clean.shape)
    for i, (src, tgt) in enumerate(zip(sources, targets)):
        n_points_in_tgt = np.size(np.where(y_clean == tgt))
        num_poison = round((percent_poison * n_points_in_tgt) / (1 - percent_poison))
        src_imgs = x_clean[y_clean == src]

        n_points_in_src = np.shape(src_imgs)[0]
        indices_to_be_poisoned = np.random.choice(n_points_in_src, num_poison)

        imgs_to_be_poisoned = np.copy(src_imgs[indices_to_be_poisoned])
        if backdoor_type == "pattern":
            imgs_to_be_poisoned = add_pattern_bd(x=imgs_to_be_poisoned, pixel_value=max_val)
        elif backdoor_type == "pixel":
            imgs_to_be_poisoned = add_single_bd(imgs_to_be_poisoned, pixel_value=max_val)
        x_poison = np.append(x_poison, imgs_to_be_poisoned, axis=0)
        y_poison = np.append(y_poison, np.ones(num_poison) * tgt, axis=0)
        is_poison = np.append(is_poison, np.ones(num_poison))

    is_poison = is_poison != 0

    return is_poison, x_poison, y_poison


if __name__ == "__main__":
    main()
