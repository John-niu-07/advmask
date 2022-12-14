"""
The script demonstrates a simple example of using ART with PyTorch. The example train a small model on the MNIST dataset
and creates adversarial examples using the Fast Gradient Sign Method. Here we use the ART classifier to train the model,
it would also be possible to provide a pretrained model to the ART classifier.
The parameters are chosen for reduced computational requirements of the script and not optimised for accuracy.
"""
from __future__ import absolute_import, division, print_function, unicode_literals

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
import os.path
from art import config

#from art.attacks.evasion import FastGradientMethod
from art.attacks.evasion.universal_perturbation2 import UniversalPerturbation
from art.estimators.classification import PyTorchClassifier
from art.utils import load_mnist
from art.visualization import create_sprite, convert_to_rgb, save_image, plot_3d
import torchvision.transforms as transforms
from PIL import Image
from art.attacks.evasion import FastGradientMethod

import logging
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Flatten, Conv2D, MaxPooling2D, Activation, Dropout
from art.estimators.classification import KerasClassifier
from art.utils import load_dataset

'''
# Configure a logger to capture ART outputs; these are printed in console and the level of detail is set to INFO
logger = logging.getLogger()
logger.setLevel(logging.INFO)
handler = logging.StreamHandler()
formatter = logging.Formatter("[%(levelname)s] %(message)s")
handler.setFormatter(formatter)
logger.addHandler(handler)
'''


from art.preactresnet import PreActBlock, PreActBottleneck, PreActResNet, PreActResNet18, initialize_weights





# Step 0: Define the neural network model, return logits instead of activation in forward method


class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv_1 = nn.Conv2d(in_channels=1, out_channels=4, kernel_size=5, stride=1)
        self.conv_2 = nn.Conv2d(in_channels=4, out_channels=10, kernel_size=5, stride=1)
        self.fc_1 = nn.Linear(in_features=4 * 4 * 10, out_features=100)
        self.fc_2 = nn.Linear(in_features=100, out_features=10)

    def forward(self, x):
        x = F.relu(self.conv_1(x))
        x = F.max_pool2d(x, 2, 2)
        x = F.relu(self.conv_2(x))
        x = F.max_pool2d(x, 2, 2)
        x = x.view(-1, 4 * 4 * 10)
        x = F.relu(self.fc_1(x))
        x = self.fc_2(x)
        return x

'''
# Step 1: Load the MNIST dataset

(x_train, y_train), (x_test, y_test), min_pixel_value, max_pixel_value = load_mnist()

# Step 1a: Swap axes to PyTorch's NCHW format

x_train = np.transpose(x_train, (0, 3, 1, 2)).astype(np.float32)
x_test = np.transpose(x_test, (0, 3, 1, 2)).astype(np.float32)

# Step 2: Create the model

model = Net()

# Step 2a: Define the loss function and the optimizer

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.01)
'''

# Read CIFAR10 dataset
(x_train, y_train), (x_test, y_test), min_, max_ = load_dataset(str("cifar10"))
#x_train, y_train = x_train[:5000], y_train[:5000]
x_test, y_test = x_test[:500], y_test[:500]
im_shape = x_train[0].shape


# Step 3: Create the ART classifier
'''
classifier = PyTorchClassifier(
    model=model,
    clip_values=(min_pixel_value, max_pixel_value),
    loss=criterion,
    optimizer=optimizer,
    input_shape=(1, 28, 28),
    nb_classes=10,
)

# Step 4: Train the ART classifier

classifier.fit(x_train, y_train, batch_size=64, nb_epochs=3)

# Step 5: Evaluate the ART classifier on benign test examples

predictions = classifier.predict(x_test)
accuracy = np.sum(np.argmax(predictions, axis=1) == np.argmax(y_test, axis=1)) / len(y_test)
print("Accuracy on benign test examples: {}%".format(accuracy * 100))
'''


# Step 2: create the PyTorch model
model = PreActResNet18()
# For running on GPU replace the model with the
# model = PreActResNet18().cuda()

model.apply(initialize_weights)
model.train()

#opt = torch.optim.SGD(model.parameters(), lr=0.21, momentum=0.9, weight_decay=5e-4)
#opt = torch.optim.SGD(model.parameters(), lr=1e-2, momentum=0.9, weight_decay=5e-4)
opt = torch.optim.SGD(model.parameters(), lr=1e-2, momentum=0.9, weight_decay=5e-4)
#opt = optim.Adam(model.parameters(), lr=0.01)

# if you have apex installed, the following line should be uncommented for faster processing
# import apex.amp as amp
# model, opt = amp.initialize(model, opt, opt_level="O2", loss_scale=1.0, master_weights=False)

criterion = nn.CrossEntropyLoss()
# Step 3: Create the ART classifier




cifar_mu = np.ones((3, 32, 32))
cifar_mu[0, :, :] = 0.4914
cifar_mu[1, :, :] = 0.4822
cifar_mu[2, :, :] = 0.4465

# (0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))

cifar_std = np.ones((3, 32, 32))
cifar_std[0, :, :] = 0.2471
cifar_std[1, :, :] = 0.2435
cifar_std[2, :, :] = 0.2616

x_train = x_train.transpose(0, 3, 1, 2).astype("float32")
x_test = x_test.transpose(0, 3, 1, 2).astype("float32")

transform = transforms.Compose(
            [transforms.RandomCrop(32, padding=4), transforms.RandomHorizontalFlip(), transforms.ToTensor()]
            )

print(x_train.shape)
for idx in range(len(x_train)):
    #print(x_train[idx])
    x = Image.fromarray(((x_train[idx] * 255).round()).astype(np.uint8).transpose(1, 2, 0))
    x = transform(x)
    x_train[idx] = x
    #print(x)

print(y_train.shape)

for idx in range(len(x_test)):
    x = Image.fromarray(((x_test[idx] * 255).round()).astype(np.uint8).transpose(1, 2, 0))
    x = transform(x)
    x_test[idx] = x



'''
itm_ = x_train[0]
print(itm_.shape)
print(type(itm_))
itm_ = (itm_*255).astype(np.uint8).transpose(1, 2, 0)
print(itm_.shape)
#print(itm_)
f_name = "image_cifar10_b.png"
save_image(itm_, f_name)
'''


classifier = PyTorchClassifier(
            model=model,
            clip_values=(0.0, 1.0),
            preprocessing=(cifar_mu, cifar_std),
            loss=criterion,
            optimizer=opt,
            input_shape=(3, 32, 32),
            nb_classes=10,
            )
'''

classifier = PyTorchClassifier(
            model=model,
            clip_values=(min_, max_),
            loss=criterion,
            optimizer=opt,
            input_shape=(3, 32, 32),
            nb_classes=10,
            )
'''

classifier.fit(x_train, y_train, batch_size=64, nb_epochs=50)


#classifier = KerasClassifier(model=model, clip_values=(min_, max_))
#classifier.fit(x_train, y_train, nb_epochs=100, batch_size=128)

preds = np.argmax(classifier.predict(x_train), axis=1)
acc = np.sum(preds == np.argmax(y_train, axis=1)) / y_train.shape[0]
#logger.info("Accuracy on benign samples: %.2f%%", (acc * 100))
print("Accuracy on benign train examples: {}%".format(acc * 100))

preds = np.argmax(classifier.predict(x_test), axis=1)
acc = np.sum(preds == np.argmax(y_test, axis=1)) / y_test.shape[0]
#logger.info("Accuracy on benign samples: %.2f%%", (acc * 100))
print("Accuracy on benign test examples: {}%".format(acc * 100))


#for i in range(10):
# Step 6: Generate adversarial test examples
#attack = FastGradientMethod(estimator=classifier, eps=0.05)
#attack = UniversalPerturbation(classifier, eps=1.05, max_iter=20)
#attack = UniversalPerturbation(classifier, eps=0.5, max_iter=20)
#attack = UniversalPerturbation(classifier, eps=0.05, max_iter=20)
attack = UniversalPerturbation(classifier, attacker='pgd', eps=5, max_iter=1)
x_test_adv = attack.generate(x=x_test)





print('visualizing...')
print(len(x_test_adv))
i=0
#for itm in x_test:
for itm in x_test_adv:
    i=i+1
    #print(np.shape(itm))

    #itm_ = (itm[0]*255).astype(np.uint8)
    #itm_ = (itm*255).astype(np.uint8)
    itm_ = (itm*255).astype(np.uint8).transpose(1, 2, 0)
    #print(itm_)
    #print(np.shape(itm_))
    #x_rgb = convert_to_rgb(itm_)
    f_name = "image_"+str(i)+".png"
    save_image(itm_, f_name)
    path = os.path.join(config.ART_DATA_PATH, f_name)
    #self.assertTrue(os.path.isfile(path))
    #os.remove(path)
    #print(path)
    if i>10:
        break

'''
# Step 7: Evaluate the ART classifier on adversarial test examples
predictions = classifier.predict(x_test_adv)
accuracy = np.sum(np.argmax(predictions, axis=1) == np.argmax(y_test, axis=1)) / len(y_test)
print("Accuracy on adversarial test examples: {}%".format(accuracy * 100))
'''

# Evaluate the adversarially trained classifier on the test set
preds = np.argmax(classifier.predict(x_test_adv), axis=1)
acc = np.sum(preds == np.argmax(y_test, axis=1)) / y_test.shape[0]
#logger.info("Accuracy on adversarial samples: %.2f%%", (acc * 100))
print("Accuracy on adversarial test examples: {}%".format(acc * 100))
