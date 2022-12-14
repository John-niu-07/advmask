"""
The script demonstrates a simple example of using ART with PyTorch. The example train a small model on the MNIST dataset
and creates adversarial examples using the Fast Gradient Sign Method. Here we use the ART classifier to train the model,
it would also be possible to provide a pretrained model to the ART classifier.
The parameters are chosen for reduced computational requirements of the script and not optimised for accuracy.
"""
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
from torch import Tensor
from torch import torch
import copy

from art.attacks.evasion.projected_gradient_descent.projected_gradient_descent import ProjectedGradientDescent
from art.attacks.evasion import FastGradientMethod
from art.estimators.classification import PyTorchClassifier
from art.utils import load_mnist
from art.utils import load_dataset
from art.preactresnet import PreActBlock, PreActBottleneck, PreActResNet, PreActResNet18, initialize_weights, PreActResNet18_
import torchvision.transforms as transforms
from PIL import Image

import numpy as np

def softmax(x, axis):
        """
        x --- 一个二维矩阵, m * n,其中m表示向量个数，n表示向量维度
        """
        assert(len(x.shape) == 2)
        row_max = np.max(x, axis=axis).reshape(-1, 1)
        x -= row_max
        x_exp = np.exp(x)
        s = x_exp / np.sum(x_exp, axis=axis, keepdims=True)

        return s

# Step 0: Define the neural network model, return logits instead of activation in forward method

# Read CIFAR10 dataset
(x_train, y_train), (x_test, y_test), min_, max_ = load_dataset(str("cifar10"))
#x_train, y_train = x_train[:5000], y_train[:5000]
x_test, y_test = x_test[:500], y_test[:500]
im_shape = x_train[0].shape


# Step 2: create the PyTorch model
#model = PreActResNet18()
model = PreActResNet18_()
model.apply(initialize_weights)
model.train()
#opt = optim.Adam(model.parameters(), lr=0.01)
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


print(x_test.shape)
for idx in range(len(x_test)):
    #print(x_train[idx])
    x = Image.fromarray(((x_test[idx] * 255).round()).astype(np.uint8).transpose(1, 2, 0))
    x = transform(x)
    x_test[idx] = x
    #print(x)

print(y_test.shape)



class _Loss(nn.Module):
    reduction: str

    def __init__(self, size_average=None, reduce=None, reduction: str = 'mean') -> None:
        super(_Loss, self).__init__()
        if size_average is not None or reduce is not None:
            self.reduction: str = _Reduction.legacy_get_string(size_average, reduce)


class _Loss(nn.Module):
    reduction: str

    def __init__(self, size_average=None, reduce=None, reduction: str = 'mean') -> None:
        super(_Loss, self).__init__()
        if size_average is not None or reduce is not None:
            self.reduction: str = _Reduction.legacy_get_string(size_average, reduce)
        else:
            self.reduction = reduction

class MyLoss(_Loss):
    __constants__ = ['reduction']

    def __init__(self, size_average=None, reduce=None, reduction: str = 'mean') -> None:
        super(MyLoss, self).__init__(size_average, reduce, reduction)

    #def forward(self, input: Tensor, target: Tensor) -> Tensor:
    #    return F.mse_loss(input, target, reduction=self.reduction)
    def forward(self, input: Tensor, target: Tensor) -> Tensor:
        #print(input.size())
        #print(target.size())
        #print(input[0])
        #print(target[0])
        #input2 = F.log_softmax(input)
        #input2 = F.softmax(input)
        (var, mean) = torch.var_mean(input, 1, unbiased=False, keepdim=True)
        #print(np.shape(var))
        #print(var)

        (var2, mean2) = torch.var_mean(var, unbiased=False)
        #print(mean2)
        return -1 * mean2
        #return  mean2
        #return F.nll_loss(input, target, reduction='mean')




# Step 3: Create the ART classifier
classifier = PyTorchClassifier(
            model=model,
            clip_values=(0.0, 1.0),
            preprocessing=(cifar_mu, cifar_std),
            loss=criterion,
            optimizer=opt,
            input_shape=(3, 32, 32),
            nb_classes=10,
            )


# Step 4: Train the ART classifier

classifier.fit(x_train, y_train, batch_size=64, nb_epochs=50)

# Step 5: Evaluate the ART classifier on benign test examples

predictions = classifier.predict(x_test)

print(np.exp(predictions[0:2]))
maxv = np.max(np.exp(predictions), axis=1)
print(maxv[0:10])
mean2 = np.mean(maxv)
print(mean2)

accuracy = np.sum(np.argmax(predictions, axis=1) == np.argmax(y_test, axis=1)) / len(y_test)
print("Accuracy on benign test examples: {}%".format(accuracy * 100))

#attack = FastGradientMethod(estimator=classifier, eps=0.2)
#attack = ProjectedGradientDescent(classifier, eps=1.0, eps_step=0.1, verbose=False)
#attack = ProjectedGradientDescent(classifier, eps=0.2, eps_step=0.1, verbose=False)
attack = ProjectedGradientDescent(
            classifier,
            norm=np.inf,
            eps=8.0 / 255.0,
            eps_step=2.0 / 255.0,
            max_iter=40,
            targeted=False,
            num_random_init=5,
            batch_size=32,
            )

x_test_adv = attack.generate(x=x_test)
predictions = classifier.predict(x_test_adv)

print(np.exp(predictions[0:2]))
maxv = np.max(np.exp(predictions), axis=1)
print(maxv[0:10])
mean2 = np.mean(maxv)
print(mean2)

accuracy = np.sum(np.argmax(predictions, axis=1) == np.argmax(y_test, axis=1)) / len(y_test)
print("Model2: Accuracy on adversarial test examples: {}%".format(accuracy * 100))
print("--------")
accuracy = np.sum(np.argmax(predictions, axis=1) == np.argmax(y_test, axis=1)) / len(y_test)
print("Accuracy on adversarial test examples: {}%".format(accuracy * 100))


print("----PGD----")
# Step 2: create the PyTorch model
#model2 = PreActResNet18()
model2 = PreActResNet18_()
model2.train()
#opt2 = optim.Adam(model2.parameters(), lr=0.01)
opt2 = torch.optim.SGD(model2.parameters(), lr=0.01, momentum=0.9, weight_decay=5e-4)

#model_trained = classifier.model()
model_trained = copy.deepcopy(classifier.model)
criterion2 = MyLoss()

model2.load_state_dict(model_trained.state_dict())

classifier2 = PyTorchClassifier(
            model=model2,
            clip_values=(0.0, 1.0),
            preprocessing=(cifar_mu, cifar_std),
            loss=criterion2,
            optimizer=opt2,
            input_shape=(3, 32, 32),
            nb_classes=10,
            )


predictions = classifier.predict(x_test)
accuracy = np.sum(np.argmax(predictions, axis=1) == np.argmax(y_test, axis=1)) / len(y_test)
print("Model2: Accuracy on benign test examples: {}%".format(accuracy * 100))



# Step 6: Generate adversarial test examples
#attack = FastGradientMethod(estimator=classifier2, eps=0.2)
#x_test_adv = attack.generate(x=x_test)

# Test PGD with np.inf norm
#attack = ProjectedGradientDescent(classifier2, eps=1.0, eps_step=0.1, verbose=False)
#attack = ProjectedGradientDescent(classifier2, eps=1.0, eps_step=0.1, verbose=False)
attack = ProjectedGradientDescent(
            classifier2,
            norm=np.inf,
            eps=8.0 / 255.0,
            eps_step=2.0 / 255.0,
            max_iter=40,
            targeted=False,
            num_random_init=5,
            batch_size=32,
            )

x_test_adv = attack.generate(x_test)


# Step 7: Evaluate the ART classifier on adversarial test examples

predictions = classifier.predict(x_test_adv)

#print(np.shape(predictions))
print(np.exp(predictions[0:2]))
#print(softmax(predictions[0:4], 1))

maxv = np.max(np.exp(predictions), axis=1)
#maxv = np.max(softmax(predictions, 1), axis=1)
#print(np.shape(var))
#print(var[0:10])
#print(np.shape(maxv))
print(maxv[0:10])
#mean2 = np.mean(var)
mean2 = np.mean(maxv)
print(mean2)

accuracy = np.sum(np.argmax(predictions, axis=1) == np.argmax(y_test, axis=1)) / len(y_test)
print("Model2: Accuracy on adversarial test examples: {}%".format(accuracy * 100))
print("--------")
