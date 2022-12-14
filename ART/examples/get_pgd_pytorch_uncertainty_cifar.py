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

from art.visualization import create_sprite, convert_to_rgb, save_image, plot_3d
import os.path
from art import config
import torchvision.transforms as transforms
from PIL import Image

from art.attacks.evasion import FastGradientMethod
from art.estimators.classification import PyTorchClassifier
from art.utils import load_mnist
from art.utils import load_dataset

from art.defences.trainer import AdversarialTrainerMadryPGD
from art.attacks.evasion.projected_gradient_descent.projected_gradient_descent import ProjectedGradientDescent
from art.attacks.evasion import FastGradientMethod
from art.defences.trainer import AdversarialTrainerMadryPGDUncertainty
from art.preactresnet import PreActBlock, PreActBottleneck, PreActResNet, PreActResNet18, initialize_weights, PreActResNet18_

# Step 0: Define the neural network model, return logits instead of activation in forward method

class ResNetSimCLR_ft(nn.Module):

    def __init__(self, base_model, out_dim):
        super(ResNetSimCLR_ft, self).__init__()

        self.backbone = PreActResNet18()
        dim_mlp = self.backbone.linear.in_features
        self.backbone.linear = nn.Linear(in_features=dim_mlp, out_features=10)

    def forward(self, x):
        x = self.backbone(x)
        x = self.backbone.linear(x)
        x = F.log_softmax(x)
        return x


class ResNetSimCLR_ft2(nn.Module):

    def __init__(self, base_model, out_dim):
        super(ResNetSimCLR_ft2, self).__init__()

        self.backbone = PreActResNet18()
        dim_mlp = self.backbone.linear.in_features
        self.backbone.linear = nn.Linear(in_features=dim_mlp, out_features=10)

    def forward(self, x):
        x = self.backbone(x)
        x = self.backbone.linear(x)
        x = F.log_softmax(x)
        return x


class ResNetSimCLR_ft3(nn.Module):

    def __init__(self, base_model, out_dim):
        super(ResNetSimCLR_ft3, self).__init__()

        self.backbone = PreActResNet18()
        dim_mlp = self.backbone.linear.in_features
        self.backbone.linear = nn.Linear(in_features=dim_mlp, out_features=10)

    def forward(self, x):
        x = self.backbone(x)
        x = self.backbone.linear(x)
        x = F.log_softmax(x)
        return x


class ResNetSimCLR_ft4(nn.Module):

    def __init__(self, base_model, out_dim):
        super(ResNetSimCLR_ft4, self).__init__()

        self.backbone = PreActResNet18()
        dim_mlp = self.backbone.linear.in_features
        self.backbone.linear = nn.Linear(in_features=dim_mlp, out_features=10)

    def forward(self, x):
        x = self.backbone(x)
        x = self.backbone.linear(x)
        x = F.log_softmax(x)
        return x


class ResNetSimCLR_ft5(nn.Module):

    def __init__(self, base_model, out_dim):
        super(ResNetSimCLR_ft5, self).__init__()

        self.backbone = PreActResNet18()
        dim_mlp = self.backbone.linear.in_features
        self.backbone.linear = nn.Linear(in_features=dim_mlp, out_features=10)

    def forward(self, x):
        x = self.backbone(x)
        x = self.backbone.linear(x)
        x = F.log_softmax(x)
        return x


class ResNetSimCLR_ft6(nn.Module):

    def __init__(self, base_model, out_dim):
        super(ResNetSimCLR_ft6, self).__init__()

        self.backbone = PreActResNet18()
        dim_mlp = self.backbone.linear.in_features
        self.backbone.linear = nn.Linear(in_features=dim_mlp, out_features=10)

    def forward(self, x):
        x = self.backbone(x)
        x = self.backbone.linear(x)
        x = F.log_softmax(x)
        return x


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
        (var, mean) = torch.var_mean(input, 1, unbiased=False, keepdim=True)
        #print(np.shape(var))
        #print(var)

        (var2, mean2) = torch.var_mean(var, unbiased=False)
        #print(mean2)
        return -1 * mean2
        ##return  mean2
        #return F.nll_loss(input, target, reduction='mean')


print('              ')
print('              ')

# Step 1: Load the MNIST dataset

#(x_train, y_train), (x_test, y_test), min_pixel_value, max_pixel_value = load_mnist()
#x_train = np.transpose(x_train, (0, 3, 1, 2)).astype(np.float32)
#x_test = np.transpose(x_test, (0, 3, 1, 2)).astype(np.float32)




# Read CIFAR10 dataset
(x_train, y_train), (x_test, y_test), min_, max_ = load_dataset(str("cifar10"))
#x_train, y_train = x_train[:5000], y_train[:5000]
#x_test, y_test = x_test[:500], y_test[:500]
im_shape = x_train[0].shape



cifar_mu = np.ones((3, 32, 32))
cifar_mu[0, :, :] = 0.4914
cifar_mu[1, :, :] = 0.4822
cifar_mu[2, :, :] = 0.4465


cifar_std = np.ones((3, 32, 32))
cifar_std[0, :, :] = 0.2471
cifar_std[1, :, :] = 0.2435
cifar_std[2, :, :] = 0.2616


transform0 = transforms.Compose(
            #[transforms.RandomCrop(32, padding=4), transforms.RandomHorizontalFlip(), transforms.ToTensor()]
            #[transforms.ToTensor(), transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2471, 0.2435, 0.2616))]
            [transforms.ToTensor(), transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))]
            #[transforms.ToTensor(), transforms.Normalize((0.1, 0.2, 0.3), (1, 1, 1))]
            )

x_train_ = x_train.copy()
x_train_ = x_train_.transpose(0, 3, 1, 2).astype("float32")

print(x_train.shape)
for idx in range(len(x_train)):
    #print(x_train[idx].shape)
    x = transform0(x_train[idx])
    #x = np.clip(x, 0.0, 1.0)
    #print(x.shape)
    x_train_[idx] = x

print(x_train_.shape)
#print(x_train[0])
#print(x_train_[0].transpose(1,2,0))


x_test_ = x_test.copy()
x_test_ = x_test_.transpose(0, 3, 1, 2).astype("float32")
for idx in range(len(x_test)):
    x = transform0(x_test[idx])
    x_test_[idx] = x
x_test = x_test_


x_train = x_train_

print(x_train.min())
print(x_train.max())



'''
#### My D

modelDU = Net()
criterionDU = nn.NLLLoss()
optimizerDU = optim.Adam(modelDU.parameters(), lr=0.01)

classifierDU = PyTorchClassifier(
    model=modelDU,
    clip_values=(min_pixel_value, max_pixel_value),
    loss=criterionDU,
    optimizer=optimizerDU,
    input_shape=(1, 28, 28),
    nb_classes=10,
)


#trainer = AdversarialTrainerMadryPGDUncertainty(classifierDU, nb_epochs=20, batch_size=64, eps= 48.0 /255.0, eps_step = 2.0/255.0)
trainer = AdversarialTrainerMadryPGDUncertainty(classifierDU, nb_epochs=20, batch_size=64, eps= 28.0 /255.0, eps_step = 2.0/255.0)
trainer.fit(x_train, y_train)
predictions = classifierDU.predict(x_test)
accuracy = np.sum(np.argmax(predictions, axis=1) == np.argmax(y_test, axis=1)) / len(y_test)
print("D for Uncetianty attack: acc on benign: {}%".format(accuracy * 100))

print("---------------Attack Defended model again----")
model_trained = copy.deepcopy(classifierDU.model)

modelU2 = Net2()
criterionU2 = MyLoss()
optimizerU2 = optim.Adam(modelU2.parameters(), lr=0.01)
modelU2.load_state_dict(model_trained.state_dict())


classifierU2 = PyTorchClassifier(
    model=modelU2,
    clip_values=(min_pixel_value, max_pixel_value),
    loss=criterionU2,
    optimizer=optimizerU2,
    input_shape=(1, 28, 28),
    nb_classes=10,
)


attackU2 = ProjectedGradientDescent(classifierU2, norm=np.inf, eps=48.0 / 255.0, eps_step=2.0 / 255.0, max_iter=40, targeted=False, num_random_init=5, batch_size=32)
x_test_advU2 = attackU2.generate(x=x_test)




predictions = classifierDU.predict(x_test_advU2)

maxv = np.max(np.exp(predictions), axis=1)
print(maxv[0:10])
mean2 = np.mean(maxv)
print(mean2)

accuracy = np.sum(np.argmax(predictions, axis=1) == np.argmax(y_test, axis=1)) / len(y_test)
print("Attack against D, acc on advU2: {}%".format(accuracy * 100))


print(err)
'''


# Step 2: Create the model

#model = Net()
model = ResNetSimCLR_ft(base_model=None, out_dim=128)

# Step 2a: Define the loss function and the optimizer

#criterion = nn.CrossEntropyLoss()
criterion = nn.NLLLoss()
#criterion = MyLoss()
#optimizer = optim.Adam(model.parameters(), lr=0.01)
optimizer = torch.optim.SGD(model.parameters(), lr=0.01, momentum=0.9, weight_decay=5e-4)

# Step 3: Create the ART classifier

classifier = PyTorchClassifier(
    model=model,
    clip_values=(-3.0, 3.0),
    loss=criterion,
    optimizer=optimizer,
    input_shape=(3, 32, 32),
    nb_classes=10,
)

# Step 4: Train the ART classifier

classifier.fit(x_train, y_train, batch_size=64, nb_epochs=30)

# Step 5: Evaluate the ART classifier on benign test examples

predictions = classifier.predict(x_test)
accuracy = np.sum(np.argmax(predictions, axis=1) == np.argmax(y_test, axis=1)) / len(y_test)
print("Acc on benign: {}%".format(accuracy * 100))


if 0:
    attack = FastGradientMethod(estimator=classifier, eps=0.3)
    #attack = FastGradientMethod(estimator=classifier, eps=1.0)
else:
    attack = ProjectedGradientDescent(classifier, norm=np.inf, eps=8.0 / 255.0, eps_step=2.0 / 255.0, max_iter=40, targeted=False, num_random_init=5, batch_size=64)


x_test_adv = attack.generate(x=x_test)
predictions = classifier.predict(x_test_adv)
accuracy = np.sum(np.argmax(predictions, axis=1) == np.argmax(y_test, axis=1)) / len(y_test)
print("Acc on adv: {}%".format(accuracy * 100))




print("----------------------------Madry defense----")
#model_trained = copy.deepcopy(classifier.model)
#modelD = Net2()
modelD = ResNetSimCLR_ft2(base_model=None, out_dim=128)
criterionD = nn.NLLLoss()
#optimizerD = optim.Adam(modelD.parameters(), lr=0.01)
optimizerD = torch.optim.SGD(modelD.parameters(), lr=0.01, momentum=0.9, weight_decay=5e-4)

classifierD = PyTorchClassifier(
    model=modelD,
    clip_values=(-3.0, 3.0),
    loss=criterionD,
    optimizer=optimizerD,
    input_shape=(3, 32, 32),
    nb_classes=10,
)


trainer = AdversarialTrainerMadryPGD(classifierD, nb_epochs=20, batch_size=64, eps= 8.0 /255.0, eps_step = 2.0/255.0)
trainer.fit(x_train, y_train)
predictions = classifierD.predict(x_test)
accuracy = np.sum(np.argmax(predictions, axis=1) == np.argmax(y_test, axis=1)) / len(y_test)
print("Madry: acc on benign examples: {}%".format(accuracy * 100))


if 0:
    attackD = FastGradientMethod(estimator=classifierD, eps=0.3)
    #attackD = FastGradientMethod(estimator=classifierD, eps=1.0)
else:
    attackD = ProjectedGradientDescent(classifierD, norm=np.inf, eps=8.0 / 255.0, eps_step=2.0 / 255.0, max_iter=40, targeted=False, num_random_init=5, batch_size=64)

x_test_advD = attackD.generate(x=x_test)
predictions = classifierD.predict(x_test_advD)
accuracy = np.sum(np.argmax(predictions, axis=1) == np.argmax(y_test, axis=1)) / len(y_test)
print("Madry: acc on adv: {}%".format(accuracy * 100))




print("------------------========= Uncertainty Attack =============----")
if 1:
    print("--get Uncertainty Attack Examples w.r.t raw-model ----")
    model_trained = copy.deepcopy(classifier.model)



    #modelU = Net3()
    modelU = ResNetSimCLR_ft3(base_model=None, out_dim=128)
    criterionU = MyLoss()
    ##optimizerU = optim.Adam(modelU.parameters(), lr=0.01)
    optimizerU = torch.optim.SGD(modelU.parameters(), lr=0.01, momentum=0.9, weight_decay=5e-4)
    modelU.load_state_dict(model_trained.state_dict())


    classifierU = PyTorchClassifier(
    model=modelU,
    clip_values=(-3.0, 3.0),
    loss=criterionU,
    optimizer=optimizerU,
    input_shape=(3, 32, 32),
    nb_classes=10,
    )


    if 0:
        #attackU = FastGradientMethod(estimator=classifierU, eps=1.0)
        attackU = FastGradientMethod(estimator=classifierU, eps=0.3)
    else:
        attackU = ProjectedGradientDescent(classifierU, norm=np.inf, eps=8.0 / 255.0, eps_step=2.0 / 255.0, max_iter=40, targeted=False, num_random_init=5, batch_size=64)

    x_test_advU = attackU.generate(x=x_test)



    print("--For a raw-model (without defense), try Uncertainty Attack Examples----")
    predictions = classifier.predict(x_test_advU)
    maxv = np.max(np.exp(predictions), axis=1)
    print(maxv[0:10])
    mean2 = np.mean(maxv)
    print(mean2)
    accuracy = np.sum(np.argmax(predictions, axis=1) == np.argmax(y_test, axis=1)) / len(y_test)
    print("Acc on advU: {}%".format(accuracy * 100))


    '''
    print("--For a MadryDefensive model, try Uncertainty Attack Examples----")
    predictions = classifierD.predict(x_test_advU)

    maxv = np.max(np.exp(predictions), axis=1)
    print(maxv[0:10])
    mean2 = np.mean(maxv)
    print(mean2)

    accuracy = np.sum(np.argmax(predictions, axis=1) == np.argmax(y_test, axis=1)) / len(y_test)
    print("Acc on advU: {}%".format(accuracy * 100))
    '''


if 1:
    print("--get Uncertainty Attack Examples w.r.t MadryDefensive model ----")
    model_trained = copy.deepcopy(classifierD.model)

    #modelU = Net3()
    modelU = ResNetSimCLR_ft3(base_model=None, out_dim=128)
    criterionU = MyLoss()
    ###########optimizerU = optim.Adam(modelU.parameters(), lr=0.01)
    optimizerU = torch.optim.SGD(modelU.parameters(), lr=0.01, momentum=0.9, weight_decay=5e-4)
    modelU.load_state_dict(model_trained.state_dict())


    classifierU = PyTorchClassifier(
    model=modelU,
    clip_values=(-3.0, 3.0),
    loss=criterionU,
    optimizer=optimizerU,
    input_shape=(3, 32, 32),
    nb_classes=10,
    )


    if 0:
        #attackU = FastGradientMethod(estimator=classifierU, eps=1.0)
        attackU = FastGradientMethod(estimator=classifierU, eps=0.3)
    else:
        attackU = ProjectedGradientDescent(classifierU, norm=np.inf, eps=8.0 / 255.0, eps_step=2.0 / 255.0, max_iter=40, targeted=False, num_random_init=5, batch_size=64)

    x_test_advU = attackU.generate(x=x_test)


    '''
    print("--For a raw-model (without defense), try Uncertainty Attack Examples----")
    predictions = classifier.predict(x_test_advU)
    maxv = np.max(np.exp(predictions), axis=1)
    print(maxv[0:10])
    mean2 = np.mean(maxv)
    print(mean2)
    accuracy = np.sum(np.argmax(predictions, axis=1) == np.argmax(y_test, axis=1)) / len(y_test)
    print("Acc on advU: {}%".format(accuracy * 100))
    '''


    print("--For a MadryDefensive model, try Uncertainty Attack Examples----")
    predictions = classifierD.predict(x_test_advU)

    maxv = np.max(np.exp(predictions), axis=1)
    print(maxv[0:10])
    mean2 = np.mean(maxv)
    print(mean2)

    accuracy = np.sum(np.argmax(predictions, axis=1) == np.argmax(y_test, axis=1)) / len(y_test)
    print("Acc on advU: {}%".format(accuracy * 100))










print("------------------========= Our defense method i.e., Uncertainty Defense =============----")
'''
x_train_advU = attackU.generate(x=x_train)

x_train_advU_ = np.vstack((x_train, x_train_advU))
print(x_train_advU.shape)
y_train_ = np.vstack((y_train, y_train))
print(y_train_.shape)



modelDU = Net()
criterionDU = nn.NLLLoss()
optimizerDU = optim.Adam(modelDU.parameters(), lr=0.01)

classifierDU = PyTorchClassifier(
    model=modelDU,
    clip_values=(min_pixel_value, max_pixel_value),
    loss=criterionDU,
    optimizer=optimizerDU,
    input_shape=(1, 28, 28),
    nb_classes=10,
)



classifierDU.fit_predict(x_train_advU_, y_train_, x_test, y_test, batch_size=64, nb_epochs=11)
predictions = classifierDU.predict(x_test)
accuracy = np.sum(np.argmax(predictions, axis=1) == np.argmax(y_test, axis=1)) / len(y_test)
print("D for Uncetianty attack: acc on benign: {}%".format(accuracy * 100))


predictions = classifierDU.predict(x_test_advU)
accuracy = np.sum(np.argmax(predictions, axis=1) == np.argmax(y_test, axis=1)) / len(y_test)
print("D for Uncetianty attack: acc on advU: {}%".format(accuracy * 100))
'''


#### My D

#modelDU = Net4()
modelDU = ResNetSimCLR_ft4(base_model=None, out_dim=128)
criterionDU = nn.NLLLoss()
optimizerDU = optim.Adam(modelDU.parameters(), lr=0.01)
optimizer = torch.optim.SGD(model.parameters(), lr=0.01, momentum=0.9, weight_decay=5e-4)

classifierDU = PyTorchClassifier(
    model=modelDU,
    clip_values=(-3.0, 3.0),
    loss=criterionDU,
    optimizer=optimizerDU,
    input_shape=(3, 32, 32),
    nb_classes=10,
)


trainer = AdversarialTrainerMadryPGDUncertainty(classifierDU, nb_epochs=20, batch_size=64, eps= 8.0 /255.0, eps_step = 2.0/255.0)
trainer.fit(x_train, y_train)
predictions = classifierDU.predict(x_test)
accuracy = np.sum(np.argmax(predictions, axis=1) == np.argmax(y_test, axis=1)) / len(y_test)
print("Uncertainty Defense: acc on benign: {}%".format(accuracy * 100))


print("---------------Adv Attack against our Uncertainty Defense ----")
print("   ------------Our Defense is better than Madry Defense ----")
attackDU = ProjectedGradientDescent(classifierDU, norm=np.inf, eps=16.0 / 255.0, eps_step=2.0 / 255.0, max_iter=40, targeted=False, num_random_init=5, batch_size=64)
x_test_advDU = attackDU.generate(x=x_test)
predictions = classifierDU.predict(x_test_advDU)
accuracy = np.sum(np.argmax(predictions, axis=1) == np.argmax(y_test, axis=1)) / len(y_test)
print("Adv Attack against Uncertainty Defense, acc on advDU: {}%".format(accuracy * 100))





print("---------------Uncertainty Attack against our Uncertainty Defense ----")
model_trained = copy.deepcopy(classifierDU.model)

#modelU2 = Net5()
modelU2 = ResNetSimCLR_ft5(base_model=None, out_dim=128)
criterionU2 = MyLoss()
#optimizerU2 = optim.Adam(modelU2.parameters(), lr=0.01)
optimizerU2 = torch.optim.SGD(modelU2.parameters(), lr=0.01, momentum=0.9, weight_decay=5e-4)
modelU2.load_state_dict(model_trained.state_dict())


classifierU2 = PyTorchClassifier(
    model=modelU2,
    clip_values=(-3.0, 3.0),
    loss=criterionU2,
    optimizer=optimizerU2,
    input_shape=(3, 32, 32),
    nb_classes=10,
)


attackU2 = ProjectedGradientDescent(classifierU2, norm=np.inf, eps=8.0 / 255.0, eps_step=2.0 / 255.0, max_iter=40, targeted=False, num_random_init=5, batch_size=64)
x_test_advU2 = attackU2.generate(x=x_test)




predictions = classifierDU.predict(x_test_advU2)

maxv = np.max(np.exp(predictions), axis=1)
print(maxv[0:10])
mean2 = np.mean(maxv)
print(mean2)

accuracy = np.sum(np.argmax(predictions, axis=1) == np.argmax(y_test, axis=1)) / len(y_test)
print("Uncertainty Attack against Uncertainty Defense, acc on advU2: {}%".format(accuracy * 100))




print("---------------Adv Attack against our Uncertainty Defense ----")
print("   ------------Our Defense is better than Madry Defense ----")
model_trained = copy.deepcopy(classifierDU.model)

#modelU3 = Net6()
modelU3 = ResNetSimCLR_ft6(base_model=None, out_dim=128)
criterionU3 = nn.NLLLoss()
#optimizerU3 = optim.Adam(modelU3.parameters(), lr=0.01)
optimizerU3 = torch.optim.SGD(modelU3.parameters(), lr=0.01, momentum=0.9, weight_decay=5e-4)
modelU3.load_state_dict(model_trained.state_dict())


classifierU3 = PyTorchClassifier(
    model=modelU3,
    clip_values=(-3.0, 3.0),
    loss=criterionU3,
    optimizer=optimizerU3,
    input_shape=(3, 32, 32),
    nb_classes=10,
)


attackU3 = ProjectedGradientDescent(classifierU3, norm=np.inf, eps=8.0 / 255.0, eps_step=2.0 / 255.0, max_iter=40, targeted=False, num_random_init=5, batch_size=64)
x_test_advU3 = attackU3.generate(x=x_test)




predictions = classifierDU.predict(x_test_advU3)
accuracy = np.sum(np.argmax(predictions, axis=1) == np.argmax(y_test, axis=1)) / len(y_test)
print("Adv Attack against Uncertainty Defense, acc on advU3: {}%".format(accuracy * 100))


print("--------over---------")







'''
print('visualizing...')
print(len(x_test_adv))
i=0
#for itm in x_test:
for itm in x_test_adv:
    i=i+1
    #print(np.shape(itm))

    itm_ = (itm[0]*255).astype(np.uint8)
    #itm_ = (itm*255).astype(np.uint8)
    #itm_ = (itm*255).astype(np.uint8).transpose(1, 2, 0)
    #print(itm_)
    #print(np.shape(itm_))
    #x_rgb = convert_to_rgb(itm_)
    f_name = "uncertainty_FGM_image_"+str(i)+".png"
    save_image(itm_, f_name)
    path = os.path.join(config.ART_DATA_PATH, f_name)
    #self.assertTrue(os.path.isfile(path))
    #os.remove(path)
    #print(path)
    if i>10:
        break
'''

