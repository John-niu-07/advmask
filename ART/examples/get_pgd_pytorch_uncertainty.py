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
from art.defences.trainer import AdversarialTrainerMadryPGD
from art.attacks.evasion.projected_gradient_descent.projected_gradient_descent import ProjectedGradientDescent
from art.attacks.evasion import FastGradientMethod
from art.defences.trainer import AdversarialTrainerMadryPGDUncertainty


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
        x = F.log_softmax(x)
        return x


class Net3(nn.Module):
    def __init__(self):
        super(Net3, self).__init__()
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
        x = F.log_softmax(x)
        return x

class Net2(nn.Module):
    def __init__(self):
        super(Net2, self).__init__()
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
        x = F.log_softmax(x)
        return x

class Net4(nn.Module):
    def __init__(self):
        super(Net4, self).__init__()
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
        x = F.log_softmax(x)
        return x


class Net5(nn.Module):
    def __init__(self):
        super(Net5, self).__init__()
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
        x = F.log_softmax(x)
        return x




class Net6(nn.Module):
    def __init__(self):
        super(Net6, self).__init__()
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

(x_train, y_train), (x_test, y_test), min_pixel_value, max_pixel_value = load_mnist()

# Step 1a: Swap axes to PyTorch's NCHW format

x_train = np.transpose(x_train, (0, 3, 1, 2)).astype(np.float32)
x_test = np.transpose(x_test, (0, 3, 1, 2)).astype(np.float32)

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

model = Net()

# Step 2a: Define the loss function and the optimizer

#criterion = nn.CrossEntropyLoss()
criterion = nn.NLLLoss()
#criterion = MyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.01)

# Step 3: Create the ART classifier

classifier = PyTorchClassifier(
    model=model,
    clip_values=(min_pixel_value, max_pixel_value),
    loss=criterion,
    optimizer=optimizer,
    input_shape=(1, 28, 28),
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
    attack = ProjectedGradientDescent(classifier, norm=np.inf, eps=16.0 / 255.0, eps_step=2.0 / 255.0, max_iter=40, targeted=False, num_random_init=5, batch_size=64)


x_test_adv = attack.generate(x=x_test)
predictions = classifier.predict(x_test_adv)
accuracy = np.sum(np.argmax(predictions, axis=1) == np.argmax(y_test, axis=1)) / len(y_test)
print("Acc on adv: {}%".format(accuracy * 100))




print("----------------------------Madry defense----")
#model_trained = copy.deepcopy(classifier.model)
modelD = Net2()
criterionD = nn.NLLLoss()
optimizerD = optim.Adam(modelD.parameters(), lr=0.01)

classifierD = PyTorchClassifier(
    model=modelD,
    clip_values=(min_pixel_value, max_pixel_value),
    loss=criterionD,
    optimizer=optimizerD,
    input_shape=(1, 28, 28),
    nb_classes=10,
)


trainer = AdversarialTrainerMadryPGD(classifierD, nb_epochs=20, batch_size=64, eps= 16.0 /255.0, eps_step = 2.0/255.0)
trainer.fit(x_train, y_train)
predictions = classifierD.predict(x_test)
accuracy = np.sum(np.argmax(predictions, axis=1) == np.argmax(y_test, axis=1)) / len(y_test)
print("Madry: acc on benign examples: {}%".format(accuracy * 100))


if 0:
    attackD = FastGradientMethod(estimator=classifierD, eps=0.3)
    #attackD = FastGradientMethod(estimator=classifierD, eps=1.0)
else:
    attackD = ProjectedGradientDescent(classifierD, norm=np.inf, eps=16.0 / 255.0, eps_step=2.0 / 255.0, max_iter=40, targeted=False, num_random_init=5, batch_size=64)

x_test_advD = attackD.generate(x=x_test)
predictions = classifierD.predict(x_test_advD)
accuracy = np.sum(np.argmax(predictions, axis=1) == np.argmax(y_test, axis=1)) / len(y_test)
print("Madry: acc on adv: {}%".format(accuracy * 100))




print("------------------========= Uncertainty Attack =============----")
if 1:
    print("--get Uncertainty Attack Examples w.r.t raw-model ----")
    model_trained = copy.deepcopy(classifier.model)



    modelU = Net3()
    criterionU = MyLoss()
    optimizerU = optim.Adam(modelU.parameters(), lr=0.01)
    modelU.load_state_dict(model_trained.state_dict())


    classifierU = PyTorchClassifier(
    model=modelU,
    clip_values=(min_pixel_value, max_pixel_value),
    loss=criterionU,
    optimizer=optimizerU,
    input_shape=(1, 28, 28),
    nb_classes=10,
    )


    if 0:
        #attackU = FastGradientMethod(estimator=classifierU, eps=1.0)
        attackU = FastGradientMethod(estimator=classifierU, eps=0.3)
    else:
        attackU = ProjectedGradientDescent(classifierU, norm=np.inf, eps=16.0 / 255.0, eps_step=2.0 / 255.0, max_iter=40, targeted=False, num_random_init=5, batch_size=64)

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

    modelU = Net3()
    criterionU = MyLoss()
    optimizerU = optim.Adam(modelU.parameters(), lr=0.01)
    modelU.load_state_dict(model_trained.state_dict())


    classifierU = PyTorchClassifier(
    model=modelU,
    clip_values=(min_pixel_value, max_pixel_value),
    loss=criterionU,
    optimizer=optimizerU,
    input_shape=(1, 28, 28),
    nb_classes=10,
    )


    if 0:
        #attackU = FastGradientMethod(estimator=classifierU, eps=1.0)
        attackU = FastGradientMethod(estimator=classifierU, eps=0.3)
    else:
        attackU = ProjectedGradientDescent(classifierU, norm=np.inf, eps=16.0 / 255.0, eps_step=2.0 / 255.0, max_iter=40, targeted=False, num_random_init=5, batch_size=64)

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

modelDU = Net4()
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


trainer = AdversarialTrainerMadryPGDUncertainty(classifierDU, nb_epochs=20, batch_size=64, eps= 16.0 /255.0, eps_step = 2.0/255.0)
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






modelDU_ft = Net6()
criterionDU_ft = nn.NLLLoss()
optimizerDU_ft = optim.Adam(modelDU_ft.parameters(), lr=0.001)

model_trained = copy.deepcopy(classifierDU.model)
modelDU_ft.load_state_dict(model_trained.state_dict())

classifierDU_ft = PyTorchClassifier(
    model=modelDU_ft,
    clip_values=(min_pixel_value, max_pixel_value),
    loss=criterionDU_ft,
    optimizer=optimizerDU_ft,
    input_shape=(1, 28, 28),
    nb_classes=10,
)


classifierDU_ft.fit(x_train, y_train, batch_size=64, nb_epochs=30)

predictions = classifierDU_ft.predict(x_test)
accuracy = np.sum(np.argmax(predictions, axis=1) == np.argmax(y_test, axis=1)) / len(y_test)
print("Uncertainty Defense ft: acc on benign: {}%".format(accuracy * 100))



attack = ProjectedGradientDescent(classifierDU_ft, norm=np.inf, eps=16.0 / 255.0, eps_step=2.0 / 255.0, max_iter=40, targeted=False, num_random_init=5, batch_size=64)
x_test_adv = attack.generate(x=x_test)
predictions = classifierDU_ft.predict(x_test_adv)
accuracy = np.sum(np.argmax(predictions, axis=1) == np.argmax(y_test, axis=1)) / len(y_test)
print("Acc on adv: {}%".format(accuracy * 100))





print("---------------Uncertainty Attack against our Uncertainty Defense ----")
model_trained = copy.deepcopy(classifierDU.model)

modelU2 = Net5()
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


attackU2 = ProjectedGradientDescent(classifierU2, norm=np.inf, eps=16.0 / 255.0, eps_step=2.0 / 255.0, max_iter=40, targeted=False, num_random_init=5, batch_size=64)
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

modelU3 = Net6()
criterionU3 = nn.NLLLoss()
optimizerU3 = optim.Adam(modelU3.parameters(), lr=0.01)
modelU3.load_state_dict(model_trained.state_dict())


classifierU3 = PyTorchClassifier(
    model=modelU3,
    clip_values=(min_pixel_value, max_pixel_value),
    loss=criterionU3,
    optimizer=optimizerU3,
    input_shape=(1, 28, 28),
    nb_classes=10,
)


attackU3 = ProjectedGradientDescent(classifierU3, norm=np.inf, eps=16.0 / 255.0, eps_step=2.0 / 255.0, max_iter=40, targeted=False, num_random_init=5, batch_size=64)
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

