"""
This is an example of how to use ART for adversarial training of a model with Fast is better than free protocol
"""
import math
from PIL import Image

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as transforms
from torch.utils.data import Dataset, DataLoader

#from art.classifiers import PyTorchClassifier
from art.estimators.classification import PyTorchClassifier
from art.defences.trainer import AdversarialTrainerMadryPGD
from art.utils import load_cifar10
from art.attacks.evasion import ProjectedGradientDescent
from art.utils import load_dataset

"""
For this example we choose the PreActResNet model as used in the paper (https://openreview.net/forum?id=BJx040EFvH)
The code for the model architecture has been adopted from
https://github.com/anonymous-sushi-armadillo/fast_is_better_than_free_CIFAR10/blob/master/preact_resnet.py
"""
from art.preactresnet import PreActBlock, PreActBottleneck, PreActResNet, PreActResNet18, initialize_weights, PreActResNet18_, PreActResNet34
from art.resnet import resnet18, resnet34
from art.vgg import vgg11
from art.shufflenetv2 import shufflenet_v2_x0_5,  shufflenet_v2_x1_5


def initialize_weights(module):
    if isinstance(module, nn.Conv2d):
        n = module.kernel_size[0] * module.kernel_size[1] * module.out_channels
        module.weight.data.normal_(0, math.sqrt(2.0 / n))
        if module.bias is not None:
            module.bias.data.zero_()
    elif isinstance(module, nn.BatchNorm2d):
        module.weight.data.fill_(1)
        module.bias.data.zero_()
    elif isinstance(module, nn.Linear):
        module.bias.data.zero_()


class CIFAR10_dataset(Dataset):
    def __init__(self, data, targets, transform=None):
        self.data = data
        self.targets = torch.LongTensor(targets)
        self.transform = transform

    def __getitem__(self, index):
        x = Image.fromarray(((self.data[index] * 255).round()).astype(np.uint8).transpose(1, 2, 0))
        x = self.transform(x)
        y = self.targets[index]
        return x, y

    def __len__(self):
        return len(self.data)

# Read CIFAR10 dataset
(x_train, y_train), (x_test, y_test), min_, max_ = load_dataset(str("cifar10"))
#(x_train, y_train), (x_test, y_test), min_, max_ = load_dataset(str("tiny"))

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
            [transforms.ToTensor()]
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


transform2 = transforms.Compose(
            #[transforms.RandomCrop(32, padding=4), transforms.RandomHorizontalFlip(), transforms.ToTensor()]
            [transforms.ToTensor(), transforms.RandomCrop(32, padding=4), transforms.RandomHorizontalFlip()]
            #[transforms.ToTensor()]
            )


x_train_2 = x_train_.copy()
for idx in range(len(x_train_2)):
    #print(x_train[idx])
    #x = Image.fromarray(((x_train_2[idx] * 255).round()).astype(np.uint8).transpose(1, 2, 0))
    x = x_train_2[idx].transpose(1,2,0)
    x = transform2(x)
    #x = transform(x)
    x_train_2[idx] = x



x_train = np.vstack((x_train_, x_train_2))
print(x_train.shape)
y_train = np.vstack((y_train, y_train))
print(y_train.shape)


class ResNetSimCLR_ft(nn.Module):

    def __init__(self, base_model, out_dim):
        super(ResNetSimCLR_ft, self).__init__()

        self.backbone = PreActResNet18()
        #self.backbone = PreActResNet34()

        #self.backbone = resnet18()
        #self.backbone = resnet34()

        #self.backbone = vgg11()

        #self.backbone = shufflenet_v2_x1_5()

        dim_mlp = self.backbone.linear.in_features

        # add mlp projection head
        #self.backbone.fc = nn.Sequential(nn.Linear(dim_mlp, dim_mlp), nn.ReLU(), self.backbone.fc)
        #self.backbone.linear = nn.Sequential(nn.Linear(dim_mlp, dim_mlp), nn.ReLU(), self.backbone.linear)
        self.backbone.linear = nn.Linear(in_features=dim_mlp, out_features=10)

        #self.backbone.linear =  nn.Linear(4096, 10)

    def forward(self, x):
        x = self.backbone(x)
        x = self.backbone.linear(x)
        x = F.log_softmax(x)
        #print(x.shape)
        return x


model = ResNetSimCLR_ft(base_model=None, out_dim=128)
model.train()
opt = torch.optim.SGD(model.parameters(), lr=0.01, momentum=0.9, weight_decay=5e-4)
criterion = nn.NLLLoss()
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=10)

classifier = PyTorchClassifier(
    model=model,
    clip_values=(-3.0, 3.0),
    #preprocessing=(cifar_mu, cifar_std),
    loss=criterion,
    optimizer=opt,
    scheduler=scheduler,    
    input_shape=(3, 32, 32),
    nb_classes=10,
)


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

classifier.fit_predict(x_train, y_train, x_test, y_test, batch_size=64, nb_epochs=51)

predictions = classifier.predict(x_test)
accuracy = np.sum(np.argmax(predictions, axis=1) == np.argmax(y_test, axis=1)) / len(y_test)
print("accuracy on benign test examples: {}%".format(accuracy * 100))



x_test_adv = attack.generate(x=x_test)

predictions = classifier.predict(x_test_adv)
accuracy = np.sum(np.argmax(predictions, axis=1) == np.argmax(y_test, axis=1)) / len(y_test)
print("Accuracy on test_adv : {}%".format(accuracy * 100))



print("----------------------------AT for defense---")
# Step 2: create the PyTorch model
modelD = ResNetSimCLR_ft(base_model=None, out_dim=128)
modelD.train()
optD = torch.optim.SGD(modelD.parameters(), lr=0.01, momentum=0.9, weight_decay=5e-4)
criterionD = nn.NLLLoss()
schedulerD = torch.optim.lr_scheduler.CosineAnnealingLR(optD, T_max=10)

# if you have apex installed, the following line should be uncommented for faster processing
# import apex.amp as amp
# model, opt = amp.initialize(model, opt, opt_level="O2", loss_scale=1.0, master_weights=False)

#criterion = nn.CrossEntropyLoss()
# Step 3: Create the ART classifier

classifierD = PyTorchClassifier(
    model=modelD,
    clip_values=(-3.0, 3.0),
    #preprocessing=(cifar_mu, cifar_std),
    loss=criterionD,
    optimizer=optD,
    scheduler=schedulerD,
    input_shape=(3, 32, 32),
    nb_classes=10,
)

attackD = ProjectedGradientDescent(
    classifierD,
    norm=np.inf,
    eps=8.0 / 255.0,
    eps_step=2.0 / 255.0,
    max_iter=40,
    targeted=False,
    num_random_init=5,
    batch_size=32,
)



# Step 4: Create the trainer object - AdversarialTrainerFBFPyTorch
#trainer = AdversarialTrainerFBFPyTorch(classifier, eps=epsilon, use_amp=False)
trainer = AdversarialTrainerMadryPGD(classifierD, nb_epochs=20, batch_size=64, eps= 8.0 /255.0, eps_step = 2.0/255.0)

# Build a Keras image augmentation object and wrap it in ART
#art_datagen = PyTorchDataGenerator(iterator=dataloader, size=x_train.shape[0], batch_size=128)

# Step 5: fit the trainer
#trainer.fit_generator(art_datagen, nb_epochs=30)
trainer.fit(x_train, y_train)

x_test_pred = np.argmax(classifierD.predict(x_test), axis=1)
print(
    "Accuracy on benign test samples after adversarial training: %.2f%%"
    % (np.sum(x_test_pred == np.argmax(y_test, axis=1)) / x_test.shape[0] * 100)
)



x_test_attack = attackD.generate(x_test)
x_test_attack_pred = np.argmax(classifierD.predict(x_test_attack), axis=1)
print(
    "Accuracy on original PGD adversarial samples after adversarial training: %.2f%%"
    % (np.sum(x_test_attack_pred == np.argmax(y_test, axis=1)) / x_test.shape[0] * 100)
)


