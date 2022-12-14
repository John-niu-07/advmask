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
from art.preactresnet import PreActBlock, PreActBottleneck, PreActResNet, PreActResNet18, initialize_weights, PreActResNet18_, PreActResNet34 
from art.resnet import resnet18, resnet34
from art.vgg import vgg11
from art.shufflenetv2 import shufflenet_v2_x0_5,  shufflenet_v2_x1_5

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



#lod = torch.load('/art/SimCLR/pre-training-model/cifar/checkpoint_0200.pth.tar')
#lod = torch.load('/art/SimCLR/pre-training-model/cifar2/checkpoint_0200.pth.tar')
#lod_state = lod['state_dict']

#lod = torch.load('/art/SimCLR2/pytorch-simclr-simclr-master/checkpoint/output.pth')
lod = torch.load('/art/SimCLR2/pytorch-simclr-simclr-master/checkpoint/tiny32_2.pth')
#lod = torch.load('/art/SimCLR2/pytorch-simclr-simclr-master/checkpoint/tiny32.pth')
#lod = torch.load('/art/SimCLR2/pytorch-simclr-simclr-master/checkpoint/tiny32_resnet18.pth')
#lod = torch.load('/art/SimCLR2/pytorch-simclr-simclr-master/checkpoint/tiny32_resnet34.pth')
#lod = torch.load('/art/SimCLR2/pytorch-simclr-simclr-master/checkpoint/tiny32_vgg.pth')
#lod = torch.load('/art/SimCLR2/pytorch-simclr-simclr-master/checkpoint/tiny32_shuffle_v2.pth')
#lod = torch.load('/art/SimCLR2/pytorch-simclr-simclr-master/checkpoint/tiny32_pre34.pth')

lod_state = lod['net']

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

# Step 2: create the PyTorch model
#model = PreActResNet18()
#model = PreActResNet18_()
#model.apply(initialize_weights)
modelD = ResNetSimCLR_ft(base_model=None, out_dim=128)
modelD.train()
#model.load_state_dict(lod_state)

#opt = optim.Adam(model.parameters(), lr=0.01)
#opt = optim.Adam(model.parameters(), lr=0.001)
#opt = torch.optim.SGD(model.parameters(), lr=0.01, momentum=0.9, weight_decay=5e-4)
optD = torch.optim.SGD(modelD.parameters(), lr=0.005, momentum=0.9, weight_decay=5e-4)
#optAdvFt = torch.optim.SGD(modelAdvFt.parameters(), lr=0.01, momentum=0.9, weight_decay=5e-4)
#opt = torch.optim.SGD(model.parameters(), lr=0.0001, momentum=0.9, weight_decay=5e-4)
#criterion = nn.CrossEntropyLoss()
criterion = nn.NLLLoss()
schedulerD = torch.optim.lr_scheduler.CosineAnnealingLR(optD, T_max=10)


classifierD = PyTorchClassifier(
            model=modelD,
            clip_values=(-3.0, 3.0),
            #preprocessing=(cifar_mu, cifar_std),
            loss=criterion,
            optimizer=optD,
            scheduler=schedulerD,
            input_shape=(3, 32, 32),
            nb_classes=10,
            )


model = ResNetSimCLR_ft(base_model=None, out_dim=128)
model.train()
opt = torch.optim.SGD(model.parameters(), lr=0.005, momentum=0.9, weight_decay=5e-4)
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=10)


#model_ft = torch.load('/art/model_cifar/clr/model_cifar_AdvFt_base.model')
#model_ft = torch.load('/art/model_cifar/clr/model_cifar_AdvFt_base_res18.model')
#model.load_state_dict(model_ft)


# Step 3: Create the ART classifier
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


#predictions = classifier.predict(x_test)
#accuracy = np.sum(np.argmax(predictions, axis=1) == np.argmax(y_test, axis=1)) / len(y_test)
#print("Pre-training accuracy on benign test examples: {}%".format(accuracy * 100))


# Step 4: Train the ART classifier
classifier.fit_predict(x_train, y_train, x_test, y_test, batch_size=64, nb_epochs=51)

# Step 5: Evaluate the ART classifier on benign test examples
predictions = classifier.predict(x_test)

#print(np.exp(predictions[0:2]))
#maxv = np.max(np.exp(predictions), axis=1)
#print(maxv[0:10])
#mean2 = np.mean(maxv)
#print(mean2)

accuracy = np.sum(np.argmax(predictions, axis=1) == np.argmax(y_test, axis=1)) / len(y_test)
print("FT: accuracy on benign test examples: {}%".format(accuracy * 100))

classifier.save('model_cifar_AdvFt_base','/art/model_cifar/clr/')
#classifier.save('model_cifar_AdvFt_base_res18','/art/model_cifar/clr/')







#####
#attack = FastGradientMethod(estimator=classifier, eps=0.2)
#attack = ProjectedGradientDescent(classifier, eps=1.0, eps_step=0.1, verbose=False)
#attack = ProjectedGradientDescent(classifier, eps=0.2, eps_step=0.1, verbose=False)
attack = ProjectedGradientDescent(
            classifier,
            norm=np.inf,
            eps=20.0 / 255.0,
            eps_step=2.0 / 255.0,
            max_iter=40,
            targeted=False,
            num_random_init=5,
            batch_size=32,
            )


x_test_adv = attack.generate(x=x_test)
np.save('x_test_adv_base.npy', x_test_adv)
#np.save('x_test_adv_base_res18.npy', x_test_adv)

#x_test_adv = np.load('x_test_adv_base.npy')
#x_test_adv = np.load('x_test_adv_base_res18.npy')


predictions = classifier.predict(x_test_adv)
accuracy = np.sum(np.argmax(predictions, axis=1) == np.argmax(y_test, axis=1)) / len(y_test)
print("FT: Accuracy on test_adv : {}%".format(accuracy * 100))



print("----------------------------AT for defense---")


x_train_adv = attack.generate(x=x_train)
np.save('x_train_adv_base.npy', x_train_adv)
#np.save('x_train_adv_base_res18.npy', x_train_adv)

#x_train_adv = np.load('x_train_adv_base.npy')
#x_train_adv = np.load('x_train_adv_base_res18.npy')

x_train_adv = np.vstack((x_train, x_train_adv))
print(x_train_adv.shape)
y_train = np.vstack((y_train, y_train))
print(y_train.shape)


classifierD.fit_predict(x_train_adv, y_train, x_test, y_test, batch_size=64, nb_epochs=51)
predictions = classifierD.predict(x_test)
predictions2 = classifierD.predict(x_test_adv)

accuracy = np.sum(np.argmax(predictions, axis=1) == np.argmax(y_test, axis=1)) / len(y_test)
print("AdvFt: accuracy on benign test examples: {}%".format(accuracy * 100))


accuracy = np.sum(np.argmax(predictions2, axis=1) == np.argmax(y_test, axis=1)) / len(y_test)
print("AdvFt: Accuracy on test_adv: {}%".format(accuracy * 100))






attack2 = ProjectedGradientDescent(
            classifierD,
            norm=np.inf,
            eps=20.0 / 255.0,
            eps_step=2.0 / 255.0,
            max_iter=40,
            targeted=False,
            num_random_init=5,
            batch_size=32,
            )

x_test_adv2 = attack2.generate(x=x_test)
predictions = classifierD.predict(x_test_adv2)
accuracy = np.sum(np.argmax(predictions, axis=1) == np.argmax(y_test, axis=1)) / len(y_test)
print("AdvFt: Accuracy on test_adv2 : {}%".format(accuracy * 100))


