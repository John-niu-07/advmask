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
from art.attacks.evasion.hop_skip_jump import HopSkipJump
from art.attacks.evasion import FastGradientMethod
from art.estimators.classification import PyTorchClassifier
from art.utils import load_mnist
from art.utils import load_dataset
from art.preactresnet import PreActBlock, PreActBottleneck, PreActResNet, PreActResNet18, initialize_weights, PreActResNet18STL, PreActResNet34
import torchvision.transforms as transforms
from PIL import Image

import numpy as np



#!pip install deepface
#from deepface.basemodels import ArcFace
from deepface.deepface.basemodels import ArcFace
from deepface.deepface.commons import functions

import matplotlib.pyplot as plt

#----------------------------------------------
#build face recognition model

ArcFaceModel = ArcFace.loadModel()




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

class FaceClassifier(nn.Module):

    def __init__(self):
        super(FaceClassifier, self).__init__()

        self.embedding = ArcFace.loadModel()
        input_shape = self.embedding.layers[0].input_shape[0][1:3]


        img1 = functions.preprocess_face("dataset/img1.jpg", input_shape)
        #img1_representation = model.predict(img1)[0,:]
        self.img1_feat = self.embedding.predict(img1)[0,:]
    
        img2 = functions.preprocess_face("dataset/img8.jpg", input_shape)
        self.img2_feat = self.embedding.predict(img2)[0,:]

        img3 = functions.preprocess_face("dataset/img3.jpg", input_shape)
        self.img3_feat = self.embedding.predict(img3)[0,:]

        self.linear = nn.Linear(in_features=111, out_features=10)


    def get_logits(self, x):

        dist_vector1 = np.square(x - self.img1_feat)
        d1 = np.sqrt(dist_vector1.sum())

        dist_vector2 = np.square(x - self.img2_feat)
        d2 = np.sqrt(dist_vector2.sum())

        dist_vector3 = np.square(x - self.img3_feat)
        d3 = np.sqrt(dist_vector3.sum())

        d_all = np.array([d1, d2, d3])
        dmax = np.max(d_all)
        dmin = np.min(d_all)
        d_nor = (d_all - dmin) / (dmax - dmin)
        #dis = [d1, d2, d3]
        #return torch.stack(dis)
        return d_nor

    def forward(self, x):
        x = self.embedding.predict(x)[0,:]
        x = self.get_logits(x)
        x = softmax(x)
        return x



Embedding = ArcFace.loadModel()
input_shape = Embedding.layers[0].input_shape[0][1:3]

print("model input shape: ", Embedding.layers[0].input_shape[1:])
print("model output shape: ", Embedding.layers[-1].input_shape[-1])


test1_crop = functions.preprocess_face("dataset/img2.jpg", input_shape)



model = FaceClassifier()
model.train()

opt = torch.optim.SGD(model.parameters(), lr=0.001, momentum=0.9, weight_decay=5e-4)
criterion = nn.NLLLoss()
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=10)




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





#####
attack = ProjectedGradientDescent(
            classifier,
            norm=np.inf,
            #eps=8.0 / 255.0,
            eps=4.0 / 255.0,
            eps_step=2.0 / 255.0,
            max_iter=40,
            targeted=False,
            num_random_init=5,
            batch_size=32,
            )



'''
attack = HopSkipJump(
            classifier,
            norm=np.inf,
            #max_iter=40,
            max_iter=10,
            targeted=False,
            #max_eval=5000,
            max_eval=1000,
            init_eval=100,
            init_size=100,
            batch_size=32,
            verbose=True,
            )
'''


test1_adv = attack.generate(x=test1_crop)
predictions = classifier.predict(test1_adv)
print(predictions)

#accuracy = np.sum(np.argmax(predictions, axis=1) == np.argmax(y_test, axis=1)) / len(y_test)
#print("HSJ attack: Accuracy on adversarial test examples: {}%".format(accuracy * 100))


