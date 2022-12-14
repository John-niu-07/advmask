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
from torch.autograd import Variable

from art.attacks.evasion.projected_gradient_descent.projected_gradient_descent import ProjectedGradientDescent
from art.attacks.evasion import FastGradientMethod
from art.estimators.classification import PyTorchClassifier
from art.utils import load_mnist
from art.utils import load_dataset
from art.preactresnet import PreActBlock, PreActBottleneck, PreActResNet, PreActResNet18, initialize_weights, PreActResNet18_

from art.visualization import create_sprite, convert_to_rgb, save_image, plot_3d
import torchvision.transforms as transforms
from PIL import Image

import numpy as np


#https://github.com/foamliu/Autoencoder/blob/master/models.py
class conv2DBatchNormRelu(nn.Module):
    def __init__(
            self,
            in_channels,
            n_filters,
            k_size,
            stride,
            padding,
            bias=True,
            dilation=1,
            with_bn=True,
    ):
        super(conv2DBatchNormRelu, self).__init__()

        conv_mod = nn.Conv2d(int(in_channels),
                             int(n_filters),
                             kernel_size=k_size,
                             padding=padding,
                             stride=stride,
                             bias=bias,
                             dilation=dilation, )

        if with_bn:
            self.cbr_unit = nn.Sequential(conv_mod,
                                          nn.BatchNorm2d(int(n_filters)),
                                          #nn.ReLU(inplace=True))
                                          nn.LeakyReLU(0.2))
        else:
            self.cbr_unit = nn.Sequential(conv_mod, nn.ReLU(inplace=True))

    def forward(self, inputs):
        outputs = self.cbr_unit(inputs)
        return outputs


class segnetDown2(nn.Module):
    def __init__(self, in_size, out_size):
        super(segnetDown2, self).__init__()
        #self.conv1 = conv2DBatchNormRelu(in_size, out_size, 3 , 1, 1)
        self.conv1 = conv2DBatchNormRelu(in_size, out_size, 3 , 1, 1)
        self.conv2 = conv2DBatchNormRelu(out_size, out_size, 3, 1, 1)

    def forward(self, inputs):
        outputs = self.conv1(inputs)
        outputs = self.conv2(outputs)
        return outputs


class segnetDown3(nn.Module):
    def __init__(self, in_size, out_size):
        super(segnetDown3, self).__init__()
        self.conv1 = conv2DBatchNormRelu(in_size, out_size, 3, 1, 1)
        self.conv2 = conv2DBatchNormRelu(out_size, out_size, 3, 1, 1)
        self.conv3 = conv2DBatchNormRelu(out_size, out_size, 3, 1, 1)

    def forward(self, inputs):
        outputs = self.conv1(inputs)
        outputs = self.conv2(outputs)
        outputs = self.conv3(outputs)
        return outputs


class segnetUp2(nn.Module):
    def __init__(self, in_size, out_size):
        super(segnetUp2, self).__init__()
        #self.unpool = nn.MaxUnpool2d(2, 2)
        self.ups = nn.Upsample(scale_factor=2)
        self.conv1 = conv2DBatchNormRelu(in_size, in_size, 3, 1, 1)
        self.conv2 = conv2DBatchNormRelu(in_size, out_size, 3, 1, 1)

    def forward(self, inputs):
        #outputs = self.unpool(input=inputs, indices=indices, output_size=output_shape)
        outputs = self.ups(input=inputs)
        outputs = self.conv1(outputs)
        outputs = self.conv2(outputs)
        return outputs


class segnetUp3(nn.Module):
    def __init__(self, in_size, out_size):
        super(segnetUp3, self).__init__()
        #self.unpool = nn.MaxUnpool2d(2, 2)
        self.ups = nn.Upsample(scale_factor=2)
        self.conv1 = conv2DBatchNormRelu(in_size, in_size, 3, 1, 1)
        self.conv2 = conv2DBatchNormRelu(in_size, in_size, 3, 1, 1)
        self.conv3 = conv2DBatchNormRelu(in_size, out_size, 3, 1, 1)

    def forward(self, inputs):
        #outputs = self.unpool(input=inputs, indices=indices, output_size=output_shape)
        outputs = self.ups(input=inputs)
        outputs = self.conv1(outputs)
        outputs = self.conv2(outputs)
        outputs = self.conv3(outputs)
        return outputs


class SegNet(nn.Module):
    def __init__(self, n_classes=3, in_channels=3, is_unpooling=True):
        super(SegNet, self).__init__()

        self.in_channels = in_channels
        self.is_unpooling = is_unpooling

        self.down1 = segnetDown2(self.in_channels, 64)
        self.down2 = segnetDown2(64, 128)
        self.down3 = segnetDown3(128, 256)
        self.down4 = segnetDown3(256, 512)
        self.down5 = segnetDown3(512, 512)

        self.up5 = segnetUp3(512, 512)
        self.up4 = segnetUp3(512, 256)
        self.up3 = segnetUp3(256, 128)
        self.up2 = segnetUp2(128, 64)
        self.up1 = segnetUp2(64, n_classes)

    def forward(self, inputs):

        down1, indices_1, unpool_shape1 = self.down1(inputs)
        down2, indices_2, unpool_shape2 = self.down2(down1)
        down3, indices_3, unpool_shape3 = self.down3(down2)
        down4, indices_4, unpool_shape4 = self.down4(down3)
        down5, indices_5, unpool_shape5 = self.down5(down4)

        up5 = self.up5(down5, indices_5, unpool_shape5)
        up4 = self.up4(up5, indices_4, unpool_shape4)
        up3 = self.up3(up4, indices_3, unpool_shape3)
        up2 = self.up2(up3, indices_2, unpool_shape2)
        up1 = self.up1(up2, indices_1, unpool_shape1)

        return up1

    def init_vgg16_params(self, vgg16):
        blocks = [self.down1, self.down2, self.down3, self.down4, self.down5]

        ranges = [[0, 4], [5, 9], [10, 16], [17, 23], [24, 29]]
        features = list(vgg16.features.children())

        vgg_layers = []
        for _layer in features:
            if isinstance(_layer, nn.Conv2d):
                vgg_layers.append(_layer)

        merged_layers = []
        for idx, conv_block in enumerate(blocks):
            if idx < 2:
                units = [conv_block.conv1.cbr_unit, conv_block.conv2.cbr_unit]
            else:
                units = [
                    conv_block.conv1.cbr_unit,
                    conv_block.conv2.cbr_unit,
                    conv_block.conv3.cbr_unit,
                ]
            for _unit in units:
                for _layer in _unit:
                    if isinstance(_layer, nn.Conv2d):
                        merged_layers.append(_layer)

        assert len(vgg_layers) == len(merged_layers)

        for l1, l2 in zip(vgg_layers, merged_layers):
            if isinstance(l1, nn.Conv2d) and isinstance(l2, nn.Conv2d):
                assert l1.weight.size() == l2.weight.size()
                assert l1.bias.size() == l2.bias.size()
                l2.weight.data = l1.weight.data
                l2.bias.data = l1.bias.data


class CNN_Encoder(nn.Module):
    def __init__(self, output_size, input_size=(3, 32, 32), is_unpooling=True):
        super(CNN_Encoder, self).__init__()
        self.input_size = input_size

        #convolutions
        self.down1 = segnetDown2(3, 64)
        self.down2 = segnetDown2(64, 128)
        self.down3 = segnetDown3(128, 256)
        self.down4 = segnetDown3(256, 512)
        self.down5 = segnetDown3(512, 512)	
	
        self.flat_fts = 512*32*32

        self.linear = nn.Sequential(
            nn.Linear(self.flat_fts, output_size),
            nn.BatchNorm1d(output_size),
            nn.LeakyReLU(0.2),
        )
		
    def get_flat_fts(self, fts):
        f = fts(Variable(torch.ones(1, *self.input_size)))
        return int(np.prod(f.size()[1:]))

		
    def forward(self, x):
        #print("Encoder x: {}".format(x.shape))
        x = x.view(-1, *self.input_size)
        down1 = self.down1(x)
        #print("Encoder down1: {}".format(down1.shape))
        down2 = self.down2(down1)
        #print("Encoder down2: {}".format(down2.shape))
        down3 = self.down3(down2)
        down4 = self.down4(down3)
        down5 = self.down5(down4)        
        #print("Encoder down5: {}".format(down5.shape))
        x = down5.view(-1, self.flat_fts)
        #print("Encoder x: {}".format(x.shape))
        x= self.linear(x)
        #print("Encoder x: {}".format(x.shape))
        return x


		
class CNN_Decoder(nn.Module):
    def __init__(self, embedding_size, input_size=(3, 32, 32)):
        super(CNN_Decoder, self).__init__()
        self.input_height = 32
        self.input_width = 32
        self.input_dim = embedding_size
        self.fc_output_dim = 512

        self.fc = nn.Sequential(
            nn.Linear(self.input_dim, self.fc_output_dim),
            nn.BatchNorm1d(self.fc_output_dim),
            #nn.ReLU(True)
            nn.LeakyReLU(0.2)
        )

        self.up5 = segnetUp3(512, 512)
        self.up4 = segnetUp3(512, 256)
        self.up3 = segnetUp3(256, 128)
        self.up2 = segnetUp2(128, 64)
        self.up1 = segnetUp2(64, 3)

    def forward(self, x):
        #print("Decoder x: {}".format(x.shape))
        x = self.fc(x)
        #print("Decoder x: {}".format(x.shape))
        x = x.view(-1, self.fc_output_dim, 1, 1)
        up5 = self.up5(x)
        #print("Decoder up5: {}".format(up5.shape))
        up4 = self.up4(up5)
        up3 = self.up3(up4)
        up2 = self.up2(up3)
        up1 = self.up1(up2)
        #print("Decoder up1: {}".format(up1.shape))
		
        #return up1.view(-1, self.input_width*self.input_height)
        return up1.view(-1, 32*32*3)


class AE_net(nn.Module):
    def __init__(self, dim_latent):
        super(AE_net, self).__init__()
        output_size = dim_latent
        self.encoder = CNN_Encoder(output_size)

        self.decoder = CNN_Decoder(dim_latent)

    def encode(self, x):
        return self.encoder(x)

    def decode(self, z):
        return self.decoder(z)

    def forward(self, x):
        #z = self.encode(x.view(-1, 32*32*3))
        z = self.encode(x.reshape(-1, 32*32*3))
        return self.decode(z)

class AE_loss(nn.Module):

    def __init__(self, args=None):
        super(AE_loss, self).__init__()

    def forward(self, recon_x, x):
        #BCE = F.binary_cross_entropy(recon_x, x.view(-1, 3*32*32), reduction='sum')
        #print(x.shape)
        #print(recon_x.shape)
        #BCE = torch.sqrt((recon_x - x).pow(2).mean())
        #x = x.view(-1,3*32*32)
        x = x.reshape((-1, 3*32*32))
        BCE = F.mse_loss(recon_x, x)
        return BCE


		
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
#x_train, y_train = x_train[:50], y_train[:50]
#x_test, y_test = x_test[:500], y_test[:500]
im_shape = x_train[0].shape


# Step 2: create the PyTorch model
model = PreActResNet18()
#model = PreActResNet18_()
model.apply(initialize_weights)
model.train()
#opt = optim.Adam(model.parameters(), lr=0.01)
opt = torch.optim.SGD(model.parameters(), lr=0.01, momentum=0.9, weight_decay=5e-4)
criterion = nn.CrossEntropyLoss()
#criterion = nn.NLLLoss()
scheduler = torch.optim.lr_scheduler.MultiStepLR(opt, milestones=[20, 30, 35], gamma=0.1)

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
            [transforms.ToTensor(), transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2471, 0.2435, 0.2616))]
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




#x_train = x_train.transpose(0, 3, 1, 2).astype("float32")
#x_test = x_test.transpose(0, 3, 1, 2).astype("float32")

transform = transforms.Compose(
            #[transforms.RandomCrop(32, padding=4), transforms.RandomHorizontalFlip(), transforms.ToTensor()]
            [transforms.ToTensor()]
            )

transform2 = transforms.Compose(
            #[transforms.RandomCrop(32, padding=4), transforms.RandomHorizontalFlip(), transforms.ToTensor()]
            [transforms.ToTensor(), transforms.RandomCrop(32, padding=4), transforms.RandomHorizontalFlip()]
            #[transforms.ToTensor()]
            )

'''
for idx in range(len(x_train)):
    #print(x_train[idx])
    x = Image.fromarray(((x_train[idx] * 255).round()).astype(np.uint8).transpose(1, 2, 0))
    x = transform(x)
    x_train[idx] = x
'''

x_train_2 = x_train_.copy()
for idx in range(len(x_train_2)):
    #print(x_train[idx])
    #x = Image.fromarray(((x_train_2[idx] * 255).round()).astype(np.uint8).transpose(1, 2, 0))
    x = x_train_2[idx].transpose(1,2,0) 
    x = transform2(x)
    #x = transform(x)
    x_train_2[idx] = x

print(x_train_2.shape)
#print(x_train_2[0].transpose(1,2,0))

'''
x_train_3 = x_train_.copy()
for idx in range(len(x_train_3)):
    #print(x_train[idx])
    x = Image.fromarray(((x_train_3[idx] * 255).round()).astype(np.uint8).transpose(1, 2, 0))
    x = transform2(x)
    x_train_3[idx] = x
'''


x_train = np.vstack((x_train_, x_train_2))
print(x_train.shape)
y_train = np.vstack((y_train, y_train))
print(y_train.shape)


'''
x_train = np.vstack((x_train, x_train_3))
print(x_train.shape)
y_train = np.vstack((y_train, y_train))
print(x_train.shape)
'''








#model.load_state_dict(torch.load('/art/model_cifar/untargeted/model_pgd_untargeted_cifar_clip.model'))
model.load_state_dict(torch.load('/art/model_cifar/untargeted/model_pgd_untargeted_cifar_clip1.model'))


# Step 3: Create the ART classifier
classifier = PyTorchClassifier(
            model=model,
            #clip_values=(-3.0, 3.0),
            clip_values=(-1.0, 1.0),
            #preprocessing=(cifar_mu, cifar_std),
            loss=criterion,
            optimizer=opt,
            scheduler=scheduler,
            input_shape=(3, 32, 32),
            nb_classes=10,
            )


# Step 4: Train the ART classifier
#classifier.fit(x_train, y_train, batch_size=128, nb_epochs=40)

# Step 5: Evaluate the ART classifier on benign test examples
predictions = classifier.predict(x_test)
#predictions = classifier.predict(x_train)

#print(np.exp(predictions[0:2]))
#maxv = np.max(np.exp(predictions), axis=1)
#print(maxv[0:10])
#mean2 = np.mean(maxv)
#print(mean2)

accuracy = np.sum(np.argmax(predictions, axis=1) == np.argmax(y_test, axis=1)) / len(y_test)
#accuracy = np.sum(np.argmax(predictions, axis=1) == np.argmax(y_train, axis=1)) / len(y_train)
print("Accuracy on benign test examples: {}%".format(accuracy * 100))
#classifier.save('model_pgd_untargeted_cifar_clip','/art/model_cifar/untargeted/')
#classifier.save('model_pgd_untargeted_cifar_clip1','/art/model_cifar/untargeted/')



print("----pgd_untargeted_attack----")
#attack = FastGradientMethod(estimator=classifier, eps=0.2)
#attack = ProjectedGradientDescent(classifier, eps=1.0, eps_step=0.1, verbose=False)
#attack = ProjectedGradientDescent(classifier, eps=0.2, eps_step=0.1, verbose=False)
eps_=35
#eps_=55
print("eps = {}".format(eps_))
attack = ProjectedGradientDescent(
            classifier,
            norm=np.inf,
            #eps=5.0 / 255.0,
            #eps=8.0 / 255.0,
            #eps=40.0 / 255.0,
            #eps= float(eps_) / 255.0,
            #eps_step=2.0 / 255.0,
            eps= float(eps_) / 128.0,
            eps_step=2.0 / 128.0,
            #max_iter=40,
            max_iter=4,
            targeted=False,
            #num_random_init=5,
            num_random_init=1,
            batch_size=32,
            )

#x_test_adv = attack.generate(x=x_test)
#predictions = classifier.predict(x_test_adv)

#print(np.exp(predictions[0:2]))
#maxv = np.max(np.exp(predictions), axis=1)
#print(maxv[0:10])
#mean2 = np.mean(maxv)
#print(mean2)

#accuracy = np.sum(np.argmax(predictions, axis=1) == np.argmax(y_test, axis=1)) / len(y_test)
#print("pgd_untargeted_attack: Accuracy on adversarial test examples: {}%".format(accuracy * 100))


###########
class UnNormalize:
    def __init__(self, mean=(0.4914, 0.4822, 0.4465), std=(0.2471, 0.2435, 0.2616)):
        self.mean = torch.tensor(mean).view((1,-1,1,1))
        self.std = torch.tensor(std).view((1,-1,1,1))
    def __call__(self, x):
        x = (torch.from_numpy(x) * self.std) + self.mean
        #return torch.clip(x,0,1).numpy().reshape((3,32,32))
        print(x.min())
        print(x.max())
        #return x.numpy().reshape((3,32,32))
        return torch.clip(x,0,1).numpy().reshape((3,32,32))
UnNor = UnNormalize()

sample = x_test[1]
gt = np.argmax(y_test[1])
#sample = x_train[1]
#gt = np.argmax(y_train[1])

#print(sample.shape)
#sample = UnNor(sample)
#print(sample.shape)
sample_img = (UnNor(sample)*255).astype(np.uint8).transpose(1, 2, 0)
save_image(sample_img, "/art/vis_cifar/ae/sample.png")


sample = sample.reshape((1,3,32,32))
sample_adv = attack.generate(x=sample)
predictions = classifier.predict(sample_adv)
rst = np.argmax(predictions, axis=1)

print("pgd_untargeted_attack: class = {}  gt = {}".format(rst, gt))
#print("pgd_untargeted_attack: {}".format(np.exp(predictions)))
print("pgd_untargeted_attack: {}".format(predictions))

sample_adv_img = (UnNor(sample_adv[0])*255).astype(np.uint8).transpose(1, 2, 0)
save_image(sample_adv_img, "/art/vis_cifar/ae/sample_adv"+str(eps_)+".png")


pert = np.abs(sample_adv[0] - sample[0])
n_pert = np.linalg.norm(pert.reshape((3*32*32)), ord=2)
#print("norm of pert: {}".format(n_pert))
#print(pert)
pert_img = (pert*255).astype(np.uint8).transpose(1, 2, 0)
save_image(pert_img, "/art/vis_cifar/ae/pert_adv"+str(eps_)+".png")


#####
#load AE
dim_latent = 8
dim_latent = 16
dim_latent = 128
dim_latent = 256
model3 = AE_net(dim_latent) 
criterion3 = AE_loss()
optimizer3 = optim.Adam(model3.parameters(), lr=0.001)
#model3.load_state_dict(torch.load('/art/model/ae/model_ae_dim8.model'))
#model3.load_state_dict(torch.load('/art/model/ae/model_ae_dim16.model'))
#model3.load_state_dict(torch.load('/art/model_cifar/ae/model_ae_dim16.model'))
#model3.load_state_dict(torch.load('/art/model_cifar/ae/model_ae_dim128_mse___.model'))
#model3.load_state_dict(torch.load('/art/model_cifar/ae/model_ae_dim128_nor_clip.model'))
#model3.load_state_dict(torch.load('/art/adversarial-robustness-toolbox/examples/fit_ae_ep201.model'))
#model3.load_state_dict(torch.load('/art/model_cifar/ae/model_ae_dim128_nor_clip__.model'))
model3.load_state_dict(torch.load('/art/model_cifar/ae/model_ae_dim256_nor_clip_.model'))

# Step 3: Create the ART classifier
classifier3 = PyTorchClassifier(
        model=model3,
        clip_values=(-3.0, 3.0),
        #preprocessing=(cifar_mu, cifar_std),
        loss=criterion3,
        optimizer=optimizer3,
        input_shape=(3, 32, 32),
        nb_classes=10,
        )


sample_code = classifier3.get_code_ae(sample)
rec_sample = classifier3.input_code_ae(sample_code)
rec_sample_img = (UnNor(rec_sample.reshape((3,32,32)))*255).astype(np.uint8).transpose(1, 2, 0)
save_image(rec_sample_img, "/art/vis_cifar/ae/rec_sample.png")

sample_adv_code = classifier3.get_code_ae(sample_adv)
rec_sample_adv = classifier3.input_code_ae(sample_adv_code)
rec_sample_adv_img = (UnNor(rec_sample_adv.reshape((3,32,32)))*255).astype(np.uint8).transpose(1, 2, 0)
save_image(rec_sample_adv_img, "/art/vis_cifar/ae/rec_sample_adv"+str(eps_)+".png")


pert_code = sample_code - sample_adv_code
n_pert_code = np.linalg.norm(pert_code, ord=2)
#print(pert_code)
#print("norm of pert_code: {}".format(n_pert_code))


######get base
delta=0.001

bb_code = sample_code[0].copy()
#print("bb_code: {}".format(bb_code))

base_s_list = []
for i in range(dim_latent):
    bb_code = sample_code[0].copy()
    bb_code[i] = bb_code[i] + delta
    inpu = np.expand_dims(bb_code, axis=0)
    #print(inpu)
    rec_bb = classifier3.input_code_ae(inpu)

    
    
    #rec_bb_img = (rec_bb.reshape((28,28))*255).astype(np.uint8)
    #save_image(rec_bb_img, "/art/vis/ae/rec_bb_"+str(i)+"_img.png")
    

    #base_s = np.abs(rec_bb[0] - rec_sample[0])
    base_s = rec_bb[0] - rec_sample[0]
    #print(base_s)
    base_s_list.append(base_s)

    #base_s_img = (np.abs(base_s.reshape((3,32,32)))*255*1000).astype(np.uint8).transpose(1, 2, 0)
    #save_image(base_s_img, "/art/vis_cifar/ae/base_s"+str(i)+".png")
    
    #pos_part = np.where(base_s > 0, base_s, 0)
    #base_s_img = (pos_part.reshape((28,28))*255*1000).astype(np.uint8)
    #save_image(base_s_img, "/art/vis/ae/base_s_pos_"+str(i)+".png")
    #n_pos = np.linalg.norm(pos_part, ord=2)
    #print("norm of pos: {}".format(n_pos))

    #neg_part = np.abs(np.where(base_s < 0, base_s, 0))
    #base_s_img = (neg_part.reshape((28,28))*255*1000).astype(np.uint8)
    #save_image(base_s_img, "/art/vis/ae/base_s_neg_"+str(i)+".png")
    #n_neg = np.linalg.norm(neg_part, ord=2)
    #print("norm of neg: {}".format(n_neg))
    

#print(base_s_list)

#n_sample = np.linalg.norm(sample.reshape((784)), ord=2)
#print(n_sample)    


######lstsq

for i in range(dim_latent):
    if i==0:
        base_s = base_s_list[i]
        Base = base_s.copy()
    else:
        base_s = base_s_list[i]
        Base = np.vstack((Base, base_s))

Base = np.transpose(Base)
#print(Base.shape)

print('...lstsq  for adv ....')
#img_v = sample.reshape((784, 1))
#img_v = sample.reshape((784, 1)) - rec_sample[0].reshape((784, 1))

#for j in range(40):
#    input_sample = x_test[j]
#    input_img = (input_sample.reshape((28,28))*255).astype(np.uint8)
#    save_image(input_img, "/art/vis/ae/input_img_"+str(j)+".png")

#input_sample = x_test[17]
#input_sample = x_test[26]

#input_sample = sample
input_sample = sample_adv


n_input = np.linalg.norm(input_sample.reshape((3*32*32)), ord=2)
print("norm of input: {}".format(n_input))
input_img = (UnNor(input_sample.reshape((3,32,32)))*255).astype(np.uint8).transpose(1, 2, 0)
save_image(input_img, "/art/vis_cifar/ae/input.png")


#### Should an input be a diff?
d = input_sample.reshape((3*32*32, 1)) - rec_sample[0].reshape((3*32*32, 1))
#print(img_d.shape)
n_d = np.linalg.norm(d, ord=2)
print("norm of d: {}".format(n_d))
img_d = (np.abs(d).reshape((3,32,32))*255*10).astype(np.uint8).transpose(1, 2, 0)
save_image(img_d, "/art/vis_cifar/ae/d.png")

lstsq_rst = np.linalg.lstsq(Base, d, rcond = None)
X = lstsq_rst[0]
Res = lstsq_rst[1]
Base_rank = lstsq_rst[2]

#print("res: {}".format(Res[0]))
coeff = np.transpose( X )[0]
#print("coeff: {}".format(coeff))

### proj
d_proj = np.zeros_like(base_s_list[0])
for i in range(dim_latent):
    base_s = base_s_list[i]
    d_proj += coeff[i] * base_s

n_d_proj = np.linalg.norm(d_proj, ord=2)
print("norm of d_proj: {}".format(n_d_proj))
d_proj_img = (np.abs(d_proj).reshape((3,32,32))*255*10).astype(np.uint8).transpose(1, 2, 0)
save_image(d_proj_img, "/art/vis_cifar/ae/d_proj.png")

#add rec?
proj = d_proj + rec_sample[0]
n_proj = np.linalg.norm(proj, ord=2)
print("norm of proj: {}".format(n_proj))
#proj_img = (np.abs(proj).reshape((3,32,32))*255).astype(np.uint8).transpose(1, 2, 0)
proj_img = (UnNor(proj.reshape((3,32,32)))*255).astype(np.uint8).transpose(1, 2, 0)
save_image(proj_img, "/art/vis_cifar/ae/proj.png")


diff = np.abs(proj - input_sample.reshape((3*32*32)))
n_diff = np.linalg.norm(diff, ord=2)
print("norm of diff: {}".format(n_diff))
diff_img = (diff.reshape((3,32,32))*255).astype(np.uint8).transpose(1, 2, 0)
save_image(diff_img, "/art/vis_cifar/ae/diff.png")

ang = np.arccos(n_d_proj/n_d) / 3.1415 * 180
print("............... angle: {}".format(ang))



print('...lstsq  for noise ....')

#d = sample_adv.reshape((3*32*32, 1)) - rec_sample[0].reshape((3*32*32, 1))
#n_d = np.linalg.norm(d, ord=2)
flag = True
init_p = 0.16
#init_p = 0.60
kk=0
while(flag):
        if kk>10:
            break
        n_d_noise_test = []
        for _ in range(20):
            #noise = np.random.rand(1,3,32,32)
            #input_sample = np.clip(sample[0] + noise * init_p, 0.0, 1.0)
            noise = np.random.rand(1,3,32,32) - 0.5 
            input_sample = sample[0] + noise * init_p
            d_noise = input_sample.reshape((3*32*32, 1)) - rec_sample[0].reshape((3*32*32, 1))
            n_d_noise = np.linalg.norm(d_noise, ord=2)
            n_d_noise_test.append(n_d_noise)
        mean_test = np.mean(n_d_noise_test)
        if np.abs(n_d - mean_test) < 0.10:
            flag = False
            print('Done    n_d:{}  mean:{}'.format(n_d, mean_test))
        elif n_d > mean_test:
            init_p +=0.01
            print('+       n_d:{}  mean:{}'.format(n_d, mean_test))
        elif mean_test > n_d:
            init_p -=0.01
            print('-       n_d:{}  mean:{}'.format(n_d, mean_test))
        kk+=1

#noise = np.abs(np.random.rand(1,3,32,32))
#input_sample = np.clip(sample + noise * init_p, 0.0, 1.0)
noise = np.random.rand(1,3,32,32) - 0.5 
input_sample = sample[0] + noise * init_p


n_input = np.linalg.norm(input_sample.reshape((3*32*32)), ord=2)
print("norm of input: {}".format(n_input))
input_img = (UnNor(input_sample.reshape((3,32,32)))*255).astype(np.uint8).transpose(1, 2, 0)
save_image(input_img, "/art/vis_cifar/ae/input_noise.png")


#### Should an input be a diff?
d = input_sample.reshape((3*32*32, 1)) - rec_sample[0].reshape((3*32*32, 1))
#print(img_d.shape)
n_d = np.linalg.norm(d, ord=2)
print("norm of d: {}".format(n_d))
img_d = (np.abs(d).reshape((3,32,32))*255*10).astype(np.uint8).transpose(1, 2, 0)
save_image(img_d, "/art/vis_cifar/ae/d_noise.png")

lstsq_rst = np.linalg.lstsq(Base, d, rcond = None)
X = lstsq_rst[0]
Res = lstsq_rst[1]
Base_rank = lstsq_rst[2]

#print("res: {}".format(Res[0]))
coeff = np.transpose( X )[0]
#print("coeff: {}".format(coeff))



### proj
d_proj = np.zeros_like(base_s_list[0])
for i in range(dim_latent):
    base_s = base_s_list[i]
    d_proj += coeff[i] * base_s

n_d_proj = np.linalg.norm(d_proj, ord=2)
print("norm of d_proj: {}".format(n_d_proj))
d_proj_img = (np.abs(d_proj).reshape((3,32,32))*255*10).astype(np.uint8).transpose(1, 2, 0)
save_image(d_proj_img, "/art/vis_cifar/ae/d_proj_noise.png")

#add rec?
proj = d_proj + rec_sample[0]
n_proj = np.linalg.norm(proj, ord=2)
print("norm of proj: {}".format(n_proj))
#proj_img = (np.abs(proj).reshape((3,32,32))*255).astype(np.uint8).transpose(1, 2, 0)
proj_img = (UnNor(proj.reshape((3,32,32)))*255).astype(np.uint8).transpose(1, 2, 0)
save_image(proj_img, "/art/vis_cifar/ae/proj_noise.png")


diff = np.abs(proj - input_sample.reshape((3*32*32)))
n_diff = np.linalg.norm(diff, ord=2)
print("norm of diff: {}".format(n_diff))
diff_img = (diff.reshape((3,32,32))*255).astype(np.uint8).transpose(1, 2, 0)
save_image(diff_img, "/art/vis_cifar/ae/diff_noise.png")

ang = np.arccos(n_d_proj/n_d) / 3.1415 * 180
print("............... angle: {}".format(ang))



'''
print("----PGD_uncertainty_attack----")
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


#predictions = classifier.predict(x_test)
#accuracy = np.sum(np.argmax(predictions, axis=1) == np.argmax(y_test, axis=1)) / len(y_test)
#print("Model2: Accuracy on benign test examples: {}%".format(accuracy * 100))



# Step 6: Generate adversarial test examples
#attack = FastGradientMethod(estimator=classifier2, eps=0.2)
#x_test_adv = attack.generate(x=x_test)

# Test PGD with np.inf norm
#attack = ProjectedGradientDescent(classifier2, eps=1.0, eps_step=0.1, verbose=False)
#attack = ProjectedGradientDescent(classifier2, eps=1.0, eps_step=0.1, verbose=False)
attack = ProjectedGradientDescent(
            classifier2,
            norm=np.inf,
            #eps=8.0 / 255.0,
            eps=50.0 / 255.0,
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
print("PGD_uncertainty_attack: Accuracy on adversarial test examples: {}%".format(accuracy * 100))



"""
print("----pgd_targeted_attack: target=0----")
attack = ProjectedGradientDescent(
            classifier,
            norm=np.inf,
            #eps=8.0 / 255.0,
            eps=20.0 / 255.0,
            #eps=80.0 / 255.0,
            eps_step=2.0 / 255.0,
            max_iter=40,
            targeted=True,
            num_random_init=5,
            batch_size=32,
            )

#y_test_ = y_test.copy()
y_test_target = np.zeros_like(y_test)
print(len(y_test_target))
for i in range(len(y_test_target)):
    #print(y_test[i])
    #y_test_target[i][0] = 1
    y_test_target[i][1] = 1
    #y_test_target[i][2] = 1
    #y_test_target[i][3] = 1
    #y_test_target[i][4] = 1
    #y_test_target[i][5] = 1
    #y_test_target[i][6] = 1
    #y_test_target[i][7] = 1
    #y_test_target[i][8] = 1
    #y_test_target[i][9] = 1
    #print(y_test_target[i])


x_test_adv = attack.generate(x=x_test, y=y_test_target)
predictions = classifier.predict(x_test_adv)
accuracy = np.sum(np.argmax(predictions, axis=1) == np.argmax(y_test_target, axis=1)) / len(y_test)
print("pgd_targeted_attack: Accuracy on adversarial test examples: {}%".format(accuracy * 100))
pgd_targeted_perturbation = x_test_adv - x_test

print(len(pgd_targeted_perturbation))
for i in range(len(pgd_targeted_perturbation)):
    tmp = pgd_targeted_perturbation[i]
    #print(tmp.shape)

    itm_ = (tmp[0]*255).astype(np.uint8)
    if i < 5:
        f_name = "/art/vis/eps20/p_target_image_"+str(i)+".png"
        save_image(itm_, f_name)
"""
'''
############
'''
#train AE
dim_latent = 8
dim_latent = 16
dim_latent = 128
dim_latent = 256
model3 = AE_net(dim_latent) 
criterion3 = AE_loss()
optimizer3 = optim.Adam(model3.parameters(), lr=0.01)
#optimizer3 = optim.Adam(model3.parameters(), lr=0.005)
#optimizer3 = optim.Adam(model3.parameters(), lr=0.0005)
#optimizer3 = torch.optim.SGD(model3.parameters(), lr=0.01, momentum=0.9, weight_decay=5e-4)
scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer3, milestones=[200, 300, 350, 390], gamma=0.5)
#scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer3, milestones=[100, 170], gamma=0.1)
#scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer3, milestones=[30, 100, 170], gamma=0.1)



#model3.load_state_dict(torch.load('/art/model_cifar/ae/model_ae_dim128_nor.model'))
#model3.load_state_dict(torch.load('/art/model_cifar/ae/model_ae_dim128_nor_clip.model'))
#model3.load_state_dict(torch.load('/art/model_cifar/ae/model_ae_dim128_nor_clip_.model'))
#model3.load_state_dict(torch.load('/art/model_cifar/ae/model_ae_dim128_nor_clip__.model'))

model3.load_state_dict(torch.load('/art/model_cifar/ae/model_ae_dim256_nor_clip_.model'))

classifier = PyTorchClassifier(
            model=model3,
            clip_values=(-3.0, 3.0),
            #preprocessing=(cifar_mu, cifar_std),
            loss=criterion3,
            optimizer=optimizer3,
            scheduler=scheduler,
            input_shape=(3, 32, 32),
            nb_classes=10,
            )


class UnNormalize:
    def __init__(self, mean=(0.4914, 0.4822, 0.4465), std=(0.2471, 0.2435, 0.2616)):
        self.mean = torch.tensor(mean).view((1,-1,1,1))
        self.std = torch.tensor(std).view((1,-1,1,1))
    def __call__(self, x):
        x = (torch.from_numpy(x) * self.std) + self.mean
        #return torch.clip(x,-1,1).numpy().reshape((3,32,32))
        return x.numpy().reshape((3,32,32))
UnNor = UnNormalize()


for i in range(1):
    p = x_test[i].reshape((1,3,32,32))
    print(p.shape)
    #rsts = classifier.get_code_ae_cifar(x_test[i])
    code = classifier.get_code_ae(p)
    print(code.shape)
    #rec_code = classifier.input_code_ae_cifar(rsts[0])
    rec_code = classifier.input_code_ae(code)
    print(rec_code.shape)
    rec_img = UnNor(rec_code.reshape((3,32,32)))
    print(rec_img.max())
    print(rec_img.min())
    #print(rec_img)
    rec_img_ = (rec_img*255).astype(np.uint8).transpose(1, 2, 0)
    f_name = "/art/vis_cifar/ae/Nor_image_"+str(i)+".png"
    save_image(rec_img_, f_name)
    loss_v = classifier.compute_loss_ae_cifar(p, batch_size=1)
    print('ae loss is %f' %(loss_v))




# Step 4: Train the ART classifier
#classifier.fit_ae(x_train, y_train, batch_size=64, nb_epochs=20)
#classifier.fit_ae(x_train, y_train, batch_size=32, nb_epochs=402)
#classifier.fit_ae(x_train, y_train, batch_size=32, nb_epochs=202)

#loss_v = classifier.compute_loss_ae_cifar(x_train, batch_size=64)
#print('ae loss is %d' %(loss_v))

#classifier.save('model_ae_dim8','/art/model/ae/')
#classifier.save('model_ae_dim128_nor','/art/model_cifar/ae/')
#classifier.save('model_ae_dim128_nor_clip','/art/model_cifar/ae/')
#classifier.save('model_ae_dim128_nor_clip_','/art/model_cifar/ae/')
#classifier.save('model_ae_dim128_nor_clip__','/art/model_cifar/ae/')
#classifier.save('model_ae_dim128_nor_clip___','/art/model_cifar/ae/')


#classifier.save('model_ae_dim256_nor_clip_','/art/model_cifar/ae/')

# Step 5: Evaluate the ART classifier on benign test examples

for i in range(5):
    rec_x = classifier.predict(x_test[i])
    print(rec_x.max())
    print(rec_x.min())
    rec_img = UnNor(rec_x.reshape((3, 32, 32)))
    print(rec_img.shape)
    print(rec_img.max())
    print(rec_img.min())
    #accuracy = np.sum(np.argmax(predictions, axis=1) == np.argmax(y_test, axis=1)) / len(y_test)
    #print("Accuracy on benign test examples: {}%".format(accuracy * 100))
    img_ = (UnNor(x_test[i])*255).astype(np.uint8).transpose(1, 2, 0)
    rec_img_ = (rec_img*255).astype(np.uint8).transpose(1, 2, 0)
    
    f_name = "/art/vis_cifar/ae/input_image_"+str(i)+".png"
    save_image(img_, f_name)
    f_name = "/art/vis_cifar/ae/rec_Nor_image_"+str(i)+".png"
    save_image(rec_img_, f_name)

'''
