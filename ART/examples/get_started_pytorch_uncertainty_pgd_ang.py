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


from art.visualization import create_sprite, convert_to_rgb, save_image, plot_3d
import torchvision.transforms as transforms
from PIL import Image

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

#https://github.com/dariocazzani/pytorch-AE/blob/master/architectures.py
class FC_Encoder(nn.Module):
    def __init__(self, output_size):
        super(FC_Encoder, self).__init__()
        self.fc1 = nn.Linear(784, output_size)

    def forward(self, x):
        h1 = F.relu(self.fc1(x))
        return h1

class FC_Decoder(nn.Module):
    def __init__(self, embedding_size):
        super(FC_Decoder, self).__init__()
        self.fc3 = nn.Linear(embedding_size, 1024)
        self.fc4 = nn.Linear(1024, 784)

    def forward(self, z):
        h3 = F.relu(self.fc3(z))
        return torch.sigmoid(self.fc4(h3))

class CNN_Encoder(nn.Module):
    def __init__(self, output_size, input_size=(1, 28, 28)):
        super(CNN_Encoder, self).__init__()

        self.input_size = input_size
        self.channel_mult = 16

        #convolutions
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels=1,
                     out_channels=self.channel_mult*1,
                     kernel_size=4,
                     stride=1,
                     padding=1),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(self.channel_mult*1, self.channel_mult*2, 4, 2, 1),
            nn.BatchNorm2d(self.channel_mult*2),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(self.channel_mult*2, self.channel_mult*4, 4, 2, 1),
            nn.BatchNorm2d(self.channel_mult*4),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(self.channel_mult*4, self.channel_mult*8, 4, 2, 1),
            nn.BatchNorm2d(self.channel_mult*8),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(self.channel_mult*8, self.channel_mult*16, 3, 2, 1),
            nn.BatchNorm2d(self.channel_mult*16),
            nn.LeakyReLU(0.2, inplace=True)
        )

        self.flat_fts = self.get_flat_fts(self.conv)

        self.linear = nn.Sequential(
            nn.Linear(self.flat_fts, output_size),
            nn.BatchNorm1d(output_size),
            nn.LeakyReLU(0.2),
        )

    def get_flat_fts(self, fts):
        f = fts(Variable(torch.ones(1, *self.input_size)))
        return int(np.prod(f.size()[1:]))

    def forward(self, x):
        x = self.conv(x.view(-1, *self.input_size))
        x = x.view(-1, self.flat_fts)
        return self.linear(x)

class CNN_Decoder(nn.Module):
    def __init__(self, embedding_size, input_size=(1, 28, 28)):
        super(CNN_Decoder, self).__init__()
        self.input_height = 28
        self.input_width = 28
        self.input_dim = embedding_size
        self.channel_mult = 16
        self.output_channels = 1
        self.fc_output_dim = 512

        self.fc = nn.Sequential(
            nn.Linear(self.input_dim, self.fc_output_dim),
            nn.BatchNorm1d(self.fc_output_dim),
            nn.ReLU(True)
        )

        self.deconv = nn.Sequential(
            # input is Z, going into a convolution
            nn.ConvTranspose2d(self.fc_output_dim, self.channel_mult*4,
                                4, 1, 0, bias=False),
            nn.BatchNorm2d(self.channel_mult*4),
            nn.ReLU(True),
            # state size. self.channel_mult*32 x 4 x 4
            nn.ConvTranspose2d(self.channel_mult*4, self.channel_mult*2,
                                3, 2, 1, bias=False),
            nn.BatchNorm2d(self.channel_mult*2),
            nn.ReLU(True),
            # state size. self.channel_mult*16 x 7 x 7
            nn.ConvTranspose2d(self.channel_mult*2, self.channel_mult*1,
                                4, 2, 1, bias=False),
            nn.BatchNorm2d(self.channel_mult*1),
            nn.ReLU(True),
            # state size. self.channel_mult*8 x 14 x 14
            nn.ConvTranspose2d(self.channel_mult*1, self.output_channels, 4, 2, 1, bias=False),
            nn.Sigmoid()
            # state size. self.output_channels x 28 x 28
        )

    def forward(self, x):
        x = self.fc(x)
        x = x.view(-1, self.fc_output_dim, 1, 1)
        x = self.deconv(x)
        return x.view(-1, self.input_width*self.input_height)


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
        z = self.encode(x.view(-1, 784))
        return self.decode(z)

class AE_loss(nn.Module):

    def __init__(self, args=None):
        super(AE_loss, self).__init__()

    def forward(self, recon_x, x):
        BCE = F.binary_cross_entropy(recon_x, x.view(-1, 784), reduction='sum')
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
        (var, mean) = torch.var_mean(input, 1, unbiased=False, keepdim=True)
        #print(np.shape(var))
        #print(var)

        (var2, mean2) = torch.var_mean(var, unbiased=False)
        #print(mean2)
        return -1 * mean2
    #return  mean2
        #return F.nll_loss(input, target, reduction='mean')




# Step 1: Load the MNIST dataset

(x_train, y_train), (x_test, y_test), min_pixel_value, max_pixel_value = load_mnist()

# Step 1a: Swap axes to PyTorch's NCHW format

x_train = np.transpose(x_train, (0, 3, 1, 2)).astype(np.float32)
x_test = np.transpose(x_test, (0, 3, 1, 2)).astype(np.float32)




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
classifier.fit(x_train, y_train, batch_size=64, nb_epochs=3)

# Step 5: Evaluate the ART classifier on benign test examples
predictions = classifier.predict(x_test)
accuracy = np.sum(np.argmax(predictions, axis=1) == np.argmax(y_test, axis=1)) / len(y_test)
print("Accuracy on benign test examples: {}%".format(accuracy * 100))
#classifier.save('model_pgd_untargeted','/art/model/untargeted/')




print("----pgd_untargeted_attack----")

#attack = FastGradientMethod(estimator=classifier, eps=0.2)
#attack = ProjectedGradientDescent(classifier, eps=1.0, eps_step=0.1, verbose=False)
#attack = ProjectedGradientDescent(classifier, eps=0.2, eps_step=0.1, verbose=False)

eps_=5
attack = ProjectedGradientDescent(
            classifier,
            norm=np.inf,
            eps=5.0 / 255.0,
            #eps=8.0 / 255.0,
            #eps=40.0 / 255.0,
            eps_step=2.0 / 255.0,
            max_iter=40,
            targeted=False,
            num_random_init=5,
            batch_size=32,
            )

#x_test_adv = attack.generate(x=x_test)
#predictions = classifier.predict(x_test_adv)
#accuracy = np.sum(np.argmax(predictions, axis=1) == np.argmax(y_test, axis=1)) / len(y_test)
#print("pgd_untargeted_attack: Accuracy on adversarial test examples: {}%".format(accuracy * 100))



sample = x_test[0]
#print(sample.shape)
sample_img = (sample[0]*255).astype(np.uint8)
save_image(sample_img, "/art/vis/ae/sample.png")


sample_adv = attack.generate(x=sample)
predictions = classifier.predict(sample_adv)
rst = np.argmax(predictions, axis=1)
print("pgd_untargeted_attack: class = {}".format(rst))
print("pgd_untargeted_attack: {}".format(np.exp(predictions)))

sample_adv_img = (sample_adv[0]*255).astype(np.uint8)
save_image(sample_adv_img, "/art/vis/ae/sample_adv"+str(eps_)+".png")


pert = np.abs(sample_adv[0] - sample[0])
#print(pert)
pert_img = (pert*255).astype(np.uint8)
save_image(pert_img, "/art/vis/ae/pert_adv"+str(eps_)+".png")
n_pert = np.linalg.norm(pert.reshape((784)), ord=2)
print("norm of pert: {}".format(n_pert))


#####
#load AE
dim_latent = 8
dim_latent = 16
model3 = AE_net(dim_latent) 
criterion3 = AE_loss()
optimizer3 = optim.Adam(model3.parameters(), lr=0.001)
#model3.load_state_dict(torch.load('/art/model/ae/model_ae_dim8.model'))
model3.load_state_dict(torch.load('/art/model/ae/model_ae_dim16.model'))
# Step 3: Create the ART classifier
classifier3 = PyTorchClassifier(
        model=model3,
        clip_values=(min_pixel_value, max_pixel_value),
        loss=criterion3,
        optimizer=optimizer3,
        input_shape=(1, 28, 28),
        nb_classes=10,
        )


#sample = x_test[0]
sample_code = classifier3.get_code_ae(sample)
#print(sample_code)
rec_sample = classifier3.input_code_ae(sample_code)
rec_sample_img = (rec_sample.reshape((28,28))*255).astype(np.uint8)
save_image(rec_sample_img, "/art/vis/ae/rec_sample.png")

sample_adv_code = classifier3.get_code_ae(sample_adv)
#print(sample_adv_code)
rec_sample_adv = classifier3.input_code_ae(sample_adv_code)
rec_sample_adv_img = (rec_sample_adv.reshape((28,28))*255).astype(np.uint8)
save_image(rec_sample_adv_img, "/art/vis/ae/rec_sample_adv"+str(eps_)+".png")


pert_code = sample_code[0] - sample_adv_code[0]
n_pert_code = np.linalg.norm(pert_code, ord=2)
print(pert_code)
print("norm of pert_code: {}".format(n_pert_code))


######get base
delta=0.001

bb_code = sample_code[0].copy()
print("bb_code: {}".format(bb_code))

base_s_list = []
for i in range(dim_latent):
    bb_code = sample_code[0].copy()
    bb_code[i] = bb_code[i] + delta
    inpu = np.expand_dims(bb_code, axis=0)
    #print(inpu)
    rec_bb = classifier3.input_code_ae(inpu)

    
    '''
    rec_bb_img = (rec_bb.reshape((28,28))*255).astype(np.uint8)
    save_image(rec_bb_img, "/art/vis/ae/rec_bb_"+str(i)+"_img.png")
    '''

    #base_s = np.abs(rec_bb[0] - rec_sample[0])
    base_s = rec_bb[0] - rec_sample[0]
    #print(base_s)
    base_s_list.append(base_s)

    base_s_img = (np.abs(base_s.reshape((28,28)))*255*1000).astype(np.uint8)
    save_image(base_s_img, "/art/vis/ae/base_s"+str(i)+".png")
    '''
    pos_part = np.where(base_s > 0, base_s, 0)
    base_s_img = (pos_part.reshape((28,28))*255*1000).astype(np.uint8)
    save_image(base_s_img, "/art/vis/ae/base_s_pos_"+str(i)+".png")
    n_pos = np.linalg.norm(pos_part, ord=2)
    print("norm of pos: {}".format(n_pos))

    neg_part = np.abs(np.where(base_s < 0, base_s, 0))
    base_s_img = (neg_part.reshape((28,28))*255*1000).astype(np.uint8)
    save_image(base_s_img, "/art/vis/ae/base_s_neg_"+str(i)+".png")
    n_neg = np.linalg.norm(neg_part, ord=2)
    print("norm of neg: {}".format(n_neg))
    '''

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

print('...lstsq....')
#img_v = sample.reshape((784, 1))
#img_v = sample.reshape((784, 1)) - rec_sample[0].reshape((784, 1))
'''
for j in range(40):
    input_sample = x_test[j]
    input_img = (input_sample.reshape((28,28))*255).astype(np.uint8)
    save_image(input_img, "/art/vis/ae/input_img_"+str(j)+".png")
'''
input_sample = x_test[17]
input_sample = x_test[26]

#input_sample = x_test[0]
#input_sample = sample_adv
noise = np.abs(np.random.randn(28,28))
#input_sample = x_test[0][0] + noise * 0.05
#input_sample = x_test[0][0] + noise * 0.09
input_sample = x_test[0][0] + noise * 0.14

input_img = (input_sample.reshape((28,28))*255).astype(np.uint8)
save_image(input_img, "/art/vis/ae/input.png")
n_input = np.linalg.norm(input_sample.reshape((784)), ord=2)
print("norm of input: {}".format(n_input))


#### Should an input be a diff?
img_d = input_sample.reshape((784, 1)) - rec_sample[0].reshape((784, 1))
#print(img_d.shape)
n_img_d = np.linalg.norm(img_d, ord=2)
print("norm of d: {}".format(n_img_d))
img_d_ = (np.abs(img_d).reshape((28,28))*255*10).astype(np.uint8)
save_image(img_d_, "/art/vis/ae/input_d.png")

lstsq_rst = np.linalg.lstsq(Base, img_d, rcond = None)
X = lstsq_rst[0]
Res = lstsq_rst[1]
Base_rank = lstsq_rst[2]

print("res: {}".format(Res[0]))
#print("res: {}".format(lstsq_rst))
coeff = np.transpose( X )[0]
print("coeff: {}".format(coeff))




'''
######proj
coeff_list = []
#img_v = rec_sample[0]
img_v = sample.reshape((784))
for i in range(8):
    base_s = base_s_list[i]
    coeff = np.dot(img_v, base_s) / np.linalg.norm(base_s, ord=2)
    print("coeff_{}:  {}".format(i, coeff))
    coeff_list.append(coeff)
'''



### proj
proj_d = np.zeros_like(base_s_list[0])
for i in range(dim_latent):
    base_s = base_s_list[i]
    proj_d += coeff[i] * base_s

#proj_d_img = (proj_d.reshape((28,28))*255).astype(np.uint8)
proj_d_img = (np.abs(proj_d).reshape((28,28))*255*10).astype(np.uint8)
save_image(proj_d_img, "/art/vis/ae/proj_d.png")
n_proj_d = np.linalg.norm(proj_d, ord=2)
print("norm of proj_d: {}".format(n_proj_d))

#add rec?
proj = proj_d + rec_sample[0]
proj_img = (np.abs(proj).reshape((28,28))*255).astype(np.uint8)
#proj_img = (proj.reshape((28,28))*255).astype(np.uint8)
save_image(proj_img, "/art/vis/ae/proj.png")
n_proj = np.linalg.norm(proj, ord=2)
print("norm of proj: {}".format(n_proj))


diff = np.abs(proj - input_sample.reshape((784)))
diff_img = (diff.reshape((28,28))*255).astype(np.uint8)
save_image(diff_img, "/art/vis/ae/diff.png")
n_diff = np.linalg.norm(diff, ord=2)
print("norm of diff: {}".format(n_diff))



'''
print("----PGD_uncertainty_attack----")

#model_trained = classifier.model()
model_trained = copy.deepcopy(classifier.model)
#print(model_trained)
model2 = Net2()
#criterion2 = nn.NLLLoss()
criterion2 = MyLoss()
optimizer2 = optim.Adam(model.parameters(), lr=0.01)

model2.load_state_dict(model_trained.state_dict())


classifier2 = PyTorchClassifier(
    model=model2,
    clip_values=(min_pixel_value, max_pixel_value),
    loss=criterion2,
    optimizer=optimizer2,
    input_shape=(1, 28, 28),
    nb_classes=10,
)


# Step 6: Generate adversarial test examples
#attack = FastGradientMethod(estimator=classifier2, eps=0.2)
#x_test_adv = attack.generate(x=x_test)

# Test PGD with np.inf norm
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
predictions = classifier.predict(x_test_adv)

#print(np.shape(predictions))
print(np.exp(predictions[0:4]))
#var = np.var(np.exp(predictions), axis=1)
maxv = np.max(np.exp(predictions), axis=1)
#print(np.shape(var))
#print(var[0:10])
#print(np.shape(maxv))
print(maxv[0:10])
#mean2 = np.mean(var)
mean2 = np.mean(maxv)
print(mean2)

accuracy = np.sum(np.argmax(predictions, axis=1) == np.argmax(y_test, axis=1)) / len(y_test)
print("PGD_uncertainty_attack: Accuracy on adversarial test examples: {}%".format(accuracy * 100))
uncertainty_perturbation = x_test_adv - x_test




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
'''

############
'''
#train AE
dim_latent = 8
dim_latent = 16
model3 = AE_net(dim_latent) 
criterion3 = AE_loss()
optimizer3 = optim.Adam(model3.parameters(), lr=0.001)


#model3.load_state_dict(torch.load('/art/model/ae/model_ae_dim8.model'))
# Step 3: Create the ART classifier
classifier = PyTorchClassifier(
        model=model3,
        clip_values=(min_pixel_value, max_pixel_value),
        loss=criterion3,
        optimizer=optimizer3,
        input_shape=(1, 28, 28),
        nb_classes=10,
        )


for i in range(2):
    code = classifier.get_code_ae(x_test[i])
    print(code)
    rec_code = classifier.input_code_ae(code)
    print(rec_code.shape)
    for j in range(len(rec_code)):
        rec_img = rec_code[j].reshape((28,28))
        rec_img_ = (rec_img*255).astype(np.uint8)
        f_name = "/art/vis/ae/dec_image_"+str(i)+"_"+str(j)+".png"
        save_image(rec_img_, f_name)





# Step 4: Train the ART classifier
#classifier.fit_ae(x_train, y_train, batch_size=64, nb_epochs=20)
classifier.fit_ae(x_train, y_train, batch_size=64, nb_epochs=120)
loss_v = classifier.compute_loss_ae(x_train, y_train)
print('ae loss is %d' %(loss_v))
#classifier.save('model_ae_dim8','/art/model/ae/')
classifier.save('model_ae_dim16','/art/model/ae/')



# Step 5: Evaluate the ART classifier on benign test examples

for i in range(5):
    rec_x = classifier.predict(x_test[i])
    rec_img = rec_x.reshape((1, 28, 28))
    print(rec_img.shape)
    #accuracy = np.sum(np.argmax(predictions, axis=1) == np.argmax(y_test, axis=1)) / len(y_test)
    #print("Accuracy on benign test examples: {}%".format(accuracy * 100))
    img_ = (x_test[i][0]*255).astype(np.uint8)
    rec_img_ = (rec_img[0]*255).astype(np.uint8)
    
    f_name = "/art/vis/ae/input_image_"+str(i)+".png"
    #save_image(img_, f_name)
    f_name = "/art/vis/ae/rec_d8_image_"+str(i)+".png"
    save_image(rec_img_, f_name)
'''

