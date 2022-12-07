"""
The script demonstrates a simple example of using ART with PyTorch. The example train a small model on the MNIST dataset
and creates adversarial examples using the Fast Gradient Sign Method. Here we use the ART classifier to train the model,
it would also be possible to provide a pretrained model to the ART classifier.
The parameters are chosen for reduced computational requirements of the script and not optimised for accuracy.
"""
import sys
import os
sys.path.append('/face/Mask/AdversarialMask')

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np

from art.attacks.evasion import FastGradientMethod
from art.estimators.classification import PyTorchClassifier
from art.attacks.evasion.projected_gradient_descent.projected_gradient_descent import ProjectedGradientDescent
from art.attacks.evasion.hop_skip_jump import HopSkipJump
from art.attacks.evasion.geometric_decision_based_attack import GeoDA

import utils
from config_art import patch_config_types
from nn_modules import LandmarkExtractor, FaceXZooProjector, TotalVariation, FaceXZooProjector_align
from utils import load_embedder, EarlyStopping, get_patch


from torchvision import transforms
from PIL import Image

import matplotlib.pyplot as plt
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import label_binarize
import matplotlib
from pathlib import Path
import pickle
#import seaborn as sns
#import pandas as pd
matplotlib.use('TkAgg')
import torchvision.transforms.functional as TF


global device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")



class Net(nn.Module):
    def __init__(self, config):
        super(Net, self).__init__()

        self.config = config
        self.embedders = utils.load_embedder(self.config.test_embedder_names, device=device)	
        self.none_mask_t = utils.load_mask(self.config, self.config.blue_mask_path, device)


        face_landmark_detector = utils.get_landmark_detector(self.config, device)
        self.location_extractor = LandmarkExtractor(device, face_landmark_detector, self.config.img_size).to(device)





        img1 = transforms.ToTensor()(Image.open("/face/Mask/AdversarialMask/datasets/CASIA/1302735//001_aligned.png").convert('RGB')).to(device).unsqueeze(0)
        img1 = F.interpolate(img1, (112, 112))
        for emb_name, emb_model in self.embedders.items():
            self.img1_feat = emb_model(img1.to(device)).detach()


        #img2 = transforms.ToTensor()(Image.open("/face/Mask/AdversarialMask/datasets/CASIA/4204960/001_aligned.png").convert('RGB')).to(device).unsqueeze(0)	
        img2 = transforms.ToTensor()(Image.open("/face/Mask/AdversarialMask/datasets/CASIA/4204960/004_aligned.png").convert('RGB')).to(device).unsqueeze(0)	
        img2 = F.interpolate(img2, (112, 112))
        for emb_name, emb_model in self.embedders.items():
            self.img2_feat = emb_model(img2.to(device)).detach()

			

        self.img_input = transforms.ToTensor()(Image.open("/face/Mask/AdversarialMask/datasets/CASIA/4204960/004_aligned.png").convert('RGB')).to(device).unsqueeze(0)	
        self.img_input = F.interpolate(self.img_input, (112, 112))

       

        img_logo = transforms.ToTensor()(Image.open("/face/Mask/AdversarialMask/datasets/nio.png").convert('RGB')).to(device).unsqueeze(0)
        img_logo = F.interpolate(img_logo, (35, 35))
        self.img_logoL = torch.zeros(3,112,112)
        self.img_logoL[:, 39:39+35,  39:39+35] = img_logo[0]

        self.img_M = torch.zeros(3,112,112)
        self.img_M[:, 39:39+35,  39:39+35] = torch.ones(3,35,35)

        '''
        fig = plt.figure()
        #plt.imshow(np.transpose(self.img_input[0].cpu().detach().numpy(),[1,2,0]))
        #plt.imshow(np.transpose(img1[0].cpu().detach().numpy(),[1,2,0]))
        plt.imshow(np.transpose(self.mask[0].cpu().detach().numpy(),[1,2,0]))
        plt.axis('off')
        plt.show()
	'''

    def get_logits(self, x):


        d1 = torch.sum(x * self.img1_feat)/  (torch.norm(x) * torch.norm(self.img1_feat) )
        d2 = torch.sum(x * self.img2_feat)/  (torch.norm(x) * torch.norm(self.img2_feat) )

        d_all = torch.stack([d1, d2])

        #d_nor = d_all/torch.sum(d_all) 
        #print(d_nor.cpu())
        #return d_nor
        print(d_all.cpu().detach())
        return d_all


		
    def forward(self, x):

        print(x)        
        angle = x[0]*360 - 180
        translate_x = x[1] 
        translate_y = x[2]
        scale = x[3] * 2
        shear = x[4]*360 - 180
        logo_aff = TF.affine(self.img_logoL, angle, (translate_x, translate_y), scale, shear)
        M_aff = TF.affine(self.img_M, angle, (translate_x, translate_y), scale, shear)

        
        #img_in = self.img_input * (1 - self.mask) + x * self.mask
        img_in = self.img_input * (1 - M_aff) + logo_aff

        '''                
        fig = plt.figure()
        #plt.imshow(np.transpose(img1_align[0].cpu().detach().numpy(),[1,2,0]))
        plt.imshow(np.transpose(img_in[0].cpu().detach().numpy(),[1,2,0]))  #detach() is bad for PGD, but ok for FG
        plt.axis('off')
        plt.show()
        '''

        for emb_name, emb_model in self.embedders.items():
            img_in_feat = emb_model(img_in.to(device))
		
        x = img_in_feat[0]
        x = self.get_logits(x).unsqueeze(0)
        return x



'''
img_logo = transforms.ToTensor()(Image.open("/face/Mask/AdversarialMask/datasets/nio.png").convert('RGB')).to(device).unsqueeze(0)
img_logo = F.interpolate(img_logo, (35, 35))


img_logoL = torch.zeros(3,112,112)  
img_logoL[:, 39:39+35,  39:39+35] = img_logo[0]

img_M = torch.zeros(3,112,112)  
img_M[:, 39:39+35,  39:39+35] = torch.ones(3,35,35)

angle = 0.6
angle = angle*360 - 180

translate_x = 0
translate_y = 0
translate_x = translate_x * 112
translate_y = translate_y * 112


scale = 0.4
scale = scale * 2

shear = 0.6
shear = shear*360 - 180
img_aff = TF.affine(img_logoL, angle, (translate_x, translate_y), scale, shear)
print(img_aff.shape)


img1 = transforms.ToTensor()(Image.open("/face/Mask/AdversarialMask/datasets/CASIA/1302735//001_aligned.png").convert('RGB')).to(device).unsqueeze(0)
img1 = F.interpolate(img1, (112, 112))[0]
M_aff = TF.affine(img_M, angle, (translate_x, translate_y), scale, shear)
img1 = img1 * (1 - M_aff) + img_aff

fig = plt.figure()
ax1 = fig.add_subplot(1,3,1)
plt.imshow(np.transpose(img_logoL.cpu().detach().numpy(),[1,2,0]))  #detach() is bad for PGD, but ok for F
plt.axis('off')


ax1 = fig.add_subplot(1,3,2)
plt.imshow(np.transpose(img_aff.cpu().detach().numpy(),[1,2,0]))  #detach() is bad for PGD, but ok for FG
plt.axis('off')

ax1 = fig.add_subplot(1,3,3)
plt.imshow(np.transpose(img1.cpu().detach().numpy(),[1,2,0]))  #detach() is bad for PGD, but ok for FG
plt.axis('off')
plt.show()
'''


mode = 'universal'
config = patch_config_types[mode]()


#adv_patch_cpu = utils.get_patch(config)
adv_patch_cpu = torch.ones(5) * 0.5
adv_patch_cpu[1] = 0
adv_patch_cpu[2] = 0
	
adv_patch_cpu = adv_patch_cpu.unsqueeze(0)
print(adv_patch_cpu)

x_test = adv_patch_cpu.detach().numpy()

print(x_test.shape)
#print(x_test)

y_target = np.array([[1., 0.]])
print(y_target.shape)
print(y_target)
# Step 2: Create the model


model = Net(config)
# Step 2a: Define the loss function and the optimizer

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.01)

# Step 3: Create the ART classifier
classifier = PyTorchClassifier(
    model=model,
    #clip_values=(0, 1),
    loss=criterion,
    optimizer=optimizer,
    #input_shape=(112, 112, 3),
    input_shape=(1, 5), #HSJ
    nb_classes=2,
)




#-------------------------

#attack = FastGradientMethod(estimator=classifier, eps=0.02)
#x_test_adv = attack.generate(x=x_test)

'''
attack = FastGradientMethod(estimator=classifier, eps=0.1)
for i in range(20):
    if i==0:
        print('--1--')
        x_test_adv = attack.generate(x=x_test)
    else:
        print('--'+str(i)+'--')
        x_test_adv = attack.generate(x=x_test_adv)

'''


attack = ProjectedGradientDescent(
            classifier,
            norm=np.inf,
            #eps=80.0 / 255.0,
            #eps_step=2.0 / 255.0,
            eps=255.0 / 255.0,
            eps_step=5.0 / 255.0,
            max_iter=40,
            #max_iter=15,
            targeted=True,
            num_random_init=5,
            batch_size=1,
            )

x_test_adv = attack.generate(x=x_test, y=y_target)


'''
attack = HopSkipJump(
            classifier,
            norm=np.inf,
            #norm=2,
            max_iter=10,
            #max_iter=10,
            #targeted=False,
            targeted=True, #Cannot
            #max_eval=5000,
            max_eval=500,
            #init_eval=100,
            init_eval=10,
            #init_size=100,
            init_size=10,
            batch_size=1,
            verbose=True,
            )
x_test_adv = attack.generate(x=x_test, y=y_target)


attack = GeoDA(
            classifier,
            norm=np.inf,
            max_iter=400,
            batch_size=1,
            verbose=False,
            #targeted=True, #Cannot
            )

x_test_adv = attack.generate(x=x_test)
#x_test_adv = attack.generate(x=x_test, y=y_target)
'''

#--------
embedders = utils.load_embedder(config.test_embedder_names, device=device)

img_target = transforms.ToTensor()(Image.open("/face/Mask/AdversarialMask/datasets/CASIA/1302735/001_aligned.png").convert('RGB')).to(device).unsqueeze(0)
img_target = F.interpolate(img_target, (112, 112))
for emb_name, emb_model in embedders.items():
    img_target_feat = emb_model(img_target.to(device))




img_input = transforms.ToTensor()(Image.open("/face/Mask/AdversarialMask/datasets/CASIA/4204960/004_aligned.png").convert('RGB')).to(device).unsqueeze(0)
img_input = F.interpolate(img_input, (112, 112))


print(np.mean(x_test_adv[0]))
x = x_test_adv[0]

angle = x[0]*360 - 180
translate_x = x[1] * 112
translate_y = x[2] * 112
scale = x[3] * 2
shear = x[4]*360 - 180
logo_aff = TF.affine(self.img_logoL, angle, (translate_x, translate_y), scale, shear)
M_aff = TF.affine(self.img_M, angle, (translate_x, translate_y), scale, shear)

img_in = img_input * (1 - M_aff) + logo_aff
for emb_name, emb_model in embedders.items():
    img_in_feat = emb_model(img_in.to(device))


#d = torch.sum(img_input_feat * img_in_feat)/  (torch.norm(img_input_feat) * torch.norm(img_in_feat) )
d = torch.sum(img_target_feat * img_in_feat)/  (torch.norm(img_target_feat) * torch.norm(img_in_feat) )
print(d)

fig = plt.figure()
plt.imshow(np.transpose(img_in[0].cpu().detach().numpy(),[1,2,0]))  #detach() is bad for PGD, but ok for FG
plt.axis('off')
plt.text(0, 0., "%s " % (d.cpu().detach().numpy()))
plt.show()


fig = plt.figure()
mask_print = torch.ones(3,112,112)  * (1 - mask) + x_test_adv * mask
plt.imshow(np.transpose(mask_print[0].cpu().detach().numpy(),[1,2,0]))  #detach() is bad for PGD, but ok for FG
plt.axis('off')
plt.text(0, 0., "%s " % (d.cpu().detach().numpy()))
plt.show()


'''
print('---')        
predictions = classifier.predict(x_test_adv)
print(predictions)

accuracy = np.sum(np.argmax(predictions, axis=1) == np.argmax(y_target, axis=1)) / len(y_target)
print("FT accuracy on benign test examples: {}%".format(accuracy * 100))
'''





