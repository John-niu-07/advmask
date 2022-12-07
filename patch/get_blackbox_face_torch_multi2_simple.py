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

global device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class Net(nn.Module):
    def __init__(self, config):
        super(Net, self).__init__()

        self.config = config
        face_landmark_detector = utils.get_landmark_detector(self.config, device)
        
        self.location_extractor = LandmarkExtractor(device, face_landmark_detector, self.config.img_size).to(device)
        self.none_mask_t = utils.load_mask(self.config, self.config.blue_mask_path, device)
        self.embedders = utils.load_embedder(self.config.test_embedder_names, device=device)	


        img1 = transforms.ToTensor()(Image.open("/face/Mask/AdversarialMask/datasets/CASIA/1302735/003_aligned.png").convert('RGB')).to(device).unsqueeze(0)
        img1 = F.interpolate(img1, (112, 112))
        for emb_name, emb_model in self.embedders.items():
            self.img1_feat = emb_model(img1.to(device)).detach()

        img2 = transforms.ToTensor()(Image.open("/face/Mask/AdversarialMask/datasets/CASIA/4204960/001_aligned.png").convert('RGB')).to(device).unsqueeze(0)	
        img2 = F.interpolate(img2, (112, 112))
        for emb_name, emb_model in self.embedders.items():
            self.img2_feat = emb_model(img2.to(device)).detach()


        self.mask = transforms.ToTensor()(Image.open("/face/Mask/AdversarialMask/datasets/012_mask.png").convert('RGB')).to(device).unsqueeze(0)
        self.mask = F.interpolate(self.mask, (112, 112))


	
        self.img_input = []
        self.preds_input = []
        self.img_input.append( transforms.ToTensor()(Image.open("/face/Mask/AdversarialMask/datasets/CASIA/4204960/001_aligned.png").convert('RGB')).to(device).unsqueeze(0) )
        self.img_input.append( transforms.ToTensor()(Image.open("/face/Mask/AdversarialMask/datasets/CASIA/4204960/002_aligned.png").convert('RGB')).to(device).unsqueeze(0) )
        self.img_input.append( transforms.ToTensor()(Image.open("/face/Mask/AdversarialMask/datasets/CASIA/4204960/003_aligned.png").convert('RGB')).to(device).unsqueeze(0) )
        self.img_input.append( transforms.ToTensor()(Image.open("/face/Mask/AdversarialMask/datasets/CASIA/4204960/004_aligned.png").convert('RGB')).to(device).unsqueeze(0) )
        self.img_input.append( transforms.ToTensor()(Image.open("/face/Mask/AdversarialMask/datasets/CASIA/4204960/005_aligned.png").convert('RGB')).to(device).unsqueeze(0) )
        self.img_input.append( transforms.ToTensor()(Image.open("/face/Mask/AdversarialMask/datasets/CASIA/4204960/006_aligned.png").convert('RGB')).to(device).unsqueeze(0) )
        self.img_input.append( transforms.ToTensor()(Image.open("/face/Mask/AdversarialMask/datasets/CASIA/4204960/007_aligned.png").convert('RGB')).to(device).unsqueeze(0) )
        self.img_input.append( transforms.ToTensor()(Image.open("/face/Mask/AdversarialMask/datasets/CASIA/4204960/008_aligned.png").convert('RGB')).to(device).unsqueeze(0) )
        self.img_input.append( transforms.ToTensor()(Image.open("/face/Mask/AdversarialMask/datasets/CASIA/4204960/009_aligned.png").convert('RGB')).to(device).unsqueeze(0) )
        self.img_input.append( transforms.ToTensor()(Image.open("/face/Mask/AdversarialMask/datasets/CASIA/4204960/010_aligned.png").convert('RGB')).to(device).unsqueeze(0) )

        self.img_input.append( transforms.ToTensor()(Image.open("/face/Mask/AdversarialMask/datasets/CASIA/4204960/011_aligned.png").convert('RGB')).to(device).unsqueeze(0) )
        self.img_input.append( transforms.ToTensor()(Image.open("/face/Mask/AdversarialMask/datasets/CASIA/4204960/012_aligned.png").convert('RGB')).to(device).unsqueeze(0) )
        self.img_input.append( transforms.ToTensor()(Image.open("/face/Mask/AdversarialMask/datasets/CASIA/4204960/013_aligned.png").convert('RGB')).to(device).unsqueeze(0) )
        self.img_input.append( transforms.ToTensor()(Image.open("/face/Mask/AdversarialMask/datasets/CASIA/4204960/014_aligned.png").convert('RGB')).to(device).unsqueeze(0) )
        self.img_input.append( transforms.ToTensor()(Image.open("/face/Mask/AdversarialMask/datasets/CASIA/4204960/015_aligned.png").convert('RGB')).to(device).unsqueeze(0) )
        self.img_input.append( transforms.ToTensor()(Image.open("/face/Mask/AdversarialMask/datasets/CASIA/4204960/016_aligned.png").convert('RGB')).to(device).unsqueeze(0) )
        self.img_input.append( transforms.ToTensor()(Image.open("/face/Mask/AdversarialMask/datasets/CASIA/4204960/017_aligned.png").convert('RGB')).to(device).unsqueeze(0) )
        self.img_input.append( transforms.ToTensor()(Image.open("/face/Mask/AdversarialMask/datasets/CASIA/4204960/018_aligned.png").convert('RGB')).to(device).unsqueeze(0) )
        self.img_input.append( transforms.ToTensor()(Image.open("/face/Mask/AdversarialMask/datasets/CASIA/4204960/019_aligned.png").convert('RGB')).to(device).unsqueeze(0) )
        self.img_input.append( transforms.ToTensor()(Image.open("/face/Mask/AdversarialMask/datasets/CASIA/4204960/020_aligned.png").convert('RGB')).to(device).unsqueeze(0) )
        
        for i in range(20):
            self.img_input[i] = F.interpolate(self.img_input[i], (112, 112))


    def get_logits(self, x):


        d2 = torch.sum(x[0] * self.img2_feat)/  (torch.norm(x[0]) * torch.norm(self.img2_feat) )

        d1 = []
        for i in range(20):
            d1.append( torch.sum(x[i] * self.img1_feat)/  (torch.norm(x[i]) * torch.norm(self.img1_feat) ) )

        kk = torch.stack(d1)
        dd1 = torch.mean(kk)
        #print(kk)

        d_all = torch.stack([dd1, d2])

        print(d_all.cpu().detach())
        return d_all

		
    def forward(self, x):

        img_in = []
        img_in_feat = []
        for i in range(20):
            #img_in.append( self.fxz_projector(self.img_input[i], self.preds_input[i], adv_patch=x, uv_mask_src=self.none_mask_t[:, 3], is_3d=True) )
            img_in.append( self.img_input[i] * (1 - self.mask) + x * self.mask )
            for emb_name, emb_model in self.embedders.items():
                img_in_feat.append( emb_model(img_in[i].to(device))[0] )

        '''                
        fig = plt.figure()
        #plt.imshow(np.transpose(img1_align[0].cpu().detach().numpy(),[1,2,0]))
        plt.imshow(np.transpose(img_in[0].cpu().detach().numpy(),[1,2,0]))  #detach() is bad for PGD, but ok for FG
        plt.axis('off')
        plt.show()
        '''

		
        x = self.get_logits(img_in_feat).unsqueeze(0)

        return x



mode = 'universal'
config = patch_config_types[mode]()


#adv_patch_cpu = utils.get_patch(config)
adv_patch_cpu = torch.ones((1, 3, 112, 112), dtype=torch.float32) * 0.5
		
# Step 1a: Swap axes to PyTorch's NCHW format

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
    clip_values=(0, 1),
    loss=criterion,
    optimizer=optimizer,
    input_shape=(112, 112, 3),
    nb_classes=2,
)




#-------------------------
'''
#attack = FastGradientMethod(estimator=classifier, eps=0.8)
#x_test_adv = attack.generate(x=x_test)


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
            #eps=40.0 / 255.0,
            eps=80.0 / 255.0,
            #eps_step=2.0 / 255.0,
            eps_step=2.0 / 255.0,
            max_iter=40,
            targeted=True,
            num_random_init=1,
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
            init_eval=100,
            init_size=100,
            batch_size=1,
            verbose=True,
            )

attack = GeoDA(
            classifier,
            norm=np.inf,
            max_iter=4000,
            batch_size=1,
            verbose=False,
            )

x_test_adv = attack.generate(x=x_test, y=y_target)
'''

#--------
embedders = utils.load_embedder(config.test_embedder_names, device=device)

mask = transforms.ToTensor()(Image.open("/face/Mask/AdversarialMask/datasets/012_mask.png").convert('RGB')).to(device).unsqueeze(0)
mask = F.interpolate(mask, (112, 112))



img_target = transforms.ToTensor()(Image.open("/face/Mask/AdversarialMask/datasets/CASIA/1302735//003_aligned.png").convert('RGB')).to(device).unsqueeze(0)
img_target = F.interpolate(img_target, (112, 112))
for emb_name, emb_model in embedders.items():
    img_target_feat = emb_model(img_target.to(device))




img_input = transforms.ToTensor()(Image.open("/face/Mask/AdversarialMask/datasets/CASIA/4204960/001_aligned.png").convert('RGB')).to(device).unsqueeze(0)
img_input = F.interpolate(img_input, (112, 112))


print(np.mean(x_test_adv[0]))

x_test_adv = transforms.ToTensor()(x_test_adv[0].transpose(1,2,0)).to(device).unsqueeze(0)
#img_in = fxz_projector(img_input, preds_input, adv_patch=x_test_adv, uv_mask_src=none_mask_t[:, 3], is_3d=True)
img_in = img_input * (1 - mask) + x_test_adv * mask
for emb_name, emb_model in embedders.items():
    img_in_feat = emb_model(img_in.to(device))


#d = torch.sum(img_input_feat * img_in_feat)/  (torch.norm(img_input_feat) * torch.norm(img_in_feat) )
d = torch.sum(img_target_feat * img_in_feat)/  (torch.norm(img_target_feat) * torch.norm(img_in_feat) )
print(d)



#---
img_input2 = transforms.ToTensor()(Image.open("/face/Mask/AdversarialMask/datasets/CASIA/4204960/002_aligned.png").convert('RGB')).to(device).unsqueeze(0)
img_input2 = F.interpolate(img_input2, (112, 112))
img_in2 = img_input2 * (1 - mask) + x_test_adv * mask
for emb_name, emb_model in embedders.items():
    img_in2_feat = emb_model(img_in2.to(device))

d = torch.sum(img_target_feat * img_in2_feat)/  (torch.norm(img_target_feat) * torch.norm(img_in2_feat) )
print(d)






fig = plt.figure()
plt.imshow(np.transpose(img_in[0].cpu().detach().numpy(),[1,2,0]))  #detach() is bad for PGD, but ok for FG
plt.axis('off')
plt.show()
        
