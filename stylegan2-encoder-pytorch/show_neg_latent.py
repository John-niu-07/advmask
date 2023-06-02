#https://github.com/bryandlee/stylegan2-encoder-pytorch.git

import os
import random
import numpy as np

import torch
from torch import nn
import torch.nn.functional as F

import torchvision
from torchvision import datasets, transforms

from PIL import Image


from model import Generator, Encoder, Discriminator
from train_encoder import VGGLoss

import matplotlib.pyplot as plt


def image2tensor(image):
    image = torch.FloatTensor(image).permute(2,0,1).unsqueeze(0)/255.
    return (image-0.5)/0.5

def tensor2image(tensor):
    tensor = tensor.clamp_(-1., 1.).detach().squeeze().permute(1,2,0).cpu().numpy()
    return tensor*0.5 + 0.5

def imshow(img, size=5, cmap='jet'):
    plt.figure(figsize=(size,size))
    plt.imshow(img, cmap=cmap)
    plt.axis('off')
    plt.show()


device = 'cuda'
image_size=256

g_model_path = './checkpoint/generator_ffhq.pt'
g_ckpt = torch.load(g_model_path, map_location=device)

latent_dim = g_ckpt['args'].latent

generator = Generator(image_size, latent_dim, 8).to(device)
generator.load_state_dict(g_ckpt["g_ema"], strict=False)
generator.eval()
print('[generator loaded]')

truncation = 0.7
trunc = generator.mean_latent(4096).detach().clone()

channel_multiplier = g_ckpt['args'].channel_multiplier
discriminator = Discriminator(image_size, channel_multiplier).to(device)
discriminator.load_state_dict(g_ckpt["d"], strict=False)
discriminator.eval()
print('[discriminator loaded]')



latent_all = np.load('/face/Mask/AdversarialMask/patch/neg_latent_all.npy')

lat = torch.from_numpy(latent_all[1]).to(device)
with torch.no_grad():
    imgs_gen, _ =  generator([lat], 
                           input_is_latent=True,
                           truncation=truncation,
                           truncation_latent=trunc,
                           randomize_noise=False)

#imshow(tensor2image(imgs_gen), 10)






ll = latent_all.shape[0]
print(ll)

imgss = []
lat_old = torch.from_numpy(latent_all[0]).to(device)
for i in range(ll):
    lat = latent_all[i]
    lat = torch.from_numpy(lat).to(device)
    print(torch.norm(lat, p=2))
    print(torch.norm(lat - lat_old, p=2))
    lat_old = lat
    with torch.no_grad():
        imgs_gen, _ =  generator([lat], 
                           input_is_latent=True,
                           truncation=truncation,
                           truncation_latent=trunc,
                           randomize_noise=False)
    imgss.append( imgs_gen.detach().reshape((3,256,256)) )

k=0
img_h = []
img_v = []
tt=1
for img in imgss:
    if k<min(ll-1, 10):
    #if k<10:
        img_v.append(img)
        k+=1
        pred_inv = discriminator(img.unsqueeze(0))
        print(pred_inv.detach())
    else:
        img_v.append(img)
        pred_inv = discriminator(img.unsqueeze(0))
        print(pred_inv.detach())
        imgs = torch.cat([img_gen for img_gen in img_v], dim=2)
        img_h.append(imgs)
        img_v = []
        k=0

        tt+=1
        print(tt)
        

imgM = torch.cat([img_gen for img_gen in img_h], dim=1)
print(imgM.shape)

imshow(tensor2image(imgM), 10)



'''
vgg_loss = VGGLoss(device)

z = z0.detach().clone()
print(z.cpu().numpy().shape)

z.requires_grad = True
optimizer = torch.optim.Adam([z], lr=0.01)

for step in range(500):
    z_last = z.detach().clone().cpu().numpy()
    imgs_gen, _ = generator([z], 
                           input_is_latent=True, 
                           truncation=truncation,
                           truncation_latent=trunc, 
                           randomize_noise=False)

    z_hat = encoder(imgs_gen)
    
    loss = F.mse_loss(imgs_gen, imgs) + vgg_loss(imgs_gen, imgs) + F.mse_loss(z0, z_hat)*2.0
    
    optimizer.zero_grad()
    loss.backward()
    optimizer.step() 
    
    if (step+1)%100 == 0:
        print(f'step:{step+1}, loss:{loss.item()}')
        imgs_fakes = torch.cat([img_gen for img_gen in imgs_gen], dim=1)        
        imshow(tensor2image(torch.cat([imgs_real, imgs_fakes], dim=2)),10)

        z_now = z_hat.clone().detach().cpu().numpy()
        print(np.linalg.norm(z_now - z_last))

#np.save('/face/Mask/AdversarialMask/patch/target.npy', z_now)
np.save('/face/Mask/AdversarialMask/patch/init_face.npy', z_now)
'''
