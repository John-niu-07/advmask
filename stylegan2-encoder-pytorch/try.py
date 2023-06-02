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

e_model_path = './checkpoint/encoder_ffhq.pt'
e_ckpt = torch.load(e_model_path, map_location=device)

encoder = Encoder(image_size, latent_dim).to(device)
encoder.load_state_dict(e_ckpt['e'])
encoder.eval()
print('[encoder loaded]')

truncation = 0.7
trunc = generator.mean_latent(4096).detach().clone()



channel_multiplier = g_ckpt['args'].channel_multiplier
discriminator = Discriminator(image_size, channel_multiplier).to(device)
discriminator.load_state_dict(g_ckpt["d"], strict=False)
discriminator.eval()
print('[discriminator loaded]')


'''
with torch.no_grad():
    latent = generator.get_latent(torch.randn(4*6, latent_dim, device=device))
    print(latent.shape)
    imgs_gen, _ = generator([latent],
                              truncation=truncation,
                              truncation_latent=trunc,
                              input_is_latent=True,
                              randomize_noise=True)

    result = []
    for row in imgs_gen.chunk(4, dim=0):
        result.append(torch.cat([img for img in row], dim=2))
    result = torch.cat(result, dim=1)
    print('generated samples:')
    imshow(tensor2image(result), size=15)
'''
batch_size = 2

transform = transforms.Compose([
    transforms.Resize(image_size),
    #transforms.CenterCrop(image_size),
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
])

#dataset = datasets.ImageFolder(root='./examples', transform=transform)
#loader = iter(torch.utils.data.DataLoader(dataset=dataset, batch_size=batch_size, shuffle=True))
#imgs, _ = next(loader)

#imgs = Image.open("/face/Mask/stylegan2-encoder-pytorch/examples/000001.png").convert('RGB')
imgs = Image.open("/face/Mask/AdversarialMask/datasets/CASIA/4204960/004_aligned.png").convert('RGB')
imgs2 = Image.open("/face/Mask/AdversarialMask/datasets/CASIA/4204960_align/004_aligned.png").convert('RGB')
imgs3 = Image.open("/face/Mask/AdversarialMask/datasets/CASIA/4204960_align/001_aligned.png").convert('RGB')
imgs4 = Image.open("/face/Mask/AdversarialMask/datasets/CASIA/4204960_align/002_aligned.png").convert('RGB')
imgs5 = Image.open("/face/Mask/AdversarialMask/datasets/CASIA/4204960_align/003_aligned.png").convert('RGB')
#imgs = Image.open("/face/Mask/idinvert_pytorch/results/inversion/my_test8/000019_aligned_ori.png").convert('RGB')

#imgs2 = Image.open("/face/Mask/AdversarialMask/datasets/CASIA/4204960_b/004.jpg").convert('RGB')
#imgs3 = Image.open("/face/Mask/AdversarialMask/datasets/CASIA/4204960_b/001.jpg").convert('RGB')
#imgs4 = Image.open("/face/Mask/AdversarialMask/datasets/CASIA/4204960_b/002.jpg").convert('RGB')
#imgs5 = Image.open("/face/Mask/AdversarialMask/datasets/CASIA/4204960_b/003.jpg").convert('RGB')

#imgs2 = Image.open("/face/Mask/AdversarialMask/patch/experiments/December/14-12-2022_21-07-06/saved_patches/patch_0.png").convert('RGB')
#imgs3 = Image.open("/face/Mask/AdversarialMask/patch/experiments/December/14-12-2022_21-07-06/saved_patches/patch_1.png").convert('RGB')
#imgs4 = Image.open("/face/Mask/AdversarialMask/patch/experiments/December/14-12-2022_21-07-06/saved_patches/patch_2.png").convert('RGB')
#imgs5 = Image.open("/face/Mask/AdversarialMask/patch/experiments/December/14-12-2022_21-07-06/saved_patches/patch_3.png").convert('RGB')


imgs = transform(imgs).unsqueeze(0)
imgs = imgs.to(device)

print(imgs.shape)
pred_inv = discriminator(imgs)
D_loss = F.softplus(pred_inv[0][0])
print(pred_inv[0][0])
#print(D_loss)

transform2 = transforms.Compose([
    transforms.Resize(image_size),
    transforms.ToTensor(),
])

print('$$')
print(discriminator(transform(imgs2).unsqueeze(0).to(device)))
print(discriminator(transform(imgs3).unsqueeze(0).to(device)))
print(discriminator(transform(imgs4).unsqueeze(0).to(device)))
print(discriminator(transform(imgs5).unsqueeze(0).to(device)))

imgs = transform(imgs4).unsqueeze(0).to(device)


with torch.no_grad():
    z0 = encoder(imgs)
    print(z0.shape)
    imgs_gen, _ =  generator([z0], 
                           input_is_latent=True,
                           truncation=truncation,
                           truncation_latent=trunc,
                           randomize_noise=False)

imgs_real = torch.cat([img for img in imgs], dim=1)
imgs_fakes = torch.cat([img_gen for img_gen in imgs_gen], dim=1)




print('===')
pred_inv = discriminator(imgs_real.unsqueeze(0))
D_loss = F.softplus(pred_inv[0][0])
print(pred_inv[0][0])
#print(D_loss)

pred_inv = discriminator(imgs_fakes.unsqueeze(0))
D_loss = F.softplus(pred_inv[0][0])
print(pred_inv[0][0])
#print(D_loss)


print('initial projections:')
imshow(tensor2image(torch.cat([imgs_real, imgs_fakes], dim=2)),10)
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
