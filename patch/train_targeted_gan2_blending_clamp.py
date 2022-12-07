import sys
import os
sys.path.append('/face/Mask/AdversarialMask')
    
import random
from pathlib import Path
import pickle

import torch
import numpy as np
from tqdm import tqdm
from torchvision import transforms
import torch.optim as optim
import matplotlib.pyplot as plt
from PIL import Image

import utils
import losses
from config_gan import patch_config_types
from nn_modules import LandmarkExtractor, FaceXZooProjector, TotalVariation
from utils import load_embedder, EarlyStopping, get_patch
import skimage.io as io

import pickle
import torch.nn.functional as F
from art.estimators.classification.stylegan_encoder import StyleGANEncoder
from art.estimators.classification.stylegan_generator import StyleGANGenerator

import torchvision.models as models
from torch.autograd import Variable

from art.config import ART_NUMPY_DTYPE
from typing import Optional, Union, TYPE_CHECKING


import warnings
warnings.simplefilter('ignore', UserWarning)

global device
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print('device is {}'.format(device), flush=True)


def set_random_seed(seed_value):
    np.random.seed(seed_value)
    torch.manual_seed(seed_value)
    random.seed(seed_value)
    os.environ['PYTHONHASHSEED'] = str(seed_value)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed_value)
        torch.cuda.manual_seed_all(seed_value)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

loader = transforms.Compose([
    transforms.Resize([112,112]),  # scale imported image
    transforms.ToTensor()])  # transform it into a torch tensor

def image_loader(image_name):
    image = Image.open(image_name)
    image = Variable(loader(image))
    # fake batch dimension required to fit network's input dimensions
    image = image.unsqueeze(0)
    return image

def _get_tensor_value(tensor):
  """Gets the value of a torch Tensor."""
  return tensor.cpu().detach().numpy()


class AdversarialMask:
    def __init__(self, config):
        self.config = config
        set_random_seed(seed_value=self.config.seed)

        self.train_no_aug_loader, self.train_loader = utils.get_train_loaders(self.config)

        self.embedders = load_embedder(self.config.train_embedder_names, device)

        face_landmark_detector = utils.get_landmark_detector(self.config, device)
        self.location_extractor = LandmarkExtractor(device, face_landmark_detector, self.config.img_size).to(device)
        self.fxz_projector = FaceXZooProjector(device, self.config.img_size, self.config.patch_size).to(device)
        self.total_variation = TotalVariation(device).to(device)
        self.dist_loss = losses.get_loss(self.config)

        self.train_losses_epoch = []
        self.train_losses_iter = []
        self.dist_losses = []
        self.tv_losses = []
        self.val_losses = []

        self.create_folders()
        utils.save_class_to_file(self.config, self.config.current_dir)
        #self.target_embedding = utils.get_person_embedding(self.config, self.train_no_aug_loader, self.config.celeb_lab_mapper, self.location_extractor,
        #                                                   self.fxz_projector, self.embedders, device)
        self.target_embedding = utils.get_person_embedding_z(self.config, self.train_no_aug_loader, self.config.celeb_lab_mapper, self.location_extractor,
                                                           self.fxz_projector, self.embedders, device)
        self.best_patch = None
        print(self.config.celeb_lab_mapper)

        self.blue_mask_t = utils.load_mask(self.config, self.config.blue_mask_path, device)

        self.full_mask = utils.load_mask(self.config, self.config.full_mask_path, device)
        self.mask_t = transforms.ToTensor()(Image.open('../prnet/new_uvT.png').convert('L'))
        self.latent_z_init = None
        self.latent_z_init_np = None
        self.latent_z_mask = None
        
        self.latent_neg_z = None
        self.latent_neg_z2 = None
        self.latent_neg_z3 = None
        self.latent_neg_z4 = None
        self.latent_neg_z5 = None
        self.latent_neg_z6 = None
        self.latent_neg_z7 = None
        self.latent_neg_z8 = None

        #self.patch_mask = transforms.ToTensor()(Image.open('/face/Mask/AdversarialMask/datasets/012_mask5.png').convert('L')).to(device).unsqueeze(0)
        self.patch_mask = transforms.ToTensor()(Image.open('/face/Mask/AdversarialMask/datasets/012_mask6.png').convert('L')).to(device).unsqueeze(0)
        self.patch_mask = F.interpolate(self.patch_mask, (112, 112))        

    def create_folders(self):
        Path('/'.join(self.config.current_dir.split('/')[:2])).mkdir(parents=True, exist_ok=True)
        Path(self.config.current_dir).mkdir(parents=True, exist_ok=True)
        Path(self.config.current_dir + '/final_results/sim-boxes').mkdir(parents=True, exist_ok=True)
        Path(self.config.current_dir + '/final_results/pr-curves').mkdir(parents=True, exist_ok=True)
        Path(self.config.current_dir + '/final_results/stats/similarity').mkdir(parents=True, exist_ok=True)
        Path(self.config.current_dir + '/final_results/stats/average_precision').mkdir(parents=True, exist_ok=True)
        Path(self.config.current_dir + '/saved_preds').mkdir(parents=True, exist_ok=True)
        Path(self.config.current_dir + '/saved_patches').mkdir(parents=True, exist_ok=True)
        Path(self.config.current_dir + '/saved_similarities').mkdir(parents=True, exist_ok=True)
        Path(self.config.current_dir + '/losses').mkdir(parents=True, exist_ok=True)

    def _projection(
        self, values: "torch.Tensor", eps: Union[int, float, np.ndarray], norm_p: Union[int, float, str]
    ) -> "torch.Tensor":
        """
        Project `values` on the L_p norm ball of size `eps`.

        :param values: Values to clip.
        :param eps: Maximum norm allowed.
        :param norm_p: L_p norm to use for clipping supporting 1, 2, `np.Inf` and "inf".
        :return: Values of `values` after projection.
        """
        import torch  # lgtm [py/repeated-import]

        # Pick a small scalar to avoid division by 0
        tol = 10e-8
        values_tmp = values.reshape(values.shape[0], -1)

        if norm_p == 2:
            if isinstance(eps, np.ndarray):
                raise NotImplementedError(
                    "The parameter `eps` of type `np.ndarray` is not supported to use with norm 2."
                )

            values_tmp = (
                values_tmp
                * torch.min(
                    torch.tensor([1.0], dtype=torch.float32).to(device),
                    eps / (torch.norm(values_tmp, p=2, dim=1) + tol),
                ).unsqueeze_(-1)
            )

        elif norm_p == 1:
            if isinstance(eps, np.ndarray):
                raise NotImplementedError(
                    "The parameter `eps` of type `np.ndarray` is not supported to use with norm 1."
                )

            values_tmp = (
                values_tmp
                * torch.min(
                    torch.tensor([1.0], dtype=torch.float32).to(device),
                    eps / (torch.norm(values_tmp, p=1, dim=1) + tol),
                ).unsqueeze_(-1)
            )

        elif norm_p in [np.inf, "inf"]:
            if isinstance(eps, np.ndarray):
                eps = eps * np.ones_like(values.cpu())
                eps = eps.reshape([eps.shape[0], -1])  # type: ignore

            values_tmp = values_tmp.sign() * torch.min(
                values_tmp.abs(), torch.tensor([eps], dtype=torch.float32).to(device)
            )

        else:
            raise NotImplementedError(
                "Values of `norm_p` different from 1, 2 and `np.inf` are currently not supported."
            )

        values = values_tmp.reshape(values.shape)

        return values
                          

    def train(self):
        #adv_patch_cpu = utils.get_patch(self.config)
        print('----styleganinv_ffhq256---')
        self.E = StyleGANEncoder('styleganinv_ffhq256')

        self.G = StyleGANGenerator('styleganinv_ffhq256')


        '''        
        lat_img = image_loader("/face/Mask/AdversarialMask/datasets/CASIA/1302735//001_aligned.png")
        #lat_img = image_loader("/face/Mask/AdversarialMask/datasets/CASIA/4204960/002_aligned.png")
        print(lat_img.shape)
        lat_img256 = transforms.Resize([256,256])(lat_img)
        print(lat_img256.shape)

        latent_weight = 100
        latent_z = self.E.net(lat_img256).clone()
        self.patch_mask = transforms.ToTensor()(Image.open('/face/Mask/AdversarialMask/datasets/012_mask5.png').convert('L')).to(device).unsqueeze(0)
        print(latent_z.shape)
        print(latent_z)

        latent_z_np = latent_z.cpu().detach().numpy()
        np.save('./train_targeted_gan_1302735_001_align.npy', latent_z_np)

        print(over)
        '''

        if 0:
            #latent_z = torch.ones((1, 7168), dtype=torch.float32)
            #latent_z = torch.rand((1, 7168), dtype=torch.float32) - 0.5
            latent_z = torch.zeros((1, 7168), dtype=torch.float32)

            #control_z_np = np.load('/face/Mask/idinvert_pytorch/results/inversion/my_test/inverted_codes.npy').reshape((1, 7168))
            control_z_np = np.load('/face/Mask/idinvert_pytorch/results/inversion/my_test8/inverted_codes.npy').reshape((1, 7168))
            control_z = torch.from_numpy(control_z_np)
            self.latent_z_control = control_z.clone()

            #control_z_np2 = np.load('/face/Mask/idinvert_pytorch/results/inversion/my_test4/inverted_codes.npy').reshape((1, 7168))
            #control_z_np2 = np.load('/face/Mask/idinvert_pytorch/results/inversion/my_test6/inverted_codes.npy').reshape((1, 7168))
            control_z_np2 = np.load('/face/Mask/idinvert_pytorch/results/inversion/my_test7/inverted_codes.npy').reshape((1, 7168))
            control_z2 = torch.from_numpy(control_z_np2)
            self.latent_z_control2 = control_z2.clone()
        else:          
            #latent_z_np = np.load('/face/Mask/idinvert_pytorch/results/inversion/my_test/inverted_codes.npy').reshape((1, 7168))
            #latent_z_np = np.load('/face/Mask/idinvert_pytorch/results/inversion/my_test2/inverted_codes.npy').reshape((1, 7168))
            #latent_z_np = np.load('/face/Mask/idinvert_pytorch/results/inversion/my_test7/inverted_codes.npy').reshape((1, 7168))
            latent_z_np = np.load('/face/Mask/idinvert_pytorch/results/inversion/my_test8/inverted_codes.npy').reshape((1, 7168))
            #latent_z_np = np.load('/face/Mask/idinvert_pytorch/results/inversion/my_test6/inverted_codes.npy').reshape((1, 7168))
            
            np.random.seed(123)
            #rad = 0.03
            rad = 0.0
            latent_z_np_ = latent_z_np.reshape((7168)) + (np.random.rand((7168))*rad - np.ones((7168))*rad/2)
            latent_z_np = latent_z_np_.reshape((1, 7168))

            latent_z = torch.from_numpy(latent_z_np).float()
            print(latent_z.shape)
            print(torch.max(latent_z))
            print(torch.min(latent_z))


            #control_z_np = np.load('/face/Mask/idinvert_pytorch/results/inversion/my_test/inverted_codes.npy').reshape((1, 7168))
            #control_z_np = np.load('/face/Mask/idinvert_pytorch/results/inversion/my_test7/inverted_codes.npy').reshape((1, 7168))
            control_z_np = np.load('/face/Mask/idinvert_pytorch/results/inversion/my_test10/inverted_codes.npy').reshape((1, 7168))
            #control_z_np = np.load('/face/Mask/idinvert_pytorch/results/inversion/my_test6/inverted_codes.npy').reshape((1, 7168))


            ''' 
            #adv8 = np.load('latent_z_adv8_.npy')[0]
            #adv8 = np.load('latent_z_adv8_2.npy')[0]
            #adv8 = np.load('latent_z_adv8_2_.npy')[0]
            #adv8 = np.load('latent_z_adv8_3.npy')[0]
            adv8 = np.load('latent_z_adv8_4.npy')[0]
            img8 = np.load('/face/Mask/idinvert_pytorch/results/inversion/my_test8/inverted_codes.npy').reshape((7168))
            vec_8_adv8 = adv8 - img8
            #control_z_np = (vec_8_adv8 * 10 + img8).reshape((1, 7168))
            control_z_np = (vec_8_adv8 * 2 + img8).reshape((1, 7168))
            '''

            '''             
            np.random.seed(123)
            rad = 0.5
            #adv8 = np.load('latent_z_adv8_.npy')[0] + (np.random.rand((7168))*rad - np.ones((7168))*rad/2)
            #adv8 = np.load('latent_z_adv8_2.npy')[0] + (np.random.rand((7168))*rad - np.ones((7168))*rad/2)
            #adv8 = np.load('latent_z_adv8_2_.npy')[0] + (np.random.rand((7168))*rad - np.ones((7168))*rad/2)
            adv8 = np.load('latent_z_adv8_3.npy')[0] + (np.random.rand((7168))*rad - np.ones((7168))*rad/2)
            img8 = np.load('/face/Mask/idinvert_pytorch/results/inversion/my_test8/inverted_codes.npy').reshape((7168))
            vec_8_adv8 = adv8 - img8
            #control_z_np = (vec_8_adv8 * 10 + img8).reshape((1, 7168))
            control_z_np = (vec_8_adv8 * 2 + img8).reshape((1, 7168))
            '''

            ''' 
            img10 = np.load('/face/Mask/idinvert_pytorch/results/inversion/my_test10/inverted_codes.npy').reshape((7168))
            np.random.seed(3)
            rad = 1.0
            img10_ = img10 + (np.random.rand((7168))*rad - np.ones((7168))*rad/2)
            img8 = np.load('/face/Mask/idinvert_pytorch/results/inversion/my_test8/inverted_codes.npy').reshape((7168))
            vec_8_10 = img10_ - img8
            control_z_np = (vec_8_10 * 2 + img8).reshape((1, 7168))
            '''


            control_z = torch.from_numpy(control_z_np)
            self.latent_z_control = control_z.clone()

            #neg_z_np = np.load('/face/Mask/idinvert_pytorch/results/inversion/my_test10/inverted_codes.npy').reshape((1, 7168))
            neg_z_np = np.load('latent_z_adv8_.npy')[0].reshape((1, 7168)) 
            neg_z = torch.from_numpy(neg_z_np)
            self.latent_neg_z = neg_z.clone()

            #neg_z2_np = np.load('latent_z_adv8_2.npy')[0].reshape((1, 7168)) 
            neg_z2_np = np.load('latent_z_adv8_2_.npy')[0].reshape((1, 7168)) 
            neg_z2 = torch.from_numpy(neg_z2_np)
            self.latent_neg_z2 = neg_z2.clone()

            neg_z3_np = np.load('latent_z_adv8_3.npy')[0].reshape((1, 7168)) 
            neg_z3 = torch.from_numpy(neg_z3_np)
            self.latent_neg_z3 = neg_z3.clone()

            neg_z4_np = np.load('latent_z_adv8_4.npy')[0].reshape((1, 7168)) 
            neg_z4 = torch.from_numpy(neg_z4_np)
            self.latent_neg_z4 = neg_z4.clone()

            neg_z5_np = np.load('latent_z_adv8_5.npy')[0].reshape((1, 7168)) 
            neg_z5 = torch.from_numpy(neg_z5_np)
            self.latent_neg_z5 = neg_z5.clone()

            neg_z6_np = np.load('latent_z_adv8_6.npy')[0].reshape((1, 7168)) 
            neg_z6 = torch.from_numpy(neg_z6_np)
            self.latent_neg_z6 = neg_z6.clone()

            neg_z7_np = np.load('latent_z_adv8_7.npy')[0].reshape((1, 7168)) 
            neg_z7 = torch.from_numpy(neg_z7_np)
            self.latent_neg_z7 = neg_z7.clone()

            neg_z8_np = np.load('latent_z_adv8_8.npy')[0].reshape((1, 7168)) 
            neg_z8 = torch.from_numpy(neg_z8_np)
            self.latent_neg_z8 = neg_z8.clone()

            neg_z9_np = np.load('latent_z_adv8_9.npy')[0].reshape((1, 7168)) 
            neg_z9 = torch.from_numpy(neg_z9_np)
            self.latent_neg_z9 = neg_z9.clone()

            neg_z10_np = np.load('latent_z_adv8_10.npy')[0].reshape((1, 7168)) 
            neg_z10 = torch.from_numpy(neg_z10_np)
            self.latent_neg_z10 = neg_z10.clone()

            neg_z11_np = np.load('latent_z_adv8_11.npy')[0].reshape((1, 7168))
            neg_z11 = torch.from_numpy(neg_z11_np)
            self.latent_neg_z11 = neg_z11.clone()


            neg_z12_np = np.load('latent_z_adv8_12.npy')[0].reshape((1, 7168))
            neg_z12 = torch.from_numpy(neg_z12_np)
            self.latent_neg_z12 = neg_z12.clone()


            neg_z13_np = np.load('latent_z_adv8_13.npy')[0].reshape((1, 7168))
            neg_z13 = torch.from_numpy(neg_z13_np)
            self.latent_neg_z13 = neg_z13.clone()

            neg_z14_np = np.load('latent_z_adv8_14.npy')[0].reshape((1, 7168))
            neg_z14 = torch.from_numpy(neg_z14_np)
            self.latent_neg_z14 = neg_z14.clone()

            neg_z15_np = np.load('latent_z_adv8_15.npy')[0].reshape((1, 7168))
            neg_z15 = torch.from_numpy(neg_z15_np)
            self.latent_neg_z15 = neg_z15.clone()

            neg_z16_np = np.load('latent_z_adv8_16.npy')[0].reshape((1, 7168))
            neg_z16 = torch.from_numpy(neg_z16_np)
            self.latent_neg_z16 = neg_z16.clone()

            neg_z17_np = np.load('latent_z_adv8_17.npy')[0].reshape((1, 7168))
            neg_z17 = torch.from_numpy(neg_z17_np)
            self.latent_neg_z17 = neg_z17.clone()

            neg_z18_np = np.load('latent_z_adv8_18.npy')[0].reshape((1, 7168))
            neg_z18 = torch.from_numpy(neg_z18_np)
            self.latent_neg_z18 = neg_z18.clone()

            neg_z19_np = np.load('latent_z_adv8_19.npy')[0].reshape((1, 7168))
            neg_z19 = torch.from_numpy(neg_z19_np)
            self.latent_neg_z19 = neg_z19.clone()
            
            neg_z20_np = np.load('latent_z_adv8_20.npy')[0].reshape((1, 7168))
            neg_z20 = torch.from_numpy(neg_z20_np)
            self.latent_neg_z20 = neg_z20.clone()

            neg_z21_np = np.load('latent_z_adv8_21.npy')[0].reshape((1, 7168))
            neg_z21 = torch.from_numpy(neg_z21_np)
            self.latent_neg_z21 = neg_z21.clone()

            neg_z22_np = np.load('latent_z_adv8_22.npy')[0].reshape((1, 7168))
            neg_z22 = torch.from_numpy(neg_z22_np)
            self.latent_neg_z22 = neg_z22.clone()

            neg_z23_np = np.load('latent_z_adv8_23.npy')[0].reshape((1, 7168))
            neg_z23 = torch.from_numpy(neg_z23_np)
            self.latent_neg_z23 = neg_z23.clone()

            neg_z24_np = np.load('latent_z_adv8_24.npy')[0].reshape((1, 7168))
            neg_z24 = torch.from_numpy(neg_z24_np)
            self.latent_neg_z24 = neg_z24.clone()

            neg_z25_np = np.load('latent_z_adv8_25.npy')[0].reshape((1, 7168))
            neg_z25 = torch.from_numpy(neg_z25_np)
            self.latent_neg_z25 = neg_z25.clone()

            neg_z26_np = np.load('latent_z_adv8_26.npy')[0].reshape((1, 7168))
            neg_z26 = torch.from_numpy(neg_z26_np)
            self.latent_neg_z26 = neg_z26.clone()

            neg_z27_np = np.load('latent_z_adv8_27.npy')[0].reshape((1, 7168))
            neg_z27 = torch.from_numpy(neg_z27_np)
            self.latent_neg_z27 = neg_z27.clone()

            neg_z28_np = np.load('latent_z_adv8_28.npy')[0].reshape((1, 7168))
            neg_z28 = torch.from_numpy(neg_z28_np)
            self.latent_neg_z28 = neg_z28.clone()

            neg_z29_np = np.load('latent_z_adv8_29.npy')[0].reshape((1, 7168))
            neg_z29 = torch.from_numpy(neg_z29_np)
            self.latent_neg_z29 = neg_z29.clone()

            neg_z30_np = np.load('latent_z_adv8_30.npy')[0].reshape((1, 7168))
            self.latent_neg_z30 = torch.from_numpy(neg_z30_np).clone()

            neg_z31_np = np.load('latent_z_adv8_31.npy')[0].reshape((1, 7168))
            self.latent_neg_z31 = torch.from_numpy(neg_z31_np).clone()

            neg_z32_np = np.load('latent_z_adv8_32.npy')[0].reshape((1, 7168))
            self.latent_neg_z32 = torch.from_numpy(neg_z32_np).clone()

        self.latent_z_init = latent_z.clone()
        self.latent_z_init_np = latent_z.clone().detach().numpy()

        #self.latent_z_control = control_z.clone()

        
        latent_z_mask = torch.zeros((1, 7168), dtype=torch.float32)
        #latent_z_mask[0][0:512] = torch.ones((512), dtype=torch.float32)
        #latent_z_mask[0][0:512*4] = torch.ones((512*4), dtype=torch.float32)
        latent_z_mask[0][0:512*14] = torch.ones((512*14), dtype=torch.float32)
        #latent_z_mask[0][:] = torch.ones((7168), dtype=torch.float32)

        '''  
        print('---')
        #advbit = np.load('advbit_0.npy')
        advbit = np.load('Advbit_0.npy')
        for bit in advbit:
            latent_z_mask[0][bit] = torch.zeros(1, dtype=torch.float32)
        advbit = np.load('Advbit_0_.npy')
        for bit in advbit:
            latent_z_mask[0][bit] = torch.zeros(1, dtype=torch.float32)

        print('---')
        '''
        
        print(latent_z_mask)
        self.latent_z_mask = latent_z_mask
        
         
        x_inv = self.G.net.synthesis(latent_z.view(1, 14, 512))
        print(x_inv.shape)
       
        '''
        fig = plt.figure()
        #plt.imshow(np.transpose(x_inv[0].cpu().detach().numpy(),[1,2,0]))  #detach() is bad for PGD, but ok for FG
        x_inv= F.interpolate(x_inv, (140, 140))        
        print(x_inv.shape)
        x_inv = transforms.CenterCrop(112)(x_inv)
        print(x_inv.shape)
        encoder_out = self.G.postprocess(_get_tensor_value(x_inv))
        plt.imshow(encoder_out[0])  #detach() is bad for PGD, but ok for FG
        plt.axis('off')
        plt.show()
        print(over)
        '''


        latent_z.requires_grad_(True)
        #optimizer = optim.Adam([adv_patch_cpu], lr=self.config.start_learning_rate, amsgrad=True)

        #optimizer = optim.Adam([latent_z], lr=self.config.start_learning_rate, amsgrad=True)
        optimizer = optim.SGD([latent_z], lr=self.config.start_learning_rate, momentum=0.9, weight_decay=5e-4)

        scheduler = self.config.scheduler_factory(optimizer)
        early_stop = EarlyStopping(current_dir=self.config.current_dir, patience=self.config.es_patience, init_patch=latent_z)
        epoch_length = len(self.train_loader)

        #source_id = 4
        #target_id = 22

        source_id = 0
        target_id = 1
        #source_id = 1
        #target_id = 0
        for epoch in range(self.config.epochs):
            train_loss = 0.0
            dist_loss = 0.0
            tv_loss = 0.0
            #progress_bar = tqdm(enumerate(self.train_no_aug_loader), desc=f'Epoch {epoch}', total=epoch_length)
            progress_bar = tqdm(enumerate(self.train_loader), desc=f'Epoch {epoch}', total=epoch_length)
            prog_bar_desc = 'train-loss: {:.6}, dist-loss: {:.6}, tv-loss: {:.6}, lr: {:.6}'

            cnt = 0
            tt = 0
            for i_batch, (img_batch, _, cls_id) in progress_bar:
                #print('   ')
                #print(cls_id)
                #if cls_id != 1: #source cls_id=1

                if cls_id == target_id:
                    if epoch == 0:
                        #print('target id')
                        #print(self.config.celeb_lab_mapper[target_id])
                        io.imsave('target_img_'+str(tt)+'.png',np.transpose(img_batch[0],[1,2,0]))
                        tt += 1
                    continue
                    
                elif cls_id != source_id:
                    continue

                #print(cls_id)
                #print(cls_id.dtype)
                cls_id = torch.ones(1, dtype=torch.int64) * target_id #target cls_id=5
                #print(cls_id)

                if epoch == 0:
                    #print('source id')
                    #print(self.config.celeb_lab_mapper[source_id])
                    io.imsave('source_img_'+str(cnt)+'.png',np.transpose(img_batch[0],[1,2,0]))


                #tmp_z = latent_z.clone()
                #tmp_z_np = latent_z.clone().detach().numpy()

                cnt += 1
                (b_loss, sep_loss), vars = self.forward_step(img_batch, latent_z, cls_id, epoch)

                train_loss += b_loss.item()
                dist_loss += sep_loss[0].item()
                #tv_loss += sep_loss[1].item()
                tv_loss += sep_loss[2].item()

                optimizer.zero_grad()
                b_loss.backward()
                optimizer.step()

                #make image stay at manifold
                #latent_z.data.clamp_(-3, 3)

                '''
                perturbation_np = latent_z.clone().detach().numpy() - tmp_z_np
                perturbation = torch.tensor(perturbation_np).to(device)
                eps_step = np.array(1.0, dtype=ART_NUMPY_DTYPE)
                perturbation_step = torch.tensor(eps_step).to(device) * perturbation
                perturbation_step[torch.isnan(perturbation_step)] = 0
                print('--per : '+ str(np.linalg.norm(perturbation_step.detach().numpy())) )
                #x = x + perturbation_step
                latent_z = tmp_z + perturbation_step
                '''



                '''
                # Do projection
                perturbation = self._projection(latent_z.clone() - self.latent_z_init, 0.1, np.inf)

                # Recompute x_adv
                latent_z = perturbation + self.latent_z_init
                '''

                tmp2_z = latent_z.clone().detach().numpy()
                diff_z = tmp2_z - self.latent_z_init_np
                #diff_z = tmp2_z - tmp_z_np
                diff_z_ = np.linalg.norm(diff_z)



                progress_bar.set_postfix_str(prog_bar_desc.format(train_loss / (i_batch + 1),
                                                                  dist_loss / (i_batch + 1),
                                                                  tv_loss / (i_batch + 1),
                                                                  optimizer.param_groups[0]["lr"]))
                self.train_losses_iter.append(train_loss / (i_batch + 1))
                #print(epoch_length)
                #print(cnt)
                #if i_batch + 1 == epoch_length:
                #if cnt + 1 == 5:
                #if cnt  == 10:
                if cnt  == 1:
                    print('--diff z: '+ str(diff_z_) )
                    #print(str(np.max(latent_z.detach().numpy())) + '  '+ str(np.min(latent_z.detach().numpy())) )
                    #print(str(np.linalg.norm(latent_z.detach().numpy()))  )
                    print(torch.norm( latent_z - self.latent_z_init, p=2 ).detach().numpy())

                    self.save_losses(epoch_length, train_loss, dist_loss, tv_loss)
                    progress_bar.set_postfix_str(prog_bar_desc.format(self.train_losses_epoch[-1],
                                                                      self.dist_losses[-1],
                                                                      self.tv_losses[-1],
                                                                      optimizer.param_groups[0]["lr"], ))
                del b_loss
                torch.cuda.empty_cache()
            if early_stop(self.train_losses_epoch[-1], latent_z, epoch, self.latent_z_init, self.latent_z_mask):
                self.best_patch = latent_z
                #break

            scheduler.step(self.train_losses_epoch[-1])
            #scheduler.step()
            #print('----- '+str(torch.max(latent_z))+'  --------  '+str(torch.min(latent_z)))
        self.best_patch = early_stop.best_patch
        self.save_final_objects()
        utils.plot_train_val_loss(self.config, self.train_losses_epoch, 'Epoch')
        utils.plot_train_val_loss(self.config, self.train_losses_iter, 'Iterations')
        utils.plot_separate_loss(self.config, self.train_losses_epoch, self.dist_losses, self.tv_losses)
    
    def loss_fn(self, patch_embs, tv_loss, cls_id, latent_z, epoch):
        distance_loss = torch.empty(0, device=device)
        for target_embedding, (emb_name, patch_emb) in zip(self.target_embedding.values(), patch_embs.items()):
            target_embeddings = torch.index_select(target_embedding, index=cls_id, dim=0).squeeze(-2)
            distance = self.dist_loss(patch_emb, target_embeddings)
            single_embedder_dist_loss = torch.mean(distance).unsqueeze(0)
            distance_loss = torch.cat([distance_loss, single_embedder_dist_loss], dim=0)
        distance_loss = self.config.dist_weight * distance_loss.mean()
        tv_loss = self.config.tv_weight * tv_loss
        #total_loss = - distance_loss + tv_loss
        total_loss = - distance_loss 

        #latent_loss = torch.norm(latent_z - self.latent_z_init, p=2) 
        latent_loss = torch.norm(latent_z - self.latent_z_control, p=2) 
        #latent_loss = torch.norm(latent_z - self.latent_z_control2, p=2) 
        #latent_loss = torch.norm(latent_z - self.latent_z_control, p=2) + torch.norm(latent_z - self.latent_z_control2, p=2)
        #latent_loss = torch.min( torch.norm(latent_z - self.latent_z_control, p=2),  torch.norm(latent_z - self.latent_z_control2, p=2))

        #total_loss += 0.1 * latent_loss 
        #total_loss += 0.2 * latent_loss 
        #total_loss += 10.0 * latent_loss 


        #latent_neg_loss = - torch.norm(latent_z - self.latent_neg_z, p=2) 
        #latent_neg_loss = - torch.norm(latent_z - self.latent_neg_z, p=2) - torch.norm(latent_z - self.latent_neg_z2, p=2)
        #latent_neg_loss = - torch.norm(latent_z - self.latent_neg_z, p=2) - torch.norm(latent_z - self.latent_neg_z2, p=2) - torch.norm(latent_z - self.latent_neg_z3, p=2)
        #latent_neg_loss = - torch.norm(latent_z - self.latent_neg_z, p=2) - torch.norm(latent_z - self.latent_neg_z2, p=2) - torch.norm(latent_z - self.latent_neg_z3, p=2) - torch.norm(latent_z - self.latent_neg_z4, p=2)
        #latent_neg_loss = - torch.norm(latent_z - self.latent_neg_z, p=2) - torch.norm(latent_z - self.latent_neg_z2, p=2) - torch.norm(latent_z - self.latent_neg_z3, p=2) - torch.norm(latent_z - self.latent_neg_z4, p=2) - torch.norm(latent_z - self.latent_neg_z5, p=2)
        #latent_neg_loss = - torch.norm(latent_z - self.latent_neg_z, p=2) - torch.norm(latent_z - self.latent_neg_z2, p=2) - torch.norm(latent_z - self.latent_neg_z3, p=2) - torch.norm(latent_z - self.latent_neg_z4, p=2) - torch.norm(latent_z - self.latent_neg_z5, p=2) - torch.norm(latent_z - self.latent_neg_z6, p=2)
        #latent_neg_loss = - torch.norm(latent_z - self.latent_neg_z, p=2) - torch.norm(latent_z - self.latent_neg_z2, p=2) - torch.norm(latent_z - self.latent_neg_z3, p=2) - torch.norm(latent_z - self.latent_neg_z4, p=2) - torch.norm(latent_z - self.latent_neg_z5, p=2) - torch.norm(latent_z - self.latent_neg_z6, p=2) - torch.norm(latent_z - self.latent_neg_z7, p=2) 
        #latent_neg_loss = - 2.0 * torch.norm(latent_z - self.latent_neg_z, p=2) - torch.norm(latent_z - self.latent_neg_z2, p=2) - torch.norm(latent_z - self.latent_neg_z3, p=2) - torch.norm(latent_z - self.latent_neg_z4, p=2) - torch.norm(latent_z - self.latent_neg_z5, p=2) - torch.norm(latent_z - self.latent_neg_z6, p=2) - 2.0 * torch.norm(latent_z - self.latent_neg_z7, p=2) - 2.0* torch.norm(latent_z - self.latent_neg_z8, p=2)
        latent_neg_loss = - 2.0 * torch.norm(latent_z - self.latent_neg_z, p=2) - torch.norm(latent_z - self.latent_neg_z2, p=2) - torch.norm(latent_z - self.latent_neg_z3, p=2) - torch.norm(latent_z - self.latent_neg_z4, p=2) - torch.norm(latent_z - self.latent_neg_z5, p=2) - torch.norm(latent_z - self.latent_neg_z6, p=2) - 2.0 * torch.norm(latent_z - self.latent_neg_z7, p=2) - 2.0* torch.norm(latent_z - self.latent_neg_z8, p=2) - torch.norm(latent_z - self.latent_neg_z9, p=2) - torch.norm(latent_z - self.latent_neg_z10, p=2) - torch.norm(latent_z - self.latent_neg_z11, p=2) - torch.norm(latent_z - self.latent_neg_z12, p=2) - torch.norm(latent_z - self.latent_neg_z13, p=2) - torch.norm(latent_z - self.latent_neg_z14, p=2) - torch.norm(latent_z - self.latent_neg_z15, p=2) - torch.norm(latent_z - self.latent_neg_z16, p=2) - torch.norm(latent_z - self.latent_neg_z17, p=2)  - torch.norm(latent_z - self.latent_neg_z18, p=2)  - torch.norm(latent_z - self.latent_neg_z19, p=2) - torch.norm(latent_z - self.latent_neg_z20, p=2) - torch.norm(latent_z - self.latent_neg_z21, p=2) - torch.norm(latent_z - self.latent_neg_z22, p=2) - torch.norm(latent_z - self.latent_neg_z23, p=2) - torch.norm(latent_z - self.latent_neg_z24, p=2) - torch.norm(latent_z - self.latent_neg_z25, p=2) - torch.norm(latent_z - self.latent_neg_z26, p=2) - torch.norm(latent_z - self.latent_neg_z27, p=2)  - torch.norm(latent_z - self.latent_neg_z28, p=2) - torch.norm(latent_z - self.latent_neg_z29, p=2) - torch.norm(latent_z - self.latent_neg_z30, p=2) - torch.norm(latent_z - self.latent_neg_z31, p=2) - torch.norm(latent_z - self.latent_neg_z32, p=2)
        #latent_loss = -latent_neg_loss



        total_loss += 10.0* torch.clamp( torch.norm( latent_z - self.latent_z_init, p=2 ) - 50, min=0)


        #total_loss = 0 
        #if epoch > 100:
        if epoch > 50:
            #total_loss += 0.3* torch.norm( latent_z, p=2 )


            #total_loss += 0.5 * latent_loss 
            #total_loss += 1.0 * latent_loss 
            #total_loss += 0.5 * latent_loss 
            
            #total_loss += 0.05 * latent_neg_loss 
            total_loss += 0.05 * latent_neg_loss 
            #total_loss += 0.3 * latent_neg_loss 
        else:
            #total_loss += 0.3* torch.norm( latent_z, p=2 )


            #total_loss += 0.1 * latent_loss 
            #total_loss += 1.0 * latent_loss 
            #total_loss += 0.5 * latent_loss

            total_loss += 0.1 * latent_neg_loss 
            #total_loss += 0.3 * latent_neg_loss 

        #if epoch ==100:
        #if epoch ==50:
        if epoch ==240:
        #if epoch ==100:
            latent_z_np = latent_z.clone().detach().numpy()
        #    #np.save('latent_z_adv8_2_.npy', latent_z_np)
        #    #np.save('latent_z_adv8_3.npy', latent_z_np)
        #    np.save('latent_z_adv8_4.npy', latent_z_np)
        #    np.save('latent_z_adv8_5.npy', latent_z_np)
        #    np.save('latent_z_adv8_6.npy', latent_z_np)
        #    np.save('latent_z_adv8_7.npy', latent_z_np)
        #    np.save('latent_z_adv8_8.npy', latent_z_np)
        #    np.save('latent_z_adv8_9.npy', latent_z_np)
        #    np.save('latent_z_adv8_10.npy', latent_z_np)
        #    np.save('latent_z_adv8_11.npy', latent_z_np)
        #    np.save('latent_z_adv8_12.npy', latent_z_np)
        #    np.save('latent_z_adv8_13.npy', latent_z_np)
        #    np.save('latent_z_adv8_14.npy', latent_z_np)
        #    np.save('latent_z_adv8_15.npy', latent_z_np)
        #    np.save('latent_z_adv8_16.npy', latent_z_np)
        #    np.save('latent_z_adv8_17.npy', latent_z_np)
        #    np.save('latent_z_adv8_18.npy', latent_z_np)
        #    np.save('latent_z_adv8_19.npy', latent_z_np)
        #    np.save('latent_z_adv8_20.npy', latent_z_np)
        #    np.save('latent_z_adv8_21.npy', latent_z_np)
        #    np.save('latent_z_adv8_22.npy', latent_z_np)
        #    np.save('latent_z_adv8_23.npy', latent_z_np)
        #    np.save('latent_z_adv8_24.npy', latent_z_np)
        #    np.save('latent_z_adv8_25.npy', latent_z_np)
        #    np.save('latent_z_adv8_26.npy', latent_z_np)
        #    np.save('latent_z_adv8_27.npy', latent_z_np)
        #    np.save('latent_z_adv8_28.npy', latent_z_np)
        #    np.save('latent_z_adv8_29.npy', latent_z_np)
        #    np.save('latent_z_adv8_30.npy', latent_z_np)
        #    np.save('latent_z_adv8_31.npy', latent_z_np)
        #    np.save('latent_z_adv8_32.npy', latent_z_np)
            np.save('latent_z_adv8_33.npy', latent_z_np)

        '''
        total_loss = 0
        if epoch >200:
            total_loss = - distance_loss

            total_loss += 0.0 * latent_loss

        elif epoch > 50:
            total_loss = - distance_loss

            #total_loss += 0.5 * latent_loss 
            total_loss += 1.0 * latent_loss
            #total_loss += 0.3 * latent_loss 
        else:
            total_loss = - distance_loss
            #total_loss += 0.1 * latent_loss 
            total_loss += 0.0 * latent_loss

            #total_loss += 1.0 * latent_loss 
            #total_loss += 5.0 * latent_neg_loss 
        '''

        '''        
        if epoch == 0:
            latent_z_np = latent_z.clone().detach().numpy()
            np.save('latent_z_ep0_.npy', latent_z_np)

        if epoch == 1:
            latent_z_np = latent_z.clone().detach().numpy()
            np.save('latent_z_ep1_.npy', latent_z_np)

        if epoch == 2:
            latent_z_np = latent_z.clone().detach().numpy()
            np.save('latent_z_ep2_.npy', latent_z_np)


        if epoch == 3:
            latent_z_np = latent_z.clone().detach().numpy()
            np.save('latent_z_ep3_.npy', latent_z_np)

        if epoch == 4:
            latent_z_np = latent_z.clone().detach().numpy()
            np.save('latent_z_ep4_.npy', latent_z_np)

        if epoch == 5:
            latent_z_np = latent_z.clone().detach().numpy()
            np.save('latent_z_ep5_.npy', latent_z_np)

        if epoch == 6:
            latent_z_np = latent_z.clone().detach().numpy()
            np.save('latent_z_ep6_.npy', latent_z_np)

        if epoch == 7:
            latent_z_np = latent_z.clone().detach().numpy()
            np.save('latent_z_ep7_.npy', latent_z_np)

        if epoch == 8:
            latent_z_np = latent_z.clone().detach().numpy()
            np.save('latent_z_ep8_.npy', latent_z_np)

        if epoch == 9:
            latent_z_np = latent_z.clone().detach().numpy()
            np.save('latent_z_ep9_.npy', latent_z_np)


        if epoch == 10:
            latent_z_np = latent_z.clone().detach().numpy()
            np.save('latent_z_ep10_.npy', latent_z_np)
          
        if epoch == 11:
            latent_z_np = latent_z.clone().detach().numpy()
            np.save('latent_z_ep11_.npy', latent_z_np)
          
        if epoch == 12:
            latent_z_np = latent_z.clone().detach().numpy()
            np.save('latent_z_ep12_.npy', latent_z_np)
          
          
        if epoch == 13:
            latent_z_np = latent_z.clone().detach().numpy()
            np.save('latent_z_ep13_.npy', latent_z_np)
          
        if epoch == 14:
            latent_z_np = latent_z.clone().detach().numpy()
            np.save('latent_z_ep14_.npy', latent_z_np)
          

        if epoch == 15:
            latent_z_np = latent_z.clone().detach().numpy()
            np.save('latent_z_ep15_.npy', latent_z_np)
          
        if epoch == 16:
            latent_z_np = latent_z.clone().detach().numpy()
            np.save('latent_z_ep16_.npy', latent_z_np)
          
        if epoch == 17:
            latent_z_np = latent_z.clone().detach().numpy()
            np.save('latent_z_ep17_.npy', latent_z_np)
        ''' 
        '''
        if epoch == 50:
            latent_z_np = latent_z.clone().detach().numpy()
            np.save('latent_z_ep50.npy', latent_z_np)
            latent_z_control_np = self.latent_z_control.clone().detach().numpy()
            np.save('latent_z_control_ep50.npy', latent_z_control_np)
            np.save('latent_z_init_ep50.npy', self.latent_z_init_np)

        if epoch == 49:
            latent_z_np = latent_z.clone().detach().numpy()
            np.save('latent_z_ep49.npy', latent_z_np)

        if epoch == 48:
            latent_z_np = latent_z.clone().detach().numpy()
            np.save('latent_z_ep48.npy', latent_z_np)

        if epoch == 47:
            latent_z_np = latent_z.clone().detach().numpy()
            np.save('latent_z_ep47.npy', latent_z_np)


        if epoch == 51:
            latent_z_np = latent_z.clone().detach().numpy()
            np.save('latent_z_ep51.npy', latent_z_np)

        if epoch == 52:
            latent_z_np = latent_z.clone().detach().numpy()
            np.save('latent_z_ep52.npy', latent_z_np)

        if epoch == 53:
            latent_z_np = latent_z.clone().detach().numpy()
            np.save('latent_z_ep53.npy', latent_z_np)

        if epoch == 54:
            latent_z_np = latent_z.clone().detach().numpy()
            np.save('latent_z_ep54.npy', latent_z_np)

        if epoch == 55:
            latent_z_np = latent_z.clone().detach().numpy()
            np.save('latent_z_ep55.npy', latent_z_np)

        if epoch == 56:
            latent_z_np = latent_z.clone().detach().numpy()
            np.save('latent_z_ep56.npy', latent_z_np)

        if epoch == 57:
            latent_z_np = latent_z.clone().detach().numpy()
            np.save('latent_z_ep57.npy', latent_z_np)


        if epoch == 58:
            latent_z_np = latent_z.clone().detach().numpy()
            np.save('latent_z_ep58.npy', latent_z_np)

        if epoch == 59:
            latent_z_np = latent_z.clone().detach().numpy()
            np.save('latent_z_ep59.npy', latent_z_np)

        if epoch == 60:
            latent_z_np = latent_z.clone().detach().numpy()
            np.save('latent_z_ep60.npy', latent_z_np)

        if epoch == 61:
            latent_z_np = latent_z.clone().detach().numpy()
            np.save('latent_z_ep61.npy', latent_z_np)

        if epoch == 62:
            latent_z_np = latent_z.clone().detach().numpy()
            np.save('latent_z_ep62.npy', latent_z_np)

        if epoch == 63:
            latent_z_np = latent_z.clone().detach().numpy()
            np.save('latent_z_ep63.npy', latent_z_np)




        if epoch == 64:
            latent_z_np = latent_z.clone().detach().numpy()
            np.save('latent_z_ep64.npy', latent_z_np)

        if epoch == 65:
            latent_z_np = latent_z.clone().detach().numpy()
            np.save('latent_z_ep65.npy', latent_z_np)

        if epoch == 66:
            latent_z_np = latent_z.clone().detach().numpy()
            np.save('latent_z_ep66.npy', latent_z_np)
        '''



        '''
        if epoch == 100:
            latent_z_np = latent_z.clone().detach().numpy()
            np.save('latent_z_ep100.npy', latent_z_np)
            latent_z_control_np = self.latent_z_control.clone().detach().numpy()
            np.save('latent_z_control_ep100.npy', latent_z_control_np)
            np.save('latent_z_init_ep100.npy', self.latent_z_init_np)

        if epoch == 98:
            latent_z_np = latent_z.clone().detach().numpy()
            np.save('latent_z_ep98.npy', latent_z_np)

        if epoch == 99:
            latent_z_np = latent_z.clone().detach().numpy()
            np.save('latent_z_ep99.npy', latent_z_np)

        if epoch == 97:
            latent_z_np = latent_z.clone().detach().numpy()
            np.save('latent_z_ep97.npy', latent_z_np)

        if epoch == 101:
            latent_z_np = latent_z.clone().detach().numpy()
            np.save('latent_z_ep101.npy', latent_z_np)
        if epoch == 102:
            latent_z_np = latent_z.clone().detach().numpy()
            np.save('latent_z_ep102.npy', latent_z_np)
        if epoch == 103:
            latent_z_np = latent_z.clone().detach().numpy()
            np.save('latent_z_ep103.npy', latent_z_np)

        if epoch == 104:
            latent_z_np = latent_z.clone().detach().numpy()
            np.save('latent_z_ep104.npy', latent_z_np)

        if epoch == 105:
            latent_z_np = latent_z.clone().detach().numpy()
            np.save('latent_z_ep105.npy', latent_z_np)
        if epoch == 106:
            latent_z_np = latent_z.clone().detach().numpy()
            np.save('latent_z_ep106.npy', latent_z_np)
        if epoch == 110:
            latent_z_np = latent_z.clone().detach().numpy()
            np.save('latent_z_ep110.npy', latent_z_np)
        
        if epoch == 0:
            latent_z_np = latent_z.clone().detach().numpy()
            np.save('latent_z_ep0.npy', latent_z_np)
        if epoch == 1:
            latent_z_np = latent_z.clone().detach().numpy()
            np.save('latent_z_ep1.npy', latent_z_np)
        if epoch == 2:
            latent_z_np = latent_z.clone().detach().numpy()
            np.save('latent_z_ep2.npy', latent_z_np)
        if epoch == 3:
            latent_z_np = latent_z.clone().detach().numpy()
            np.save('latent_z_ep3.npy', latent_z_np)
        '''
        return total_loss, [distance_loss, tv_loss, latent_loss]
    

    def forward_step(self, img_batch, latent_z, cls_id, epoch):
        img_batch = img_batch.to(device)

        #adv_patch = adv_patch_cpu.to(device)
        latent_z = self.latent_z_init * (1-self.latent_z_mask) + latent_z * self.latent_z_mask

        x_inv = self.G.net.synthesis(latent_z.view(1, 14, 512))

        if 0: #zoom in 
            x_inv= F.interpolate(x_inv, (140, 140))
            adv_patch = transforms.CenterCrop(112)(x_inv)
        else: #w/o zoom in
            adv_patch = F.interpolate(x_inv, (112, 112))


        #print(adv_patch.shape)
        adv_patch = (adv_patch - self.G.min_val) * 255 / (self.G.max_val - self.G.min_val)
        adv_patch = torch.clip( adv_patch + 0.5, 0, 255)/255.

        #if epoch == 110:
        #    latent_z_np = latent_z.clone().detach().numpy()
        #    np.save('latent_z_ep110.npy', latent_z_np)
        return total_loss, [distance_loss, tv_loss, latent_loss]
    

    def forward_step(self, img_batch, latent_z, cls_id, epoch):
        img_batch = img_batch.to(device)

        #adv_patch = adv_patch_cpu.to(device)
        latent_z = self.latent_z_init * (1-self.latent_z_mask) + latent_z * self.latent_z_mask

        x_inv = self.G.net.synthesis(latent_z.view(1, 14, 512))

        if 0: #zoom in 
            x_inv= F.interpolate(x_inv, (140, 140))
            adv_patch = transforms.CenterCrop(112)(x_inv)
        else: #w/o zoom in
            adv_patch = F.interpolate(x_inv, (112, 112))


        #print(adv_patch.shape)
        adv_patch = (adv_patch - self.G.min_val) * 255 / (self.G.max_val - self.G.min_val)
        adv_patch = torch.clip( adv_patch + 0.5, 0, 255)/255.


        cls_id = cls_id.to(device)

        preds = self.location_extractor(img_batch)

        #img_batch_applied = self.fxz_projector(img_batch, preds, adv_patch, self.blue_mask_t[:, 3], is_3d=True, do_aug=self.config.mask_aug)
        #img_batch_applied = self.fxz_projector(img_batch, preds, adv_patch, self.mask_t, is_3d=True, do_aug=self.config.mask_aug)
        #img_batch_applied = adv_patch
        img_batch_applied = img_batch * (1 - self.patch_mask) + adv_patch * self.patch_mask

        '''  
        fig = plt.figure()
        #plt.imshow(np.transpose(x_inv[0].cpu().detach().numpy(),[1,2,0]))  #detach() is bad for PGD, but ok for FG
        encoder_out = self.G.postprocess(_get_tensor_value(img_batch_applied))
        plt.imshow(encoder_out[0])  #detach() is bad for PGD, but ok for FG
        plt.axis('off')
        plt.show()
        '''


        patch_embs = {}
        for embedder_name, emb_model in self.embedders.items():
            patch_embs[embedder_name] = emb_model(img_batch_applied)

        tv_loss = self.total_variation(adv_patch)
        loss = self.loss_fn(patch_embs, tv_loss, cls_id, latent_z, epoch)


        return loss, [img_batch, adv_patch, img_batch_applied, patch_embs, tv_loss]

    def save_losses(self, epoch_length, train_loss, dist_loss, tv_loss):
        train_loss /= epoch_length
        dist_loss /= epoch_length
        tv_loss /= epoch_length
        self.train_losses_epoch.append(train_loss)
        self.dist_losses.append(dist_loss)
        self.tv_losses.append(tv_loss)

    def save_final_objects(self):
        alpha = transforms.ToTensor()(Image.open('../prnet/new_uv.png').convert('L'))
        #final_patch = torch.cat([self.best_patch.squeeze(0), alpha])
        final_patch = self.G.net.synthesis(self.best_patch.view(1, 14, 512))

        #final_patch = F.interpolate(final_patch, (112, 112))
        if 0: #zoom in
            x_inv= F.interpolate(final_patch, (140, 140))
            final_patch = transforms.CenterCrop(112)(x_inv)
        else: #w/o zoom in 
            final_patch = F.interpolate(final_patch, (112, 112))



        final_patch = self.G.postprocess(_get_tensor_value(final_patch))
        final_patch = torch.from_numpy(final_patch).permute(0,3,1,2)/255.

        print(final_patch.shape)
        print(self.patch_mask.shape)
        final_patch = final_patch * self.patch_mask + torch.ones(3,112,112) * (1 - self.patch_mask)


        final_patch_img = transforms.ToPILImage()(final_patch.squeeze(0))




        final_patch_img = transforms.ToPILImage()(final_patch.squeeze(0))
        final_patch_img.save(self.config.current_dir + '/final_results/final_patch.png', 'PNG')
        new_size = tuple(self.config.magnification_ratio * s for s in self.config.img_size)
        transforms.Resize(new_size)(final_patch_img).save(self.config.current_dir + '/final_results/final_patch_magnified.png', 'PNG')
        #torch.save(self.best_patch, self.config.current_dir + '/final_results/final_patch_raw.pt')

        with open(self.config.current_dir + '/losses/train_losses', 'wb') as fp:
            pickle.dump(self.train_losses_epoch, fp)
        with open(self.config.current_dir + '/losses/val_losses', 'wb') as fp:
            pickle.dump(self.val_losses, fp)
        with open(self.config.current_dir + '/losses/dist_losses', 'wb') as fp:
            pickle.dump(self.dist_losses, fp)
        with open(self.config.current_dir + '/losses/tv_losses', 'wb') as fp:
            pickle.dump(self.tv_losses, fp)


def main():
    mode = 'universal'
    config = patch_config_types[mode]()
    print('Starting train...', flush=True)
    adv_mask = AdversarialMask(config)
    adv_mask.train()
    print('Finished train...', flush=True)


if __name__ == '__main__':
    main()
