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

import utilsD as utils
import losses
from config_ganD import patch_config_types
from nn_modules import LandmarkExtractor, FaceXZooProjector, TotalVariation
from utilsD import load_embedder, EarlyStopping, get_patch
import skimage.io as io

import pickle
import torch.nn.functional as F
#from art.estimators.classification.stylegan_encoder import StyleGANEncoder
#from art.estimators.classification.stylegan_generator import StyleGANGenerator

import torchvision.models as models
from torch.autograd import Variable

from art.config import ART_NUMPY_DTYPE
from typing import Optional, Union, TYPE_CHECKING

from train_encoder import VGGLoss
from model import Encoder, Generator, Discriminator

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
        self.tv_losses2 = []
        self.tv_losses3 = []
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
        
        self.latent_neg_z = []
        #self.latent_neg_number = 18
        #self.latent_neg_number = 22
        self.latent_neg_number = 2
        self.rad = 8
        self.rad = 60
        #self.rad = 80
        self.rad = 12

        self.dist_n = [0,0,0,0,0,0,0,0,0,0,0,0,0]
        self.latent_neg_idx = []
        self.len_new_neg = 1
        self.th=0.48*2
        self.th=0.498*2
        self.th=0.496*2
        #self.th=0.499*2


        #self.patch_mask = transforms.ToTensor()(Image.open('/face/Mask/AdversarialMask/datasets/012_mask5.png').convert('L')).to(device).unsqueeze(0)
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
        #self.E = StyleGANEncoder('styleganinv_ffhq256')
        #self.G = StyleGANGenerator('styleganinv_ffhq256')




        image_size=256
        g_model_path = '/face/Mask/stylegan2-encoder-pytorch/checkpoint/generator_ffhq.pt'
        g_ckpt = torch.load(g_model_path, map_location=device)
        latent_dim = g_ckpt['args'].latent
        self.generator = Generator(image_size, latent_dim, 8, device=device).to(device)
        self.generator.load_state_dict(g_ckpt["g_ema"], strict=False)
        self.generator.eval()
        print('[generator loaded]')

        
        channel_multiplier = g_ckpt['args'].channel_multiplier
        self.discriminator = Discriminator(image_size, channel_multiplier).to(device)
        self.discriminator.load_state_dict(g_ckpt["d"], strict=False)
        self.discriminator.eval()
        print('[discriminator loaded]')

        '''
        e_model_path = '/face/Mask/stylegan2-encoder-pytorch/checkpoint/encoder_ffhq.pt'
        e_ckpt = torch.load(e_model_path, map_location=device)
        self.encoder = Encoder(image_size, latent_dim).to(device)
        self.encoder.load_state_dict(e_ckpt['e'])
        self.encoder.eval()
        print('[encoder loaded]')
        '''


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

        if 1:
            #latent_z = torch.zeros((1, 14, 512), dtype=torch.float32)

            #latent_z_np = np.load('./init_face_002.npy').reshape((1, 14, 512))
            #latent_z_np = np.load('./init_img1.npy').reshape((1, 14, 512))
            #latent_z_np = np.load('./init_img3.npy').reshape((1, 14, 512))
            #latent_z_np = np.load('./init_img46.npy').reshape((1, 14, 512))
            #latent_z_np = np.load('./init_img20.npy').reshape((1, 14, 512))
            #latent_z_np = np.load('./init_img32.npy').reshape((1, 14, 512))
            latent_z_np = np.load('./init_face_005.npy').reshape((1, 14, 512))

            latent_z_np = np.load('/face/Mask/AdversarialMask/patch/init_face000005.npy').reshape((1, 14, 512))


            #latent_z_np = np.load('./init_face_004.npy').reshape((1, 14, 512))
            #latent_z_np = np.load('./init_face_003.npy').reshape((1, 14, 512))
            #latent_z_np = np.load('./init_face_001.npy').reshape((1, 14, 512))
            #latent_z_np = np.load('./init_face_015.npy').reshape((1, 14, 512))
            #latent_z_np = np.load('./init_face_004_.npy').reshape((1, 14, 512))
            latent_z = torch.from_numpy(latent_z_np).to(device).float()


            ''' 
            np.random.seed(123)
            #latent_z_np = (np.random.rand((7168))*rad - np.ones((7168))*rad/2).reshape((1, 14, 512))
            latent_z_np = (np.random.rand((7168))*3 ).reshape((1, 14, 512))
            #latent_z_np = (np.random.rand((7168))*3).reshape((1, 14, 512))
            #latent_z_np = (np.random.rand((7168))*rad).reshape((1, 14, 512))
            latent_z = torch.from_numpy(latent_z_np).to(device).float()
            '''



            #control_z_np = np.load('/face/Mask/idinvert_pytorch/results/inversion/my_test/inverted_codes.npy').reshape((1, 7168))
            control_z_np = np.load('./target.npy').reshape((1, 14, 512))
            control_z = torch.from_numpy(control_z_np)
            self.latent_z_control = control_z.clone().to(device)


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
        '''
        for i in range(self.latent_neg_number):
            print(i)
            tpp = np.load('neg_latent_'+str(i+1)+'.npy')
            ll = tpp.shape[0]
            for r in range(ll):
                self.latent_neg_z.append( torch.from_numpy(tpp[r].reshape((1, 14, 512))).clone().to(device) )
        '''
        tpp = np.load('neg_latent_all.npy')
        ll = tpp.shape[0]
        for r in range(ll):
            self.latent_neg_z.append( torch.from_numpy(tpp[r].reshape((1, 14, 512))).clone().to(device) )

            #self.latent_neg_z18 = torch.from_numpy(np.load('latent_z_adv8C_18.npy')[0].reshape((1, 7168))).clone()


        self.latent_z_init = latent_z.clone().to(device)
        self.latent_z_init_np = latent_z.clone().detach().cpu().numpy()

        #self.latent_z_control = control_z.clone()

        
        latent_z_mask = torch.ones((1, 14, 512), dtype=torch.float32).to(device)

        '''
        latent_z_mask = torch.zeros((1, 14, 512), dtype=torch.float32).to(device)

        adv_latent = np.load('adv_latent.npy')
        for k in adv_latent:
            latent_z_mask[0][int(np.floor(k/512))][int(np.mod(k, 512))] = torch.ones((1), dtype=torch.float32)

        print(latent_z_mask)
        print(latent_z_mask[0][0][353])
        print(latent_z_mask[0][5][22])
        print(latent_z_mask[0][5][424])

        latent_z_mask[0][0][:] = torch.ones((512), dtype=torch.float32)
        '''
        #latent_z_mask[0][0:512] = torch.ones((512), dtype=torch.float32)
        #latent_z_mask[0][0:512*4] = torch.ones((512*4), dtype=torch.float32)
        #latent_z_mask[0][0:512*24] = torch.ones((512*24), dtype=torch.float32)
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
        
         
        #x_inv = self.G.net.synthesis(latent_z.view(1, 14, 512))
        #print(x_inv.shape)
        print(device)
        truncation = 0.7
        trunc = self.generator.mean_latent(4096)
        x_inv, _ =  self.generator([latent_z], input_is_latent=True, truncation=truncation, truncation_latent=trunc, randomize_noise=False)
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
        early_stop = EarlyStopping(current_dir=self.config.current_dir, patience=self.config.es_patience, init_patch=latent_z, device=device)
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
            tv_loss2 = 0.0
            tv_loss3 = 0.0
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
                tmp_z_np = latent_z.clone().detach().cpu().numpy()

                cnt += 1
                (b_loss, sep_loss), vars = self.forward_step(img_batch, latent_z, cls_id, epoch)

                train_loss += b_loss.item()
                dist_loss += sep_loss[0].item()
                #tv_loss += sep_loss[1].item()
                #tv_loss += sep_loss[2].item()
                tv_loss += sep_loss[2]

                tv_loss2 += sep_loss[3].item()
                tv_loss3 += sep_loss[4].item()

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




                progress_bar.set_postfix_str(prog_bar_desc.format(train_loss / (i_batch + 1),
                                                                  dist_loss / (i_batch + 1),
                                                                  tv_loss / (i_batch + 1),
                                                                  optimizer.param_groups[0]["lr"]))
                #self.train_losses_iter.append(train_loss / (i_batch + 1))
                self.train_losses_iter.append(dist_loss / (i_batch + 1))
                #print(epoch_length)
                #print(cnt)
                #if i_batch + 1 == epoch_length:
                #if cnt + 1 == 5:
                #if cnt  == 10:
                if cnt  == 1:
                    #print('--diff z: '+str(np.linalg.norm(latent_z.detach().cpu().numpy() - tmp_z_np)))
                    #print(str(np.max(latent_z.detach().numpy())) + '  '+ str(np.min(latent_z.detach().numpy())) )
                    #print(str(np.linalg.norm(latent_z.detach().numpy()))  )
                    #print('--norm z : ' + str(torch.norm( latent_z - self.latent_z_init, p=2 ).detach().numpy()))

                    #self.save_losses(epoch_length, train_loss, dist_loss, tv_loss)
                    self.save_losses(epoch_length, train_loss, dist_loss, tv_loss, tv_loss2, tv_loss3)
                    progress_bar.set_postfix_str(prog_bar_desc.format(self.train_losses_epoch[-1],
                                                                      self.dist_losses[-1],
                                                                      self.tv_losses[-1],
                                                                      optimizer.param_groups[0]["lr"], ))
                del b_loss
                torch.cuda.empty_cache()
            if early_stop(self.train_losses_epoch[-1], latent_z, epoch, self.latent_z_init, self.latent_z_mask):
                self.best_patch = latent_z
                #break

            #scheduler.step(self.train_losses_epoch[-1])
            scheduler.step()
            #print('----- '+str(torch.max(latent_z))+'  --------  '+str(torch.min(latent_z)))

            if dist_loss > self.th:
            #if dist_loss > 0.4*2:
                over = 0
                for tp_z in self.latent_neg_z:
                    if torch.norm(latent_z.clone().to(device) - tp_z.to(device), p=2) < self.rad:
                        over = 1
                        #print('***************************************')
                        #print('***************************************')
                        #print('***************************************')
                        break
                if over == 0:
                    print('                                      ')
                    print('                                      ')
                    print('                                      ')
                    print('                                      ')
                    print('                                      ')
                    #self.latent_neg_z.append( latent_z.to(device) ) #error
                    self.latent_neg_z.append( latent_z.clone().to(device) )
                    self.latent_neg_idx.append(epoch)
                    print(len(self.latent_neg_z))

                #print(len(self.latent_neg_z))
                self.dist_n.append(1)
            else:
                self.dist_n.append(0)

        self.best_patch = early_stop.best_patch
        self.save_final_objects()
        utils.plot_train_val_loss(self.config, self.train_losses_epoch, 'Epoch')
        utils.plot_train_val_loss(self.config, self.train_losses_iter, 'Iterations')
        #utils.plot_separate_loss(self.config, self.train_losses_epoch, self.dist_losses, self.tv_losses)
        utils.plot_separate_loss(self.config, self.train_losses_epoch, self.dist_losses, self.tv_losses, self.tv_losses2, self.tv_losses3)
    
    #def loss_fn(self, patch_embs, tv_loss, cls_id, latent_z, epoch):
    def loss_fn(self, patch_embs, D_loss, cls_id, latent_z, epoch):
        distance_loss = torch.empty(0, device=device)
        for target_embedding, (emb_name, patch_emb) in zip(self.target_embedding.values(), patch_embs.items()):
            target_embeddings = torch.index_select(target_embedding, index=cls_id, dim=0).squeeze(-2)
            distance = self.dist_loss(patch_emb, target_embeddings)
            single_embedder_dist_loss = torch.mean(distance).unsqueeze(0)
            distance_loss = torch.cat([distance_loss, single_embedder_dist_loss], dim=0)
        distance_loss = self.config.dist_weight * distance_loss.mean()
        #tv_loss = self.config.tv_weight * tv_loss
        #total_loss = - distance_loss + tv_loss


        stable = sum(self.dist_n[-10:])

        #total_loss = - distance_loss
        #the larger the better: real:-2.7122/0.0643   fake:-3.0716/0.0453

        #if stable >8:
        if 0:
            total_loss = - 200 * distance_loss - 1 * torch.clamp( D_loss + 3.0 , max=0, min=-10)
            #total_loss = - 100 * distance_loss - 1 * torch.clamp( D_loss + 3.0 , max=0, min=-10) #10, 20
            #total_loss = - 200 * distance_loss - 1 * torch.clamp( D_loss + 3.0 , max=0, min=-10) #30
            latent_loss3 =  torch.clamp( D_loss + 3.0 , max=0, min=-10)


            total_loss += 100*torch.clamp( torch.norm( latent_z - self.latent_z_init, p=2 ) - 160, min=0)
            latent_loss = torch.norm( latent_z - self.latent_z_init, p=2 ) 


            neg_loss=0
            for tp_z in self.latent_neg_z:
                neg_loss += 200*torch.clamp( self.rad -torch.norm(latent_z - tp_z, p=2), min=0)
            total_loss += neg_loss
            print(neg_loss)
            latent_loss2 = neg_loss 


        else:
            #total_loss = - 20 * distance_loss
            #total_loss = - 20 * distance_loss - 1 * torch.clamp( D_loss + 3.0 , max=0, min=-10)
            total_loss = - 20 * distance_loss + 3 * torch.clamp( D_loss + 3.0 , max=10, min=0) #D_loss the less the better
            latent_loss3 =  torch.clamp( D_loss + 3.0 , max=10, min=0)



            #total_loss += 100*torch.clamp( torch.norm( latent_z - self.latent_z_init, p=2 ) - 320, min=0)
            #total_loss += 100*torch.clamp( torch.norm( latent_z - self.latent_z_init, p=2 ) - 200, min=0)
            #total_loss += 100*torch.clamp( torch.norm( latent_z - self.latent_z_init, p=2 ) - 20, min=0)
            #total_loss += 100*torch.clamp( torch.norm( latent_z - self.latent_z_init, p=2 ) - 15, min=0)
            #total_loss += 100*torch.clamp( torch.norm( latent_z - self.latent_z_init, p=2 ) - 12, min=0)
            #total_loss += 100*torch.clamp( torch.norm( latent_z - self.latent_z_init, p=2 ) - 9, min=0)
            #total_loss += 100*torch.clamp( torch.norm( latent_z - self.latent_z_init, p=2 ) - 12, min=0)
            #total_loss += 100*torch.clamp( torch.norm( latent_z - self.latent_z_init, p=2 ) - 14, min=0)
            total_loss += 100*torch.clamp( torch.norm( latent_z - self.latent_z_init, p=2 ) - 350, min=0)
            latent_loss = torch.norm( latent_z - self.latent_z_init, p=2 ) 


            neg_loss=0
            for tp_z in self.latent_neg_z:
                neg_loss += 30*torch.clamp( self.rad -torch.norm(latent_z - tp_z, p=2), min=0)

            total_loss += neg_loss
            #print(neg_loss)
            latent_loss2 = neg_loss 





        #return total_loss, [distance_loss, tv_loss, latent_loss]
        #return total_loss, [distance_loss, tv_loss, latent_loss, latent_loss2]
        #return total_loss, [distance_loss, D_loss, latent_loss, latent_loss2, D_loss]
        return total_loss, [distance_loss, D_loss, latent_loss, latent_loss2, latent_loss3]
    

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
        latent_z = latent_z.to(device)
        latent_z = self.latent_z_init * (1-self.latent_z_mask) + latent_z * self.latent_z_mask

        #x_inv = self.G.net.synthesis(latent_z.view(1, 14, 512))
        truncation = 0.7
        trunc = self.generator.mean_latent(4096)
        #trunc = torch.zeros((1, 512)).to(device)
        x_inv, _ =  self.generator([latent_z], input_is_latent=True, truncation=truncation, truncation_latent=trunc, randomize_noise=False)



        #print('    ')
        #print(torch.min(x_inv))
        #print(torch.max(x_inv))
        #print(torch.mean(x_inv))

        pred_inv = self.discriminator(x_inv)
        #print(pred_inv)

        if 0: #zoom in 
            x_inv= F.interpolate(x_inv, (140, 140))
            adv_patch = transforms.CenterCrop(112)(x_inv)
        else: #w/o zoom in
            adv_patch = F.interpolate(x_inv, (112, 112))


        #print(adv_patch.shape)
        #adv_patch = (adv_patch - self.G.min_val) * 255 / (self.G.max_val - self.G.min_val)
        #adv_patch = torch.clip( adv_patch + 0.5, 0, 255)/255.
        adv_patch = adv_patch.clamp_(-1., 1.) *0.5 + 0.5

        cls_id = cls_id.to(device)

        #preds = self.location_extractor(img_batch)

        #img_batch_applied = self.fxz_projector(img_batch, preds, adv_patch, self.blue_mask_t[:, 3], is_3d=True, do_aug=self.config.mask_aug)
        #img_batch_applied = self.fxz_projector(img_batch, preds, adv_patch, self.mask_t, is_3d=True, do_aug=self.config.mask_aug)
        #img_batch_applied = adv_patch

        img_batch_applied = img_batch * (1 - self.patch_mask) + adv_patch * self.patch_mask


        '''
        #final_patch = torch.cat([patch.squeeze(0), self.alpha])
        transforms.ToPILImage()(img_batch_applied[0]).save(self.config.current_dir+
                                                  '/saved_patches' +
                                                  '/patch_' +
                                                  str(epoch) +
                                                  '.png', 'PNG')
        '''


        patch_embs = {}
        for embedder_name, emb_model in self.embedders.items():
            patch_embs[embedder_name] = emb_model(img_batch_applied)

        #tv_loss = self.total_variation(adv_patch)
        #loss = self.loss_fn(patch_embs, tv_loss, cls_id, latent_z, epoch)
        #D_loss = F.softplus(pred_inv[0][0])
        D_loss = pred_inv[0][0]
        loss = self.loss_fn(patch_embs, D_loss, cls_id, latent_z, epoch)

        dist_loss = loss[1][0].item()
        if dist_loss > self.th and len(self.latent_neg_z)>self.len_new_neg:
            self.len_new_neg = len(self.latent_neg_z)
            fig = plt.figure()
            plt.ion()
            #plt.imshow(np.transpose(x_inv[0].cpu().detach().numpy(),[1,2,0]))  #detach() is bad for PGD, but ok for FG
            #encoder_out = self.G.postprocess(_get_tensor_value(img_batch_applied))
            plt.imshow( np.transpose(img_batch_applied[0].cpu().detach().numpy(), [1,2,0]) )  #detach() is bad for PGD, but ok for FG
            plt.axis('off')
            plt.show()

        #return loss, [img_batch, adv_patch, img_batch_applied, patch_embs, tv_loss]
        return loss, [img_batch, adv_patch, img_batch_applied, patch_embs, D_loss]

    #def save_losses(self, epoch_length, train_loss, dist_loss, tv_loss):
    def save_losses(self, epoch_length, train_loss, dist_loss, tv_loss, tv_loss2, tv_loss3):
        train_loss /= epoch_length
        dist_loss /= epoch_length
        tv_loss /= epoch_length
        tv_loss2 /= epoch_length
        self.train_losses_epoch.append(train_loss)
        self.dist_losses.append(dist_loss)
        self.tv_losses.append(tv_loss)
        self.tv_losses2.append(tv_loss2)
        self.tv_losses3.append(tv_loss3)

    def save_final_objects(self):
        alpha = transforms.ToTensor()(Image.open('../prnet/new_uv.png').convert('L'))
        #final_patch = torch.cat([self.best_patch.squeeze(0), alpha])
        #final_patch = self.G.net.synthesis(self.best_patch.view(1, 14, 512))
        truncation = 0.7
        trunc = self.generator.mean_latent(4096)
        x_inv, _ =  self.generator([self.best_patch], input_is_latent=True, truncation=truncation, truncation_latent=trunc, randomize_noise=False)
        final_patch = x_inv

        #final_patch = F.interpolate(final_patch, (112, 112))
        if 0: #zoom in
            x_inv= F.interpolate(final_patch, (140, 140))
            final_patch = transforms.CenterCrop(112)(x_inv)
        else: #w/o zoom in 
            final_patch = F.interpolate(final_patch, (112, 112))


        #need?
        #final_patch = self.G.postprocess(_get_tensor_value(final_patch))
        #final_patch = torch.from_numpy(final_patch).permute(0,3,1,2)/255.
        final_patch = final_patch.clamp_(-1., 1.)*0.5 + 0.5


        print(final_patch.shape)
        print(self.patch_mask.shape)
        final_patch = final_patch.cpu()
        self.patch_mask = self.patch_mask.cpu()
        final_patch = final_patch * self.patch_mask + torch.ones(3,112,112) * (1 - self.patch_mask)


        print(final_patch.shape)
        final_patch_img = transforms.ToPILImage()(final_patch[0])

        neg_latent = []
        for tp_z in self.latent_neg_z:
            neg_latent.append( tp_z.detach().cpu().numpy()  )

        print(len(self.latent_neg_z))
        np.save(self.config.current_dir + '/final_results/neg_latent.npy', np.array(neg_latent))
        np.save('/face/Mask/AdversarialMask/patch/neg_latent_all.npy', np.array(neg_latent))
        np.save('/face/Mask/AdversarialMask/patch/neg_latent_idx.npy', np.array(self.latent_neg_idx))


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
