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
from art.estimators.classification.editor import manipulate

import torchvision.models as models
from torch.autograd import Variable



import warnings
warnings.simplefilter('ignore', UserWarning)

global device
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print('device is {}'.format(device), flush=True)



def linear_interpolate(latent_code,
                       boundary,
                       start_distance=-3.0,
                       end_distance=3.0,
                       steps=10):
  """Manipulates the given latent code with respect to a particular boundary.
  Args:
    latent_code: The input latent code for manipulation.
    boundary: The semantic boundary as reference.
    start_distance: The distance to the boundary where the manipulation starts.
      (default: -3.0)
    end_distance: The distance to the boundary where the manipulation ends.
      (default: 3.0)
    steps: Number of steps to move the latent code from start position to end
      position. (default: 10)
  """
  #assert (latent_code.shape[0] == 1 and boundary.shape[0] == 1 and
  #        len(boundary.shape) == 2 and
  #        boundary.shape[1] == latent_code.shape[-1])

  linspace = np.linspace(start_distance, end_distance, steps)
  #if len(latent_code.shape) == 2:
  if 1:
    k = latent_code.dot(boundary)
    print(k)
    linspace = linspace - latent_code.dot(boundary.T)
    linspace = linspace.reshape(-1, 1).astype(np.float32)
    return latent_code + linspace * boundary
  if len(latent_code.shape) == 3:
    linspace = linspace.reshape(-1, 1, 1).astype(np.float32)
    return latent_code + linspace * boundary.reshape(1, 1, -1)
  #raise ValueError(f'Input `latent_code` should be with shape '
  #                 f'[1, latent_space_dim] or [1, N, latent_space_dim] for '
  #                 f'W+ space in Style GAN!\n'
  #                 f'But {latent_code.shape} is received.')



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
        self.latent_z_mask = None

        self.patch_mask = transforms.ToTensor()(Image.open('/face/Mask/AdversarialMask/datasets/012_mask5.png').convert('L')).to(device).unsqueeze(0)
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
        print(latent_z.shape)
        print(latent_z)

        latent_z_np = latent_z.cpu().detach().numpy()
        np.save('./train_targeted_gan_1302735_001_align.npy', latent_z_np)

        print(over)
        '''

        if 1:
            #latent_z = torch.ones((1, 7168), dtype=torch.float32)
            #latent_z = torch.rand((1, 7168), dtype=torch.float32) - 0.5
            latent_z = torch.zeros((1, 7168), dtype=torch.float32)


            #latent_z_np = np.load('../datasets/inverted_codes.npy')
            #print(latent_z_np.shape)

            latent_z_np = np.load('/face/Mask/idinvert_pytorch/results/inversion/my_test3/inverted_codes.npy').reshape((1, 14, 512))
            #latent_z_np = np.zeros((1,14,512))

            #boundary_file = np.load('/face/Mask/idinvert_pytorch/boundaries/stylegan_ffhq256/age.npy', allow_pickle=True)[()]
            #boundary_file = np.load('/face/Mask/idinvert_pytorch/boundaries/stylegan_ffhq256/eyeglasses.npy', allow_pickle=True)[()]
            boundary_file = np.load('/face/Mask/idinvert_pytorch/boundaries/stylegan_ffhq256/pose.npy', allow_pickle=True)[()]
            #boundary_file = np.load('/face/Mask/idinvert_pytorch/boundaries/stylegan_ffhq256/gender.npy', allow_pickle=True)[()]
            #boundary_file = np.load('/face/Mask/idinvert_pytorch/boundaries/stylegan_ffhq256/age.npy', allow_pickle=True)[()]
            #boundary_file = np.load('/face/Mask/idinvert_pytorch/boundaries/stylegan_ffhq256/expression.npy', allow_pickle=True)[()]
            boundary = boundary_file['boundary']
            manipulate_layers = boundary_file['meta_data']['manipulate_layers']



            codes = manipulate(latent_codes=latent_z_np,
                     boundary=boundary,
                     start_distance=-2.0,
                     end_distance=2.0,
                     step=14,
                     layerwise_manipulation=True,
                     num_layers=14,
                     manipulate_layers=manipulate_layers,
                     is_code_layerwise=True,
                     is_boundary_layerwise=True)



            #print(codes)
            print(codes.shape)



            for i in range(codes.shape[1]):
                latent_w_tmp = torch.tensor(torch.from_numpy(codes[0][i]), dtype=torch.float32).unsqueeze(0)

                x_inv = self.G.net.synthesis(latent_w_tmp.view(1, 14, 512))
                #encoder_out = self.G.postprocess(_get_tensor_value(latent_w_tmp))
                encoder_out = self.G.postprocess(_get_tensor_value(x_inv))

                '''
                fig = plt.figure()
                plt.imshow(encoder_out[0])  #detach() is bad for PGD, but ok for FG
                plt.axis('off')
                plt.show()
                '''

                final_patch_img = transforms.ToPILImage()(encoder_out.squeeze(0))
                final_patch_img.save('/face/Mask/AdversarialMask/patch/glass/'+str(i)+'.png', 'PNG')


            print(over)




            '''
            #latent_w = np.random.rand(512)-0.5
            #np.save('latent_w_rand.npy', latent_w)
            
            latent_w_tmp = torch.zeros((1, 7168), dtype=torch.float32)
            latent_w_short = np.load('latent_w_rand.npy')
            latent_w_short_ = torch.tensor( torch.from_numpy(latent_w_short), dtype=torch.float32)
            latent_w_tmp[0][0:512] = latent_w_short_
    
            #fig = plt.figure()
            #x_inv = self.G.net.synthesis(latent_w_tmp.view(1, 14, 512))
            #encoder_out = self.G.postprocess(_get_tensor_value(x_inv))
            #plt.imshow(encoder_out[0])  #detach() is bad for PGD, but ok for FG
            #plt.axis('off')
            #plt.show()
            

            steps = 12
            #boundary = np.load('/face/Mask/interfacegan/boundaries/stylegan_ffhq_eyeglasses_w_boundary.npy')
            boundary = np.load('/face/Mask/interfacegan/boundaries/stylegan_ffhq_gender_w_boundary.npy')
            #boundary = torch.tensor(torch.from_numpy(boundary[0]), dtype=torch.float32)
            interpolations = linear_interpolate(latent_w_short, boundary[0], steps =steps)

            print(interpolations.shape)

            for i in range(steps):
                latent_w_short = interpolations[i]


                latent_w_tmp = torch.zeros((1, 7168), dtype=torch.float32)
                latent_w_short_ = torch.tensor( torch.from_numpy(latent_w_short), dtype=torch.float32)
                latent_w_tmp[0][0:512] = latent_w_short_

                x_inv = self.G.net.synthesis(latent_w_tmp.view(1, 14, 512))
                #encoder_out = self.G.postprocess(_get_tensor_value(latent_w_tmp))
                encoder_out = self.G.postprocess(_get_tensor_value(x_inv))

                #fig = plt.figure()
                #plt.imshow(encoder_out[0])  #detach() is bad for PGD, but ok for FG
                #plt.axis('off')
                #plt.show()

                final_patch_img = transforms.ToPILImage()(encoder_out.squeeze(0))
                final_patch_img.save('/face/Mask/AdversarialMask/patch/glass/'+str(i)+'.png', 'PNG')


            print(over)
            '''


        else:          
            #latent_z_np = np.load('./train_targeted_gan_1302735_001_align.npy')
            latent_z_np = np.load('../datasets/inverted_codes.npy').reshape((1, 7168))
            #latent_z_np = np.load('/face/Mask/idinvert_pytorch/results/inversion/my_test2/inverted_codes.npy').reshape((1, 7168))
            

            latent_z = torch.from_numpy(latent_z_np)
            print(latent_z.shape)
            print(torch.max(latent_z))
            print(torch.min(latent_z))


            latent_w_short = latent_z_np[0][0:512]
            latent_w_tmp = torch.zeros((1, 7168), dtype=torch.float32)
            latent_w_short_ = torch.tensor( torch.from_numpy(latent_w_short), dtype=torch.float32)
            latent_w_tmp[0][0:512] = latent_w_short_
            latent_z = latent_w_tmp


        self.latent_z_init = latent_z.clone()

        
        latent_z_mask = torch.zeros((1, 7168), dtype=torch.float32)
        #latent_z_mask[0][0:512] = torch.ones((512), dtype=torch.float32)
        #latent_z_mask[0][0:512*14] = torch.ones((512*14), dtype=torch.float32)
        latent_z_mask[0][0:512*1] = torch.ones((512*1), dtype=torch.float32)
        #latent_z_mask[0][:] = torch.ones((7168), dtype=torch.float32)
        print(latent_z_mask)
        self.latent_z_mask = latent_z_mask
        
         
        x_inv = self.G.net.synthesis(latent_z.view(1, 14, 512))
        print(x_inv.shape)
      


        ''' 
        fig = plt.figure()
        #plt.imshow(np.transpose(x_inv[0].cpu().detach().numpy(),[1,2,0]))  #detach() is bad for PGD, but ok for FG
        encoder_out = self.G.postprocess(_get_tensor_value(x_inv))
        plt.imshow(encoder_out[0])  #detach() is bad for PGD, but ok for FG
        plt.axis('off')
        plt.show()
        print(over)
        '''


        latent_z.requires_grad_(True)
        #optimizer = optim.Adam([adv_patch_cpu], lr=self.config.start_learning_rate, amsgrad=True)
        optimizer = optim.Adam([latent_z], lr=self.config.start_learning_rate, amsgrad=True)
        #optimizer = optim.SGD([latent_z], lr=self.config.start_learning_rate, momentum=0.9, weight_decay=5e-4)

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



                cnt += 1
                (b_loss, sep_loss), vars = self.forward_step(img_batch, latent_z, cls_id)

                train_loss += b_loss.item()
                dist_loss += sep_loss[0].item()
                tv_loss += sep_loss[1].item()

                optimizer.zero_grad()
                b_loss.backward()
                optimizer.step()

                #latent_z.data.clamp_(0, 1)

                progress_bar.set_postfix_str(prog_bar_desc.format(train_loss / (i_batch + 1),
                                                                  dist_loss / (i_batch + 1),
                                                                  tv_loss / (i_batch + 1),
                                                                  optimizer.param_groups[0]["lr"]))
                self.train_losses_iter.append(train_loss / (i_batch + 1))
                #print(epoch_length)
                #print(cnt)
                #if i_batch + 1 == epoch_length:
                #if cnt + 1 == 5:
                if cnt  == 10:
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
        #self.save_final_objects()
        self.save_final_objects(latent_z)
        utils.plot_train_val_loss(self.config, self.train_losses_epoch, 'Epoch')
        utils.plot_train_val_loss(self.config, self.train_losses_iter, 'Iterations')
        utils.plot_separate_loss(self.config, self.train_losses_epoch, self.dist_losses, self.tv_losses)
    
    def loss_fn(self, patch_embs, tv_loss, cls_id):
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
        return total_loss, [distance_loss, tv_loss]
    

    def forward_step(self, img_batch, latent_z, cls_id):
        img_batch = img_batch.to(device)

        #adv_patch = adv_patch_cpu.to(device)
        latent_z = self.latent_z_init * (1-self.latent_z_mask) + latent_z * self.latent_z_mask

        x_inv = self.G.net.synthesis(latent_z.view(1, 14, 512))
        adv_patch = F.interpolate(x_inv, (112, 112))

        #print(adv_patch.shape)


        cls_id = cls_id.to(device)

        preds = self.location_extractor(img_batch)

        #img_batch_applied = self.fxz_projector(img_batch, preds, adv_patch, self.blue_mask_t[:, 3], is_3d=True, do_aug=self.config.mask_aug)
        #img_batch_applied = self.fxz_projector(img_batch, preds, adv_patch, self.mask_t, is_3d=True, do_aug=self.config.mask_aug)
        img_batch_applied = adv_patch
        #img_batch_applied = img_batch * (1 - self.patch_mask) + adv_patch * self.patch_mask





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
        loss = self.loss_fn(patch_embs, tv_loss, cls_id)

        return loss, [img_batch, adv_patch, img_batch_applied, patch_embs, tv_loss]

    def save_losses(self, epoch_length, train_loss, dist_loss, tv_loss):
        train_loss /= epoch_length
        dist_loss /= epoch_length
        tv_loss /= epoch_length
        self.train_losses_epoch.append(train_loss)
        self.dist_losses.append(dist_loss)
        self.tv_losses.append(tv_loss)

    def save_final_objects(self, latent_z):
        alpha = transforms.ToTensor()(Image.open('../prnet/new_uv.png').convert('L'))
        #final_patch = torch.cat([self.best_patch.squeeze(0), alpha])
        final_patch = self.G.net.synthesis(self.best_patch.view(1, 14, 512))

        final_patch = self.G.postprocess(_get_tensor_value(final_patch))
        final_patch_img = transforms.ToPILImage()(final_patch.squeeze(0))
        final_patch_img.save(self.config.current_dir + '/final_results/final_patch.png', 'PNG')
        #new_size = tuple(self.config.magnification_ratio * s for s in self.config.img_size)
        #transforms.Resize(new_size)(final_patch_img).save(self.config.current_dir + '/final_results/final_patch_magnified.png', 'PNG')
        #torch.save(self.best_patch, self.config.current_dir + '/final_results/final_patch_raw.pt')
    
        np.save(self.config.current_dir + '/final_results/latent_z.npy', latent_z.cpu().detach().numpy())

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
