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
from config import patch_config_types
from nn_modules import LandmarkExtractor, FaceXZooProjector, TotalVariation
from utils import load_embedder, EarlyStopping, get_patch
import skimage.io as io

import warnings
warnings.simplefilter('ignore', UserWarning)

global device
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print('device is {}'.format(device), flush=True)

import torch.nn as nn
import torch.nn.functional as F

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

class Net(nn.Module):
    def __init__(self, theta):
        super(Net, self).__init__()

        '''
        self.conv1 = nn.Conv2d(1, 10, kernel_size=5)
        self.conv2 = nn.Conv2d(10, 20, kernel_size=5)
        self.conv2_drop = nn.Dropout2d()
        self.fc1 = nn.Linear(320, 50)
        self.fc2 = nn.Linear(50, 10)

        # Spatial transformer localization-network
        self.localization = nn.Sequential(
            nn.Conv2d(1, 8, kernel_size=7),
            nn.MaxPool2d(2, stride=2),
            nn.ReLU(True),
            nn.Conv2d(8, 10, kernel_size=5),
            nn.MaxPool2d(2, stride=2),
            nn.ReLU(True)
        )

        # Regressor for the 3 * 2 affine matrix
        self.fc_loc = nn.Sequential(
            nn.Linear(10 * 3 * 3, 32),
            nn.ReLU(True),
            nn.Linear(32, 3 * 2)
        )

        # Initialize the weights/bias with identity transformation
        self.fc_loc[2].weight.data.zero_()
        self.fc_loc[2].bias.data.copy_(torch.tensor([1, 0, 0, 0, 1, 0], dtype=torch.float))
        '''
        self.theta = theta

    # Spatial transformer network forward function
    def stn(self, x):
        #xs = self.localization(x)
        #xs = xs.view(-1, 10 * 3 * 3)
        #theta = self.fc_loc(xs)
        theta = self.theta.view(-1, 2, 3)

        grid = F.affine_grid(theta, x.size())
        x = F.grid_sample(x, grid)

        return x

    def forward(self, x):
        # transform the input
        x = self.stn(x)
        
        '''
        # Perform the usual forward pass
        x = F.relu(F.max_pool2d(self.conv1(x), 2))
        x = F.relu(F.max_pool2d(self.conv2_drop(self.conv2(x)), 2))
        x = x.view(-1, 320)
        x = F.relu(self.fc1(x))
        x = F.dropout(x, training=self.training)
        x = self.fc2(x)
        return F.log_softmax(x, dim=1)
        '''
        return x


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
        self.target_embedding = utils.get_person_embedding(self.config, self.train_no_aug_loader, self.config.celeb_lab_mapper, self.location_extractor,
                                                           self.fxz_projector, self.embedders, device)
        self.best_patch = None
        print(self.config.celeb_lab_mapper)

        self.blue_mask_t = utils.load_mask(self.config, self.config.blue_mask_path, device)

        self.theta = torch.Tensor([1., 0., 0., 0., 1., 0.])
        self.stn = Net(self.theta).to(device)

        self.patch_mask = transforms.ToTensor()(Image.open('/face/Mask/AdversarialMask/datasets/012_mask7.png').convert('L')).to(device).unsqueeze(0)
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



        adv_patch_cpu = utils.get_patch(self.config)
        optimizer = optim.Adam([adv_patch_cpu], lr=self.config.start_learning_rate, amsgrad=True)
        #optimizer = optim.SGD([adv_patch_cpu], lr=self.config.start_learning_rate, momentum=0.9)
        scheduler = self.config.scheduler_factory(optimizer)
        early_stop = EarlyStopping(current_dir=self.config.current_dir, patience=self.config.es_patience, init_patch=adv_patch_cpu)
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
                #if cls_id != 1: #source cls_id=1

                if cls_id == target_id:
                    if epoch == 0:
                        print('target id')
                        print(self.config.celeb_lab_mapper[target_id])
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
                    print('source id')
                    print(self.config.celeb_lab_mapper[source_id])
                    io.imsave('source_img_'+str(cnt)+'.png',np.transpose(img_batch[0],[1,2,0]))



                cnt += 1
                (b_loss, sep_loss), vars = self.forward_step(img_batch, adv_patch_cpu, cls_id)

                train_loss += b_loss.item()
                dist_loss += sep_loss[0].item()
                tv_loss += sep_loss[1].item()

                optimizer.zero_grad()
                b_loss.backward()
                optimizer.step()

                adv_patch_cpu.data.clamp_(0, 1)

                progress_bar.set_postfix_str(prog_bar_desc.format(train_loss / (i_batch + 1),
                                                                  dist_loss / (i_batch + 1),
                                                                  tv_loss / (i_batch + 1),
                                                                  optimizer.param_groups[0]["lr"]))
                self.train_losses_iter.append(train_loss / (i_batch + 1))
                #print(epoch_length)
                #print(cnt)
                #if i_batch + 1 == epoch_length:
                #if cnt + 1 == 5:
                if cnt  == 20:
                    self.save_losses(epoch_length, train_loss, dist_loss, tv_loss)
                    progress_bar.set_postfix_str(prog_bar_desc.format(self.train_losses_epoch[-1],
                                                                      self.dist_losses[-1],
                                                                      self.tv_losses[-1],
                                                                      optimizer.param_groups[0]["lr"], ))
                del b_loss
                torch.cuda.empty_cache()
            if early_stop(self.train_losses_epoch[-1], adv_patch_cpu, epoch):
                self.best_patch = adv_patch_cpu
                #break

            scheduler.step(self.train_losses_epoch[-1])
        self.best_patch = early_stop.best_patch
        self.save_final_objects()
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

    def forward_step(self, img_batch, adv_patch_cpu, cls_id):
        img_batch = img_batch.to(device)
        adv_patch = adv_patch_cpu.to(device)
        cls_id = cls_id.to(device)

        #preds = self.location_extractor(img_batch)
        #img_batch_applied = self.fxz_projector(img_batch, preds, adv_patch, self.blue_mask_t[:, 3], is_3d=True, do_aug=self.config.mask_aug)
        #img_batch_applied = self.fxz_projector(img_batch, preds, adv_patch, do_aug=self.config.mask_aug)
        #img_batch_applied = adv_patch

        batch = 1
        scale = np.random.uniform(1./1.04,1.04, size=(batch,1))
        y1 = np.random.uniform(-10., 10., size=(batch,1))
        x1 = np.random.uniform(-10., 10., size=(batch,1))
        alpha = np.random.uniform(-5.,5., size=(batch,1))/180.*np.pi

        '''
        scale = np.random.uniform(1./1.1,1.1, size=(batch,1))
        y1 = np.random.uniform(-20., 20., size=(batch,1))
        x1 = np.random.uniform(-20., 20., size=(batch,1))
        alpha = np.random.uniform(-10.,10., size=(batch,1))/180.*np.pi
        '''

        theta = scale * np.hstack([np.cos(alpha), np.sin(alpha), -x1/112.,  -np.sin(alpha), np.cos(alpha), -y1/112.])
        self.stn.theta = torch.Tensor(theta)

        img_batch_applied = self.stn(adv_patch)
        img_batch_applied = img_batch * (1 - self.patch_mask) + self.stn(adv_patch) * self.patch_mask
        #img_batch_applied = img_batch * (1 - self.patch_mask) + adv_patch * self.patch_mask

        ''' 
        fig = plt.figure()
        plt.imshow(np.transpose(img_batch_applied[0].cpu().detach().numpy(),[1,2,0]))  #detach() is bad for PGD, but ok for FG
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

    def save_final_objects(self):
        #alpha = transforms.ToTensor()(Image.open('../prnet/new_uv.png').convert('L'))
        #final_patch = torch.cat([self.best_patch.squeeze(0), alpha])

        alpha = transforms.ToTensor()(Image.open('/face/Mask/AdversarialMask/datasets/012_mask7.png').convert('L')).unsqueeze(0)
        alpha = F.interpolate(alpha, (112, 112))
        final_patch = torch.cat([self.best_patch.squeeze(0), alpha.squeeze(0)])

        print(final_patch.shape)
        final_patch_img = transforms.ToPILImage()(final_patch.squeeze(0))
        final_patch_img.save(self.config.current_dir + '/final_results/final_patch.png', 'PNG')
        new_size = tuple(self.config.magnification_ratio * s for s in self.config.img_size)
        transforms.Resize(new_size)(final_patch_img).save(self.config.current_dir + '/final_results/final_patch_magnified.png', 'PNG')
        torch.save(self.best_patch, self.config.current_dir + '/final_results/final_patch_raw.pt')

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
