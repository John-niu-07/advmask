import sys
import os
sys.path.append('/face/Mask/AdversarialMask')

import warnings
import utils
import torch
from nn_modules import LandmarkExtractor, FaceXZooProjector

from config_gan import patch_config_types
from torchvision import transforms
from PIL import Image
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import label_binarize
import matplotlib
from pathlib import Path
import pickle
#import seaborn as sns
import pandas as pd
#matplotlib.use('Agg')
matplotlib.use('TkAgg')
import skimage.io as io
import torch.nn.functional as F


global device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class Evaluator:
    def __init__(self, config, best_patch) -> None:
        super().__init__()
        self.config = config
        self.best_patch = best_patch
        face_landmark_detector = utils.get_landmark_detector(self.config, device)
        self.location_extractor = LandmarkExtractor(device, face_landmark_detector, self.config.img_size).to(device)
        self.fxz_projector = FaceXZooProjector(device, self.config.img_size, self.config.patch_size).to(device)
        self.transform = transforms.Compose([transforms.Resize(self.config.patch_size), transforms.ToTensor()])
        self.embedders = utils.load_embedder(self.config.test_embedder_names, device=device)
        emb_loaders, self.test_loaders = utils.get_test_loaders(self.config, self.config.test_celeb_lab.keys())

        self.target_embedding_w_mask, self.target_embedding_wo_mask = {}, {}
        '''
        for dataset_name, loader in emb_loaders.items():
            self.target_embedding_w_mask[dataset_name] = utils.get_person_embedding(self.config, loader, self.config.test_celeb_lab_mapper[dataset_name], self.location_extractor,
                                                                                    self.fxz_projector, self.embedders, device, include_others=True)
            self.target_embedding_wo_mask[dataset_name] = utils.get_person_embedding(self.config, loader, self.config.test_celeb_lab_mapper[dataset_name], self.location_extractor,
                                                                                     self.fxz_projector, self.embedders, device, include_others=False)
        '''
        self.random_mask_t = utils.load_mask(self.config, self.config.random_mask_path, device)
        self.blue_mask_t = utils.load_mask(self.config, self.config.blue_mask_path, device)
        self.full_mask_t = utils.load_mask(self.config, self.config.full_mask_path, device)
        self.face1_mask_t = utils.load_mask(self.config, self.config.face1_mask_path, device)
        self.face3_mask_t = utils.load_mask(self.config, self.config.face3_mask_path, device)
        self.mask_names = ['Clean', 'Adv', 'Random', 'Blue', 'Face1', 'Face3']

        self.mask_t = transforms.ToTensor()(Image.open('../prnet/new_uvT.png').convert('L'))


        #self.patch_mask = transforms.ToTensor()(Image.open('/face/Mask/AdversarialMask/datasets/012_mask5.png').convert('L')).to(device).unsqueeze(0)
        self.patch_mask = transforms.ToTensor()(Image.open('/face/Mask/AdversarialMask/datasets/012_mask6.png').convert('L')).to(device).unsqueeze(0)
        self.patch_mask = F.interpolate(self.patch_mask, (112, 112))

        #Path(self.config.current_dir).mkdir(parents=True, exist_ok=True)
        #utils.save_class_to_file(self.config, self.config.current_dir)
        
    def test(self):
        self.calc_overall_similarity()
        for dataset_name in self.test_loaders.keys():
            similarities_target_with_mask_by_person = self.get_final_similarity_from_disk('with_mask', dataset_name=dataset_name, by_person=True)
            similarities_target_without_mask_by_person = self.get_final_similarity_from_disk('without_mask', dataset_name=dataset_name, by_person=True)
            self.calc_similarity_statistics(similarities_target_with_mask_by_person, target_type='with', dataset_name=dataset_name, by_person=True)
            self.calc_similarity_statistics(similarities_target_without_mask_by_person, target_type='without', dataset_name=dataset_name, by_person=True)
            self.plot_sim_box(similarities_target_with_mask_by_person, target_type='with', dataset_name=dataset_name, by_person=True)
            self.plot_sim_box(similarities_target_without_mask_by_person, target_type='without', dataset_name=dataset_name, by_person=True)

    @torch.no_grad()
    def calc_overall_similarity(self):
        with warnings.catch_warnings():
            warnings.simplefilter('ignore', UserWarning)
            k=0
            adv_patch = self.best_patch.to(device)
            for dataset_name, loader in self.test_loaders.items():
                df_with_mask = pd.DataFrame(columns=['y_true', 'y_pred'])
                df_without_mask = pd.DataFrame(columns=['y_true', 'y_pred'])
                for img_batch, img_names, cls_id in tqdm(loader):
                    img_batch = img_batch.to(device)
                    cls_id = cls_id.to(device).type(torch.int32)

                    # Apply different types of masks
                    img_batch_applied = self.apply_all_masks(img_batch, adv_patch)

                    '''
                    #save masked_img
                    print(len(img_batch_applied))
                    for i in range(len(img_batch_applied)):
                        tmp_img = img_batch_applied[i].cpu()
                        for j in range(tmp_img.shape[0]):
                            print(np.transpose(tmp_img[j],[1,2,0]).shape)
                            k+=1
                            io.imsave('masked_img_'+str(k)+'.png',np.transpose(tmp_img[j],[1,2,0]))
                            #io.imsave('masked_img_'+str(i)+'_a.png',np.transpose(tmp_img[0],[1,2,0]))
                            #io.imsave('masked_img_'+str(i)+'_b.png',np.transpose(tmp_img[1],[1,2,0]))
                            #io.imsave('masked_img_'+str(i)+'_c.png',np.transpose(tmp_img[2],[1,2,0]))
                    '''


                    # Get embedding
                    all_embeddings = self.get_all_embeddings(img_batch, img_batch_applied)

                    self.calc_all_similarity(all_embeddings, img_names, cls_id, 'with_mask', dataset_name)
                    self.calc_all_similarity(all_embeddings, img_names, cls_id, 'without_mask', dataset_name)

                    df_with_mask = df_with_mask.append(self.calc_preds(cls_id, all_embeddings, target_type='with_mask', dataset_name=dataset_name))
                    df_without_mask = df_without_mask.append(self.calc_preds(cls_id, all_embeddings, target_type='without_mask', dataset_name=dataset_name))

                Path(os.path.join(self.config.current_dir, 'saved_preds', dataset_name)).mkdir(parents=True, exist_ok=True)
                df_with_mask.to_csv(os.path.join(self.config.current_dir, 'saved_preds', dataset_name, 'preds_with_mask.csv'), index=False)
                df_without_mask.to_csv(os.path.join(self.config.current_dir, 'saved_preds', dataset_name, 'preds_without_mask.csv'), index=False)

    def plot_sim_box(self, similarities, target_type, dataset_name, by_person=False):
        Path(os.path.join(self.config.current_dir, 'final_results', 'sim-boxes', dataset_name, target_type)).mkdir(parents=True, exist_ok=True)
        for emb_name in self.config.test_embedder_names:
            sim_df = pd.DataFrame()
            for i in range(len(similarities[emb_name])):
                sim_df[self.mask_names[i]] = similarities[emb_name][i]
            sorted_index = sim_df.mean().sort_values(ascending=False).index
            sim_df_sorted = sim_df[sorted_index]
            sns.boxplot(data=sim_df_sorted).set_title('Similarities for Different Masks')
            plt.xlabel('Mask Type')
            plt.ylabel('Similarity')
            avg_type = 'person' if by_person else 'image'
            plt.savefig(os.path.join(self.config.current_dir, 'final_results', 'sim-boxes', dataset_name, target_type, avg_type + '_' + emb_name + '.png'))
            plt.close()

    def write_similarities_to_disk(self, sims, img_names, cls_ids, sim_type, emb_name, dataset_name):
        Path(os.path.join(self.config.current_dir, 'saved_similarities', dataset_name, emb_name)).mkdir(parents=True, exist_ok=True)
        for i, lab in self.config.test_celeb_lab_mapper[dataset_name].items():
            Path(os.path.join(self.config.current_dir, 'saved_similarities', dataset_name, emb_name, lab)).mkdir(parents=True, exist_ok=True)
            for similarity, mask_name in zip(sims, self.mask_names):
                sim = similarity[cls_ids.cpu().numpy() == i].tolist()
                sim = {img_name: s for img_name, s in zip(img_names, sim)}
                with open(os.path.join(self.config.current_dir, 'saved_similarities', dataset_name, emb_name, lab, sim_type + '_' + mask_name + '.pickle'), 'ab') as f:
                    pickle.dump(sim, f)
        for similarity, mask_name in zip(sims, self.mask_names):
            sim = {img_name: s for img_name, s in zip(img_names, similarity.tolist())}
            with open(os.path.join(self.config.current_dir, 'saved_similarities', dataset_name, emb_name, sim_type + '_' + mask_name + '.pickle'), 'ab') as f:
                pickle.dump(sim, f)

    #def apply_all_masks(self, img_batch, adv_patch):
    def apply_all_masks(self, img_batch):
        adv_patch = self.best_patch.to(device)
        '''
        img_batch_applied_adv = utils.apply_mask(self.location_extractor,
                                                 self.fxz_projector, img_batch, adv_patch)
                                                 #self.fxz_projector, img_batch, adv_patch, self.blue_mask_t[:, 3], is_3d=True)
        
        img_batch_applied_random = utils.apply_mask(self.location_extractor,
                                                    self.fxz_projector, img_batch,
                                                    self.random_mask_t)
        
        img_batch_applied_blue = utils.apply_mask(self.location_extractor,
                                                  self.fxz_projector, img_batch,
                                                  self.blue_mask_t[:, :3],
                                                  self.blue_mask_t[:, 3], is_3d=True)
        
        img_batch_applied_face1 = utils.apply_mask(self.location_extractor,
                                                   self.fxz_projector, img_batch,
                                                   self.face1_mask_t[:, :3],
                                                   self.face1_mask_t[:, 3], is_3d=True)
        img_batch_applied_face3 = utils.apply_mask(self.location_extractor,
                                                   self.fxz_projector, img_batch,
                                                   self.face3_mask_t[:, :3],
                                                   self.face3_mask_t[:, 3], is_3d=True)

        return img_batch_applied_adv, img_batch_applied_random, img_batch_applied_blue, img_batch_applied_face1, img_batch_applied_face3
        '''

        '''
        img_batch_applied = utils.apply_mask(self.location_extractor,
                                                  self.fxz_projector, img_batch,
                                                  #adv_patch, self.mask_t, is_3d=True)
                                                  adv_patch, self.blue_mask_t[:, 3], is_3d=True)
        '''
        
        #img_batch_applied = adv_patch

        img_batch_applied = img_batch * (1 - self.patch_mask) + adv_patch * self.patch_mask

        return img_batch_applied

    def get_all_embeddings(self, img_batch, img_batch_applied_masks):
        batch_embs = {}
        for emb_name, emb_model in self.embedders.items():
            #batch_embs[emb_name] = [emb_model(img_batch.to(device)).cpu().numpy()]
            batch_embs[emb_name] = [emb_model(img_batch.to(device)).cpu().detach().numpy()]
            for img_batch_applied_mask in img_batch_applied_masks:
                #batch_embs[emb_name].append(emb_model(img_batch_applied_mask.to(device)).cpu().numpy())
                batch_embs[emb_name].append(emb_model(img_batch_applied_mask.to(device)).cpu().detach().numpy())
        return batch_embs

    def calc_all_similarity(self, all_embeddings, img_names, cls_id, target_type, dataset_name):
        for emb_name in self.config.test_embedder_names:
            target = self.target_embedding_w_mask[dataset_name][emb_name] if target_type == 'with_mask' else self.target_embedding_wo_mask[dataset_name][emb_name]
            target_embedding = torch.index_select(target, index=cls_id, dim=0).cpu().numpy().squeeze(-2)
            sims = []
            for emb in all_embeddings[emb_name]:
                sims.append(np.diag(cosine_similarity(emb, target_embedding)))
            self.write_similarities_to_disk(sims, img_names, cls_id, sim_type=target_type, emb_name=emb_name, dataset_name=dataset_name)

    def get_final_similarity_from_disk(self, sim_type, dataset_name, by_person=False):
        sims = {}
        for emb_name in self.config.test_embedder_names:
            sims[emb_name] = []
            for i, mask_name in enumerate(self.mask_names):
                if not by_person:
                    with open(os.path.join(self.config.current_dir, 'saved_similarities', dataset_name, emb_name, sim_type + '_' + mask_name + '.pickle'), 'rb') as f:
                        sims[emb_name].append([])
                        while True:
                            try:
                                data = pickle.load(f).values()
                                sims[emb_name][i].extend(list(data))
                            except EOFError:
                                break
                else:
                    sims[emb_name].append([])
                    for lab in self.config.test_celeb_lab[dataset_name]:
                        with open(os.path.join(self.config.current_dir, 'saved_similarities', dataset_name, emb_name, lab, sim_type + '_' + mask_name + '.pickle'), 'rb') as f:
                            person_sims = []
                            while True:
                                try:
                                    data = pickle.load(f).values()
                                    person_sims.extend(list(data))
                                except EOFError:
                                    break
                            print('---')
                            print(person_sims)
                            if len(person_sims) == 0:
                                person_avg_sim =0
                            else:
                                person_avg_sim = sum(person_sims) / (len(person_sims) + 1)
                            #person_avg_sim = sum(person_sims) 
                            sims[emb_name][i].append(person_avg_sim)
        return sims


    def calc_preds(self, cls_id, all_embeddings, target_type, dataset_name):
        df = pd.DataFrame(columns=['emb_name', 'mask_name', 'y_true', 'y_pred'])
        class_labels = list(range(0, len(self.config.test_celeb_lab_mapper[dataset_name])))
        y_true = label_binarize(cls_id.cpu().numpy(), classes=class_labels)
        y_true = [lab.tolist() for lab in y_true]
        for emb_name in self.config.test_embedder_names:
            target_embedding = self.target_embedding_w_mask[dataset_name][emb_name] \
                if target_type == 'with_mask' else self.target_embedding_wo_mask[dataset_name][emb_name]
            target_embedding = target_embedding.cpu().numpy().squeeze(-2)
            for i, mask_name in enumerate(self.mask_names):
                emb = all_embeddings[emb_name][i]
                cos_sim = cosine_similarity(emb, target_embedding)
                y_pred = [lab.tolist() for lab in cos_sim]
                new_rows = pd.DataFrame({
                    'emb_name': [emb_name] * len(y_true),
                    'mask_name': [mask_name] * len(y_true),
                    'y_true': y_true,
                    'y_pred': y_pred
                })
                df = df.append(new_rows, ignore_index=True)
        return df

    def calc_similarity_statistics(self, sim_dict, target_type, dataset_name, by_person=False):
        df_mean = pd.DataFrame(columns=['emb_name'] + self.mask_names)
        df_std = pd.DataFrame(columns=['emb_name'] + self.mask_names)
        for emb_name, sim_values in sim_dict.items():
            sim_values = np.array([np.array(lst) for lst in sim_values])
            sim_mean = np.round(sim_values.mean(axis=1), decimals=3)
            sim_std = np.round(sim_values.std(axis=1), decimals=3)
            df_mean = df_mean.append(pd.Series([emb_name] + sim_mean.tolist(), index=df_mean.columns), ignore_index=True)
            df_std = df_std.append(pd.Series([emb_name] + sim_std.tolist(), index=df_std.columns), ignore_index=True)

        avg_type = 'person' if by_person else 'image'
        Path(os.path.join(self.config.current_dir, 'final_results', 'stats', 'similarity', dataset_name, target_type)).mkdir(parents=True, exist_ok=True)
        df_mean.to_csv(os.path.join(self.config.current_dir, 'final_results', 'stats', 'similarity', dataset_name, target_type, 'mean_df' + '_' + avg_type + '.csv'), index=False)
        df_std.to_csv(os.path.join(self.config.current_dir, 'final_results', 'stats', 'similarity', dataset_name, target_type, 'std_df' + '_' + avg_type + '.csv'), index=False)


def main():
    mode = 'universal'
    config = patch_config_types[mode]()
    #adv_mask = Image.open('../data/masks/final_patch.png').convert('RGB')
    #adv_mask = Image.open('/face/Mask/AdversarialMask/patch/experiments/October/12-10-2022_11-22-59/final_results/final_patch.png').convert('RGB')
    #adv_mask = Image.open('/face/Mask/AdversarialMask/patch/experiments/October/13-10-2022_21-33-08/final_results/final_patch.png').convert('RGB')
    #adv_mask = Image.open('/face/Mask/AdversarialMask/patch/experiments/October/21-10-2022_08-32-29/saved_patches/patch_94.png').convert('RGB')
    #adv_mask = Image.open('/face/Mask/AdversarialMask/patch/experiments/October/21-10-2022_09-26-47/saved_patches/patch_90.png').convert('RGB')
    #adv_mask = Image.open('/face/Mask/AdversarialMask/patch/experiments/October/21-10-2022_11-42-50/saved_patches/patch_94.png').convert('RGB')


    #adv_mask = Image.open('/face/Mask/AdversarialMask/patch/experiments/October/21-10-2022_16-11-41/saved_patches/patch_33.png').convert('RGB')
    #adv_mask = Image.open('/face/Mask/AdversarialMask/patch/experiments/October/21-10-2022_16-22-22/saved_patches/patch_33.png').convert('RGB')
    #adv_mask = Image.open('/face/Mask/AdversarialMask/patch/experiments/October/21-10-2022_16-36-20/saved_patches/patch_99.png').convert('RGB')
    #adv_mask = Image.open('/face/Mask/AdversarialMask/patch/experiments/October/21-10-2022_17-01-26/saved_patches/patch_96.png').convert('RGB')
    #adv_mask = Image.open('/face/Mask/AdversarialMask/patch/experiments/October/21-10-2022_17-26-31/saved_patches/patch_95.png').convert('RGB')
    #adv_mask = Image.open('/face/Mask/AdversarialMask/patch/experiments/October/21-10-2022_17-52-30/saved_patches/patch_33.png').convert('RGB')
    #adv_mask = Image.open('/face/Mask/AdversarialMask/patch/experiments/October/21-10-2022_18-00-51/saved_patches/patch_85.png').convert('RGB')
    #adv_mask = Image.open('/face/Mask/AdversarialMask/patch/experiments/October/21-10-2022_18-41-28/saved_patches/patch_94.png').convert('RGB')
    #adv_mask = Image.open('/face/Mask/AdversarialMask/patch/experiments/October/21-10-2022_20-21-39/saved_patches/patch_80.png').convert('RGB')
    #adv_mask = Image.open('/face/Mask/AdversarialMask/patch/experiments/October/21-10-2022_21-02-24/saved_patches/patch_98.png').convert('RGB')

    #gan wrong
    #adv_mask = Image.open('/face/Mask/AdversarialMask/patch/experiments/October/22-10-2022_08-54-49/saved_patches/patch_97.png').convert('RGB')
    #adv_mask = Image.open('/face/Mask/AdversarialMask/patch/experiments/October/22-10-2022_09-54-40/saved_patches/patch_96.png').convert('RGB')
    #adv_mask = Image.open('/face/Mask/AdversarialMask/patch/experiments/October/22-10-2022_10-27-06/saved_patches/patch_94.png').convert('RGB')
    #adv_mask = Image.open('/face/Mask/AdversarialMask/patch/experiments/October/22-10-2022_10-56-52/saved_patches/patch_94.png').convert('RGB')
    #adv_mask = Image.open('/face/Mask/AdversarialMask/patch/experiments/October/22-10-2022_12-49-32/saved_patches/patch_98.png').convert('RGB')
    #adv_mask = Image.open('/face/Mask/AdversarialMask/patch/experiments/October/22-10-2022_13-36-52/saved_patches/patch_94.png').convert('RGB')
    #adv_mask = Image.open('/face/Mask/AdversarialMask/patch/experiments/October/22-10-2022_14-20-32/saved_patches/patch_95.png').convert('RGB')
    #adv_mask = Image.open('/face/Mask/AdversarialMask/patch/experiments/October/22-10-2022_14-45-29/saved_patches/patch_96.png').convert('RGB')
    #adv_mask = Image.open('/face/Mask/AdversarialMask/patch/experiments/October/22-10-2022_15-15-05/saved_patches/patch_94.png').convert('RGB')
    #adv_mask = Image.open('/face/Mask/AdversarialMask/patch/experiments/October/22-10-2022_15-47-38/saved_patches/patch_94.png').convert('RGB')


    #adv_mask = Image.open('/face/Mask/AdversarialMask/patch/experiments/November/01-11-2022_08-49-44/saved_patches/patch_94.png').convert('RGB') #niu_14layer
    #adv_mask = Image.open('/face/Mask/AdversarialMask/patch/experiments/November/01-11-2022_09-33-22/saved_patches/patch_98.png').convert('RGB') #niu_2layer
    #adv_mask = Image.open('/face/Mask/AdversarialMask/patch/experiments/November/01-11-2022_10-00-32/saved_patches/patch_90.png').convert('RGB') #niu_7layer

    #adv_mask = Image.open('/face/Mask/AdversarialMask/patch/experiments/November/01-11-2022_10-32-53/saved_patches/patch_94.png').convert('RGB') #random_14layer
    #adv_mask = Image.open('/face/Mask/AdversarialMask/patch/experiments/November/01-11-2022_10-32-53/final_results/final_patch.png').convert('RGB') 
    #adv_mask = Image.open('/face/Mask/AdversarialMask/patch/experiments/November/01-11-2022_11-38-39/final_results/final_patch.png').convert('RGB') #random_14layer + stn
    #adv_mask = Image.open('/face/Mask/AdversarialMask/patch/experiments/November/01-11-2022_10-56-57/saved_patches/patch_97.png').convert('RGB') #random_2layer


    #adv_mask = Image.open('/face/Mask/AdversarialMask/patch/experiments/October/30-10-2022_17-59-00/saved_patches/patch_295.png').convert('RGB') #johnny init 2 layer
    #adv_mask = Image.open('/face/Mask/AdversarialMask/patch/experiments/October/30-10-2022_19-19-32/saved_patches/patch_98.png').convert('RGB') #14 layer
    #adv_mask = Image.open('/face/Mask/AdversarialMask/patch/experiments/October/30-10-2022_19-46-59/saved_patches/patch_98.png').convert('RGB') #7 layer


    #adv_mask = Image.open('/face/Mask/AdversarialMask/patch/experiments/October/31-10-2022_15-11-36/saved_patches/patch_94.png').convert('RGB') #zhao init 14layer w/o stn
    #adv_mask = Image.open('/face/Mask/AdversarialMask/patch/experiments/October/31-10-2022_15-26-42/saved_patches/patch_96.png').convert('RGB') #zhao init 2layer w/o stn
    #adv_mask = Image.open('/face/Mask/AdversarialMask/patch/experiments/October/31-10-2022_15-43-15/saved_patches/patch_99.png').convert('RGB') #zhao init 14layer + stn
    #adv_mask = Image.open('/face/Mask/AdversarialMask/patch/experiments/November/01-11-2022_12-06-53/final_results/final_patch.png').convert('RGB') #zhao_14layer + stn
    #adv_mask = Image.open('/face/Mask/AdversarialMask/patch/experiments/October/31-10-2022_16-48-19/saved_patches/patch_90.png').convert('RGB') #zhao init 2layer + stn
    
    

    #adv_mask = Image.open('/face/Mask/AdversarialMask/patch/experiments/November/24-11-2022_16-19-21/final_results/final_patch.png').convert('RGB') #full face

    #adv_mask = Image.open('/face/Mask/AdversarialMask/patch/experiments/November/25-11-2022_11-03-22/saved_patches/patch_30.png').convert('RGB') #full face
    #adv_mask = Image.open('/face/Mask/AdversarialMask/patch/experiments/November/25-11-2022_11-03-22/saved_patches/patch_45.png').convert('RGB') #full face
    #adv_mask = Image.open('/face/Mask/AdversarialMask/patch/experiments/November/29-11-2022_15-41-32/saved_patches/patch_43.png').convert('RGB') #full face
    adv_mask = Image.open('/face/Mask/AdversarialMask/patch/experiments/November/30-11-2022_21-48-08/saved_patches/patch_43.png').convert('RGB') #full face


    #adv_mask = Image.open('/face/Mask/AdversarialMask/patch/October/04-10-2022_10-49-24/final_results/final_patch.png').convert('RGB')
    #adv_mask = Image.open('/face/Mask/AdversarialMask/patch/October/04-10-2022_11-13-25/final_results/final_patch.png').convert('RGB')
    adv_mask_t = transforms.ToTensor()(adv_mask).unsqueeze(0)
    adv_mask_t = F.interpolate(adv_mask_t, (112, 112))
    print('Starting test...', flush=True)
    evaluator = Evaluator(config, adv_mask_t)
    
    #img = Image.open('./john_hat_aligned.png').convert('RGB')
    #img = Image.open('/art/adversarial-robustness-toolbox/examples/dataset/img32_aligned.png').convert('RGB')
    img = Image.open('/face/Mask/AdversarialMask/datasets/CASIA/1302735/011_aligned.png').convert('RGB')
    #img = Image.open('/face/Mask/AdversarialMask/datasets/advmask_aligned.png').convert('RGB')
    #img = Image.open('/face/Mask/AdversarialMask/datasets/CASIA/0005238/047_aligned.png').convert('RGB')
    imgP = Image.open('/face/Mask/AdversarialMask/datasets/012_aligned.png').convert('RGB')
    imgP2 = Image.open('/face/Mask/AdversarialMask/datasets/009_aligned.png').convert('RGB')

    img2 = Image.open('/face/Mask/AdversarialMask/datasets/CASIA/4204960_align/001_aligned.png').convert('RGB')
    img3 = Image.open('/face/Mask/AdversarialMask/datasets/CASIA/4204960_align/002_aligned.png').convert('RGB')
    img4 = Image.open('/face/Mask/AdversarialMask/datasets/CASIA/4204960_align/003_aligned.png').convert('RGB')
    img5 = Image.open('/face/Mask/AdversarialMask/datasets/CASIA/4204960_align/004_aligned.png').convert('RGB')
    img6 = Image.open('/face/Mask/AdversarialMask/datasets/CASIA/4204960_align/005_aligned.png').convert('RGB')
    img7 = Image.open('/face/Mask/AdversarialMask/datasets/CASIA/4204960_align/006_aligned.png').convert('RGB')
    #img2 = Image.open('/face/Mask/AdversarialMask/datasets/CASIA/0001152/110_aligned.png').convert('RGB')

    img_np = np.transpose(np.array(img, dtype=np.float32)/255., [2,0,1])
    img_np = np.expand_dims(img_np,0)
    img_t = torch.from_numpy(img_np).to(device)
    img_input = F.interpolate(img_t, (112, 112))
    img_applied = evaluator.apply_all_masks(img_input)

    img_applied_ = img_applied.unsqueeze(0)
    all_embeddings = evaluator.get_all_embeddings(img_input, img_applied_)
    embeddings = all_embeddings['resnet50_arcface']
    cos = np.sum( embeddings[0]*embeddings[1] )/ (np.linalg.norm(embeddings[1]) *np.linalg.norm(embeddings[0]) )
    print(cos)
    #print(np.sum( embeddings[1]*embeddings[1] )/ (np.linalg.norm(embeddings[1]) *np.linalg.norm(embeddings[0]) ))


    imgP_np = np.transpose(np.array(imgP, dtype=np.float32)/255., [2,0,1])
    imgP_np = np.expand_dims(imgP_np,0)
    imgP_t = torch.from_numpy(imgP_np).to(device)
    imgP_input = F.interpolate(imgP_t, (112, 112))
    imgP_applied = evaluator.apply_all_masks(imgP_input)
    imgP_applied_ = imgP_applied.unsqueeze(0)
    allP_embeddings = evaluator.get_all_embeddings(imgP_input, imgP_applied_)
    embeddingsP = allP_embeddings['resnet50_arcface']
    cosP = np.sum( embeddings[0]*embeddingsP[0] )/ (np.linalg.norm(embeddings[0]) *np.linalg.norm(embeddingsP[0]) )

    imgP2_np = np.transpose(np.array(imgP2, dtype=np.float32)/255., [2,0,1])
    imgP2_np = np.expand_dims(imgP2_np,0)
    imgP2_t = torch.from_numpy(imgP2_np).to(device)
    imgP2_input = F.interpolate(imgP2_t, (112, 112))
    imgP2_applied = evaluator.apply_all_masks(imgP2_input)
    imgP2_applied_ = imgP2_applied.unsqueeze(0)
    allP2_embeddings = evaluator.get_all_embeddings(imgP2_input, imgP2_applied_)
    embeddingsP2 = allP2_embeddings['resnet50_arcface']
    cosP2 = np.sum( embeddings[0]*embeddingsP2[0] )/ (np.linalg.norm(embeddings[0]) *np.linalg.norm(embeddingsP2[0]) )





    img_np2 = np.transpose(np.array(img2, dtype=np.float32)/255., [2,0,1])
    img_np2 = np.expand_dims(img_np2,0)
    img_t2 = torch.from_numpy(img_np2).to(device)
    img_input2 = F.interpolate(img_t2, (112, 112))
    img_applied2 = evaluator.apply_all_masks(img_input2)
    img_applied2_ = img_applied2.unsqueeze(0)
    all_embeddings2 = evaluator.get_all_embeddings(img_input2, img_applied2_)
    embeddings2 = all_embeddings2['resnet50_arcface']
    cos2 = np.sum( embeddings2[0]*embeddings2[1] )/ (np.linalg.norm(embeddings2[1]) *np.linalg.norm(embeddings2[0]) )
    cos2_source_target = np.sum( embeddings[1]*embeddings2[0] )/ (np.linalg.norm(embeddings[1]) *np.linalg.norm(embeddings2[0]) )
    print(cos2)


    img_np3 = np.transpose(np.array(img3, dtype=np.float32)/255., [2,0,1])
    img_np3 = np.expand_dims(img_np3,0)
    img_t3 = torch.from_numpy(img_np3).to(device)
    img_input3 = F.interpolate(img_t3, (112, 112))
    img_applied3 = evaluator.apply_all_masks(img_input3)
    img_applied3_ = img_applied3.unsqueeze(0)
    all_embeddings3 = evaluator.get_all_embeddings(img_input3, img_applied3_)
    embeddings3 = all_embeddings3['resnet50_arcface']
    cos3 = np.sum( embeddings3[0]*embeddings3[1] )/ (np.linalg.norm(embeddings3[1]) *np.linalg.norm(embeddings3[0]) )
    cos3_source_target = np.sum( embeddings[1]*embeddings3[0] )/ (np.linalg.norm(embeddings[1]) *np.linalg.norm(embeddings3[0]) )
    print(cos3)


    img_np4 = np.transpose(np.array(img4, dtype=np.float32)/255., [2,0,1])
    img_np4 = np.expand_dims(img_np4,0)
    img_t4 = torch.from_numpy(img_np4).to(device)
    img_input4 = F.interpolate(img_t4, (112, 112))
    img_applied4 = evaluator.apply_all_masks(img_input4)
    img_applied4_ = img_applied4.unsqueeze(0)
    all_embeddings4 = evaluator.get_all_embeddings(img_input4, img_applied4_)
    embeddings4 = all_embeddings4['resnet50_arcface']
    cos4 = np.sum( embeddings4[0]*embeddings4[1] )/ (np.linalg.norm(embeddings4[1]) *np.linalg.norm(embeddings4[0]) )
    cos4_source_target = np.sum( embeddings[1]*embeddings4[0] )/ (np.linalg.norm(embeddings[1]) *np.linalg.norm(embeddings4[0]) )
    print(cos4)

    img_np5 = np.transpose(np.array(img5, dtype=np.float32)/255., [2,0,1])
    img_np5 = np.expand_dims(img_np5,0)
    img_t5 = torch.from_numpy(img_np5).to(device)
    img_input5 = F.interpolate(img_t5, (112, 112))
    img_applied5 = evaluator.apply_all_masks(img_input5)
    img_applied5_ = img_applied5.unsqueeze(0)
    all_embeddings5 = evaluator.get_all_embeddings(img_input5, img_applied5_)
    embeddings5 = all_embeddings5['resnet50_arcface']
    cos5 = np.sum( embeddings5[0]*embeddings5[1] )/ (np.linalg.norm(embeddings5[1]) *np.linalg.norm(embeddings5[0]) )
    cos5_source_target = np.sum( embeddings[1]*embeddings5[0] )/ (np.linalg.norm(embeddings[1]) *np.linalg.norm(embeddings5[0]) )
    print(cos5)

    img_np6 = np.transpose(np.array(img6, dtype=np.float32)/255., [2,0,1])
    img_np6 = np.expand_dims(img_np6,0)
    img_t6 = torch.from_numpy(img_np6).to(device)
    img_input6 = F.interpolate(img_t6, (112, 112))
    img_applied6 = evaluator.apply_all_masks(img_input6)
    img_applied6_ = img_applied6.unsqueeze(0)
    all_embeddings6 = evaluator.get_all_embeddings(img_input6, img_applied6_)
    embeddings6 = all_embeddings6['resnet50_arcface']
    cos6 = np.sum( embeddings6[0]*embeddings6[1] )/ (np.linalg.norm(embeddings6[1]) *np.linalg.norm(embeddings6[0]) )
    cos6_source_target = np.sum( embeddings[1]*embeddings6[0] )/ (np.linalg.norm(embeddings[1]) *np.linalg.norm(embeddings6[0]) )
    print(cos6)

    img_np7 = np.transpose(np.array(img7, dtype=np.float32)/255., [2,0,1])
    img_np7 = np.expand_dims(img_np7,0)
    img_t7 = torch.from_numpy(img_np7).to(device)
    img_input7 = F.interpolate(img_t7, (112, 112))
    img_applied7 = evaluator.apply_all_masks(img_input7)
    img_applied7_ = img_applied7.unsqueeze(0)
    all_embeddings7 = evaluator.get_all_embeddings(img_input7, img_applied7_)
    embeddings7 = all_embeddings7['resnet50_arcface']
    cos7 = np.sum( embeddings7[0]*embeddings7[1] )/ (np.linalg.norm(embeddings7[1]) *np.linalg.norm(embeddings7[0]) )
    cos7_source_target = np.sum( embeddings[1]*embeddings7[0] )/ (np.linalg.norm(embeddings[1]) *np.linalg.norm(embeddings7[0]) )
    print(cos7)

    #cos_orig = np.sum( embeddings[0]*embeddings2[0] )/ (np.linalg.norm(embeddings[0]) *np.linalg.norm(embeddings2[0]) )
    #cos_masked = np.sum( embeddings[1]*embeddings2[1] )/ (np.linalg.norm(embeddings[1]) *np.linalg.norm(embeddings2[1]) )
    #cos_source_target = np.sum( embeddings[1]*embeddings2[0] )/ (np.linalg.norm(embeddings[1]) *np.linalg.norm(embeddings2[0]) )


    #img_applied2 = F.interpolate(img_applied2, (512, 512)).cpu()
    #print(img_applied2.shape)
    #io.imsave('out2_img.png',np.transpose(img_applied2[0],[1,2,0]))
    
    #img_applied = F.interpolate(img_applied, (512, 512)).cpu()
    #print(img_applied.shape)
    #io.imsave('out_img.png',np.transpose(img_applied[0],[1,2,0]))

    fig = plt.figure()


    ax1 = fig.add_subplot(3,4,1)
    #plt.imshow(img)
    plt.imshow(np.transpose(img_input[0].cpu(),[1,2,0]))
    plt.axis('off')


    fig.add_subplot(3,4,2)
    #plt.imshow(imgP)
    plt.imshow(np.transpose(imgP_input[0].cpu(),[1,2,0]))
    #plt.imshow(np.transpose(imgP_applied[0].cpu(),[1,2,0]))
    plt.text(0, 0., "%s " % (cosP))
    plt.axis('off')

    ax1 = fig.add_subplot(3,4,3)
    #plt.imshow(imgP)
    plt.imshow(np.transpose(imgP2_input[0].cpu(),[1,2,0]))
    #plt.imshow(np.transpose(imgP2_applied[0].cpu(),[1,2,0]))
    plt.axis('off')
    plt.text(0, 0., "%s " % (cosP2))




    ax1 = fig.add_subplot(3,4,4)
    plt.imshow(np.transpose(img_applied[0].cpu(),[1,2,0]))
    plt.axis('off')
    plt.text(0, 0., "%s " % (cos))
    





    ax1 = fig.add_subplot(3,4,5)
    #plt.imshow(np.transpose(img_applied2[0].cpu(),[1,2,0]))
    plt.imshow(np.transpose(img_input2[0].cpu(),[1,2,0]))
    plt.axis('off')
    plt.text(0, 0., "%s " % (cos2_source_target))

    ax1 = fig.add_subplot(3,4,6)
    #plt.imshow(np.transpose(img_applied3[0].cpu(),[1,2,0]))
    plt.imshow(np.transpose(img_input3[0].cpu(),[1,2,0]))
    plt.axis('off')
    plt.text(0, 0., "%s " % (cos3_source_target))

    ax1 = fig.add_subplot(3,4,7)
    #plt.imshow(np.transpose(img_applied4[0].cpu(),[1,2,0]))
    plt.imshow(np.transpose(img_input4[0].cpu(),[1,2,0]))
    plt.axis('off')
    plt.text(0, 0., "%s " % (cos4_source_target))


    ax1 = fig.add_subplot(3,4,8)
    #plt.imshow(np.transpose(img_applied5[0].cpu(),[1,2,0]))
    plt.imshow(np.transpose(img_input5[0].cpu(),[1,2,0]))
    plt.axis('off')
    plt.text(0, 0., "%s " % (cos5_source_target))

    ax1 = fig.add_subplot(3,4,9)
    #plt.imshow(np.transpose(img_applied6[0].cpu(),[1,2,0]))
    plt.imshow(np.transpose(img_input6[0].cpu(),[1,2,0]))
    plt.axis('off')
    plt.text(0, 0., "%s " % (cos6_source_target))

    ax1 = fig.add_subplot(3,4,10)
    #plt.imshow(np.transpose(img_applied7[0].cpu(),[1,2,0]))
    plt.imshow(np.transpose(img_input7[0].cpu(),[1,2,0]))
    plt.axis('off')
    plt.text(0, 0., "%s " % (cos7_source_target))



    plt.show()



    #evaluator = Evaluator(config, adv_mask_t)
    #evaluator.test()
    print('Finished test...', flush=True)


if __name__ == '__main__':
    main()
