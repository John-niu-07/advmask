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


sys.path.append('/face/Mask/stylegan2-encoder-pytorch')
from model import Generator, Encoder, Discriminator


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
        #self.patch_mask = transforms.ToTensor()(Image.open('/face/Mask/AdversarialMask/datasets/012_mask.png').convert('L')).to(device).unsqueeze(0)
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



    #adv_mask = Image.open('/face/Mask/AdversarialMask/patch/experiments/November/25-11-2022_11-03-22/saved_patches/patch_30.png').convert('RGB') #full face
    #adv_mask = Image.open('/face/Mask/AdversarialMask/patch/experiments/November/25-11-2022_11-03-22/saved_patches/patch_45.png').convert('RGB') #full face
    #adv_mask = Image.open('/face/Mask/AdversarialMask/patch/experiments/November/29-11-2022_15-41-32/saved_patches/patch_43.png').convert('RGB') #full face
    #adv_mask = Image.open('/face/Mask/AdversarialMask/patch/experiments/December/27-12-2022_20-25-24/final_results/final_patch.png').convert('RGB') #full face





    device = 'cuda'
    image_size=256

    g_model_path = '/face/Mask/stylegan2-encoder-pytorch/checkpoint/generator_ffhq.pt'
    g_ckpt = torch.load(g_model_path, map_location=device)

    latent_dim = g_ckpt['args'].latent

    generator = Generator(image_size, latent_dim, 8).to(device)
    generator.load_state_dict(g_ckpt["g_ema"], strict=False)
    generator.eval()
    print('[generator loaded]')
    truncation = 0.7
    trunc = generator.mean_latent(4096).detach().clone()


    latent_all = np.load('/face/Mask/AdversarialMask/patch/neg_latent_all.npy')
    #latent_all = np.load('/face/Mask/AdversarialMask/patch/neg_latent_all_t.npy')
    ll = latent_all.shape[0]


    latent_z_np = np.load('/face/Mask/AdversarialMask/patch/init_face000005.npy').reshape((1, 14, 512))

    latent_zt_np = np.load('/face/Mask/AdversarialMask/patch/init_face000019.npy').reshape((1, 14, 512))



    adv_latent1 = []
    adv_latent3 = []
    adv_latent5 = []
    adv_latent7 = []
    adv_latent9 = []
    adv_latent50 = []

    adv_vector = []
    print('=========')
    for i in range(ll):
        if i == 0:
            continue

        lat = latent_all[i]
        k = np.reshape(lat-latent_z_np, (-1))
        kn = np.linalg.norm(k, ord=2)
        k_norm = k/kn
        adv_vector.append(k_norm)

        lat = torch.from_numpy(lat).to(device)
        print('---')
        for j in range(ll):
            if j == 0:
                continue

            lat2 = latent_all[j]
            lat2 = torch.from_numpy(lat2).to(device)
            print(torch.norm(lat - lat2, p=2))

            diff = np.reshape(np.abs((lat-lat2).cpu().numpy()), (-1))
            diff_idx = np.argsort(-diff)

            #fig = plt.figure()
            #plt.plot(diff[diff_idx[0:40]])
            #plt.show()
            if i!= j:
                adv_latent1.append(diff_idx[0:1])
                adv_latent3.append(diff_idx[0:3])
                adv_latent5.append(diff_idx[0:5])
                adv_latent7.append(diff_idx[0:7])
                adv_latent9.append(diff_idx[0:9])
                adv_latent50.append(diff_idx[0:50])

    print('==== z_init =====')
    for i in range(ll):
        if i == 0:
            continue

        lat = latent_all[i]
        print(np.linalg.norm(np.reshape(lat, (-1)) - np.reshape(latent_z_np,(-1)), ord=2))


    print('=========')
    print(np.linalg.norm(np.reshape(latent_z_np, (-1)) - np.reshape(latent_zt_np,(-1)), ord=2))
    print('===== z_t ====')
    for i in range(ll):
        if i == 0:
            continue

        lat = latent_all[i]
        print(np.linalg.norm(np.reshape(lat, (-1)) - np.reshape(latent_zt_np,(-1)), ord=2))
                
    '''
    print('==== cos =====')
    for i in adv_vector:
        print('---')
        for j in adv_vector:
            cos_sim = np.dot(i, j)
            print(cos_sim)

    np.save('adv_vec.npy', adv_vector )


     
    rand_vec = np.random.rand(100, 3) #- np.ones((100, 14*512))*0.5
    #rand_vec = np.random.rand(100, 14*512) #- np.ones((100, 14*512))*0.5

    print(rand_vec)

    cos_list = []
    print('==== cos =====')
    for i in range(20):
        print('---')
        for j in range(20):
            if i != j:
                cos_sim = np.dot(rand_vec[i], rand_vec[j])/(np.linalg.norm(rand_vec[i], ord=2)*np.linalg.norm(rand_vec[j], ord=2) )
                cos_list.append(cos_sim)
                print(cos_sim)

    print( sum(cos_list)/len(cos_list) )
    '''


    '''
    adv_final= []

    print(np.array(adv_latent1).shape)
    print(ll)
    #print( np.reshape(adv_latent1, (-1)) )
    print('******')
    adv1, freq = np.unique( np.reshape(adv_latent1, (-1)), return_counts=True)
    adv_idx = np.argsort(-freq)
    print(adv1[adv_idx[0:9]])
    print(freq[adv_idx[0:9]])
    adv_final.append(adv1[adv_idx[0:9]])

    print('******')
    adv3, freq = np.unique( np.reshape(adv_latent3, (-1)), return_counts=True)
    adv_idx = np.argsort(-freq)
    print(adv3[adv_idx[0:9]])
    print(freq[adv_idx[0:9]])
    adv_final.append(adv3[adv_idx[0:9]])
    print('******')
    adv5, freq = np.unique( np.reshape(adv_latent5, (-1)), return_counts=True)
    adv_idx = np.argsort(-freq)
    print(adv5[adv_idx[0:9]])
    print(freq[adv_idx[0:9]])
    adv_final.append(adv5[adv_idx[0:9]])
    print('******')
    adv7, freq = np.unique( np.reshape(adv_latent7, (-1)), return_counts=True)
    adv_idx = np.argsort(-freq)
    print(adv7[adv_idx[0:9]])
    print(freq[adv_idx[0:9]])
    adv_final.append(adv7[adv_idx[0:9]])
    print('******')
    adv9, freq = np.unique( np.reshape(adv_latent9, (-1)), return_counts=True)
    adv_idx = np.argsort(-freq)
    print(adv9[adv_idx[0:9]])
    print(freq[adv_idx[0:9]])
    adv_final.append(adv9[adv_idx[0:9]])


    print('***----------***')
    adv_f, freq = np.unique( np.reshape(adv_final, (-1)), return_counts=True)
    adv_idx = np.argsort(-freq)
    print(adv_f[adv_idx[0:9]])
    print(freq[adv_idx[0:9]])
    np.save('adv_latent.npy', adv_f[adv_idx[0:9]] )

    #for k in adv_f[adv_idx[0:9]]:
    #    print(np.mod(k,512))

    print('\n******')
    adv50, freq = np.unique( np.reshape(adv_latent50, (-1)), return_counts=True)
    adv_idx = np.argsort(-freq)
    print(adv50[adv_idx[0:29]])
    print(freq[adv_idx[0:29]])

    
    fig = plt.figure()
    ax1 = fig.add_subplot(1,1,1)
    ax1.set_ylim(0,5)
    plt.hist(np.reshape(adv_latent1, (-1)), bins=8000)

    
    ax2 = fig.add_subplot(3,1,2)
    ax2.set_ylim(0,40)
    plt.hist(np.reshape(adv_latent10, (-1)), bins=8000)

    ax3 = fig.add_subplot(3,1,3)
    ax3.set_ylim(0,120)
    plt.hist(np.reshape(adv_latent50, (-1)), bins=8000)
    '''

    plt.show()

    '''
    z_now = np.load('/face/Mask/AdversarialMask/patch/init_face000005.npy').reshape((1, 14, 512))
    adv_vec = np.load('/face/Mask/AdversarialMask/patch/adv_vec.npy').reshape((18, 14, 512))
    adv_v = adv_vec[9]

    imgs = []
    for i in range(20):
        z_now_v = z_now + 1.2 * i * adv_v
        z_now_ = torch.from_numpy(z_now_v).to(device)

        with torch.no_grad():
            imgs_gen, _ =  generator([z_now_],
                           input_is_latent=True,
                           truncation=truncation,
                           truncation_latent=trunc,
                           randomize_noise=False)
        imgs.append( imgs_gen.detach().reshape((3,256,256)) )


    '''


    imgs = []
    lat_old = torch.from_numpy(latent_all[0]).to(device)
    for i in range(ll):
        lat = latent_all[i]
        lat = torch.from_numpy(lat).to(device)
        #print(torch.norm(lat, p=2))
        #print(torch.norm(lat - lat_old, p=2))
        lat_old = lat
        with torch.no_grad():
            imgs_gen, _ =  generator([lat],
                           input_is_latent=True,
                           truncation=truncation,
                           truncation_latent=trunc,
                           randomize_noise=False)
        imgs.append( imgs_gen.detach().reshape((3,256,256)) )
    


    adv_mask = imgs[1].clamp_(-1., 1.)*0.5 + 0.5
    adv_mask_t = adv_mask.unsqueeze(0)
    print(adv_mask_t.shape)

    #adv_mask = Image.open('/face/Mask/AdversarialMask/patch/experiments/December/27-12-2022_20-25-24/final_results/final_patch.png').convert('RGB') #full face
    #adv_mask_t = transforms.ToTensor()(adv_mask).unsqueeze(0)
    #print(adv_mask_t.shape)



    adv_mask_t = F.interpolate(adv_mask_t, (112, 112))
    print('Starting test...', flush=True)
    evaluator = Evaluator(config, adv_mask_t)
    

    print(ll)





    img2 = Image.open('/face/Mask/AdversarialMask/datasets/CASIA/4204960/000019_aligned.png').convert('RGB')
    #img2 = Image.open('/face/Mask/AdversarialMask/datasets/CASIA/4204960_align/001_aligned.png').convert('RGB')
    #img3 = Image.open('/face/Mask/AdversarialMask/datasets/CASIA/4204960_align/002_aligned.png').convert('RGB')
    #img4 = Image.open('/face/Mask/AdversarialMask/datasets/CASIA/4204960_align/003_aligned.png').convert('RGB')
    #img5 = Image.open('/face/Mask/AdversarialMask/datasets/CASIA/4204960_align/004_aligned.png').convert('RGB')
    #img6 = Image.open('/face/Mask/AdversarialMask/datasets/CASIA/4204960_align/005_aligned.png').convert('RGB')
    #img7 = Image.open('/face/Mask/AdversarialMask/datasets/CASIA/4204960_align/006_aligned.png').convert('RGB')



    '''
    latent_z_np = np.load('./init_face_003.npy').reshape((1, 14, 512))
    latent_z = torch.from_numpy(latent_z_np).to(device).float()
    with torch.no_grad():
            imgs_gen, _ =  generator([latent_z],
                           input_is_latent=True,
                           truncation=truncation,
                           truncation_latent=trunc,
                           randomize_noise=False)
    img_ = imgs_gen[0].clamp_(-1., 1.)*0.5 + 0.5
    img_input = F.interpolate(img_.unsqueeze(0), (112, 112))
    '''
    img = Image.open('/face/Mask/AdversarialMask/datasets/CASIA/1302735/000005_aligned.png').convert('RGB')
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





    img_t = F.interpolate(torch.from_numpy(np.expand_dims(np.transpose(np.array(img2, dtype=np.float32)/255., [2,0,1]), 0)).to(device), (112, 112))
    img_t_applied = evaluator.apply_all_masks(img_t).unsqueeze(0)
    all_embeddingsT = evaluator.get_all_embeddings(img_t, img_t_applied)
    embeddingsT = all_embeddingsT['resnet50_arcface']
    cos_source_target = np.sum( embeddings[1]*embeddingsT[0] )/ (np.linalg.norm(embeddings[1]) *np.linalg.norm(embeddingsT[0]) )

    if ll > 2:
        adv_mask2 = imgs[2].clamp_(-1., 1.)*0.5 + 0.5
        adv_mask2_ = F.interpolate(adv_mask2.unsqueeze(0), (112, 112))
        evaluator2 = Evaluator(config, adv_mask2_)

        img_applied2 = evaluator2.apply_all_masks(img_input).unsqueeze(0)
        all_embeddings2 = evaluator2.get_all_embeddings(img_input, img_applied2)
        embeddings2 = all_embeddings2['resnet50_arcface']
        cos_source2_target = np.sum( embeddings2[1]*embeddingsT[0] )/ (np.linalg.norm(embeddings2[1]) *np.linalg.norm(embeddingsT[0]) )

        cos_source2_source1 = np.sum( embeddings2[1]*embeddings[1] )/ (np.linalg.norm(embeddings2[1]) *np.linalg.norm(embeddings[1]) )
        print('src2_src1: '+ str(cos_source2_source1))
        print(' ')


    if ll > 3:
        adv_mask3 = imgs[3].clamp_(-1., 1.)*0.5 + 0.5
        adv_mask3_ = F.interpolate(adv_mask3.unsqueeze(0), (112, 112))
        evaluator3 = Evaluator(config, adv_mask3_)

        img_applied3 = evaluator3.apply_all_masks(img_input).unsqueeze(0)
        all_embeddings3 = evaluator3.get_all_embeddings(img_input, img_applied3)
        embeddings3 = all_embeddings3['resnet50_arcface']
        cos_source3_target = np.sum( embeddings3[1]*embeddingsT[0] )/ (np.linalg.norm(embeddings3[1]) *np.linalg.norm(embeddingsT[0]) )

        cos_source3_source1 = np.sum( embeddings3[1]*embeddings[1] )/ (np.linalg.norm(embeddings3[1]) *np.linalg.norm(embeddings[1]) )
        print('src3_src1: '+ str(cos_source3_source1))
        cos_source3_source2 = np.sum( embeddings3[1]*embeddings2[1] )/ (np.linalg.norm(embeddings3[1]) *np.linalg.norm(embeddings2[1]) )
        print('src3_src2:'+ str(cos_source3_source2))
        print(' ')

    if ll > 4:
        adv_mask4 = imgs[4].clamp_(-1., 1.)*0.5 + 0.5
        adv_mask4_ = F.interpolate(adv_mask4.unsqueeze(0), (112, 112))
        evaluator4 = Evaluator(config, adv_mask4_)

        img_applied4 = evaluator4.apply_all_masks(img_input).unsqueeze(0)
        all_embeddings4 = evaluator4.get_all_embeddings(img_input, img_applied4)
        embeddings4 = all_embeddings4['resnet50_arcface']
        cos_source4_target = np.sum( embeddings4[1]*embeddingsT[0] )/ (np.linalg.norm(embeddings4[1]) *np.linalg.norm(embeddingsT[0]) )

        cos_source4_source1 = np.sum( embeddings4[1]*embeddings[1] )/ (np.linalg.norm(embeddings4[1]) *np.linalg.norm(embeddings[1]) )
        print('src4_src1: '+ str(cos_source4_source1))
        cos_source4_source2 = np.sum( embeddings4[1]*embeddings2[1] )/ (np.linalg.norm(embeddings4[1]) *np.linalg.norm(embeddings2[1]) )
        print('src4_src2: '+ str(cos_source4_source2))
        cos_source4_source3 = np.sum( embeddings4[1]*embeddings3[1] )/ (np.linalg.norm(embeddings4[1]) *np.linalg.norm(embeddings3[1]) )
        print('src4_src3: '+ str(cos_source4_source3))
        print(' ')



    if ll > 5:
        adv_mask5 = imgs[5].clamp_(-1., 1.)*0.5 + 0.5
        adv_mask5_ = F.interpolate(adv_mask5.unsqueeze(0), (112, 112))
        evaluator5 = Evaluator(config, adv_mask5_)

        img_applied5 = evaluator5.apply_all_masks(img_input).unsqueeze(0)
        all_embeddings5 = evaluator5.get_all_embeddings(img_input, img_applied5)
        embeddings5 = all_embeddings5['resnet50_arcface']
        cos_source5_target = np.sum( embeddings5[1]*embeddingsT[0] )/ (np.linalg.norm(embeddings5[1]) *np.linalg.norm(embeddingsT[0]) )

        cos_source5_source1 = np.sum( embeddings5[1]*embeddings[1] )/ (np.linalg.norm(embeddings5[1]) *np.linalg.norm(embeddings[1]) )
        print('src5_src1: '+ str(cos_source5_source1))
        cos_source5_source2 = np.sum( embeddings5[1]*embeddings2[1] )/ (np.linalg.norm(embeddings5[1]) *np.linalg.norm(embeddings2[1]) )
        print('src5_src2: '+ str(cos_source5_source2))
        cos_source5_source3 = np.sum( embeddings5[1]*embeddings3[1] )/ (np.linalg.norm(embeddings5[1]) *np.linalg.norm(embeddings3[1]) )
        print('src5_src3: '+ str(cos_source5_source3))
        cos_source5_source4 = np.sum( embeddings5[1]*embeddings4[1] )/ (np.linalg.norm(embeddings5[1]) *np.linalg.norm(embeddings4[1]) )
        print('src5_src4: '+ str(cos_source5_source4))
        print(' ')




    if ll > 6:
        adv_mask6 = imgs[6].clamp_(-1., 1.)*0.5 + 0.5
        adv_mask6_ = F.interpolate(adv_mask6.unsqueeze(0), (112, 112))
        evaluator6 = Evaluator(config, adv_mask6_)

        img_applied6 = evaluator6.apply_all_masks(img_input).unsqueeze(0)
        all_embeddings6 = evaluator6.get_all_embeddings(img_input, img_applied6)
        embeddings6 = all_embeddings6['resnet50_arcface']
        cos_source6_target = np.sum( embeddings6[1]*embeddingsT[0] )/ (np.linalg.norm(embeddings6[1]) *np.linalg.norm(embeddingsT[0]) )

        cos_source6_source1 = np.sum( embeddings6[1]*embeddings[1] )/ (np.linalg.norm(embeddings6[1]) *np.linalg.norm(embeddings[1]) )
        print('src6_src1: '+ str(cos_source6_source1))
        cos_source6_source2 = np.sum( embeddings6[1]*embeddings2[1] )/ (np.linalg.norm(embeddings6[1]) *np.linalg.norm(embeddings2[1]) )
        print('src6_src2: '+ str(cos_source6_source2))
        cos_source6_source3 = np.sum( embeddings6[1]*embeddings3[1] )/ (np.linalg.norm(embeddings6[1]) *np.linalg.norm(embeddings3[1]) )
        print('src6_src3: '+ str(cos_source6_source3))
        cos_source6_source4 = np.sum( embeddings6[1]*embeddings4[1] )/ (np.linalg.norm(embeddings6[1]) *np.linalg.norm(embeddings4[1]) )
        print('src6_src4: '+ str(cos_source6_source4))
        cos_source6_source5 = np.sum( embeddings6[1]*embeddings5[1] )/ (np.linalg.norm(embeddings6[1]) *np.linalg.norm(embeddings5[1]) )
        print('src6_src5: '+ str(cos_source6_source5))
        print(' ')






    if ll > 7:
        adv_mask7 = imgs[7].clamp_(-1., 1.)*0.5 + 0.5
        adv_mask7_ = F.interpolate(adv_mask7.unsqueeze(0), (112, 112))
        evaluator7 = Evaluator(config, adv_mask7_)

        img_applied7 = evaluator7.apply_all_masks(img_input).unsqueeze(0)
        all_embeddings7 = evaluator7.get_all_embeddings(img_input, img_applied7)
        embeddings7 = all_embeddings7['resnet50_arcface']
        cos_source7_target = np.sum( embeddings7[1]*embeddingsT[0] )/ (np.linalg.norm(embeddings7[1]) *np.linalg.norm(embeddingsT[0]) )

        cos_source7_source1 = np.sum( embeddings7[1]*embeddings[1] )/ (np.linalg.norm(embeddings7[1]) *np.linalg.norm(embeddings[1]) )
        print('src7_src1: '+ str(cos_source7_source1))
        cos_source7_source2 = np.sum( embeddings7[1]*embeddings2[1] )/ (np.linalg.norm(embeddings7[1]) *np.linalg.norm(embeddings2[1]) )
        print('src7_src2: '+ str(cos_source7_source2))
        cos_source7_source3 = np.sum( embeddings7[1]*embeddings3[1] )/ (np.linalg.norm(embeddings7[1]) *np.linalg.norm(embeddings3[1]) )
        print('src7_src3: '+ str(cos_source7_source3))
        cos_source7_source4 = np.sum( embeddings7[1]*embeddings4[1] )/ (np.linalg.norm(embeddings7[1]) *np.linalg.norm(embeddings4[1]) )
        print('src7_src4: '+ str(cos_source7_source4))
        cos_source7_source5 = np.sum( embeddings7[1]*embeddings5[1] )/ (np.linalg.norm(embeddings7[1]) *np.linalg.norm(embeddings5[1]) )
        print('src7_src5: '+ str(cos_source7_source5))
        cos_source7_source6 = np.sum( embeddings7[1]*embeddings6[1] )/ (np.linalg.norm(embeddings7[1]) *np.linalg.norm(embeddings6[1]) )
        print('src7_src6: '+ str(cos_source7_source6))
        print(' ')




    if ll > 8:
        adv_mask8 = imgs[8].clamp_(-1., 1.)*0.5 + 0.5
        adv_mask8_ = F.interpolate(adv_mask8.unsqueeze(0), (112, 112))
        evaluator8 = Evaluator(config, adv_mask8_)

        img_applied8 = evaluator8.apply_all_masks(img_input).unsqueeze(0)
        all_embeddings8 = evaluator8.get_all_embeddings(img_input, img_applied8)
        embeddings8 = all_embeddings8['resnet50_arcface']
        cos_source8_target = np.sum( embeddings8[1]*embeddingsT[0] )/ (np.linalg.norm(embeddings8[1]) *np.linalg.norm(embeddingsT[0]) )

        cos_source8_source1 = np.sum( embeddings8[1]*embeddings[1] )/ (np.linalg.norm(embeddings8[1]) *np.linalg.norm(embeddings[1]) )
        print('src8_src1: '+ str(cos_source8_source1))
        cos_source8_source2 = np.sum( embeddings8[1]*embeddings2[1] )/ (np.linalg.norm(embeddings8[1]) *np.linalg.norm(embeddings2[1]) )
        print('src8_src2: '+ str(cos_source8_source2))
        cos_source8_source3 = np.sum( embeddings8[1]*embeddings3[1] )/ (np.linalg.norm(embeddings8[1]) *np.linalg.norm(embeddings3[1]) )
        print('src8_src3: '+ str(cos_source8_source3))
        cos_source8_source4 = np.sum( embeddings8[1]*embeddings4[1] )/ (np.linalg.norm(embeddings8[1]) *np.linalg.norm(embeddings4[1]) )
        print('src8_src4: '+ str(cos_source8_source4))
        cos_source8_source5 = np.sum( embeddings8[1]*embeddings5[1] )/ (np.linalg.norm(embeddings8[1]) *np.linalg.norm(embeddings5[1]) )
        print('src8_src5: '+ str(cos_source8_source5))
        cos_source8_source6 = np.sum( embeddings8[1]*embeddings6[1] )/ (np.linalg.norm(embeddings8[1]) *np.linalg.norm(embeddings6[1]) )
        print('src8_src6: '+ str(cos_source8_source6))
        cos_source8_source7 = np.sum( embeddings8[1]*embeddings7[1] )/ (np.linalg.norm(embeddings8[1]) *np.linalg.norm(embeddings7[1]) )
        print('src8_src7: '+ str(cos_source8_source7))
        print(' ')












    fig = plt.figure()


    ax1 = fig.add_subplot(3,4,1)
    plt.imshow(np.transpose(img_input[0].cpu(),[1,2,0]))
    plt.axis('off')

    ax1 = fig.add_subplot(3,4,4)
    plt.imshow(np.transpose(img_t[0].cpu(),[1,2,0]))
    plt.axis('off')





    ax1 = fig.add_subplot(3,4,5)
    plt.imshow(np.transpose(img_applied[0].cpu(),[1,2,0]))
    plt.axis('off')
    #plt.text(0, 0., "%s " % (cos))
    plt.text(0, 0., "%s " % (cos_source_target))
    

    if ll > 2:
        ax1 = fig.add_subplot(3,4,6)
        plt.imshow(np.transpose(img_applied2[0][0].cpu(),[1,2,0]))
        plt.axis('off')
        plt.text(0, 0., "%s " % (cos_source2_target))

    if ll>3:
        ax1 = fig.add_subplot(3,4,7)
        plt.imshow(np.transpose(img_applied3[0][0].cpu(),[1,2,0]))
        plt.axis('off')
        plt.text(0, 0., "%s " % (cos_source3_target))



    if ll>4:
        ax1 = fig.add_subplot(3,4,8)
        plt.imshow(np.transpose(img_applied4[0][0].cpu(),[1,2,0]))
        plt.axis('off')
        plt.text(0, 0., "%s " % (cos_source4_target))



    if ll>5:
        ax1 = fig.add_subplot(3,4,9)
        plt.imshow(np.transpose(img_applied5[0][0].cpu(),[1,2,0]))
        plt.axis('off')
        plt.text(0, 0., "%s " % (cos_source5_target))

    if ll>6:
        ax1 = fig.add_subplot(3,4,10)
        plt.imshow(np.transpose(img_applied6[0][0].cpu(),[1,2,0]))
        plt.axis('off')
        plt.text(0, 0., "%s " % (cos_source6_target))


    if ll>7:
        ax1 = fig.add_subplot(3,4,11)
        plt.imshow(np.transpose(img_applied7[0][0].cpu(),[1,2,0]))
        plt.axis('off')
        plt.text(0, 0., "%s " % (cos_source7_target))


    if ll>8:
        ax1 = fig.add_subplot(3,4,12)
        plt.imshow(np.transpose(img_applied8[0][0].cpu(),[1,2,0]))
        plt.axis('off')
        plt.text(0, 0., "%s " % (cos_source8_target))


    plt.show()



    #evaluator = Evaluator(config, adv_mask_t)
    #evaluator.test()
    print('Finished test...', flush=True)


if __name__ == '__main__':
    main()
