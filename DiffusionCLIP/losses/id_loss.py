import torch
from torch import nn
from configs.paths_config import MODEL_PATHS
from models.insight_face.model_irse import Backbone, MobileFaceNet

import face_recognition.insightface_torch.backbones as InsightFaceResnetBackbone
import face_recognition.magface_torch.backbones as MFBackbone
from collections import OrderedDict

import os
import torch.nn.functional as F
from torch.nn import CosineSimilarity, L1Loss, MSELoss




embedders_dict = {
    'resnet18': {
        'layers': [2, 2, 2, 2],
        'heads': {
            'arcface': {
                'weights_path': os.path.join('/face/Mask/AdversarialMask', 'face_recognition', 'insightface_torch', 'weights', 'ms1mv3_arcface_resnet18.pth')
            },
            'cosface': {
                'weights_path': os.path.join('/face/Mask/AdversarialMask', 'face_recognition', 'insightface_torch', 'weights', 'glint360k_cosface_resnet18.pth')
            },
            'magface': {
                'weights_path': os.path.join('/face/Mask/AdversarialMask', 'face_recognition', 'magface_torch', 'weights',
                                             'magface_iresnet18_casia_dp.pth')
            }
        }
    },
    'resnet34': {
        'layers': [3, 4, 6, 3],
        'heads': {
            'arcface': {
                'weights_path': os.path.join('/face/Mask/AdversarialMask', 'face_recognition', 'insightface_torch', 'weights',
                                             'ms1mv3_arcface_resnet34.pth')
            },
            'cosface': {
                'weights_path': os.path.join('/face/Mask/AdversarialMask', 'face_recognition', 'insightface_torch', 'weights',
                                             'glint360k_cosface_resnet34.pth')
            }
        }
    },
    'resnet50': {
        'layers': [3, 4, 14, 3],
        'heads': {
            'arcface': {
                'weights_path': os.path.join('/face/Mask/AdversarialMask', 'face_recognition', 'insightface_torch', 'weights',
                                             'ms1mv3_arcface_resnet50.pth')
            },
            'cosface': {
                'weights_path': os.path.join('/face/Mask/AdversarialMask', 'face_recognition', 'insightface_torch', 'weights',
                                             'glint360k_cosface_resnet50.pth')
            },
            'magface': {
                'weights_path': os.path.join('/face/Mask/AdversarialMask', 'face_recognition', 'magface_torch', 'weights',
                                             'magface_iresnet50_MS1MV2_ddp_fp32.pth')
            }
        }
    },
    'resnet100': {
        'layers': [3, 13, 30, 3],
        'heads': {
            'arcface': {
                'weights_path': os.path.join('/face/Mask/AdversarialMask', 'face_recognition', 'insightface_torch', 'weights',
                                             'ms1mv3_arcface_resnet100.pth')
            },
            'cosface': {
                'weights_path': os.path.join('/face/Mask/AdversarialMask', 'face_recognition', 'insightface_torch', 'weights',
                                             'glint360k_cosface_resnet100.pth')
            },
            'magface': {
                'weights_path': os.path.join('/face/Mask/AdversarialMask', 'face_recognition', 'magface_torch', 'weights',
                                             'magface_epoch_00025.pth')
            }
        }
    }
}



def rewrite_weights_dict(sd):
    sd.pop('fc.weight')
    sd_new = OrderedDict()
    for key, value in sd.items():
        new_key = key.replace('features.module.', '')
        sd_new[new_key] = value
    return sd_new



#magface18
def rewrite_weights_dict_mag18(sd):
    #sd.pop('fc.weight')
    sd_new = OrderedDict()
    for key, value in sd.items():
        #print(key)
        if key == 'module.fc.weight' or key == 'parallel_fc.weight':
            print(key)
        else:
            new_key = key.replace('module.features.', '')
            sd_new[new_key] = value
    return sd_new


#magface50
def rewrite_weights_dict_mag50(sd):
    #sd.pop('fc.weight')
    sd_new = OrderedDict()
    for key, value in sd.items():
        #print(key)
        if key == 'fc.weight' or key == 'parallel_fc.weight':
            print(key)
        else:
            new_key = key.replace('features.module.', '')
            sd_new[new_key] = value
    return sd_new



#embedder_names = ['resnet50_arcface']
#embedder_names = ['resnet50_arcface', 'resnet50_cosface']

#embedder_names = ['resnet100_magface', 'resnet50_magface', 'resnet18_magface']
#embedder_names = ['resnet34_cosface', 'resnet18_arcface']
#embedder_names = ['resnet34_arcface', 'resnet50_arcface']
#embedder_names = ['resnet18_arcface', 'resnet18_arcface']

#embedder_names = ['resnet100_arcface', 'resnet50_arcface', 'resnet34_arcface', 'resnet18_arcface',
#                  'resnet100_cosface', 'resnet50_cosface', 'resnet34_cosface', 'resnet18_cosface',
#                  'resnet100_magface', 'resnet50_magface', 'resnet18_magface'
#                 ]
embedder_names = ['resnet50_arcface', 'resnet34_arcface', 'resnet18_arcface',
                  'resnet100_cosface', 'resnet50_cosface', 'resnet34_cosface', 'resnet18_cosface',
                  'resnet100_magface', 'resnet50_magface', 'resnet18_magface'
                 ]

#embedder_names = ['resnet50_arcface', 'resnet34_arcface', 'resnet18_arcface',
#                  'resnet50_cosface', 'resnet34_cosface', 'resnet18_cosface',
#                 ]

def load_embedder():
    embedders = {}

    #embedder_names = ['resnet100_arcface', 'resnet50_arcface', 'resnet34_arcface', 'resnet18_arcface',
    #                  'resnet100_cosface', 'resnet50_cosface', 'resnet34_cosface', 'resnet18_cosface'
    #                 ]


    #if 1:
    for embedder_name in embedder_names:

        #embedder_name = 'resnet50_arcface'
        backbone, head = embedder_name.split('_')
        weights_path = embedders_dict[backbone]['heads'][head]['weights_path']
        #sd = torch.load(weights_path, map_location=device)
        sd = torch.load(weights_path)
        if 'magface' in embedder_name:
            embedder = MFBackbone.IResNet(MFBackbone.IBasicBlock, layers=embedders_dict[backbone]['layers']).to('cuda').eval()
            print(embedder_name)
            if embedder_name == 'resnet18_magface':
                sd = rewrite_weights_dict_mag18(sd['state_dict'])
            elif embedder_name == 'resnet50_magface':
                sd = rewrite_weights_dict_mag50(sd['state_dict'])
            else:
                sd = rewrite_weights_dict(sd['state_dict'])
        else:
            embedder = InsightFaceResnetBackbone.IResNet(InsightFaceResnetBackbone.IBasicBlock,
                                                         layers=embedders_dict[backbone]['layers']).to('cuda').eval()
        embedder.load_state_dict(sd)
        embedders[embedder_name] = embedder
    return embedders



class IDLoss(nn.Module):
    def __init__(self, use_mobile_id=False):
        super(IDLoss, self).__init__()
        #print('Loading ResNet ArcFace')
        #self.facenet = Backbone(input_size=112, num_layers=50, drop_ratio=0.6, mode='ir_se')
        #self.facenet.load_state_dict(torch.load(MODEL_PATHS['ir_se50']))

        #self.face_pool = torch.nn.AdaptiveAvgPool2d((112, 112))
        #self.facenet.eval()


        print('Loading ResNet ArcFace')
        self.embedders = load_embedder()

        self.cos_sim = CosineSimilarity()



    def extract_feats(self, x, flg=False):
        #x = x[:, :, 35:223, 32:220]  # Crop interesting region
        #x = self.face_pool(x)
        #x_feats = self.facenet(x)
        #return x_feats


        patch_embs = {}
        for embedder_name, emb_model in self.embedders.items():
            img_batch_applied = F.interpolate(x, (112, 112))
            if flg:
                #print('detach')
                patch_embs[embedder_name] = emb_model(img_batch_applied).unsqueeze(0).detach()
            else:
                #print('no detach')
                patch_embs[embedder_name] = emb_model(img_batch_applied).unsqueeze(0)
            #x_feats = emb_model(img_batch_applied).unsqueeze(0)

        #return x_feats
        return patch_embs


    def forward(self, x, x_hat):
        n_samples = x.shape[0]
        x_feats = self.extract_feats(x, flg=True)
        #x_feats = x_feats.detach()

        x_hat_feats = self.extract_feats(x_hat, flg=False)
        losses = []
        #for i in range(n_samples):
        #    #loss_sample = 1 - x_hat_feats[i].dot(x_feats[i])
        #    loss_sample = 1 - (self.cos_sim(x_hat_feats[i], x_feats[i]) + 1) / 2
        #    losses.append(loss_sample.unsqueeze(0))


        for embedder_name in embedder_names:
            x_hat = x_hat_feats[embedder_name]
            x = x_feats[embedder_name]
            for i in range(n_samples):
                loss_sample = 1 - (self.cos_sim(x_hat[i], x[i]) + 1) / 2
                losses.append(loss_sample.unsqueeze(0))

        losses = torch.cat(losses, dim=0)
        return losses
