# ---------------------------------------------------------------
# Copyright (c) 2022, NVIDIA CORPORATION. All rights reserved.
#
# This work is licensed under the NVIDIA Source Code License
# for DiffPure. To view a copy of this license, see the LICENSE file.
# ---------------------------------------------------------------

import argparse
import logging
import yaml
import os
import time

import random
import numpy as np
from PIL import Image

from torchvision.datasets import ImageFolder

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.utils as tvu
from torchvision import transforms, datasets
from bpda_eot.bpda_eot_attack import BPDA_EOT_Attack

import utils
from utils import str2bool, get_accuracy, get_image_classifier, load_data

from runners.diffpure_ddpm import Diffusion
from runners.diffpure_guided import GuidedDiffusion
from runners.diffpure_sde import RevGuidedDiffusion


class ResNet_Adv_Model(nn.Module):
    def __init__(self, args, config):
        super().__init__()
        # image classifier
        self.resnet = get_image_classifier(args.classifier_name).to(config.device)

    def purify(self, x):
        return x

    def forward(self, x, mode='purify_and_classify'):
        if mode == 'purify':
            out = self.purify(x)
        elif mode == 'classify':
            out = self.resnet(x)  # x in [0, 1]
        elif mode == 'purify_and_classify':
            x = self.purify(x)
            out = self.resnet(x)  # x in [0, 1]
        else:
            raise NotImplementedError(f'unknown mode: {mode}')
        return out


class SDE_Adv_Model(nn.Module):
    def __init__(self, args, config):
        super().__init__()
        self.args = args

        # image classifier
        self.resnet = get_image_classifier(args.classifier_name).to(config.device)

        # diffusion model
        print(f'diffusion_type: {args.diffusion_type}')
        if args.diffusion_type == 'ddpm':
            self.runner = GuidedDiffusion(args, config, device=config.device)
        elif args.diffusion_type == 'sde':
            self.runner = RevGuidedDiffusion(args, config, device=config.device)
        elif args.diffusion_type == 'celebahq-ddpm':
            self.runner = Diffusion(args, config, device=config.device)
        else:
            raise NotImplementedError('unknown diffusion type')

        self.register_buffer('counter', torch.zeros(1, device=config.device))
        self.tag = None

    # use `counter` to record the the sampling time every 5 NFEs (note we hardcoded print freq to 5,
    # and you may want to change the freq)
    def reset_counter(self):
        self.counter = torch.zeros(1, dtype=torch.int, device=config.device)

    def set_tag(self, tag=None):
        self.tag = tag

    def purify(self, x):
        counter = self.counter.item()
        if counter % 5 == 0:
            print(f'diffusion times: {counter}')

        # imagenet [3, 224, 224] -> [3, 256, 256] -> [3, 224, 224]
        if 'imagenet' in self.args.domain:
            x = F.interpolate(x, size=(256, 256), mode='bilinear', align_corners=False)

        start_time = time.time()
        x_re = self.runner.image_editing_sample((x - 0.5) * 2, bs_id=counter, tag=self.tag)
        minutes, seconds = divmod(time.time() - start_time, 60)

        if 'imagenet' in self.args.domain:
            x_re = F.interpolate(x_re, size=(224, 224), mode='bilinear', align_corners=False)

        if counter % 5 == 0:
            print(f'x shape (before diffusion models): {x.shape}')
            print(f'x shape (before resnet): {x_re.shape}')
            print("Sampling time per batch: {:0>2}:{:05.2f}".format(int(minutes), seconds))

        self.counter += 1

        return (x_re + 1) * 0.5

    def forward(self, x, mode='purify_and_classify'):
        if mode == 'purify':
            out = self.purify(x)
        elif mode == 'classify':
            out = self.resnet(x)  # x in [0, 1]
        elif mode == 'purify_and_classify':
            x = self.purify(x)
            out = self.resnet(x)  # x in [0, 1]
        else:
            raise NotImplementedError(f'unknown mode: {mode}')
        return out


def parse_args_and_config():
    parser = argparse.ArgumentParser(description=globals()['__doc__'])
    # diffusion models
    parser.add_argument('--config', type=str, default='celeba.yml', help='Path to the config file')
    parser.add_argument('--data_seed', type=int, default=0, help='Random seed')
    parser.add_argument('--seed', type=int, default=1234, help='Random seed')
    parser.add_argument('--exp', type=str, default='exp_rst', help='Path for saving running related data.')
    parser.add_argument('--verbose', type=str, default='info', help='Verbose level: info | debug | warning | critical')
    parser.add_argument('-i', '--image_folder', type=str, default='celebahq-adv-2x4-disc-bpda-rev', help="The folder name of samples")
    parser.add_argument('--ni', action='store_true', help="No interaction. Suitable for Slurm Job launcher")
    parser.add_argument('--sample_step', type=int, default=1, help='Total sampling steps')
    parser.add_argument('--t', type=int, default=500, help='Sampling noise scale')
    parser.add_argument('--t_delta', type=int, default=15, help='Perturbation range of sampling noise scale')
    parser.add_argument('--rand_t', type=str2bool, default=False, help='Decide if randomize sampling noise scale')
    parser.add_argument('--diffusion_type', type=str, default='celebahq-ddpm', help='[ddpm, sde, celebahq-ddpm]')
    parser.add_argument('--score_type', type=str, default='guided_diffusion', help='[guided_diffusion, score_sde]')
    parser.add_argument('--eot_iter', type=int, default=20, help='only for rand version of autoattack')
    parser.add_argument('--use_bm', action='store_true', help='whether to use brownian motion')

    parser.add_argument('--eot_defense_reps', type=int, default=20)
    parser.add_argument('--eot_attack_reps', type=int, default=1)

    # adv
    parser.add_argument('--domain', type=str, default='celebahq', help='which domain: celebahq, cat, car, imagenet')
    parser.add_argument('--classifier_name', type=str, default='celebahq__Eyeglasses', help='which classifier to use')
    parser.add_argument('--partition', type=str, default='val')
    parser.add_argument('--adv_batch_size', type=int, default=1)
    parser.add_argument('--num_sub', type=int, default=1000, help='imagenet subset')
    parser.add_argument('--adv_eps', type=float, default=0.062745098)

    parser.add_argument(
        '--input', type=str,
        default='/face/Mask/AdversarialMask/patch/experiments/April/13-04-2023_13-48-33/saved_patches/patch_499.png'  #mask9
    )
    parser.add_argument('--input_label_id', type=int, default=0)
    parser.add_argument(
        '--output_dir', type=str,
        default='./outputs')
    args = parser.parse_args()

    # parse config file
    with open(os.path.join('configs', args.config), 'r') as f:
        config = yaml.safe_load(f)
    new_config = utils.dict2namespace(config)

    level = getattr(logging, args.verbose.upper(), None)
    if not isinstance(level, int):
        raise ValueError('level {} not supported'.format(args.verbose))

    handler1 = logging.StreamHandler()
    formatter = logging.Formatter('%(levelname)s - %(filename)s - %(asctime)s - %(message)s')
    handler1.setFormatter(formatter)
    logger = logging.getLogger()
    logger.addHandler(handler1)
    logger.setLevel(level)

    args.image_folder = os.path.join(args.exp, args.image_folder)
    os.makedirs(args.image_folder, exist_ok=True)
    args.log_dir = args.output_dir
    os.makedirs(args.log_dir, exist_ok=True)

    # add device
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    logging.info("Using device: {}".format(device))
    new_config.device = device

    # set random seed
    torch.manual_seed(args.seed)
    random.seed(args.seed)
    np.random.seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    torch.backends.cudnn.benchmark = True

    return args, new_config


if __name__ == '__main__':
    args, config = parse_args_and_config()

    # load model
    print('starting the model and loader...')
    model = SDE_Adv_Model(args, config)
    model = model.eval().to(config.device)

    # load data
    transforms = transforms.Compose([
        transforms.Resize([256, 256]),
        transforms.ToTensor(),
    ])
    image = Image.open(args.input).convert('RGB')
    x_val = transforms(image)
    x_val = x_val.unsqueeze(dim=0)
    y_val = torch.ones([1]) * args.input_label_id
    # x_val, y_val = load_data(args, args.adv_batch_size)
    x_val, y_val = x_val.to(config.device), y_val.to(config.device)

    # tvu.save_image((x_val + 1) * 0.5, args.output_dir + f'/before_defended.png')
    tvu.save_image(x_val, args.output_dir + f'/before_defended.png')
    X_purified = model(x_val, mode='purify').detach().clone()
    tvu.save_image(X_purified, args.output_dir + f'/defended.png')
    # X_purified.requires_grad_()

    with torch.no_grad():
        logits = model(X_purified.data, mode='classify')
    rep = args.eot_attack_reps
    logits_pred = logits.view([rep, int(logits.shape[0] / rep), logits.shape[1]]).mean(0)
    _, y_pred = torch.max(logits_pred, 1)
    defended = torch.eq(y_pred, y_val).detach()
    print(f'defend result: {defended}')

