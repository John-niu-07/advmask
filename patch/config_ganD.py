from torch import optim
import os
import datetime
import time


embedders_dict = {
    'resnet18': {
        'layers': [2, 2, 2, 2],
        'heads': {
            'arcface': {
                'weights_path': os.path.join('..', 'face_recognition', 'insightface_torch', 'weights', 'ms1mv3_arcface_resnet18.pth')
            },
            'cosface': {
                'weights_path': os.path.join('..', 'face_recognition', 'insightface_torch', 'weights', 'glint360k_cosface_resnet18.pth')
            }
        }
    },
    'resnet34': {
        'layers': [3, 4, 6, 3],
        'heads': {
            'arcface': {
                'weights_path': os.path.join('..', 'face_recognition', 'insightface_torch', 'weights',
                                             'ms1mv3_arcface_resnet34.pth')
            },
            'cosface': {
                'weights_path': os.path.join('..', 'face_recognition', 'insightface_torch', 'weights',
                                             'glint360k_cosface_resnet34.pth')
            }
        }
    },
    'resnet50': {
        'layers': [3, 4, 14, 3],
        'heads': {
            'arcface': {
                'weights_path': os.path.join('..', 'face_recognition', 'insightface_torch', 'weights',
                                             'ms1mv3_arcface_resnet50.pth')
            },
            'cosface': {
                'weights_path': os.path.join('..', 'face_recognition', 'insightface_torch', 'weights',
                                             'glint360k_cosface_resnet50.pth')
            }
        }
    },
    'resnet100': {
        'layers': [3, 13, 30, 3],
        'heads': {
            'arcface': {
                'weights_path': os.path.join('..', 'face_recognition', 'insightface_torch', 'weights',
                                             'ms1mv3_arcface_resnet100.pth')
            },
            'cosface': {
                'weights_path': os.path.join('..', 'face_recognition', 'insightface_torch', 'weights',
                                             'glint360k_cosface_resnet100.pth')
            },
            'magface': {
                'weights_path': os.path.join('..', 'face_recognition', 'magface_torch', 'weights',
                                             'magface_epoch_00025.pth')
            }
        }
    }
}


class BaseConfiguration:
    def __init__(self):
        self.seed = 42
        self.patch_name = 'base'

        # Train dataset options
        self.is_real_person = False
        self.train_dataset_name = 'CASIA'
        self.train_img_dir = os.path.join('..', 'datasets', self.train_dataset_name)
        #self.train_number_of_people = 100
        self.train_number_of_people = 2
        self.celeb_lab = os.listdir(self.train_img_dir)[:self.train_number_of_people]
        self.celeb_lab_mapper = {i: lab for i, lab in enumerate(self.celeb_lab)}
        #self.num_of_train_images = 5
        #self.num_of_train_images = 10
        self.num_of_train_images = 1

        self.shuffle = True
        self.img_size = (112, 112)
        self.train_batch_size = 1
        self.test_batch_size = 32
        self.magnification_ratio = 35

        # Attack options
        self.mask_aug = True
        self.patch_size = (112, 112)  # height, width
        self.initial_patch = 'white'  # body, white, random, stripes, l_stripes
        #self.initial_patch = 'grey'  # body, white, random, stripes, l_stripes
        #self.initial_patch = 'init'  # body, white, random, stripes, l_stripes
        #self.epochs = 6450
        #self.epochs = 1450
        self.epochs = 86800 
        self.epochs = 6800 
        self.epochs = 2000 
        #self.epochs = 8800 
        #self.epochs = 20
        #self.start_learning_rate = 1e-2
        #self.start_learning_rate = 0.01

        #self.start_learning_rate = 0.4
        self.start_learning_rate = 0.01

        #self.start_learning_rate = 1e-1
        #self.es_patience = 7
        self.es_patience = 30
        #self.sc_patience = 2
        self.sc_patience = 20
        self.sc_min_lr = 1e-6

        '''
        self.scheduler_factory = lambda optimizer: optim.lr_scheduler.ReduceLROnPlateau(optimizer,
                                                                                        patience=self.sc_patience,
                                                                                        min_lr=self.sc_min_lr,
                                                                                        mode='min')
        '''
        #self.scheduler_factory = lambda optimizer: optim.lr_scheduler.MultiStepLR(optimizer, milestones=[9800, 9900], gamma=0.1)
        #self.scheduler_factory = lambda optimizer: optim.lr_scheduler.MultiStepLR(optimizer, milestones=[19800, 19900], gamma=0.1)
        self.scheduler_factory = lambda optimizer: optim.lr_scheduler.MultiStepLR(optimizer, milestones=[2000, 16800], gamma=4)
        self.scheduler_factory = lambda optimizer: optim.lr_scheduler.MultiStepLR(optimizer, milestones=[2000, 86800], gamma=3)
        self.scheduler_factory = lambda optimizer: optim.lr_scheduler.MultiStepLR(optimizer, milestones=[2000, 86800], gamma=2.2)
        self.scheduler_factory = lambda optimizer: optim.lr_scheduler.MultiStepLR(optimizer, milestones=[2000, 86800], gamma=2.3)
        self.scheduler_factory = lambda optimizer: optim.lr_scheduler.MultiStepLR(optimizer, milestones=[800, 86800], gamma=2.5)
        self.scheduler_factory = lambda optimizer: optim.lr_scheduler.MultiStepLR(optimizer, milestones=[800, 86800], gamma=1.5)
        self.scheduler_factory = lambda optimizer: optim.lr_scheduler.MultiStepLR(optimizer, milestones=[800, 86800], gamma=2.2)
        self.scheduler_factory = lambda optimizer: optim.lr_scheduler.MultiStepLR(optimizer, milestones=[800, 86800], gamma=2.5)
        #self.scheduler_factory = lambda optimizer: optim.lr_scheduler.MultiStepLR(optimizer, milestones=[800, 86800], gamma=2.2)

        # Landmark detection options
        self.landmark_detector_type = 'mobilefacenet'  # face_alignment, mobilefacenet

        # Embedder options
        #self.train_embedder_names = ['resnet100_magface']
        #self.train_embedder_names = ['resnet18_arcface']
        self.train_embedder_names = ['resnet50_arcface']

        #self.test_embedder_names = ['resnet100_magface']
        #self.test_embedder_names = ['resnet18_arcface']
        self.test_embedder_names = ['resnet50_arcface']

        # Loss options
        self.dist_loss_type = 'cossim'
        self.dist_weight = 1
        self.tv_weight = 0.1
        #self.tv_weight = 0.5

        # Test options
        self.masks_path = os.path.join('..', 'data', 'masks')
        self.random_mask_path = os.path.join(self.masks_path, 'random.png')
        self.blue_mask_path = os.path.join(self.masks_path, 'blue.png')
        self.full_mask_path = os.path.join(self.masks_path, 'full.png')

        self.black_mask_path = os.path.join(self.masks_path, 'black.png')
        self.white_mask_path = os.path.join(self.masks_path, 'white.png')
        self.face1_mask_path = os.path.join(self.masks_path, 'face1.png')
        self.face3_mask_path = os.path.join(self.masks_path, 'face3.png')

        self.update_current_dir()

    def set_attribute(self, name, value):
        setattr(self, name, value)

    def update_current_dir(self):
        my_date = datetime.datetime.now()
        month_name = my_date.strftime("%B")
        self.current_dir = os.path.join("experiments", month_name, time.strftime("%d-%m-%Y") + '_' + time.strftime("%H-%M-%S"))
        if 'SLURM_JOBID' in os.environ.keys():
            self.current_dir += '_' + os.environ['SLURM_JOBID']


class UniversalAttack(BaseConfiguration):
    def __init__(self):
        super(UniversalAttack, self).__init__()
        self.patch_name = 'universal'
        #self.num_of_train_images = 10
        self.num_of_train_images = 1
        #self.num_of_train_images = 20
        self.train_batch_size = 1
        self.test_batch_size = 32
        #self.test_batch_size = 3

        # Test dataset options
        self.test_num_of_images_for_emb = 5
        self.test_dataset_names = ['CASIA']
        self.test_img_dir = {name: os.path.join('..', 'datasets', name) for name in self.test_dataset_names}
        self.test_number_of_people = 200
        self.test_celeb_lab = {}
        for dataset_name, img_dir in self.test_img_dir.items():
            label_list = os.listdir(img_dir)[:self.test_number_of_people]
            if dataset_name == self.train_dataset_name:
                label_list = os.listdir(img_dir)[-self.test_number_of_people:]
            self.test_celeb_lab[dataset_name] = label_list
        self.test_celeb_lab_mapper = {dataset_name: {i: lab for i, lab in enumerate(self.test_celeb_lab[dataset_name])}
                                      for dataset_name in self.test_dataset_names}


class TargetedAttack(BaseConfiguration):
    def __init__(self):
        super(TargetedAttack, self).__init__()
        self.patch_name = 'targeted'
        self.num_of_train_images = 50
        self.train_batch_size = 1
        self.test_batch_size = 4
        self.test_img_dir = {self.train_dataset_name: self.train_img_dir}

    def update_test_celeb_lab(self):
        self.test_celeb_lab = {self.train_dataset_name: self.celeb_lab}
        self.test_celeb_lab_mapper = {self.train_dataset_name: self.celeb_lab_mapper}


patch_config_types = {
    "base": BaseConfiguration,
    "universal": UniversalAttack,
    "targeted": TargetedAttack,
}
