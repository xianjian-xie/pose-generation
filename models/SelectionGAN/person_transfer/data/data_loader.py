import torch.utils.data as data
import os

import torchvision.transforms as transforms
from PIL import Image

import random
import pandas as pd
import numpy as np
import torch

def CreateDataLoader(opt):
    data_loader = CustomDatasetDataLoader()
    print(data_loader.name())
    data_loader.initialize(opt)
    return data_loader

def CreateDataset(opt):
    dataset = None

    if opt.dataset_mode == 'hmubi':
        dataset = HMUBIDataset()

    else:
        raise ValueError("Dataset [%s] not recognized." % opt.dataset_mode)

    print("dataset [%s] was created" % (dataset.name()))
    dataset.initialize(opt)
    return dataset

class BaseDataLoader():
    def __init__(self):
        pass
    
    def initialize(self, opt):
        self.opt = opt
        pass

    def load_data():
        return None

class CustomDatasetDataLoader(BaseDataLoader):
    def name(self):
        return 'CustomDatasetDataLoader'

    def initialize(self, opt):
        BaseDataLoader.initialize(self, opt)
        self.dataset = CreateDataset(opt)
        self.dataloader = torch.utils.data.DataLoader(
            self.dataset,
            batch_size=opt.batchSize,
            shuffle=not opt.serial_batches,
            # num_workers=int(opt.nThreads)
        )

    def load_data(self):
        return self

    def __len__(self):
        return min(len(self.dataset), self.opt.max_dataset_size)

    def __iter__(self):
        for i, data in enumerate(self.dataloader):
            if i >= self.opt.max_dataset_size:
                break
            yield data

class BaseDataset(data.Dataset):
    def __init__(self):
        super(BaseDataset, self).__init__()

    def name(self):
        return 'BaseDataset'

    def initialize(self, opt):
        pass

def get_transform(opt):
    transform_list = []
    if opt.resize_or_crop == 'resize_and_crop':
        osize = [opt.loadSize, opt.loadSize]
        transform_list.append(transforms.Scale(osize, Image.BICUBIC))
        transform_list.append(transforms.RandomCrop(opt.fineSize))
    elif opt.resize_or_crop == 'crop':
        transform_list.append(transforms.RandomCrop(opt.fineSize))
    elif opt.resize_or_crop == 'scale_width':
        transform_list.append(transforms.Lambda(
            lambda img: __scale_width(img, opt.fineSize)))
    elif opt.resize_or_crop == 'scale_width_and_crop':
        transform_list.append(transforms.Lambda(
            lambda img: __scale_width(img, opt.loadSize)))
        transform_list.append(transforms.RandomCrop(opt.fineSize))

    transform_list += [transforms.ToTensor(),
                       transforms.Normalize((0.5, 0.5, 0.5),
                                            (0.5, 0.5, 0.5))]
    return transforms.Compose(transform_list)

def __scale_width(img, target_width):
    ow, oh = img.size
    if (ow == target_width):
        return img
    w = target_width
    h = int(target_width * oh / ow)
    return img.resize((w, h), Image.BICUBIC)

class HMUBIDataset(BaseDataset):
    def initialize(self, opt):
        self.opt = opt
        if opt.phase == 'test':
            self.root = os.path.join(opt.dataroot, 'test')
        elif opt.phase == 'train':
            self.root = os.path.join(opt.dataroot, opt.phase)
        self.dir_P = os.path.join(self.root, 'image')
        self.dir_K = os.path.join(self.root, 'keypoints')

        self.init_categories(opt.pairLst)
        self.transform = get_transform(opt)

    def init_categories(self, pairLst):
        pairs_file_train = pd.read_csv(pairLst, sep=' ', header=None)
        self.size = len(pairs_file_train)
        self.pairs = []
        print('Loading data pairs ...')
        for i in range(self.size):
            pair = [pairs_file_train.iloc[i][0], pairs_file_train.iloc[i][1]]
            self.pairs.append(pair)

        print('Loading data pairs finished ...')

    def __getitem__(self, index):
        if self.opt.phase == 'train':
            index = random.randint(0, self.size-1)

        P1_name, P2_name = self.pairs[index]
        P1_name = P1_name.split('/')[1].split('.')[0]
        P2_name = P2_name.split('/')[1].split('.')[0]
        P1_path = os.path.join(self.dir_P, P1_name + '.jpg') # person 1
        BP1_path = os.path.join(self.dir_K, P1_name + '.txt') # bone of person 1

        # person 2 and its bone
        P2_path = os.path.join(self.dir_P, P2_name + '.jpg') # person 2
        BP2_path = os.path.join(self.dir_K, P2_name + '.txt') # bone of person 2


        P1_img = Image.open(P1_path).convert('RGB')
        if self.opt.phase == 'test':
            P2_img = Image.open(P1_path).convert('RGB')
        else:
            P2_img = Image.open(P2_path).convert('RGB')

        BP1 = np.genfromtxt(BP1_path) # h, w, c
        BP2 = np.genfromtxt(BP2_path) 

        BP1_img = np.zeros((18, P1_img.size[0], P1_img.size[1]))
        for i, bp in enumerate(BP1):
            BP1_img[i, np.round(bp[1]).astype(int), np.round(bp[0]).astype(int)] = 1
        BP2_img = np.zeros((18, P1_img.size[0], P1_img.size[1]))
        for i, bp in enumerate(BP2):
            BP2_img[i, np.round(bp[1]).astype(int), np.round(bp[0]).astype(int)] = 1
        # use flip
        if self.opt.phase == 'train' and self.opt.use_flip:
            # print ('use_flip ...')
            flip_random = random.uniform(0,1)
            
            if flip_random > 0.5:
                # print('fliped ...')
                P1_img = P1_img.transpose(Image.FLIP_LEFT_RIGHT)
                P2_img = P2_img.transpose(Image.FLIP_LEFT_RIGHT)

                BP1_img = np.array(BP1_img[:, ::-1, :]) # flip
                BP2_img = np.array(BP2_img[:, ::-1, :]) # flip

            BP1 = torch.from_numpy(BP1_img).float() #h, w, c
            BP1 = BP1.transpose(2, 0) #c,w,h
            BP1 = BP1.transpose(2, 1) #c,h,w 

            BP2 = torch.from_numpy(BP2_img).float()
            BP2 = BP2.transpose(2, 0) #c,w,h
            BP2 = BP2.transpose(2, 1) #c,h,w 

            P1 = self.transform(P1_img)
            P2 = self.transform(P2_img)

        else:
            BP1 = torch.from_numpy(BP1_img).float() #h, w, c
            BP2 = torch.from_numpy(BP2_img).float()

            P1 = self.transform(P1_img)
            P2 = self.transform(P2_img)

        return {'P1': P1, 'BP1': BP1, 'P2': P2, 'BP2': BP2,
                'P1_path': P1_name, 'P2_path': P2_name}
                

    def __len__(self):
        if self.opt.phase == 'train':
            return 4000
        elif self.opt.phase == 'test':
            return self.size

    def name(self):
        return 'HMUBIDataset'