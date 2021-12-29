import os
import torch.utils.data as data
from PIL import Image
import torchvision.transforms as transforms
import numpy as np
import random
import torchvision.transforms.functional as F
# from util import pose_utils
from PIL import Image
import pandas as pd
import torch
import math
import numbers

import cv2

class BaseDataset(data.Dataset):
    def __init__(self):
        super(BaseDataset, self).__init__()

    @staticmethod
    def modify_commandline_options(parser, is_train):
        parser.add_argument('--angle', type=float, default=False)
        parser.add_argument('--shift', type=float, default=False)
        parser.add_argument('--scale', type=float, default=False)
        return parser

    def initialize(self, opt):
        self.opt = opt
        self.image_dir, self.bone_file, self.mask_file, self.name_pairs = self.get_paths(opt)
        size = len(self.name_pairs)
        self.dataset_size = size

        if isinstance(opt.load_size, int):
            self.load_size = (opt.load_size, opt.load_size)
        else:
            self.load_size = opt.load_size


        transform_list=[]
        # transform_list.append(transforms.Resize(size=self.load_size))
        transform_list.append(transforms.ToTensor())
        transform_list.append(transforms.Normalize((0.5, 0.5, 0.5),(0.5, 0.5, 0.5)))
        self.trans = transforms.Compose(transform_list) 

    def get_paths(self, opt):
        label_paths = []
        image_paths = []
        instance_paths = []
        assert False, "A subclass of MarkovAttnDataset must override self.get_paths(self, opt)"
        return label_paths, image_paths, instance_paths

    def __getitem__(self, index):
        P1_name, P2_name = self.name_pairs[index]
        P1_name = P1_name.split('/')[1].split('.')[0]
        P2_name = P2_name.split('/')[1].split('.')[0]
        P1_path = os.path.join(self.image_dir, P1_name + '.jpg') # person 1
        BP1_path = os.path.join(self.bone_file, P1_name + '.txt') # bone of person 1
        M1_path = os.path.join(self.mask_file, P1_name + '.png')

        if self.opt.phase == 'test':
            P2_path = os.path.join(self.image_dir, P1_name + '.jpg') # person 2
            M2_path = os.path.join(self.mask_file, P1_name + '.png')
        else:
            P2_path = os.path.join(self.image_dir, P2_name + '.jpg') # person 2
            M2_path = os.path.join(self.mask_file, P2_name + '.png')
        BP2_path = os.path.join(self.bone_file, P2_name + '.txt') # bone of person 2

        P1_img = Image.open(P1_path).convert('RGB')
        M1_img = Image.open(M1_path).convert('1')

        P2_img = Image.open(P2_path).convert('RGB')
        M2_img = Image.open(M2_path).convert('1')
        
        M1 = torch.tensor(np.where(np.array(M1_img), 1, 0))
        M2 = torch.tensor(np.where(np.array(M2_img), 1, 0))

        if self.opt.phase == 'test':
            P1 = np.array(P1_img)
            P2 = np.array(P2_img)
        else:
            P1 = self.trans(P1_img)
            P2 = self.trans(P2_img)

        BP1 = np.genfromtxt(BP1_path)
        BP2 = np.genfromtxt(BP2_path) 

        BP1_img = np.zeros((18, P1_img.size[0], P1_img.size[1]))
        for i, bp in enumerate(BP1):
            BP1_img[i, np.round(bp[1]).astype(int), np.round(bp[0]).astype(int)] = 1
        BP2_img = np.zeros((18, P1_img.size[0], P1_img.size[1]))
        for i, bp in enumerate(BP2):
            BP2_img[i, np.round(bp[1]).astype(int), np.round(bp[0]).astype(int)] = 1

        BP1 = torch.from_numpy(BP1_img).float()
        BP2 = torch.from_numpy(BP2_img).float()

        return {'P1': P1, 'BP1': BP1, 'P2': P2, 'BP2': BP2, 'M1': M1, 'M2': M2,
                'P1_path': P1_name, 'P2_path': P2_name}

   

    def __len__(self):
        return self.dataset_size

    def name(self):
        assert False, "A subclass of BaseDataset must override self.name"

    def getRandomAffineParam(self):
        if self.opt.angle is not False:
            angle = np.random.uniform(low=self.opt.angle[0], high=self.opt.angle[1])
        else:
            angle = 0
        if self.opt.scale is not False:
            scale   = np.random.uniform(low=self.opt.scale[0], high=self.opt.scale[1])
        else:
            scale=1
        if self.opt.shift is not False:
            shift_x = np.random.uniform(low=self.opt.shift[0], high=self.opt.shift[1])
            shift_y = np.random.uniform(low=self.opt.shift[0], high=self.opt.shift[1])
        else:
            shift_x=0
            shift_y=0
        return angle, (shift_x,shift_y), scale

    def get_inverse_affine_matrix(self, center, angle, translate, scale, shear):
        # code from https://pytorch.org/docs/stable/_modules/torchvision/transforms/functional.html#affine
        # Helper method to compute inverse matrix for affine transformation

        # As it is explained in PIL.Image.rotate
        # We need compute INVERSE of affine transformation matrix: M = T * C * RSS * C^-1
        # where T is translation matrix: [1, 0, tx | 0, 1, ty | 0, 0, 1]
        #       C is translation matrix to keep center: [1, 0, cx | 0, 1, cy | 0, 0, 1]
        #       RSS is rotation with scale and shear matrix
        #       RSS(a, scale, shear) = [ cos(a + shear_y)*scale    -sin(a + shear_x)*scale     0]
        #                              [ sin(a + shear_y)*scale    cos(a + shear_x)*scale     0]
        #                              [     0                  0          1]
        # Thus, the inverse is M^-1 = C * RSS^-1 * C^-1 * T^-1


        angle = math.radians(angle)
        if isinstance(shear, (tuple, list)) and len(shear) == 2:
            shear = [math.radians(s) for s in shear]
        elif isinstance(shear, numbers.Number):
            shear = math.radians(shear)
            shear = [shear, 0]
        else:
            raise ValueError(
                "Shear should be a single value or a tuple/list containing " +
                "two values. Got {}".format(shear))
        scale = 1.0 / scale

        # Inverted rotation matrix with scale and shear
        d = math.cos(angle + shear[0]) * math.cos(angle + shear[1]) + \
            math.sin(angle + shear[0]) * math.sin(angle + shear[1])
        matrix = [
            math.cos(angle + shear[0]), math.sin(angle + shear[0]), 0,
            -math.sin(angle + shear[1]), math.cos(angle + shear[1]), 0
        ]
        matrix = [scale / d * m for m in matrix]

        # Apply inverse of translation and of center translation: RSS^-1 * C^-1 * T^-1
        matrix[2] += matrix[0] * (-center[0] - translate[0]) + matrix[1] * (-center[1] - translate[1])
        matrix[5] += matrix[3] * (-center[0] - translate[0]) + matrix[4] * (-center[1] - translate[1])

        # Apply center translation: C * RSS^-1 * C^-1 * T^-1
        matrix[2] += center[0]
        matrix[5] += center[1]
        return matrix

    def get_affine_matrix(self, center, angle, translate, scale, shear):
        matrix_inv = self.get_inverse_affine_matrix(center, angle, translate, scale, shear)

        matrix_inv = np.matrix(matrix_inv).reshape(2,3)
        pad = np.matrix([0,0,1])
        matrix_inv = np.concatenate((matrix_inv, pad), 0)
        matrix = np.linalg.inv(matrix_inv)
        return matrix