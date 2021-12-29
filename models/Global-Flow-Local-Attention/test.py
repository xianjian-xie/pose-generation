from options.test_options import TestOptions
import data as Dataset
from model import create_model
from util import visualizer
from itertools import islice
import numpy as np
import torch

from silnet_model import SilNet
import os
from tqdm.autonotebook import tqdm

if __name__=='__main__':
    # get testing options
    opt = TestOptions().parse()
    # creat a dataset
    dataset = Dataset.create_dataloader(opt)

    dataset_size = len(dataset) * opt.batchSize
    print('testing images = %d' % dataset_size)
    # create a model
    silnet = SilNet()
    silnet.cuda()
    silnet.eval()
    if os.path.exists(opt.silnet_dataroot):
        print('Loading SilNet state dict...')
        silnet.load_state_dict(torch.load(opt.silnet_dataroot))
    model = create_model(opt)

    with torch.no_grad():
        for data in tqdm(dataset):
            data['M2'] = silnet.forward(data).argmax(dim=1).float()
            data['P1'] = torch.FloatTensor(np.multiply(np.array(data['P1'].cpu()), np.expand_dims(np.array(data['M1'].cpu()), 1)))
            data['P2'] = torch.FloatTensor(np.multiply(np.array(data['P2'].cpu()), np.expand_dims(np.array(data['M2'].cpu()), 1)))
            
            model.set_input(data)
            model.test()