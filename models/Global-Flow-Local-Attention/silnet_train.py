import torch
from torch import nn
import torch.nn.functional as F
import numpy as np

import time
import sys
import os
import data as Dataset

import argparse
import logging
from tqdm.autonotebook import tqdm

from silnet_model import SilNet

def parseOpts():
    parser = argparse.ArgumentParser()
    parser.add_argument('--name', type=str, default='experiment_name', help='name of the experiment.')
    parser.add_argument('--model', type=str, default='rec', help='name of the model type.')
    parser.add_argument('--checkpoints_dir', type=str, default='./result', help='models are save here')
    parser.add_argument('--which_iter', type=str, default='latest', help='which iterations to load')
    parser.add_argument('--gpu_ids', type=str, default='0', help='gpu ids: e.g. 0, 1, 2 use -1 for CPU')
    parser.add_argument('--phase', type=str, default='train', help='train, val, test, etc')
    parser.add_argument('--continue_train', action='store_true', help='continue training: load the latest model')


    # input/output sizes
    parser.add_argument('--batchSize', type=int, help='input batch size', default=8)
    parser.add_argument('--old_size', type=int, default=(256,256), help='Scale images to this size. The final image will be cropped to --crop_size.')
    parser.add_argument('--load_size', type=int, default=1024, help='Scale images to this size. The final image will be cropped to --crop_size.')
    parser.add_argument('--structure_nc', type=int, default=18 )
    parser.add_argument('--image_nc', type=int, default=3 )

    # for setting inputs
    parser.add_argument('--dataroot', type=str, default='datasets/')
    parser.add_argument('--dataset_mode', type=str, default='hmubi')
    parser.add_argument('--fid_gt_path', type=str)
    parser.add_argument('--serial_batches', action='store_true', help='if true, takes images in order to make batches, otherwise takes them randomly')
    parser.add_argument('--nThreads', default=8, type=int, help='# threads for loading data')
    parser.add_argument('--max_dataset_size', type=int, default=sys.maxsize, help='Maximum number of samples allowed per dataset. If the dataset directory contains more than max_dataset_size, only a subset is loaded.')

    # display parameter define
    parser.add_argument('--display_winsize', type=int, default=256, help='display window size')
    parser.add_argument('--display_id', type=int, default=1, help='display id of the web')
    parser.add_argument('--display_port', type=int, default=8096, help='visidom port of the web display')
    parser.add_argument('--display_single_pane_ncols', type=int, default=0, help='if positive, display all images in a single visidom web panel')
    parser.add_argument('--display_env', type=str, default=parser.parse_known_args()[0].name.replace('_',''), help='the environment of visidom display')

    parser.add_argument('--iter_count', type=int, default=1, help='the starting epoch count')
    parser.add_argument('--niter', type=int, default=5000000, help='# of iter with initial learning rate')
    parser.add_argument('--niter_decay', type=int, default=0, help='# of iter to decay learning rate to zero')

    # learning rate and loss weight
    parser.add_argument('--lr_policy', type=str, default='lambda', help='learning rate policy[lambda|step|plateau]')
    parser.add_argument('--lr', type=float, default=1e-4, help='initial learning rate for adam')
    parser.add_argument('--gan_mode', type=str, default='lsgan', choices=['wgan-gp', 'hinge', 'lsgan'])

    # display the results
    parser.add_argument('--display_freq', type=int, default=1000, help='frequency of showing training results on screen')
    parser.add_argument('--eval_iters_freq', type=int, default=15000, help='frequency of showing training results on screen')
    parser.add_argument('--print_freq', type=int, default=1000, help='frequency of showing training results on console')
    parser.add_argument('--save_latest_freq', type=int, default=1000, help='frequency of saving the latest results')
    parser.add_argument('--save_iters_freq', type=int, default=10000, help='frequency of saving checkpoints at the end of epochs')
    parser.add_argument('--no_html', action='store_true', help='do not save intermediate training results')

    return parser

if __name__ == '__main__':
    # logging.basicConfig(
    #     level=logging.info,
    #     format='%(asctime)s - %(levelname)s - %(message)s'
    # )
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # get training options
    opt = parseOpts().parse_args()
    opt.isTrain=True

    # create a dataset
    logging.info("Loading dataset...")
    before_load_data = time.time()
    dataset = None
    if not os.path.exists('dataset.pth') and not dataset:
        dataset = Dataset.create_dataloader(opt)
        torch.save(dataset, 'dataset.pth')
    else:
        dataset = torch.load('dataset.pth')

    after_load_data = time.time()
    logging.info(f"Dataset loaded. Time = {after_load_data - before_load_data}s")
    
    
    # dataset = {'P1': P1, 'BP1': BP1, 'P2': P2, 'BP2': BP2, 'M1': M1_img, 'M2': M2_img,
    #                           'P1_path': P1_name, 'P2_path': P2_name}
    dataset_size = len(dataset) * opt.batchSize
    logging.info('training images = %d' % dataset_size)
    keep_training = True
    epoch = 0

    # training process
    model = SilNet()
    model.cuda()

    if os.path.exists('SilNet.pt'):
        print('Loading state dict...')
        model.load_state_dict(torch.load('SilNet.pt'))

    optimizer = torch.optim.Adam(model.parameters(), lr=opt.lr, betas=(0.0, 0.999))
    criterion = nn.CrossEntropyLoss()

    total_iteration=0
    model.train()
    while(keep_training):
        epoch_start_time = time.time()
        epoch+=1
        logging.info('\n Training epoch: %d' % epoch)

        pbar = tqdm(enumerate(dataset))
        for i, data in pbar:
            label = data['M2'].to(device)
            
            iter_start_time = time.time()
            total_iteration += 1
            
            y_preds = model.forward(data)
            loss = criterion(y_preds, label)
            pbar.set_description(f'Iteration {i+1}/{len(dataset)}; Loss: {loss.item()}')
            loss.backward()
            
            optimizer.step()
            optimizer.zero_grad()

            if i%opt.display_freq==0:
                import matplotlib
                import matplotlib.pyplot as plt
                matplotlib.use('TkAgg')
                img_pred = y_preds[0].argmax(dim=0).cpu().numpy()
                img_label = label[0].cpu().numpy()
                img_display = np.hstack((img_pred, img_label))
                plt.title(f'Iteration {i+1} mask prediction')
                plt.imshow(img_display, cmap='gray')
                plt.show()

        torch.save(model.state_dict(), 'SilNet.pt')

        epoch_end_time = time.time()
        if epoch > 0:
            logging.info(f'Epoch {epoch} runtime: {epoch_end_time-epoch_start_time}s')