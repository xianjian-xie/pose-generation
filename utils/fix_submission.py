# from options.test_options import TestOptions
import data as Dataset
import numpy as np
import torch
import sys

from silnet_model import SilNet
import os
from tqdm.autonotebook import tqdm
import argparse
from PIL import Image

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

if __name__=='__main__':
    # get testing options
    opt = parseOpts().parse_args()
    opt.isTrain=False
    opt.phase='test'
    # creat a dataset
    dataset = Dataset.create_dataloader(opt)

    dataset_size = len(dataset) * opt.batchSize
    print('testing images = %d' % dataset_size)
    # create a model
    silnet = SilNet()
    silnet.cuda()
    silnet.eval()
    if os.path.exists('C:\\Users\\Carl\\Desktop\\5561_Project\\SilNet.pt'):
        print('Loading SilNet state dict...')
        silnet.load_state_dict(torch.load('C:\\Users\\Carl\\Desktop\\5561_Project\\SilNet.pt'))
    # model = create_model(opt)

    gt_root = 'gt'
    for data in tqdm(dataset):
        masks = silnet.forward(data).argmax(dim=1).float()
        for i in range(len(data['P1_path'])):
            mask_image = Image.fromarray(masks[i].cpu().numpy().astype(bool))
            gt_image_file = data['P1_path'][i]+'_'+data['P2_path'][i]+'.jpg'
            gt_image_path = os.path.join(gt_root, gt_image_file)
            gt_image = Image.open(gt_image_path)
            gt_image_masked = Image.fromarray(np.multiply(np.expand_dims(np.array(mask_image), 2), np.array(gt_image)))
            gt_image_masked.save(gt_image_path)


