# -*- coding: utf-8 -*-
# @Date    : 2019-07-25
# @Author  : Xinyu Gong (xy_gong@tamu.edu)
# @Link    : None
# @Version : 0.0

import argparse

def str2bool(v):
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--max_epoch',
        type=int,
        default=200,
        help='number of epochs of training')
    parser.add_argument(
        '--num_iter',
        type=int,
        default=None,
        help='set the max iteration number')
    parser.add_argument(
        '-train_bs',
        '--train_batch_size',
        type=int,
        default=64,
        help='size of the batches')
    parser.add_argument(
        '--g_lr',
        type=float,
        default=0.0001,
        help='adam: generator learning rate')
    parser.add_argument(
        '--d_lr',
        type=float,
        default=0.0001,
        help='adam: discriminator learning rate')
    parser.add_argument(
        '--ae_recon_lr',
        type=float,
        default=0.0001,
        help='adam: autoencoder (reconstruction loss) learning rate')
    parser.add_argument(
        '--ae_reg_lr',
        type=float,
        default=0.0001,
        help='adam: autoencoder (regularization loss) learning rate')                              
    parser.add_argument(
        '--lambda_d',
        type=float,
        default=0.5)
    parser.add_argument(
        '--lambda_g',
        type=float,
        default=0.1)    
    parser.add_argument(
        '--beta1',
        type=float,
        default=0.0,
        help='adam: decay of first order momentum of gradient')
    parser.add_argument(
        '--beta2',
        type=float,
        default=0.9,
        help='adam: decay of second order momentum of gradient')
    parser.add_argument(
        '--num_workers',
        type=int,
        default=8,
        help='number of cpu threads to use during batch generation')
    parser.add_argument(
        '--latent_dim',
        type=int,
        default=128,
        help='dimensionality of the latent space')
    parser.add_argument(
        '--img_size',
        type=int,
        default=32,
        help='size of each image dimension')
    parser.add_argument(
        '--channels',
        type=int,
        default=3,
        help='number of image channels')
    parser.add_argument(
        '--val_freq',
        type=int,
        default=3,
        help='interval between each validation')    
    parser.add_argument(
        '--print_freq',
        type=int,
        default=10,
        help='interval between each verbose')
    parser.add_argument(
        '--load_path',
        type=str,
        help='The reload model path')
    parser.add_argument(
        '--exp_name',
        type=str,
        help='The name of exp')
    parser.add_argument(
        '--dataset',
        type=str,
        default='cifar10',
        help='dataset type')
    parser.add_argument(
        '--data_path',
        type=str,
        default='./data',
        help='The path of data set')
    parser.add_argument('--init_type', type=str, default='normal',
                        choices=['normal', 'orth', 'xavier_uniform', 'false'],
                        help='The init type')
    parser.add_argument('--gf_dim', type=int, default=256,
                        help='The base channel num of gen')
    parser.add_argument('--df_dim', type=int, default=128,
                        help='The base channel num of disc')
    parser.add_argument('--ef_dim', type=int, default=256,
                        help='The base channel num of enc')                    
    parser.add_argument('--eval_batch_size', type=int, default=200)
    parser.add_argument('--num_eval_imgs', type=int, default=10000)
    parser.add_argument(
        '--bottom_width',
        type=int,
        default=4,
        help="the base resolution of the GAN")
    parser.add_argument('--random_seed', type=int, default=12345)
    
    opt = parser.parse_args()

    return opt