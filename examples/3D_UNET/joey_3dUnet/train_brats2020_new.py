# Python libraries
import argparse
import os

import lib.medloaders as medical_loaders
import lib.medzoo as medzoo
import lib.train as train
# Lib files
import lib.utils as utils
from  lib.losses3D import DiceLoss
import torch.nn as nn
from  lib.medzoo.Unet3D import UNet3D
import torch.optim as optim
import time
seed = 1777777
import torch
import numpy as np
def main():
    args = get_arguments()
    utils.reproducibility(args, seed)
    utils.make_dirs(args.save)
    start_time = time.time()
    training_generator, val_generator, full_volume, affine = medical_loaders.generate_datasets(args,path='examples/3D_UNET/joey_3dUnet/dataset')
     
    input_s = (args.batchSz,args.inChannels,*args.dim)    
    print("generating model", input_s)
                                                                                        
    model = UNet3D(in_channels=args.inChannels, input_size = input_s, n_classes=args.classes, base_n_filter=2).net

    weight_decay = 0.0000000001

    if args.opt == 'sgd':
        optimizer = optim.SGD(model._parameters, lr=args.lr, momentum=0.5, weight_decay=weight_decay)
    elif args.opt == 'adam':
        optimizer = optim.Adam(model._parameters, lr=args.lr, weight_decay=weight_decay)
    elif args.opt == 'rmsprop':
        optimizer = optim.RMSprop(model._parameters, lr=args.lr, weight_decay=weight_decay)

  
    
    criterion = DiceLoss(classes=args.classes)

    trainer = train.Trainer(args, model, criterion, optimizer, train_data_loader=training_generator,
                            valid_data_loader=val_generator)
    trainer.training()


def get_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument('--batchSz', type=int, default=2)
    parser.add_argument('--dataset_name', type=str, default="brats2020")
    parser.add_argument('--dim', nargs="+", type=int, default=(64 , 64, 64))
    parser.add_argument('--nEpochs', type=int, default=10)
    parser.add_argument('--classes', type=int, default=4)
    parser.add_argument('--samples_train', type=int, default=10)
    parser.add_argument('--samples_val', type=int, default=10)
    parser.add_argument('--inChannels', type=int, default=4)
    parser.add_argument('--inModalities', type=int, default=4)
    parser.add_argument('--threshold', default=0.0000001, type=float)
    parser.add_argument('--terminal_show_freq', default=50)
    parser.add_argument('--augmentation', action='store_true', default=True)
    parser.add_argument('--normalization', default='brats', type=str,
                        help='Tensor normalization: options ,max_min,',
                        choices=('max_min', 'full_volume_mean', 'brats', 'max', 'mean'))
    parser.add_argument('--split', default=0.8, type=float, help='Select percentage of training data(default: 0.8)')
    parser.add_argument('--lr', default=1e-2, type=float,
                        help='learning rate (default: 1e-3)')
    parser.add_argument('--cuda', action='store_true', default=True)
    parser.add_argument('--loadData', default=False)
    parser.add_argument('--resume', default='', type=str, metavar='PATH',
                        help='path to latest checkpoint (default: none)')
    parser.add_argument('--model', type=str, default='UNET3D',
                        choices=('VNET', 'VNET2', 'UNET3D', 'DENSENET1', 'DENSENET2', 'DENSENET3', 'HYPERDENSENET'))
    parser.add_argument('--opt', type=str, default='adam',
                        choices=('sgd', 'adam', 'rmsprop'))
    parser.add_argument('--log_dir', type=str,
                        default='./runs/')
    parser.add_argument('--time', type=int,
                        default=time.time())

    args = parser.parse_args()

    args.save = './saved_models/' + args.model + '_checkpoints/' + args.model + '_{}_{}_'.format(
        utils.datestr(), args.dataset_name)
    return args


if __name__ == '__main__':
    main()
