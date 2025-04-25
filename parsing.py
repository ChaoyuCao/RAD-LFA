<<<<<<< HEAD
from argparse import ArgumentParser
from data_processing.settings import *


def parse_args():
    parser = ArgumentParser(description='Args for ablation study')
    
    #! If retrain on timesaver and hbv datasets
    parser.add_argument('--time_saver', type=bool, choices=[True, False], default=False,
                        help='Load arguments for training timer saver dataset')
    parser.add_argument('--hbv', type=bool, choices=[True, False], default=False,
                        help='Load arguments for training hbv dataset')
    
    #! logging
    parser.add_argument('--expname', default='Training', help='Experiment name')
    parser.add_argument('--dataset_dir', default='dataset/Timesaver_data/Standard_sample', help='Main dir for saving datasets')
    parser.add_argument('--log_dir', default='logs', help='Dir for saving logs')
    parser.add_argument('--ckp_dir', default='ckps', help='Dir for saving best trained model')
    
    #! Data
    parser.add_argument('--threshold', type=float, default=4 / 12.0, 
                        help='Clinical concentration threshold for positive and negative samples')
    parser.add_argument('--series', type=int, default=18, help='Time series number for training')
    parser.add_argument('--batch_size', type=int, default=8, help='Batch size')
    parser.add_argument('--split', type=float, default=0.8, help='Train/validation split ratio')
    parser.add_argument('--startpoint', type=int, default=10, help='Min time (sec)')
    parser.add_argument('--endpoint', type=int, default=900, help='Max time (sec)')
    parser.add_argument('--steps', type=int, default=90, help='Times')
    
    #! Model 
    parser.add_argument('--dim', type=int, default=256, help='Feature embedding dim')
    parser.add_argument('--num_heads', type=int, default=8, help='Head number of multi-head attention')
    parser.add_argument('--out_dim', type=int, default=1, 
                        help='For training quantification model, set to 1. For classification model, set to number of classes')
    parser.add_argument('--depth', type=int, default=54, help='Number of differenitall transformer layers')
    parser.add_argument('--resnet_name', type=str, default=None, help='ResNet names, including resnet18, resnet34, resnet50, resnet101')
    parser.add_argument('--in_channels', type=int, default=6, help='Input image channels. 6 for HSV + RGB, or 3 for RGB only')
    parser.add_argument('--resize_shape', type=int, default=48, help='Resize input image shape to (resize_shape, resize_shape)')
    parser.add_argument('--frequency_embedding_size', type=int, default=256, 
                        help='The model use time as positional embeddings. Set it to 0 to close time positional embedding')
    parser.add_argument('--apply_attn', type=bool, choices=[True, False], default=True, help='If use differential transformer')
    parser.add_argument('--outer', type=bool, choices=[True, False], default=True, help='If use outer product for information transition between x and t')
    parser.add_argument('--dropout', type=float, default=0.2, help='Dropout rate')
    
    #! Training
    parser.add_argument('--seed', default=2024, type=int)
    parser.add_argument('--epochs', default=100, type=int, help='Epochs for training')
    parser.add_argument('--weight_decay', type=float, default=1e-6)
    parser.add_argument('--step_size', type=int, default=1000)
    parser.add_argument('--gamma', type=float, default=0.2)
    parser.add_argument('--lr', default=1e-4, help='Learning Rate')

    args = parser.parse_args()
    args.shape = (args.resize_shape, args.resize_shape, 3)
    
    if args.frequency_embedding_size == 0:
        args.outer = False
        
    assert not (args.time_saver and args.hbv)
    if args.time_saver:
        args.__dict__.update(TimeSaverArgs)
    if args.hbv:
        args.__dict__.update(HBVArgs)
    
=======
from argparse import ArgumentParser
from data_processing.settings import *


def parse_args():
    parser = ArgumentParser(description='Args for ablation study')
    
    #! If retrain on timesaver and hbv datasets
    parser.add_argument('--time_saver', type=bool, choices=[True, False], default=False,
                        help='Load arguments for training timer saver dataset')
    parser.add_argument('--hbv', type=bool, choices=[True, False], default=False,
                        help='Load arguments for training hbv dataset')
    
    #! logging
    parser.add_argument('--expname', default='Training', help='Experiment name')
    parser.add_argument('--dataset_dir', default='dataset/Timesaver_data/Standard_sample', help='Main dir for saving datasets')
    parser.add_argument('--log_dir', default='logs', help='Dir for saving logs')
    parser.add_argument('--ckp_dir', default='ckps', help='Dir for saving best trained model')
    
    #! Data
    parser.add_argument('--threshold', type=float, default=4 / 12.0, 
                        help='Clinical concentration threshold for positive and negative samples')
    parser.add_argument('--series', type=int, default=18, help='Time series number for training')
    parser.add_argument('--batch_size', type=int, default=8, help='Batch size')
    parser.add_argument('--split', type=float, default=0.8, help='Train/validation split ratio')
    parser.add_argument('--startpoint', type=int, default=10, help='Min time (sec)')
    parser.add_argument('--endpoint', type=int, default=900, help='Max time (sec)')
    parser.add_argument('--steps', type=int, default=90, help='Times')
    
    #! Model 
    parser.add_argument('--dim', type=int, default=256, help='Feature embedding dim')
    parser.add_argument('--num_heads', type=int, default=8, help='Head number of multi-head attention')
    parser.add_argument('--out_dim', type=int, default=1, 
                        help='For training quantification model, set to 1. For classification model, set to number of classes')
    parser.add_argument('--depth', type=int, default=54, help='Number of differenitall transformer layers')
    parser.add_argument('--resnet_name', type=str, default=None, help='ResNet names, including resnet18, resnet34, resnet50, resnet101')
    parser.add_argument('--in_channels', type=int, default=6, help='Input image channels. 6 for HSV + RGB, or 3 for RGB only')
    parser.add_argument('--resize_shape', type=int, default=48, help='Resize input image shape to (resize_shape, resize_shape)')
    parser.add_argument('--frequency_embedding_size', type=int, default=256, 
                        help='The model use time as positional embeddings. Set it to 0 to close time positional embedding')
    parser.add_argument('--apply_attn', type=bool, choices=[True, False], default=True, help='If use differential transformer')
    parser.add_argument('--outer', type=bool, choices=[True, False], default=True, help='If use outer product for information transition between x and t')
    parser.add_argument('--dropout', type=float, default=0.2, help='Dropout rate')
    
    #! Training
    parser.add_argument('--seed', default=2024, type=int)
    parser.add_argument('--epochs', default=100, type=int, help='Epochs for training')
    parser.add_argument('--weight_decay', type=float, default=1e-6)
    parser.add_argument('--step_size', type=int, default=1000)
    parser.add_argument('--gamma', type=float, default=0.2)
    parser.add_argument('--lr', default=1e-4, help='Learning Rate')

    args = parser.parse_args()
    args.shape = (args.resize_shape, args.resize_shape, 3)
    
    if args.frequency_embedding_size == 0:
        args.outer = False
        
    assert not (args.time_saver and args.hbv)
    if args.time_saver:
        args.__dict__.update(TimeSaverArgs)
    if args.hbv:
        args.__dict__.update(HBVArgs)
    
>>>>>>> 12dbf5c (revision)
    return args