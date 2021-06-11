import argparse
import os
import os.path as osp
from vectorrvnn.utils import *

class Options():
    """
    This class defines options used during both training 
    and test time.

    It also implements several helper functions such as 
    parsing, printing, and saving the options. 
    """

    def __init__(self):
        """
        Reset the class; indicates the class hasn't
        been initialized
        """
        self.initialized = False

    def initialize(self, parser):
        # basic parameters
        parser.add_argument(
            '--dataroot', 
            type=str,
            default='./',
            help='path to graphics (should have subfolders Train, Val)'
        )
        parser.add_argument(
            '--name', 
            type=str, 
            default='experiment_name', 
            help='name of the experiment. It decides where to store samples and models'
        )
        parser.add_argument(
            '--checkpoints_dir', 
            type=str, 
            default='./results', 
            help='models are saved here'
        )
        parser.add_argument(
            '--load_ckpt', 
            default=None,
            help='load checkpoint from this path'
        )
        # model parameters
        parser.add_argument(
            '--embedding_size',
            type=int,
            default=128, 
            help='size of the path encoding'
        )
        parser.add_argument(
            '--max_margin',
            type=float,
            default=1.0,
            help='margin for triplet loss'
        )
        parser.add_argument(
            '--hidden_size', 
            type=int,
            default=None,
            help='size of hidden layer before output'
        )
        # if number of input channels is 3, remember to 
        # apply some alphacompositing rule and also 
        # normalize the image correctly.
        parser.add_argument(
            '--input_nc', 
            type=int, 
            default=3, 
            help='# of input image channels: 3 for RGB and 4 for RGBA'
        )

        parser.add_argument(
            '--train_epoch_length',
            type=int,
            default=25600,
            help='number of triplets per epoch for training'
        )

        parser.add_argument(
            '--val_epoch_length',
            type=int,
            default=2560,
            help='number of triplets per epoch for validation'
        )
        parser.add_argument(
            '--max_len',
            type=int,
            default=500,
            help='maximum number of paths in a graphic'
        )
        parser.add_argument(
            '--mean', 
            type=list,
            default=[0.485, 0.456, 0.406],
            help='mean to normalize rasters before forward pass'
        )
        parser.add_argument(
            '--std',
            type=list,
            default=[0.229, 0.224, 0.225],
            help='std to normalize rasters before forward pass'
        )

        # According to facenet 
        #   (https://arxiv.org/pdf/1503.03832.pdf),
        # we need large batch sizes when learning based 
        # on triplet loss with hard/semi-hard 
        # negative mining. They use a batch size of 1800.
        # I'll go with 512.
        parser.add_argument(
            '--batch_size', 
            type=int, 
            default=512, 
            help='input batch size'
        )
        
        # The pattern grouping paper: 
        #   (https://people.cs.umass.edu/~kalo/papers/PatternGrouping/PatternGrouping.pdf)
        # used 800 by 800 rasters.
        parser.add_argument(
            '--raster_size', 
            type=int, 
            default=256, 
            help='scale rasters to this size'
        )
        # logging parameters
        parser.add_argument(
            '--frequency', 
            type=int, 
            default=50, 
            help='frequency of showing visualizations on screen'
        )
        # training parameters
        parser.add_argument(
            '--n_epochs', 
            type=int, 
            default=100, 
            help='number of epochs with the initial learning rate'
        )
        parser.add_argument(
            '--n_epochs_decay', 
            type=int, 
            default=100, 
            help='number of epochs to linearly decay learning rate to zero'
        )
        parser.add_argument(
            '--beta1', 
            type=float, 
            default=0.5, 
            help='momentum term of adam'
        )
        parser.add_argument(
            '--lr', 
            type=float, 
            default=0.0002, 
            help='initial learning rate for adam'
        )
        parser.add_argument(
            '--lr_policy', 
            type=str, 
            default='linear', 
            help='learning rate policy. [linear | step | plateau | cosine]'
        )
        parser.add_argument(
            '--lr_decay_iters', 
            type=int, 
            default=50, 
            help='multiply by a gamma every lr_decay_iters iterations'
        )
        
        self.initialized = True
        return parser

    def gather_options(self, testing=[]):
        """Initialize our parser with basic options(only once).
        Add additional model-specific and dataset-specific options.
        These options are defined in the <modify_commandline_options> function
        in model and dataset classes.
        """
        if not self.initialized:  # check if it has been initialized
            parser = argparse.ArgumentParser(
                formatter_class=argparse.ArgumentDefaultsHelpFormatter
            )
            parser = self.initialize(parser)

        # get the basic options
        opt, _ = parser.parse_known_args()

        # save and return the parser
        self.parser = parser
        if testing : 
            return parser.parse_args(testing)
        return parser.parse_args()

    def validate(self, opt): 
        assert(len(opt.std) == opt.input_nc)
        assert(len(opt.mean) == opt.input_nc)

    def print_options(self, opt):
        """Print and save options

        It will print both current options and default values(if different).
        It will save options into a text file / [checkpoints_dir] / opt.txt
        """
        message = ''
        message += '----------------- Options ---------------\n'
        for k, v in sorted(vars(opt).items()):
            comment = ''
            default = self.parser.get_default(k)
            if v != default:
                comment = '\t[default: %s]' % str(default)
            message += '{:>25}: {:<30}{}\n'.format(str(k), str(v), comment)
        message += '----------------- End -------------------'
        print(message)

        # save to the disk
        expr_dir = osp.join(opt.checkpoints_dir, opt.name)
        mkdir(expr_dir)
        file_name = osp.join(expr_dir, '{}_opt.txt'.format(opt.name))
        with open(file_name, 'wt') as opt_file:
            opt_file.write(message)
            opt_file.write('\n')

    def parse(self, testing=[]):
        """Parse our options, create checkpoints directory suffix, and set up gpu device."""
        opt = self.gather_options(testing=testing)
        self.validate(opt)
        # process opt.suffix
        self.print_options(opt)
        self.opt = opt
        return self.opt
