import argparse
import os
import os.path as osp
from vectorrvnn.utils import *
from collections import namedtuple

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
        # Basic Parameters
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
        parser.add_argument(
            '--device',
            type=str,
            default='cuda:0',
            choices=['cpu', 'cuda:0', 'cuda:1'],
            help='device to run training on.'
        )
        # Model Parameters
        parser.add_argument(
            '--modelcls',
            type=str,
            default='TwoBranch',
            help='model class to use'
        )
        parser.add_argument(
            '--structure_embedding_size',
            type=int,
            default=None,
            help='size of the structure embedding'
        )
        parser.add_argument(
            '--embedding_size',
            type=int,
            default=128, 
            help='size of the path encoding'
        )
        parser.add_argument(
            '--hidden_size', 
            type=int,
            default=None,
            help='size of hidden layer before output'
        )
        parser.add_argument(
            '--phase',
            type=str,
            default='train',
            choices=['train', 'test'],
            help='phase of the experiment'
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
            '--max_len',
            type=int,
            default=500,
            help='maximum number of paths in a graphic'
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
        self.add_loss_args(parser)
        self.add_batch_args(parser)
        parser.add_argument(
            '--init_type',
            type=str,
            default='kaiming',
            choices=['normal', 'xavier', 'kaiming', 'orthogonal'],
            help='initialized modules that are not pretrained'
        )
        parser.add_argument(
            '--augmentation', 
            type=str,
            default='none',
            choices=['none', 'simple', 'oneof', 'compose', 'multiaug'],
            help='Augmentation applied to data'
        )
        parser.add_argument(
            '--samplercls',
            type=str,
            default='AllSampler', 
            choices=['AllSampler', 'SiblingSampler', 'DiscriminativeSampler'],
            help='Class to use to sample triplets'
        )
        parser.add_argument(
            '--train_epoch_length',
            type=int,
            default=25600,
            help='number of triplets per epoch for training'
        )
        parser.add_argument(
            '--mean', 
            type=tuple,
            default=(0.485, 0.456, 0.406),
            help='mean to normalize rasters before forward pass'
        )
        parser.add_argument(
            '--std',
            type=tuple,
            default=(0.229, 0.224, 0.225),
            help='std to normalize rasters before forward pass'
        )
        parser.add_argument(
            '--val_epoch_length',
            type=int,
            default=2560,
            help='number of triplets per epoch for validation'
        )
        parser.add_argument(
            '--n_epochs', 
            type=int, 
            default=100, 
            help='total number of epochs'
        )
        parser.add_argument(
            '--decay_start', 
            type=int, 
            default=50, 
            help='epoch from which to start lr decay'
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
            '--use_swa',
            type=str,
            default='false',
            choices=['true', 'false'],
            help='whether to use stochastic weight averaging'
        )
        parser.add_argument(
            '--lr_policy', 
            type=str, 
            default='linear', 
            choices=['linear', 'step', 'plateau', 'cosine', 'swalr'],
            help='learning rate policy.'
        )
        self.initialized = True
        return parser
    
    def add_batch_args(self, parser): 
        parser.add_argument(
            '--batch_size', 
            type=int, 
            default=32, 
            help='input batch size'
        )
        parser.add_argument(
            '--base_size',
            type=int,
            default=32,
            help='process batches of base size to create minibatch (to avoid GPU OOM)'
        )

    def add_loss_args(self, parser) : 
        parser.add_argument(
            '--loss',
            type=str,
            default='maxMarginLoss',
            choices=[
                'maxMarginLoss', 
                'tripletLoss',
                'hardSemiHardMaxMarginLoss', 
                'hardTripletLoss'
            ],
            help='loss function for training'
        )
        parser.add_argument(
            '--hard_threshold',
            type=float,
            default=None,
            help='margin for triplet loss'
        )
        parser.add_argument(
            '--max_margin',
            type=float,
            default=1.0,
            help='margin for triplet loss'
        )

    def validate_batch_size_args (self, opt) : 
        assert opt.batch_size % opt.base_size == 0

    def validate_loss_args (self, opt) : 
        if opt.loss.endswith('MarginLoss') : 
            assert (opt.max_margin is not None)
        elif opt.loss == 'HardTripletLoss' : 
            assert (opt.hard_threshold is not None)

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
        self.validate_loss_args(opt)
        self.validate_batch_size_args(opt)
        assert((opt.lr_policy == 'swalr') == (opt.use_swa == 'true'))
        assert((opt.structure_embedding_size is not None) \
                == (opt.modelcls.startswith('PatternGrouping')))
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
        if osp.exists(file_name) : 
            file_name += '.swp'
        with open(file_name, 'wt') as opt_file:
            opt_file.write(message)
            opt_file.write('\n')

    def toNamedTuple (self, opt) : 
        items = sorted(vars(opt).items())
        Option = namedtuple('Option', [k for k, _ in items])
        return Option(*[v for _, v in items])

    def parse(self, testing=[]):
        """Parse our options, create checkpoints directory suffix, and set up gpu device."""
        opt = self.gather_options(testing=testing)
        self.validate(opt)
        # process opt.suffix
        self.print_options(opt)
        self.opt = self.toNamedTuple(opt)
        return self.opt
