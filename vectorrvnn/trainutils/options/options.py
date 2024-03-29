import argparse
import os
import os.path as osp
import pickle
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
        self.add_basic_parameters(parser)
        self.add_model_parameters(parser)
        self.add_loss_args(parser)
        self.add_batch_args(parser)
        self.add_raster_args(parser)
        self.add_optimizer_args(parser)
        self.initialized = True
        return parser

    def add_model_parameters(self, parser): 
        parser.add_argument(
            '--modelcls',
            type=str,
            default='ThreeBranch',
            help='model class to use'
        )
        parser.add_argument(
            '--backbone',
            type=str,
            default='resnet18',
            help='Convolutional backbone'
        )
        parser.add_argument(
            '--freeze_layers',
            type=str,
            nargs='*',
            default=[],
            help='list of names of modules to freeze'
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
            nargs='*',
            default=[],
            help='size of hidden layer before output'
        )
        parser.add_argument(
            '--max_len',
            type=int,
            default=50,
            help='maximum number of paths in a graphic'
        )
        parser.add_argument(
            '--init_type',
            type=str,
            default='kaiming',
            choices=['normal', 'xavier', 'kaiming', 'orthogonal'],
            help='initialized modules that are not pretrained'
        )
        parser.add_argument(
            '--dropout',
            type=float,
            default=0.0,
            help='dropout on linear layers'
        )
        parser.add_argument(
            '--use_layer_norm', 
            type=bool,
            default=False,
            help='whether to use layer norm'
        )


    def add_basic_parameters (self, parser) : 
        parser.add_argument(
            '--dataroot', 
            type=str,
            default='./',
            help='path to graphics'
        )
        # parser.add_argument(
        #     '--otherdata',
        #     type=str,
        #     default='./',
        #     help='path to graphics for node overlap'
        # )
        # parser.add_argument(
        #     '--n_otherdata',
        #     type=int,
        #     default=100,
        #     help='number of points from other data'
        # )
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
            help='device to run training on.'
        )
        parser.add_argument(
            '--seed', 
            type=int,
            default=1000,
            help='seed for the random number generator'
        )
        parser.add_argument(
            '--frequency', 
            type=int, 
            default=50, 
            help='frequency of showing visualizations on screen'
        )
        parser.add_argument(
            '--phase',
            type=str,
            default='train',
            choices=['train', 'test'],
            help='phase of the experiment'
        )

    def add_optimizer_args (self, parser) : 
        """ 
        We are using adam for all experiments. Even so, we can 
        use these arguments to control the learning rate, beta1 and 
        learning rate schedule.
        """
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
    
    def add_batch_args(self, parser): 
        """
        Arguments related to training epochs
        and batch construction. 
            1. Which sampler to use? 
            2. How many samples per epoch?
            3. Batch size?
            4. What augmentation to use?
        """
        parser.add_argument(
            '--n_epochs', 
            type=int, 
            default=100, 
            help='total number of epochs'
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
            '--samplercls',
            type=str,
            default='TripletSampler', 
            help='Class to use to sample triplets'
        )
        parser.add_argument(
            '--dataloadercls',
            type=str,
            default='TripletDataLoader',
            help='Custom data loader class'
        )
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
            help='blocks used to construct minibatch'
        )
        parser.add_argument(
            '--augmentation', 
            type=str,
            default='none',
            help='Augmentation applied to data'
        )

    def add_raster_args (self, parser) : 
        """
        These parameters are to do with the vector 
        image. Here we add:
            1. Size of the raster image
            2. Number of channels in the image
            3. Normalization to use for ImageNet pretrained networks.
        """
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
            '--raster_size', 
            type=int, 
            default=256, 
            help='scale rasters to this size'
        )
        parser.add_argument(
            '--rasterize_thread_local',
            type=bool,
            default=False,
            help='whether to create a global or thread local raster context'
        )

    def add_loss_args(self, parser) : 
        parser.add_argument(
            '--loss',
            type=str,
            default='ncs',
            help='loss function for training'
        )
        parser.add_argument(
            '--temperature',
            type=float,
            default=0.1,
            help='temperature control for cosine similarity loss'
        )
        parser.add_argument(
            '--wd',
            type=float,
            default=0.0001,
            help='weight decay for optimizer'
        )

    def validate_batch_size_args (self, opt) : 
        assert (opt.batch_size % opt.base_size == 0)

    def validate_loss_args (self, opt) : 
        if opt.loss.endswith('Margin') : 
            assert (opt.max_margin is not None)
        elif opt.loss == 'hardTriplet' : 
            assert (opt.hard_threshold is not None)
        elif opt.loss == 'infoNCE' : 
            assert (opt.temperature is not None)
        l2 = ['maxMargin', 'hardMaxMarginLoss', 'triplet', 'hardTriplet']
        if opt.loss in l2 : 
            opt.sim_criteria = 'l2'
        else : 
            opt.sim_criteria = 'ncs'

    def gather_options(self, testing=[]):
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

    def toNamedTuple (self, opt) : 
        items = sorted(vars(opt).items())
        Option = namedtuple('Option', [k for k, _ in items])
        globals()['Option'] = Option
        return Option(*[v for _, v in items])

    def parse(self, testing=[]):
        """Parse our options, create checkpoints directory suffix, and set up gpu device."""
        opt = self.gather_options(testing=testing)
        self.validate(opt)
        # process opt.suffix
        self.print_options(opt)
        self.opt = self.toNamedTuple(opt)
        # save to the disk for easy reload 
        expr_dir = osp.join(opt.checkpoints_dir, opt.name)
        mkdir(expr_dir)
        file_name = osp.join(expr_dir, 'opts.pkl')
        # only save training options
        if opt.phase == 'train' : 
            with open(file_name, 'wb') as fp :
                # convert opt to dictionary 
                pickle.dump(dict(self.opt._asdict()), fp) 
        return self.opt
