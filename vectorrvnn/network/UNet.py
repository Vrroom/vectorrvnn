""" also stolen from pytorch-CycleGAN-and-pix2pix """ 
import torch
import torch.nn as nn
from functools import partial

class UNet(nn.Module):
    """Create a UNet"""

    def __init__ (self, opts) : 
        """Construct a Unet 
        We construct the U-Net from the innermost layer to the outermost layer.
        It is a recursive process.
        """
        super(UNet, self).__init__()
        self.raster_size = opts.raster_size
        num_downs = 8 if self.raster_size == 256 else 7
        input_nc = opts.input_nc
        output_nc = opts.structure_embedding_size
        ngf = 64
        norm_layer=nn.BatchNorm2d
        use_dropout=False
        # construct unet structure
        unet_block = UnetSkipConnectionBlock(
            ngf * 8, 
            ngf * 8, 
            input_nc=None, 
            submodule=None, 
            norm_layer=norm_layer, 
            innermost=True
        )  # add the innermost layer
        for i in range(num_downs - 5):          # add intermediate layers with ngf * 8 filters
            unet_block = UnetSkipConnectionBlock(
                ngf * 8, 
                ngf * 8, 
                input_nc=None, 
                submodule=unet_block, 
                norm_layer=norm_layer, 
                use_dropout=use_dropout
            )
        # gradually reduce the number of filters from ngf * 8 to ngf
        unet_block = UnetSkipConnectionBlock(
            ngf * 4, 
            ngf * 8, 
            input_nc=None, 
            submodule=unet_block, 
            norm_layer=norm_layer
        )
        unet_block = UnetSkipConnectionBlock(
            ngf * 2, 
            ngf * 4, 
            input_nc=None, 
            submodule=unet_block, 
            norm_layer=norm_layer
        )
        unet_block = UnetSkipConnectionBlock(
            ngf, 
            ngf * 2, 
            input_nc=None, 
            submodule=unet_block, 
            norm_layer=norm_layer
        )
        self.model = UnetSkipConnectionBlock(
            output_nc, 
            ngf, 
            input_nc=input_nc, 
            submodule=unet_block, 
            outermost=True, 
            norm_layer=norm_layer
        )  # add the outermost layer

    def forward(self, input):
        """Standard forward"""
        return self.model(input)

class UnetSkipConnectionBlock(nn.Module):
    """Defines the Unet submodule with skip connection.
        X -------------------identity----------------------
        |-- downsampling -- |submodule| -- upsampling --|
    """

    def __init__(self, outer_nc, inner_nc, input_nc=None,
                 submodule=None, outermost=False, innermost=False, norm_layer=nn.BatchNorm2d, use_dropout=False):
        """Construct a Unet submodule with skip connections.
        Parameters:
            outer_nc (int) -- the number of filters in the outer conv layer
            inner_nc (int) -- the number of filters in the inner conv layer
            input_nc (int) -- the number of channels in input images/features
            submodule (UnetSkipConnectionBlock) -- previously defined submodules
            outermost (bool)    -- if this module is the outermost module
            innermost (bool)    -- if this module is the innermost module
            norm_layer          -- normalization layer
            use_dropout (bool)  -- if use dropout layers.
        """
        super(UnetSkipConnectionBlock, self).__init__()
        self.outermost = outermost
        if type(norm_layer) == partial:
            use_bias = norm_layer.func == nn.InstanceNorm2d
        else:
            use_bias = norm_layer == nn.InstanceNorm2d
        if input_nc is None:
            input_nc = outer_nc
        downconv = nn.Conv2d(
            input_nc, 
            inner_nc, 
            kernel_size=4,
            stride=2, 
            padding=1, 
            bias=use_bias
        )
        downrelu = nn.LeakyReLU(0.2, True)
        downnorm = norm_layer(inner_nc)
        uprelu = nn.ReLU(True)
        upnorm = norm_layer(outer_nc)

        if outermost:
            upconv = nn.ConvTranspose2d(
                inner_nc * 2, 
                outer_nc,
                kernel_size=4, 
                stride=2,
                padding=1
            )
            down = [downconv]
            up = [uprelu, upconv, nn.Tanh()]
            model = down + [submodule] + up
        elif innermost:
            upconv = nn.ConvTranspose2d(
                inner_nc, 
                outer_nc,
                kernel_size=4, 
                stride=2,
                padding=1, 
                bias=use_bias
            )
            down = [downrelu, downconv]
            up = [uprelu, upconv, upnorm]
            model = down + up
        else:
            upconv = nn.ConvTranspose2d(
                inner_nc * 2, 
                outer_nc,
                kernel_size=4, 
                stride=2,
                padding=1, 
                bias=use_bias
            )
            down = [downrelu, downconv, downnorm]
            up = [uprelu, upconv, upnorm]

            if use_dropout:
                model = down + [submodule] + up + [nn.Dropout(0.5)]
            else:
                model = down + [submodule] + up

        self.model = nn.Sequential(*model)

    def forward(self, x):
        if self.outermost:
            return self.model(x)
        else:   # add skip connections
            return torch.cat([x, self.model(x)], 1)