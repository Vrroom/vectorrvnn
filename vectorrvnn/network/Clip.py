import torch
import clip
from torch import nn
from vectorrvnn.utils import * 
from vectorrvnn.trainutils import *
from PIL import Image
from .TripletBase import TripletBase

class Clip (TripletBase) : 

    def __init__ (self, opts) : 
        super(Clip, self).__init__(opts)
        # load CLIP on CPU. Some performance bug on GPUs with torch==1.7.0
        # this is noted in their github repo as well
        #   https://github.com/openai/CLIP/issues/13
        self.model, self.preprocess = clip.load("ViT-B/32", device="cpu")

    def embedding (self, node, **kwargs) : 
        """ for some reason, we have to load the model on each run """
        pil = node['pil']
        pp_image = self.preprocess(pil).unsqueeze(0)
        embed = self.model.encode_image(pp_image)
        return embed.to(self.opts.device)

    def nodeFeatures (cls, t, ps, opts) : 
        image = rasterize(
            subsetSvg(t.doc, ps),
            opts.raster_size,
            opts.raster_size,
            opts.rasterize_thread_local
        )
        eightBit = (image * 255).astype(np.uint8)
        pil = Image.fromarray(eightBit, "RGBA")
        data = dict(pil=pil)
        return data
