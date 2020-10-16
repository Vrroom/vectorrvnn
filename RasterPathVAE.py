from osTools import *
import argparse
from more_itertools import collapse
import os
import matplotlib.image as image
import numpy as np
import torch as th
from torch.utils.data import DataLoader
import torchvision.datasets as dset
import torchvision.transforms as transforms
import ttools
import ttools.interfaces
from ttools.modules import networks
import pydiffvg

LOG = ttools.get_logger(__name__)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
VAE_OUTPUT = os.path.join(BASE_DIR, "results", "mnist_vae")
AE_OUTPUT = os.path.join(BASE_DIR, "results", "mnist_ae")
GAN_OUTPUT = os.path.join(BASE_DIR, "results", "mnist_gan")
RASTER_GAN_OUTPUT = os.path.join(BASE_DIR, "results", "mnist_gan_raster")

def render(canvas_width, canvas_height, shapes, shape_groups, samples=2):
    _render = pydiffvg.RenderFunction.apply
    scene_args = pydiffvg.RenderFunction.serialize_scene(\
        canvas_width, canvas_height, shapes, shape_groups)
    img = _render(canvas_width, # width
                 canvas_height, # height
                 samples,   # num_samples_x
                 samples,   # num_samples_y
                 0,   # seed
                 *scene_args)
    return img

class Flatten(th.nn.Module):
    def __init__(self):
        super(Flatten, self).__init__()

    def forward(self, x):
        bs = x.shape[0]
        return x.view(bs, -1)

def alphaComposite (destination, source) : 
    alpha = source[:, 3:, :, :]
    d_ = destination[:, :3, :, :]
    s_ = source[:, :3, :, :]
    return (d_ * (1 - alpha) + s_ * alpha)

class MNISTCallback(ttools.callbacks.ImageDisplayCallback):
    """Simple callback that visualize images."""
    def visualized_image(self, batch, fwd_result):
        fwd_result = fwd_result[0]
        ref = batch.cpu()
        out = fwd_result.cpu()
        ref = alphaComposite(th.ones_like(ref), ref)
        out = alphaComposite(th.ones_like(out), out)
        vizdata = [out, ref]
        # tensor to visualize, concatenate images
        viz = th.clamp(th.cat(vizdata, 2), 0, 1)
        return viz

    def caption(self, batch, fwd_result):
        # write some informative caption into the visdom window
        s = "fake, real"
        if fwd_result[1] is not None:
            s += ", raw"
        return s

class VAEInterface(ttools.ModelInterface):
    def __init__(self, model, lr=1e-4, cuda=True, max_grad_norm=10,
                 variational=True, w_kld=1.0):
        super(VAEInterface, self).__init__()

        self.max_grad_norm = max_grad_norm

        self.model = model

        self.w_kld = w_kld

        self.variational = variational

        self.device = "cpu"
        if cuda:
            self.device = "cuda"

        self.model.to(self.device)

        # self.opt = th.optim.Adamax(self.model.parameters(), lr=lr)
        # self.opt = th.optim.RMSprop(self.model.parameters(), lr=lr, alpha=0.0)
        # self.opt = th.optim.SGD(self.model.parameters(), lr=lr, momentum=0.9, nesterov=True)
        self.opt = th.optim.Adam(self.model.parameters(), lr=lr)

    def forward(self, batch):
        im = batch
        im = im.to(self.device)
        out, auxdata = self.model(im)
        return out, auxdata

    def backward(self, batch, fwd_data):
        rendering, aux_data = fwd_data
        im = batch
        im = im.to(self.device)

        logvar = aux_data["logvar"]
        mu = aux_data["mu"]

        data_loss = th.nn.functional.mse_loss(rendering, im)

        ret = {}
        if self.variational:
            kld = -0.5 * th.sum(1 + logvar - mu.pow(2) - logvar.exp(), 1)
            kld = kld.mean()
            loss = data_loss + kld*self.w_kld
            ret["kld"] = kld.item()

            # Weight decay
            reg_loss = 0
            for p in self.model.parameters():
                reg_loss += p.pow(2).sum()

            # loss = loss + 1e-4*reg_loss

            ret["wd"] = reg_loss.item()
        else:
            loss = data_loss

        # optimize
        self.opt.zero_grad()
        loss.backward()
        if self.max_grad_norm is not None:
            nrm = th.nn.utils.clip_grad_norm_(self.model.parameters(), self.max_grad_norm)
            if nrm > self.max_grad_norm:
                LOG.warning("Clipping generator gradients. norm = %.3f > %.3f", nrm, self.max_grad_norm)
        self.opt.step()

        ret["loss"] = loss.item()
        ret["data_loss"] = data_loss.item()

        return ret

    def init_validation(self):
        return {"count": 0, "loss": 0}

    def update_validation(self, batch, fwd, running_data):
        with th.no_grad():
            ref = batch[1].to(self.device)
            loss = th.nn.functional.mse_loss(fwd, ref)
            n = ref.shape[0]

        return {
            "loss": running_data["loss"] + loss.item()*n,
            "count": running_data["count"] + n
        }

    def finalize_validation(self, running_data):
        return {
            "loss": running_data["loss"] / running_data["count"]
        }

class VectorMNISTVAE(th.nn.Module):
    def __init__(self, imsize=28, paths=4, segments=5, samples=2, zdim=128,
                 conditional=False, variational=True, raster=False, fc=False):
        super(VectorMNISTVAE, self).__init__()

        self.samples = samples
        self.imsize = imsize
        self.paths = paths
        self.segments = segments
        self.zdim = zdim
        self.conditional = conditional
        self.variational = variational

        ncond = 0
        if self.conditional:  # one hot encoded input for conditional model
            ncond = 10

        self.fc = fc
        mult = 1
        nc = 1024

        if not self.fc:  # conv model
            self.encoder = th.nn.Sequential(
                # 32x32
                th.nn.Conv2d(4, mult*64, 4, padding=0, stride=2),
                th.nn.LeakyReLU(0.2, inplace=True),

                # 16x16
                th.nn.Conv2d(mult*64, mult*128, 4, padding=0, stride=2),
                th.nn.LeakyReLU(0.2, inplace=True),

                # 8x8
                th.nn.Conv2d(mult*128, mult*256, 4, padding=0, stride=2),
                th.nn.LeakyReLU(0.2, inplace=True),
                Flatten(),
            )
        else:
            self.encoder = th.nn.Sequential(
                # 32x32
                Flatten(),
                th.nn.Linear(28*28 + ncond, mult*256),
                th.nn.LeakyReLU(0.2, inplace=True),

                # 8x8
                th.nn.Linear(mult*256, mult*256, 4),
                th.nn.LeakyReLU(0.2, inplace=True),
            )

        self.mu_predictor = th.nn.Linear(256*1*1, zdim)
        if self.variational:
            self.logvar_predictor = th.nn.Linear(256*1*1, zdim)

        self.decoder = th.nn.Sequential(
            th.nn.Linear(zdim + ncond, nc),
            th.nn.SELU(inplace=True),

            th.nn.Linear(nc, nc),
            th.nn.SELU(inplace=True),
        )


        self.raster = raster
        if self.raster:
            self.raster_decoder = th.nn.Sequential(
                th.nn.Linear(nc, imsize*imsize),
            )
        else:
            # 4 points bezier with n_segments -> 3*n_segments + 1 points
            self.point_predictor = th.nn.Sequential(
                th.nn.Linear(nc, 2*self.paths*(self.segments*3+1)),
                th.nn.Tanh()  # bound spatial extent
            )

            self.width_predictor = th.nn.Sequential(
                th.nn.Linear(nc, self.paths),
                th.nn.Sigmoid()
            )

            self.alphaStroke_predictor = th.nn.Sequential(
                th.nn.Linear(nc, self.paths),
                th.nn.Sigmoid()
            )

            self.rStroke_predictor = th.nn.Sequential(
                th.nn.Linear(nc, self.paths),
                th.nn.Sigmoid()
            )

            self.gStroke_predictor = th.nn.Sequential(
                th.nn.Linear(nc, self.paths),
                th.nn.Sigmoid()
            )

            self.bStroke_predictor = th.nn.Sequential(
                th.nn.Linear(nc, self.paths),
                th.nn.Sigmoid()
            )

            self.alphaFill_predictor = th.nn.Sequential(
                th.nn.Linear(nc, self.paths),
                th.nn.Sigmoid()
            )

            self.rFill_predictor = th.nn.Sequential(
                th.nn.Linear(nc, self.paths),
                th.nn.Sigmoid()
            )

            self.gFill_predictor = th.nn.Sequential(
                th.nn.Linear(nc, self.paths),
                th.nn.Sigmoid()
            )

            self.bFill_predictor = th.nn.Sequential(
                th.nn.Linear(nc, self.paths),
                th.nn.Sigmoid()
            )

            self._reset_weights()

    def _orthogonal_init (self, model) : 
        for n, p in model.named_parameters():
            if 'bias' in n:
                p.data.zero_()
            elif 'weight' in n:
                th.nn.init.orthogonal_(p)

    def _reset_weights(self):
        for n, p in self.encoder.named_parameters():
            if 'bias' in n:
                p.data.zero_()
            elif 'weight' in n:
                th.nn.init.kaiming_normal_(p.data, nonlinearity="leaky_relu")

        th.nn.init.kaiming_normal_(self.mu_predictor.weight.data, nonlinearity="linear")
        if self.variational:
            th.nn.init.kaiming_normal_(self.logvar_predictor.weight.data, nonlinearity="linear")

        for n, p in self.decoder.named_parameters():
            if 'bias' in n:
                p.data.zero_()
            elif 'weight' in n:
                th.nn.init.kaiming_normal_(p, nonlinearity="linear")

        if not self.raster:
            self._orthogonal_init(self.width_predictor)
            # for n, p in self.alphaStroke_predictor.named_parameters():
            #     if 'bias' in n:
            #         p.data.zero_()
            #     elif 'weight' in n:
            #         th.nn.init.orthogonal_(p)
            self._orthogonal_init(self.alphaStroke_predictor)
            self._orthogonal_init(self.rStroke_predictor)
            self._orthogonal_init(self.gStroke_predictor)
            self._orthogonal_init(self.bStroke_predictor)
            self._orthogonal_init(self.alphaFill_predictor)
            self._orthogonal_init(self.rFill_predictor)
            self._orthogonal_init(self.gFill_predictor)
            self._orthogonal_init(self.bFill_predictor)

    def encode(self, im):
        bs, _, h, w = im.shape
        if self.conditional:
            label_onehot = _onehot(label)
            if not self.fc:
                label_onehot = label_onehot.view(bs, 10, 1, 1).repeat(1, 1, h, w)
                out = self.encoder(th.cat([im, label_onehot], 1))
            else:
                out = self.encoder(th.cat([im.view(bs, -1), label_onehot], 1))
        else:
            out = self.encoder(im)
        mu = self.mu_predictor(out)
        if self.variational:
            logvar = self.logvar_predictor(out)
            return mu, logvar
        else:
            return mu

    def reparameterize(self, mu, logvar):
        std = th.exp(0.5*logvar)
        eps = th.randn_like(logvar)
        return mu + std*eps

    def _decode_features(self, z):
        decoded = self.decoder(z)
        return decoded

    def decode(self, z):
        bs = z.shape[0]
        feats = self._decode_features(z)

        if self.raster:
            out = self.raster_decoder(feats).view(bs, 1, self.imsize, self.imsize)
            return out, {}

        all_points = self.point_predictor(feats)
        all_points = all_points.view(bs, self.paths, -1, 2)

        all_points = all_points*(self.imsize//2-2) + self.imsize//2 

        all_widths = self.width_predictor(feats) * 1.5 + .25
        all_alphaStrokes = self.alphaStroke_predictor(feats)
        all_rStrokes = self.rStroke_predictor(feats)
        all_gStrokes = self.gStroke_predictor(feats)
        all_bStrokes = self.bStroke_predictor(feats)
        all_alphaFills = self.alphaFill_predictor(feats)
        all_rFills = self.rFill_predictor(feats)
        all_gFills = self.gFill_predictor(feats)
        all_bFills = self.bFill_predictor(feats)

        # Process the batch sequentially
        outputs = []
        scenes = []
        for k in range(bs):
            # Get point parameters from network
            shapes = []
            shape_groups = []
            for p in range(self.paths):
                points = all_points[k, p].contiguous().cpu()
                width = all_widths[k, p].cpu()

                alphaStroke = all_alphaStrokes[k, p].cpu()
                rStroke = all_rStrokes[k, p].cpu()
                gStroke = all_gStrokes[k, p].cpu()
                bStroke = all_bStrokes[k, p].cpu()

                alphaFill = all_alphaFills[k, p].cpu()
                rFill = all_rFills[k, p].cpu()
                gFill = all_gFills[k, p].cpu()
                bFill = all_bFills[k, p].cpu()

                strokeColor = th.cat([rStroke.view(1,), gStroke.view(1,), bStroke.view(1,), alphaStroke.view(1,)])
                fillColor = th.cat([rFill.view(1,), gFill.view(1,), bFill.view(1,), alphaFill.view(1,)])
                num_ctrl_pts = th.zeros(self.segments, dtype=th.int32) + 2

                path = pydiffvg.Path(
                    num_control_points=num_ctrl_pts, points=points,
                    stroke_width=width, is_closed=False)

                shapes.append(path)
                path_group = pydiffvg.ShapeGroup(
                    shape_ids=th.tensor([len(shapes) - 1]),
                    fill_color=fillColor,
                    stroke_color=strokeColor)
                shape_groups.append(path_group)

            scenes.append(
                [shapes, shape_groups, (self.imsize, self.imsize)])

            # Rasterize
            out = render(self.imsize, self.imsize, shapes, shape_groups, samples=self.samples)

            # Torch format, discard alpha, make gray
            out = out.permute(2, 0, 1)
            outputs.append(out)

        output =  th.stack(outputs).to(z.device)

        aux_data = {
            "points": all_points,
            "scenes": scenes,
        }

        # map to [-1, 1]
        return output, aux_data

    def forward(self, im):
        bs = im.shape[0]

        if self.variational:
            mu, logvar = self.encode(im)
            z = self.reparameterize(mu, logvar)
        else:
            mu = self.encode(im)
            z = mu
            logvar = None

        if self.conditional:
            output, aux =  self.decode(z)
        else:
            output, aux =  self.decode(z)

        aux["logvar"] = logvar
        aux["mu"] = mu

        return output, aux

class Dataset(th.utils.data.Dataset):
    def __init__(self, dataDir):
        super(Dataset, self).__init__()
        dataPts = map(listdir, listdir(dataDir))
        dataPts = collapse(dataPts)
        dataPts = filter(lambda p : p.endswith('png'), dataPts)
        self.images = [image.imread(pt) for pt in dataPts]

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        im = np.transpose(self.images[idx], (2, 0, 1))
        return im

def train(args):
    th.manual_seed(0)
    np.random.seed(0)

    pydiffvg.set_use_gpu(args.cuda)

    # Initialize datasets
    imsize = 28
    dataset = Dataset(args.data_dir)
    dataloader = DataLoader(dataset, batch_size=args.bs,
                            num_workers=4, shuffle=True)

    if args.generator in ["vector_gan", "vae", "ae"]:
        LOG.info("Vector config:\n  samples %d\n"
                 "  paths: %d\n  segments: %d\n"
                 "  zdim: %d\n"
                 "  conditional: %d\n"
                 "  fc: %d\n",
                 args.samples, args.paths, args.segments,
                 args.zdim, args.conditional, args.fc)

    model_params = dict(samples=args.samples, paths=args.paths,
                        segments=args.segments, conditional=args.conditional,
                        zdim=args.zdim, fc=args.fc)

    if args.generator == "vector_gan":
        model = VectorMNISTGenerator(**model_params)
        chkpt = GAN_OUTPUT
        name = "mnist_gan"
    elif args.generator == "vae":
        model = VectorMNISTVAE(variational=True, **model_params)
        chkpt = VAE_OUTPUT
        name = "mnist_vae"
    elif args.generator == "ae":
        model = VectorMNISTVAE(variational=False, **model_params)
        chkpt = AE_OUTPUT
        name = "mnist_ae"
    elif args.generator == "raster_gan":
        model = MNISTGenerator()
        chkpt = RASTER_GAN_OUTPUT
        name = "mnist_gan_raster"
    else:
        raise ValueError("unknown generator")

    if args.conditional:
        name += "_conditional"
        chkpt += "_conditional"

    if args.fc:
        name += "_fc"
        chkpt += "_fc"

    # Resume from checkpoint, if any
    checkpointer = ttools.Checkpointer(
        chkpt, model, meta=model_params, prefix="g_")
    extras, meta = checkpointer.load_latest()

    if meta is not None and meta != model_params:
        LOG.info("Checkpoint's metaparams differ from CLI, aborting: %s and %s",
                 meta, model_params)

    # Hook interface
    if args.generator in ["vae", "ae"]:
        variational = args.generator == "vae"
        if variational:
            LOG.info("Using a VAE")
        else:
            LOG.info("Using an AE")
        interface = VAEInterface(model, lr=args.lr, cuda=args.cuda,
                                 variational=variational, w_kld=args.kld_weight)
    else:
        LOG.info("Using a GAN")

        # Init discriminator
        discrim = Discriminator(conditional=args.conditional, fc=args.fc)
        checkpointer_d = ttools.Checkpointer(
            chkpt, discrim, prefix="d_")
        checkpointer_d.load_latest()

        interface = Interface(model, discrim,
                              conditional=args.conditional,
                              fc=args.fc,
                              lr=args.lr,
                              cuda=args.cuda)

    trainer = ttools.Trainer(interface)

    # Add callbacks
    keys = ["loss_g", "loss_d"]
    if args.generator == "vae":
        keys = ["kld", "data_loss", "loss", "wd"]
    elif args.generator == "ae":
        keys = ["data_loss", "loss"]
    port = 8097
    trainer.add_callback(ttools.callbacks.ProgressBarCallback(
        keys=keys, val_keys=keys))
    trainer.add_callback(ttools.callbacks.VisdomLoggingCallback(
        keys=keys, val_keys=keys, env=name, port=port))
    trainer.add_callback(MNISTCallback(
        env=name, win="samples", port=port, frequency=args.freq))
    trainer.add_callback(ttools.callbacks.CheckpointingCallback(
        checkpointer, max_files=2, max_epochs=50))
    if args.generator not in ["ae", "vae"]:
        trainer.add_callback(ttools.callbacks.CheckpointingCallback(
            checkpointer_d, max_files=2, interval=600, max_epochs=50))

    # Start training
    trainer.train(dataloader, num_epochs=args.num_epochs)


def sample_vae(args):
    chkpt = VAE_OUTPUT
    if args.conditional:
        chkpt += "_conditional"
    if args.fc:
        chkpt += "_fc"

    meta = ttools.Checkpointer.load_meta(chkpt, prefix="g_")
    if meta is None:
        LOG.info("No metadata in checkpoint (or no checkpoint), aborting.")
        return

    model = VectorMNISTVAE(**meta)
    checkpointer = ttools.Checkpointer(chkpt, model, prefix="g_")
    checkpointer.load_latest()
    model.eval()

    # Sample some latent vectors
    n = 8
    bs = n*n

    imsize = 28
    dataset = Dataset(args.data_dir)
    dataloader = DataLoader(dataset, batch_size=bs,
                            num_workers=1, shuffle=True)

    for batch in dataloader:
        ref = batch
        break

    LOG.info("Sampling with auto-encoder code")
    mu, logvar = model.encode(ref)
    z = model.reparameterize(mu, logvar)

    with th.no_grad():
        images, aux = model.decode(z)
        scenes = aux["scenes"]

    h = w = model.imsize

    images = alphaComposite(th.ones_like(images), images)
    for i in range(bs) : 
        image = images[i]
        image = image.permute(1, 2, 0)
        path = os.path.join(chkpt, f'samples{i}.png')
        pydiffvg.imwrite(image, path, gamma=2.2)

    ref = alphaComposite(th.ones_like(ref), ref)
    for i in range(bs) : 
        image = ref[i]
        image = image.permute(1, 2, 0)
        path = os.path.join(chkpt, f'ref{i}.png')
        pydiffvg.imwrite(image, path, gamma=2.2)


def interpolate_vae(args):
    chkpt = VAE_OUTPUT
    if args.conditional:
        chkpt += "_conditional"
    if args.fc:
        chkpt += "_fc"

    meta = ttools.Checkpointer.load_meta(chkpt, prefix="g_")
    if meta is None:
        LOG.info("No metadata in checkpoint (or no checkpoint), aborting.")
        return

    model = VectorMNISTVAE(imsize=128, **meta)
    checkpointer = ttools.Checkpointer(chkpt, model, prefix="g_")
    checkpointer.load_latest()
    model.eval()

    # Sample some latent vectors
    bs = 10
    z = th.randn(bs, model.zdim)

    label = None
    label = th.arange(0, 10)

    animation = []
    nframes = 60
    with th.no_grad():
        for idx, _z in enumerate(z):
            if idx == 0:  # skip first
                continue
            _z0 = z[idx-1].unsqueeze(0).repeat(nframes, 1)
            _z = _z.unsqueeze(0).repeat(nframes, 1)
            if args.conditional:
                _label = label[idx].unsqueeze(0).repeat(nframes)
            else:
                _label = None

            # interp weights
            alpha = th.linspace(0, 1, nframes).view(nframes,  1)
            batch = alpha*_z + (1.0 - alpha)*_z0
            images, aux = model.decode(batch, label=_label)
            images += 1.0
            images /= 2.0
            animation.append(images)

    anim_dir = os.path.join(chkpt, "interpolation")
    os.makedirs(anim_dir, exist_ok=True)
    animation = th.cat(animation, 0)
    h = w = model.imsize
    for idx, frame in enumerate(animation):
        frame = frame.squeeze()
        frame = th.clamp(frame, 0, 1).cpu().numpy()
        path = os.path.join(anim_dir, "frame%03d.png" % idx)
        pydiffvg.imwrite(frame, path, gamma=2.2)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    subs = parser.add_subparsers()

    parser.add_argument("--cpu", dest="cuda", action="store_false", default=th.cuda.is_available())
    parser.add_argument("--conditional", action="store_true", default=False)
    parser.add_argument("--fc", action="store_true", default=False)
    parser.add_argument("--data_dir", default="./ManuallyAnnotatedDataset/Train")

    # -- Train ----------------------------------------------------------------
    parser_train = subs.add_parser("train")
    parser_train.add_argument("generator", choices=["vae", "ae"])
    parser_train.add_argument("--freq", type=int, default=100)
    parser_train.add_argument("--lr", type=float, default=1e-4)
    parser_train.add_argument("--kld_weight", type=float, default=1.0)
    parser_train.add_argument("--bs", type=int, default=8)
    parser_train.add_argument("--num_epochs", type=int)
    # Vector configs
    parser_train.add_argument("--paths", type=int, default=1)
    parser_train.add_argument("--segments", type=int, default=3)
    parser_train.add_argument("--samples", type=int, default=2)
    parser_train.add_argument("--zdim", type=int, default=20)
    parser_train.set_defaults(func=train)

    # -- Eval -----------------------------------------------------------------
    parser_vae = subs.add_parser("sample_vae")
    parser_vae.add_argument("--digit", type=int, choices=list(range(10)))
    parser_vae.set_defaults(func=sample_vae)

    parser_vae = subs.add_parser("interpolate_vae")
    parser_vae.set_defaults(func=interpolate_vae)

    args = parser.parse_args()

    ttools.set_logger(True)
    args.func(args)
