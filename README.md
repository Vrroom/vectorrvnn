# vectorrvnn

![image](https://user-images.githubusercontent.com/7254326/143408586-494c7a2c-bcee-4656-b8c2-336ac0587c6d.png)

We present a data-driven method for hierarchical grouping of paths in vector graphics. 

Paths in a graphic have a natural hierarchical order. They combine to make up parts that we visually recognize (eg. a flower petal, a face, a bike). These parts combine with others to create a scene (eg. a woman cycling through a garden). We want to infer such an organization automatically from a given graphic.

This repository contains:

* Dataset of graphics with annotated hierarchical decompositions.
* Code for training and testing ReGroup.
* An application to simplify selection of parts in vector graphics.

**[Paper](https://arxiv.org/abs/2111.11759)**

## Prerequisites

* Linux
* CUDA
* python3.6
* rust
* EGL

## Getting Started

### Installation

* Clone this repository:
```
https://github.com/Vrroom/vectorrvnn.git
cd vectorrvnn
```
* Install vectorrvnn library: 
```
python3 setup.py install --user
```

Hopefully, things go smoothly. If not, open an Issue!

### Training

The main script for training is `TripletInterface.py`. It can be called with a bunch of options. These are enumerated on: 

```
python3 vectorrvnn/TripletInterface.py -h 
```

The options we used for the results presented in the paper are:

```
python3 vectorrvnn/interfaces/TripletInterface.py \
    --frequency 50 \
    --dataroot ./data/All \
    --name expt-none1 \
    --backbone resnet18 \
    --modelcls ThreeBranch \
    --freeze_layers conv1 bn1 layer1 layer2 \
    --embedding_size 64 \
    --loss cosineSimilarity \
    --temperature 0.1 \
    --samplercls DiscriminativeSampler \
    --sim_criteria negativeCosineSimilarity \
    --n_epochs 28 \
    --hidden_size 128 128 \
    --lr 0.0002 \
    --batch_size 32 \
    --wd 0.00001 \
    --augmentation none \
    --use_layer_norm true \
    --lr_policy step \
    --decay_start 7 \
    --phase train \
    --seed 0 \
```

## Organization

* `data` - The dataset we collected. More information in `./data/README.md`.
* `notebooks` - Jupyter Notebooks for testing and visualization.
    * `Clip.ipynb` - Pairing ReGroup with CLIP to make text based selection app.
    * `MultiGraphic.ipynb` - Visualizations for Multi-Graphic data augmentation strategy.
    * `NodeOverlap.ipynb` - Node Overlap metric discussed in Section 7 of the paper.
    * `TreeEditDistances.ipynb` - Visualizations of tree-to-tree matching.
* `selection_app` - Vector Graphic selection app powered by our method. More information in `./selection_app/README.md`.
* `tests` - Tests for the vectorrvnn library. Can be run by `python3 -m pytest --cov=vectorrvnn ./tests/`.
* `vectorrvnn` - The main library for training, testing, visualizations, baselines etc.
    * `baselines` - Implementation of Suggero and Fisher et. al.
    * `data` - Custom data loaders, triplet samplers and data augmentation pipeline.
    * `geometry` - Methods for analysing bezier path geometry. These are used to implement the baselines.
    * `interfaces` - Torch-tools interface for training. It helps logging, visualizing on visdom, saving checkpoints etc.
    * `networks` - Definitions of neural network architectures. All of them inherit from `TripletBase.py`. This class defines various contrastive losses and the hierarchy inference procedure discussed in Section 5 of the paper.
    * `trainutils` - Utility functions for pytorch, callbacks, command line options, network initialization.
    * `utils` - Utility functions for manipulating XML DOM of SVG files, algebra on bounding boxes, visualizing hierarchies, computing tree edit distances, bipartite matching etc. 



