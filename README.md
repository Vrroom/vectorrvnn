# vectorrvnn

## Path distance functions

1. color histogram distance.
2. shape descriptors - d2, fd, distance histogram around centroid
3. stroke attribute discriptors - color, width, linecap, linejoin, dasharray
4. parallelism distance
5. curve proximity distances 
6. endpoint connectivity distance
7. isometric transformation distance
8. learnt distance function based on Triplet Loss

## Hierarchy comparison metrics

1. Unordered tree edit distance
2. Fowlkes Mallow's Index

## Baselines

1. Suggero
2. Autogroup (my name for adobe method)

## TODO: 

- [x] Clean any useless code lying around. 
- [ ] Figure out which graphics to use and do their preprocessing. <switch>, size of graphic, number of paths, size of paths.
- [ ] Write data augmentation code.
- [x] Add different metric learning loss functions for experimentation.
- [x] fix callbacks
- [ ] Run experiments more than cleaning code.
- [ ] Blind method evaluation tool (can be given to anyone later also). Add apps
- [ ] Answer Sid's questions on T.E.Ds - for that I think I'll have to obtain the matching map 
- [x] Analyse Mike's clusters
```
TED   = 0.0822 (down)
FMI 1 = 0.8669 (up) 
FMI 2 = 0.5320 (up)
FMI 3 = 0.1371 (up)
```
- [x] Write Adobe autogroup
- [ ] Is it possible to make ilength faster? -> Yes I think so. There is a bug in svgpathtools where the tolerance for the length function invoked by inv_arclength is much less meaning that the heavy length function will spend too much time in computation which is not required. Benchmark it!!
- [x] Figure out if adobe method is viable for tree generation. (not so unless ilength can be made faster)
- [ ] Take all the annotated data and put it somewhere nice so that analysis is easy
- [x] autoprepare stuff for device and evaluation method.
- [x] cleaner interface for training/testing.
- [x] add vis for distance distribution, hierarchies, lr
- [x] remember to update the options correctly for loss/maxmargin etc and validate
- [x] ModelCls + Checkpointing in build model
- [x] NodeFeatures in data
- [ ] Fix lcaScore by converting forest to tree
- [ ] Run the Pattern Grouping Model
