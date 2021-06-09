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
- [x] Write data augmentation code.
- [ ] Blind method evaluation tool (can be given to anyone later also)
- [ ] Experiment with conventional metric learning CNNs
- [ ] Answer Sid's questions on T.E.Ds - for that I think I'll have to obtain the matching map 
- [ ] Analyse Mike's clusters
- [x] Write Adobe autogroup
- [ ] Is it possible to make ilength faster?
