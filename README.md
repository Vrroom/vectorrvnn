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
- [ ] Blind method evaluation tool (can be given to anyone later also). Add apps
- [x] Analyse Mike's clusters
```
TED   = 0.0822 (down)
FMI 1 = 0.8669 (up) 
FMI 2 = 0.5320 (up)
FMI 3 = 0.1371 (up)
```
- [ ] Check scores on only parts of Val/Test data which don't overlap with training data
