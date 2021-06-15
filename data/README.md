# Perceptually organized hierarchies

I collected such hierarchies for emojis. Most of them have been 
annotated by me. Data was collected in 3 phases:

1. _MyAnnotations_: My annotations of approximately 1000 graphics.
2. _CrowdSourcedAnnotations_ : Around 300 annotations from other people.
3. _MikeAnnotations_ : Around 150 annotations by Mike and I. 

I have combined all this data into the _All_ directory. The complete
collection is partitioned into 3 categories - _Train_, _Test_ and _Val_.

## Data organization

```
./datadir
  |-- 0/
  ...|-- 0.pkl
  ...|-- 0.svg
  ...|-- id.txt (or metadata.txt)
  |-- 1/
  ...
```

The `pickle` file is a `networkx` directed graph that can 
be loaded with `nx.read_gpickle`. More often than not the directed 
graph will be a tree, otherwise it'll be a forest. The leaf nodes
in the graph index the paths in the graphic (the `.svg` file). 
See `data.py` for more information on how data is loaded.
