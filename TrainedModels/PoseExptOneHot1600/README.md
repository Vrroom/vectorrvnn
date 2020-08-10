# Expt Models

1. Trained a RvNN model on synthetic pose dataset.
2. 1600 Training Samples
3. 300 Test Samples
4. 100 CV Samples
5. Trained models in this directory mastered it.
6. Training trees were the ones I annotated specifically 
for this dataset.
7. One Hot Encoding of the different body parts was used.

```
{
	"path_code_size": 11, 
	"feature_size": 80, 
	"hidden_size": 200, 
	"epochs": 100, 
	"batch_size": 10, 
	"lr": 0.004, 
	"lr_decay_by": 2, 
	"lr_decay_every": 50, 
	"desc_functions": ["oneHot"], 
	"relation_functions": ["areaGraph"], 
	"graph_cluster_algo": ["kernighan_lin_bisection"]
}
```
