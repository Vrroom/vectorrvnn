Vectorrvnn
==========

Usage instructions
------------------

1. `$ git clone https://github.com/Vrroom/svgpathtools.git`
2. Go to the cloned directory (svgpathtools)
3. `$ git checkout usefulTools`
4. `$ python3 setup.py install --user`
5. `$ git clone https://sumit2993@bitbucket.org/sidch/vectorrvnn.git` 
6. Go to the cloned directory (vectorrvnn)
7. `$ mkdir Trees && cd Trees && mkdir Train && mkdir Test && cd -`
8. `$ mkdir Models`
9. `$ python3 Train.py`

Organisation
------------

1. Config.py: Contains the parameters for the current experiment.
..* Change network hyperparameters.
..* Specify descriptors from the list of functions in Utilities.py or write your own.
..* Specify how different SVG paths are related.
..* Specify which graph clustering algorithms to be used to create training trees.
2. Utilities.py: Contains all Utility functions that I could think of. These range
from descriptor calculation to data generation.
3. Model.py: Contains the Recursive Network Architecture and a loss computation
function.
4. Data.py: Contains the classes compliant with how PyTorch's way of handling
data. 
5. Train.py: Runs the training loop based on the configuration in Config.py. 
One interesting thing that is also done is that the status of Config.py is
saved in the same folder as where the trained models are stored. This way,
it will become easier to keep track of experiments.
6. Emojis: Directory containing train and test emojis.
