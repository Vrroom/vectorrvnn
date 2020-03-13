from Data import GRASSDataset
import Data
from functools import reduce, partial
import torch.multiprocessing
import Model
from Model import GRASSEncoder, GRASSDecoder
import torch
import torch.optim as optim
import json
import time
import os
import os.path as osp
import datetime
import logging
import Utilities
from Utilities import * 
import matplotlib.pyplot as plt
import multiprocessing as mp
from more_itertools import collapse
import svgpathtools as svg
from copy import deepcopy
import math

def findTree (config, svgFile, encoder, decoder) :
    """
    Do a local greedy search to find a hierarchy.
    
    Roughly, the algorithm is the follows:
        1) Keep a list of all sub-trees formed so far.
           Initially this is a list of leaf nodes.
        2) Try all possible pairs of trees in the list
           and find the one whose merge results in the
           lowest loss.
        3) Remove the 2 trees that were merged and add
           the merged tree back.
        4) If the list has only one tree left, return
           this tree.

    Parameters
    ----------
    config : dict
        Configuration to figure
        out which descriptors to use.
    svgFile : str
        Path to svgFile
    encoder : GRASSEncoder
        Encoding model
    decoder : GRASSDecoder
        Decoding model

    """
    functionGetter = lambda x : getattr(Utilities, x) 
    descFunctions = list(map(functionGetter, config['desc_functions']))

    doc = svg.Document(svgFile)
    paths = svg.Document(svgFile).flatten_all_paths()
    vb = doc.get_viewbox()

    trees = [] 

    # Populate the candidate list with leaves
    for idx, path in enumerate(paths) :
        descriptors = [f(paths[idx].path, vb) for f in descFunctions]
        flattened = list(more_itertools.collapse(descriptors))

        nxTree = nx.DiGraph()
        nxTree.add_node(idx)

        nxTree.nodes[idx]['desc'] = flattened
        nxTree.nodes[idx]['pathSet'] = [idx]
        nxTree.nodes[idx]['svg'] = getSubsetSvg(paths, [idx], vb)
        
        tree = Data.Tree()
        tree.setTree(nxTree, idx)
        tree.tensorify()

        trees.append(tree)

    # Do local greedy search over the space of 
    # candidate trees.
    st = time.time()
    ll = len(trees)
    while len(trees) > 1 : 
        minI, minJ = -1, -1
        minLoss = math.inf
        bestTree = None
        for i in range(len(trees)) :
            for j in range(len(trees)) :
                if i != j : 
                    treeI = deepcopy(trees[i])
                    treeJ = deepcopy(trees[j])

                    treeI.merge([treeJ], paths, vb)

                    loss = Model.treeLoss(treeI, encoder, decoder).item()
                    
                    if loss < minLoss :
                        minI = min(i, j)
                        minJ = max(i, j)
                        minLoss = loss
                        bestTree = treeI
         
        trees.remove(trees[minI])
        trees.remove(trees[minJ - 1])
        trees.append(bestTree)

    print(st - time.time(), svgFile, ll)
    trees[0].relabel()
    return trees[0]

def compareNetTreeWithGroundTruth (sample, encoder, decoder, config, path=None) :
    """
    Use the encoder and decoder to find
    the optimal tree and compare it
    with the ground truth using tree edit
    distance

    Parameters
    ----------
    sample : tuple
        (svgFile, groundTruthTree)
    encoder : GRASSEncoder
        Encoder network.
    decoder : GRASSDecoder
        Decoder network.
    config : dict
        Configuration dictionary.
    path : None or str
        If path is None, don't save the inferred
        tree. Else do save it at the specified
        location.
    """
    svgFile, gt = sample

    netTree = findTree(config, svgFile, encoder, decoder)
    netRoot = netTree.root
    gtRoot = findRoot(gt)

    if path is not None: 
        netTree.untensorify()
        _, svgFile = osp.split(svgFile)
        svgName, ext = osp.splitext(svgFile)
        savePath = osp.join(path, svgName) + '.json'
        GraphReadWrite('tree').write((netTree.tree, netRoot), savePath)

    return match((netRoot, netTree.tree), (gtRoot, gt))

class Trainer () :
    """
    Conduct a bunch of experiments using
    the different configurations given
    as dictionaries.

    The cross-validation of different 
    models is done by calculating the average
    tree-edit distance of the ground truth 
    (extracted from the SVG) and the trees
    inferred on the cross-validation dataset.

    When this class is run, a directory for
    this experiment will be created in the
    ./Expts/ directory with the following
    structure: 

    Expt_<today's date>
    |--Models
    |  |-- Config1 
    |  |   |-- TrainingTrees
    |  |   |-- TrainingPlot.png
    |  |   |-- CVHistogram.png
    |  |   |-- Models
    |  .
    |  .
    |  .
    |  |-- Confign 
    |--Test
       |--BestModel
       |--FinalTrees
    """

    def __init__ (self, commonConfig, configs) :
        """
        Constructor. 

        Initialize a Trainer with dictionaries
        containing training hyperparameters.

        Set up the logging utility. 

        Don't make any directories or make
        any hint of starting any experiment.

        Parameters
        ----------
        commonConfig : dict
            Configurations which are common across
            all experiments. These include the dataset
            directories, saving directories, logging
            preferences and model saving preferences.
        configs : list
            List of configuration dictionaries.
        """
        self.configs = configs

        self.gpu = commonConfig['gpu'] 
        self.cuda = commonConfig['cuda']

        self.exptDir = osp.join(commonConfig['expt_path'], 'Expt_' + str(datetime.date.today()))
        self.modelDir = osp.join(self.exptDir, 'Models')

        self.trainDir = commonConfig['train_directory']
        self.cvDir = commonConfig['cv_directory']
        self.testDir = commonConfig['test_directory']

        self.trainData = None
        self.cvData = None
        self.testData = None

        self.logFrequency  = commonConfig['show_log_every']
        self.saveFrequency = commonConfig['save_snapshot_every']

        self.models = []
        self.modelScores = []

        logging.basicConfig(filename=commonConfig['log_file'], level=logging.DEBUG)

    def run (self) :
        """
        Run experiments for all 
        configuration files.
        """
        logging.info('Starting Expt')

        self.makeDir(self.exptDir)
        self.makeDir(self.modelDir)

        logging.info('Loading Cross-validation Data')
        self.cvData = GRASSDataset(self.cvDir)
        logging.info('Loading Test Data')
        self.testData = GRASSDataset(self.testDir)

        for i, config in enumerate(self.configs) :
            self.runExpt(i + 1, config)

    def runExpt (self, i, config) :
        """
        Run the i-th experiment using the
        configurations present in config.

        Parameters
        ----------
        i : int
            Index of the experiment.
        config : dict
            Dictionary containing parameters.
        """
        try :
            configPath, trainingTreesPath, modelPath = self.createDirectories(i, config)

            logging.info(f'Starting Expt {i}')

            self.setTrainDataLoader(config)
            self.setModel(config)

            encoder, decoder = self.models[-1]

            self.startTrainingLoop(encoder, decoder, config, modelPath, configPath)
            self.crossValidate(config, encoder, decoder, configPath)
        except Exception :
            logging.error(f'Failed at config {i}')

    def createDirectories (self, i, config) : 
        """
        Create a set of directories for each 
        sub-experiment. The structure is mentioned
        in the docstring for this class: 

        |--confign
           |--TrainingTrees
           |--Models

        Parameters
        ----------
        i : int
            Index of config dictionary in self.configs.
        config : dict
            Configuration for the current sub-experiment.
        """
        configPath = osp.join(self.modelDir, f'config{i}')
        self.makeDir(configPath)

        trainingTreesPath = osp.join(configPath, 'TrainingTrees')
        self.makeDir(trainingTreesPath)

        modelPath = osp.join(configPath, 'Models')
        self.makeDir(modelPath)

        return configPath, trainingTreesPath, modelPath

    def setTrainDataLoader (self, config) : 
        """
        Load training data and batch loader.
        
        Parameters
        ----------
        config : dict
            Configuration for this sub-experiment.
        """
        functionGetter = lambda x : getattr(Utilities, x) 

        descFunctions = list(map(functionGetter, config['desc_functions']))
        relationFunctions = list(map(functionGetter, config['relation_functions']))
        graphClusterAlgos = list(map(functionGetter, config['graph_cluster_algo']))

        logging.info('Loading Training Data')
        self.trainData = GRASSDataset(
            self.trainDir, 
            makeTrees=True, 
            graphClusterAlgos=graphClusterAlgos,
            relationFunctions=relationFunctions,
            descFunctions=descFunctions
        )
        self.trainDataLoader = torch.utils.data.DataLoader(
            self.trainData, 
            batch_size=config['batch_size'], 
            shuffle=True,
            collate_fn=lambda x : x
        )

    def setModel (self, config) : 
        """
        Set up the network architecture
        for this configuration.

        Parameters
        ----------
        config : dict
            Configuration for this sub-experiment.
        """
        torch.cuda.set_device(self.gpu)

        if self.cuda and torch.cuda.is_available() : 
            logging.info(f'Using CUDA on GPU {self.gpu}')
        else :
            logging.info('Not using CUDA')

        encoder = GRASSEncoder(config)
        decoder = GRASSDecoder(config)

        if self.cuda :
            encoder = encoder.cuda()
            decoder = decoder.cuda()

        self.models.append((encoder, decoder))

    def saveSnapshots (self, path, encoder, decoder, encName, decName) : 
        """
        Convenience function to save model snapshots
        at any moment.

        Parameters
        ----------
        path : str
            Path in which the two networks
            have to be saved.
        encoder : GRASSEncoder
            Encoder network.
        decoder : GRASSDecoder
            Decoder network.
        encName : str
            File name for output.
        decName : str
            File name for output.
        """
        logging.info(f'Saving Models {encName}, {decName}') 
        torch.save(encoder, osp.join(path, encName))
        torch.save(decoder, osp.join(path, decName))

    def startTrainingLoop(self, encoder, decoder, config, modelPath, configPath) :
        """
        Training Loop. Probably hotspot for bugs.
        Uses the specs given in the config dictionary
        to set learning rate, learning rate decay, 
        epochs etc.

        Use of multiple inner functions to keep the
        code neat.

        Parameters
        ----------
        encoder : GRASSEncoder
            Encoder Network.
        decoder : GRASSDecoder
            Decoder Network.
        config : dict
            Configuration for this sub-experiment.
        modelPath : str
            Path to where we want to save models.
        configPath : str
            Path to where the results of current
            sub-experiment are being stored.
        """

        def trainOneEpoch () :

            def reportStatistics () : 
                elapsedTime = time.strftime("%H:%M:%S",time.gmtime(time.time()-start))
                donePercent = 100. * (1 + batchIdx + nBatches * epoch) / totalIter
                logging.info(logTemplate.format(
                    elapsedTime, 
                    epoch, 
                    epochs, 
                    1+batchIdx, 
                    nBatches, 
                    donePercent, 
                    totalLoss.data.item()
                ))

            for batchIdx, batch in enumerate(self.trainDataLoader):
                trees = filter(lambda x : type(x) is Data.Tree, collapse(batch))
                losses = map(lambda x : Model.treeLoss(x, encoder, decoder), trees)
                totalLoss = reduce(lambda x, y : x + y, losses)

                encoderOpt.zero_grad()
                decoderOpt.zero_grad()

                totalLoss.backward()

                encoderOpt.step()
                decoderOpt.step()

                if batchIdx % self.logFrequency == 0:
                    reportStatistics()

            return totalLoss.data.item()

        def createAndSaveTrainingPlot () :
            plt.plot(range(epochs), losses) 
            plt.xlabel('Epochs')
            plt.ylabel('Training Loss')
            plt.savefig(osp.join(configPath, 'TrainingPlot'))

        nBatches = len(self.trainDataLoader)
        epochs = config['epochs']

        encoderOpt = optim.Adam(encoder.parameters(), lr=config['lr'])
        decoderOpt = optim.Adam(decoder.parameters(), lr=config['lr'])

        gamma = 1 / config['lr_decay_by']
        decayEvery = config['lr_decay_every']

        encoderSched = optim.lr_scheduler.StepLR(encoderOpt, decayEvery, gamma=gamma)
        decoderSched = optim.lr_scheduler.StepLR(decoderOpt, decayEvery, gamma=gamma)

        logging.info('Starting Training')

        start = time.time()

        totalIter = epochs * nBatches

        header = '     Time    Epoch     Iteration    Progress(%)  TotalLoss'
        logTemplate = '{:>9s} {:>5.0f}/{:<5.0f} {:>5.0f}/{:<5.0f} {:>9.1f}% {:>10.2f}'

        losses = []

        for epoch in range(epochs):
            logging.info(header)
            loss = trainOneEpoch()
            encoderSched.step()
            decoderSched.step()

            if (epoch + 1) % self.saveFrequency == 0 :
                encName = 'encoder_epoch{}_loss_{:.2f}.pkl'.format(epoch+1, loss)
                decName = 'decoder_epoch{}_loss_{:.2f}.pkl'.format(epoch+1, loss)
                self.saveSnapshots(modelPath, encoder, decoder, encName, decName)

            losses.append(loss)
        
        self.saveSnapshots(modelPath, encoder, decoder, 'encoder.pkl', 'decoder.pkl')
        createAndSaveTrainingPlot()

    def treeDistHistogram(self, treeDist, path) :
        """
        Plot and save a histogram of tree edit 
        distances. 

        Parameters
        ----------
        treeDist : list
            List of distances.
        path : str
            Where to save the plot.
        """
        plt.hist(treeDist)
        plt.xlabel('Tree Edit Distance')
        plt.ylabel('Frequency')
        plt.savefig(path)

    def crossValidate(self, config, encoder, decoder, configPath) :
        """
        Compute the tree edit distances to the
        ground truth and average them. This is the 
        score of this model.

        Parameters
        ----------
        config : dict
            Configuration for this sub-experiment.
        encoder : GRASSEncoder
            Encoder Network.
        decoder : GRASSDecoder
            Decoder Network.
        configPath : str
            Path to where we are storing this
            experiment's results.
        """
        with torch.multiprocessing.Pool(mp.cpu_count()) as p : 
            treeDist = p.map(
                    partial(compareNetTreeWithGroundTruth, 
                        encoder=encoder, decoder=decoder, config=config), 
                    self.cvData)

        self.treeDistHistogram(treeDist, osp.join(configPath, 'CVHistogram'))
        
        score = sum(treeDist) / len(treeDist)
        logging.info(f'Cross Validation Score : {score}')
        self.modelScores.append(score)

    def test(self) : 
        """
        Select the model with the smallest 
        average tree edit distance. Compute
        tree edit distances for it. Also store
        the inferred trees in the directory.
        """
        testDir = osp.join(self.exptDir, 'Test')
        self.makeDir(testDir)

        finalTreesDir = osp.join(testDir, 'FinalTrees')
        self.makeDir(finalTreesDir)

        bestEncoder, bestDecoder = self.models[argmin(self.modelScores)]
        config = self.configs[argmin(self.modelScores)]

        self.saveSnapshots(testDir, bestEncoder, bestDecoder, 'bestEnc.pkl', 'bestDec.pkl')
         
        with torch.multiprocessing.Pool(mp.cpu_count()) as p : 
            treeDist = p.map(
                    partial(compareNetTreeWithGroundTruth, 
                        encoder=bestEncoder, decoder=bestDecoder, config=config, path=finalTreesDir), 
                    self.testData)

        self.treeDistHistogram(treeDist, osp.join(testDir, 'Histogram'))

        score = sum(treeDist) / len(treeDist)
        logging.info(f'Test Score : {score}')

    def makeDir (self, path) :
        """
        Make directory and log this information.

        Parameters
        ----------
        path : str
            New directories path.
        """
        os.mkdir(path)
        logging.info(f'Made directory: {path}')
    
def main () :
    torch.multiprocessing.set_start_method('spawn')
    with open('commonConfig.json') as fd :
        commonConfig = json.load(fd)

    configs = []

    for configFile in os.listdir('./Configs/'):
        configFilePath = osp.join('./Configs/', configFile)
        with open(configFilePath) as fd : 
            configs.append(json.load(fd))

    trainer = Trainer(commonConfig, configs)
    trainer.run()
    trainer.test()

if __name__ == "__main__" :
    main()
