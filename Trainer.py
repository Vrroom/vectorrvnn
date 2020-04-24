from Data import GRASSDataset
import resource
import sys
import pickle
import Data
from functools import reduce, partial
import torch.multiprocessing
import Model
from Model import GRASSAutoEncoder
import torch
import torch.nn as nn
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
from more_itertools import collapse, unzip
import svgpathtools as svg
from copy import deepcopy
import math
from Test import findTree
from torchfold import Fold

def compareNetTreeWithGroundTruth (sample, autoencoder, config, cuda, path=None) :
    """
    Use the encoder and decoder to find
    the optimal tree and compare it
    with the ground truth using tree edit
    distance

    Parameters
    ----------
    sample : tuple
        (svgFile, groundTruthTree)
    autoencoder : GRASSAutoEncoder
        AutoEncoder network.
    config : dict
        Configuration dictionary.
    cuda : bool
        Whether to use CUDA.
    path : None or str
        If path is None, don't save the inferred
        tree. Else do save it at the specified
        location.
    """
    svgFile, gt = sample
    netTree = findTree(config, svgFile, autoencoder, cuda)

    if path is not None: 
        netTree.untensorify()
        _, end = osp.split(svgFile)
        svgName, ext = osp.splitext(end)
        savePath = osp.join(path, svgName) + '.json'
        GraphReadWrite('tree').write((netTree.tree, netTree.root), savePath)

    paths = svg.Document(svgFile).flatten_all_paths()
    return svgTreeEditDistance(netTree, gt, paths)

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

        logging.basicConfig(filename=commonConfig['log_file'], level=logging.INFO)
        self.logger = logging.getLogger('Trainer')

    def run (self) :
        """
        Run experiments for all 
        configuration files.
        """
        self.logger.info('Starting Expt')

        self.makeDir(self.exptDir)
        self.makeDir(self.modelDir)

        self.logger.info('Loading Cross-validation Data')
        self.cvData = GRASSDataset(self.cvDir)

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
        configPath, trainingTreesPath, modelPath = self.createDirectories(i, config)
        self.logger.info(f'Starting Expt {i}')
        self.setTrainDataLoader(config)
        self.setModel(config)
        autoencoder = self.models[-1]
        self.startTrainingLoop(autoencoder, config, modelPath, configPath)
        self.crossValidate(config, autoencoder, configPath)

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

        self.logger.info('Loading Training Data')
        self.trainData = GRASSDataset(
            self.trainDir, 
            makeTrees=True, 
            graphClusterAlgos=graphClusterAlgos,
            relationFunctions=relationFunctions,
            descFunctions=descFunctions
        )

        self.trainData.save('trainData.pkl')

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
            self.logger.info(f'Using CUDA on GPU {self.gpu}')
        else :
            self.logger.info('Not using CUDA')

        autoencoder = GRASSAutoEncoder(config)

        if self.cuda :
            autoencoder = autoencoder.cuda()

        self.models.append(autoencoder)

    def saveSnapshots (self, path, autoencoder, name) : 
        """
        Convenience function to save model snapshots
        at any moment.

        Parameters
        ----------
        path : str
            Path in which the two networks
            have to be saved.
        autoencoder : GRASSAutoEncoder
            AutoEncoder Network.
        name : str
            File name for output.
        """
        self.logger.info(f'Saving Model {name}') 
        torch.save(autoencoder, osp.join(path, name))

    def startTrainingLoop(self, autoencoder, config, modelPath, configPath) :
        """
        Training Loop. Probably hotspot for bugs.
        Uses the specs given in the config dictionary
        to set learning rate, learning rate decay, 
        epochs etc.

        Use of multiple inner functions to keep the
        code neat.

        Parameters
        ----------
        autoencoder : GRASSAutoEncoder
            AutoEncoder network.
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
                self.logger.info(logTemplate.format(
                    elapsedTime, 
                    epoch, 
                    epochs, 
                    1+batchIdx, 
                    nBatches, 
                    donePercent, 
                    totalLoss.data.item()
                ))

            for batchIdx, batch in enumerate(self.trainDataLoader):

                fold = Fold(cuda=self.cuda)

                trees = unzip(collapse(unzip(batch)[2], base_type=tuple))[0]
                nodes = [Model.lossFold(fold, tree) for tree in trees]
                
                opt.zero_grad()
                totalLoss, *_ = fold.apply(autoencoder, [nodes])
                totalLoss = sum(totalLoss)
                totalLoss.backward()
                opt.step()

                if batchIdx % self.logFrequency == 0:
                    reportStatistics()

            return totalLoss.data.item()

        def createAndSaveTrainingPlot () :
            fig, axes = plt.subplots()
            axes.plot(range(epochs), losses) 
            axes.set_xlabel('Epochs')
            axes.set_ylabel('Training Loss')
            fig.savefig(osp.join(configPath, 'TrainingPlot'))
            plt.close(fig)

        nBatches = len(self.trainDataLoader)
        epochs = config['epochs']

        opt = optim.Adam(autoencoder.parameters(), lr=config['lr'])

        gamma = 1 / config['lr_decay_by']
        decayEvery = config['lr_decay_every']

        sched = optim.lr_scheduler.StepLR(opt, decayEvery, gamma=gamma)

        self.logger.info('Starting Training')

        start = time.time()

        totalIter = epochs * nBatches

        header = '     Time    Epoch     Iteration    Progress(%)  TotalLoss'
        logTemplate = '{:>9s} {:>5.0f}/{:<5.0f} {:>5.0f}/{:<5.0f} {:>9.1f}% {:>10.2f}'

        losses = []

        for epoch in range(epochs):
            self.logger.info(header)
            loss = trainOneEpoch()
            sched.step()

            if (epoch + 1) % self.saveFrequency == 0 :
                name = 'autoencoder_epoch{}_loss_{:.2f}.pkl'.format(epoch+1, loss)
                self.saveSnapshots(modelPath, autoencoder, name)

            losses.append(loss)
        
        self.saveSnapshots(modelPath, autoencoder, 'autoencoder.pkl')
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
        fig, axes = plt.subplots()
        axes.hist(treeDist)
        axes.set_xlabel('Tree Edit Distance')
        axes.set_ylabel('Frequency')
        fig.savefig(path)
        plt.close(fig)

    def crossValidate(self, config, autoencoder, configPath) :
        """
        Compute the tree edit distances to the
        ground truth and average them. This is the 
        score of this model.

        Parameters
        ----------
        config : dict
            Configuration for this sub-experiment.
        autoencoder : GRASSAutoEncoder
            Encoder Network.
        configPath : str
            Path to where we are storing this
            experiment's results.
        """
        with torch.multiprocessing.Pool(mp.cpu_count()) as p : 
            treeDist = p.map(
                    partial(compareNetTreeWithGroundTruth, 
                        autoencoder=autoencoder, config=config,
                        cuda=self.cuda), 
                    self.cvData)

        self.treeDistHistogram(treeDist, osp.join(configPath, 'CVHistogram'))
        
        score = sum(treeDist) / (len(treeDist) + 1)
        self.logger.info(f'Cross Validation Score : {score}')
        self.modelScores.append(score)

    def test(self) : 
        """
        Select the model with the smallest 
        average tree edit distance. Compute
        tree edit distances for it. Also store
        the inferred trees in the directory.
        """
        self.logger.info('Loading Test Data')
        self.testData = GRASSDataset(self.testDir)
        testDir = osp.join(self.exptDir, 'Test')
        self.makeDir(testDir)
 
        finalTreesDir = osp.join(testDir, 'FinalTrees')
        self.makeDir(finalTreesDir)
 
        bestAutoEncoder = self.models[argmin(self.modelScores)]
        config = self.configs[argmin(self.modelScores)]
 
        self.saveSnapshots(testDir, bestAutoEncoder, 'bestAutoEncoder.pkl')
         
        with torch.multiprocessing.Pool(mp.cpu_count()) as p : 
            treeDist = p.map(
                    partial(compareNetTreeWithGroundTruth, 
                        autoencoder=bestAutoEncoder, config=config, 
                        cuda=self.cuda, path=finalTreesDir), 
                    self.testData)
 
        self.treeDistHistogram(treeDist, osp.join(testDir, 'Histogram'))
 
        score = sum(treeDist) / len(treeDist)
        self.logger.info(f'Test Score : {score}')
 
    def makeDir (self, path) :
        """
        Make directory and log this information.

        Parameters
        ----------
        path : str
            New directories path.
        """
        os.mkdir(path)
        self.logger.info(f'Made directory: {path}')
    
def main () :
    torch.multiprocessing.set_start_method('spawn')
    rlimit = resource.getrlimit(resource.RLIMIT_NOFILE)
    resource.setrlimit(resource.RLIMIT_NOFILE, (1000000, rlimit[1]))

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
