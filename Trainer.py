from Dataset import *
from tqdm import tqdm
import shutil
import resource
import sys
import pickle
from functools import reduce, partial
import torch.multiprocessing
from Model import VectorRvNNAutoEncoder
import torch
import torch.nn as nn
import torch.optim as optim
import json
import time
import os
import os.path as osp
import datetime
import logging
import matplotlib.pyplot as plt
import multiprocessing as mp
from more_itertools import collapse, unzip
import svgpathtools as svg
from copy import deepcopy
import math
from torchfold import Fold
from torch_geometric.data import Batch
from dictOps import aggregateDict

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

        self.commonConfig = commonConfig
        self.gpu  = commonConfig['gpu'] 
        self.cuda = commonConfig['cuda']

        trainDir = commonConfig['train_directory']
        cvDir    = commonConfig['cv_directory']
        testDir  = commonConfig['test_directory']

        self.trainCache = DatasetCache(trainDir)
        self.cvCache    = DatasetCache(cvDir)
        self.testCache  = DatasetCache(testDir)

        self.logFrequency  = commonConfig['show_log_every']
        self.saveFrequency = commonConfig['save_snapshot_every']

        self.models = []
        self.modelScores = []

        logging.basicConfig(filename=commonConfig['log_file'], level=logging.INFO)
        self.logger = logging.getLogger('Trainer')

        self._makeAllDirs()

    def _makeDir (self, path) :
        """
        Make directory and log this information.

        Parameters
        ----------
        path : str
            New directories path.
        """
        os.mkdir(path)
        self.logger.info(f'Made directory: {path}')

    def _makeAllDirs(self) :
        """
        Helper function to make all directories for
        current experiment.
        """
        self.exptDir  = osp.join(self.commonConfig['expt_path'], 'Expt_' + str(datetime.date.today()))
        if osp.exists(self.exptDir): 
            shutil.rmtree(self.exptDir, ignore_errors=True)

        self.modelDir = osp.join(self.exptDir, 'Models')
        self.testDir = osp.join(self.exptDir, 'Test')
        self.finalTreesDir = osp.join(self.testDir, 'FinalTrees')

        self._makeDir(self.exptDir)
        self._makeDir(self.modelDir)
        self._makeDir(self.testDir)
        self._makeDir(self.finalTreesDir)

        self.configPaths = []
        self.trainingTreesPaths = []
        self.modelPaths = [] 

        for i, config in enumerate(self.configs): 
            configPath = osp.join(self.modelDir, f'config{i}')
            self._makeDir(configPath)
            trainingTreesPath = osp.join(configPath, 'TrainingTrees')
            self._makeDir(trainingTreesPath)
            modelPath = osp.join(configPath, 'Models')
            self._makeDir(modelPath)
            self.configPaths.append(configPath)
            self.trainingTreesPaths.append(trainingTreesPath)
            self.modelPaths.append(modelPath)

    def run (self) :
        """
        Run experiments for all 
        configuration files.
        """
        self.logger.info('Starting Expt')
        for i, config in tqdm(list(enumerate(self.configs))) :
            self.logger.info(f'Starting Expt {config}')
            self.setTrainDataLoader(config)
            self.setModel(config)
            autoencoder = self.models[-1]
            self.startTrainingLoop(autoencoder, config, self.modelPaths[i], self.configPaths[i])
            self.crossValidate(config, autoencoder, self.configPaths[i])

    def setTrainDataLoader (self, config) : 
        """
        Load training data and batch loader.
        
        Parameters
        ----------
        config : dict
            Configuration for this sub-experiment.
        """
        self.logger.info('Loading Training Data')
        self.trainData = self.trainCache.dataset(config)
        self.trainData.toTensor()
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
        if self.cuda and torch.cuda.is_available() : 
            torch.cuda.set_device(self.gpu)
            self.logger.info(f'Using CUDA on GPU {self.gpu}')
        else :
            self.logger.info('Not using CUDA')
        autoencoder = VectorRvNNAutoEncoder(config)
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
        autoencoder.train()
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
                nLosses = len(combinedLosses)
                lossTemplate = '{:>10.2f} ' * nLosses
                logTemplate = '{:>9s} {:>5.0f}/{:<5.0f} {:>5.0f}/{:<5.0f} {:>9.1f}% ' + lossTemplate
                lossHeaders = '    '.join(combinedLosses.keys())
                header = '     Time    Epoch     Iteration    Progress(%)   ' + lossHeaders
                elapsedTime = time.strftime("%H:%M:%S",time.gmtime(time.time()-start))
                donePercent = 100. * (1 + batchIdx + nBatches * epoch) / totalIter
                self.logger.info(header)
                self.logger.info(logTemplate.format(
                    elapsedTime, 
                    epoch, 
                    epochs, 
                    1+batchIdx, 
                    nBatches, 
                    donePercent, 
                    *map(lambda x : x.data.item(), combinedLosses.values())
                ))

            for batchIdx, batch in enumerate(self.trainDataLoader):
                opt.zero_grad()
                losses = [autoencoder(item) for item in batch]
                combinedLosses = aggregateDict(losses, sum)
                totalLoss = sum(combinedLosses.values()) / len(batch)
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
        opt = optim.Adam(autoencoder.parameters(), lr=config['lr'], weight_decay=config['weight_decay'])
        gamma = 1 / config['lr_decay_by']
        decayEvery = config['lr_decay_every']
        sched = optim.lr_scheduler.StepLR(opt, decayEvery, gamma=gamma)
        self.logger.info('Starting Training')
        start = time.time()
        totalIter = epochs * nBatches
        losses = []
        for epoch in range(epochs):
            loss = trainOneEpoch()
            sched.step()
            if (epoch + 1) % self.saveFrequency == 0 :
                name = 'autoencoder_epoch{}_loss_{:.2f}.pkl'.format(epoch+1, loss)
                self.saveSnapshots(modelPath, autoencoder, name)
            losses.append(loss)
        self.saveSnapshots(modelPath, autoencoder, 'autoencoder.pkl')
        createAndSaveTrainingPlot()

    def bkFrequencyHistogram(self, bkFrequency, path) :
        """
        Plot and save a histogram of tree edit 
        distances. 

        Parameters
        ----------
        bkFrequency : list
            List of bk values greater than 0.7.
        path : str
            Where to save the plot.
        """
        fig, axes = plt.subplots()
        axes.hist(bkFrequency, bins=range(11))
        axes.set_xlabel('bk > 0.7')
        axes.set_ylabel('Frequency')
        fig.savefig(path)
        plt.close(fig)

    def _score (self, config, autoencoder, dataHandler) : 
        data = dataHandler.dataset(config)
        data.toTensor()
        self.logger.info(f'Classification : {autoencoder.classificationAccuracy(data)}')
        self.logger.info(f'iouAvg : {autoencoder.iouAvg(data)}')
        self.logger.info(f'iouConsistency : {autoencoder.iouConsistency(data)}')
        with torch.multiprocessing.Pool(maxtasksperchild=30) as p: 
            trees = p.map(autoencoder.sample, data)
            scores = p.map(autoencoder.score, data)
        return scores, trees

    def crossValidate(self, config, autoencoder, configPath) :
        """
        Compute the bk frequency to the
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
        scores, _ = self._score(config, autoencoder, self.cvCache)
        self.bkFrequencyHistogram(scores, osp.join(configPath, 'CVHistogram'))
        score = sum(scores) / len(scores)
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
        bestAutoEncoder = self.models[argmax(self.modelScores)]
        config = self.configs[argmax(self.modelScores)]
        self.saveSnapshots(self.testDir, bestAutoEncoder, 'bestAutoEncoder.pkl')
        scores, trees = self._score(config, bestAutoEncoder, self.testCache) 
        self.drawTrees(trees, self.testCache.svgFiles, self.finalTreesDir)
        self.bkFrequencyHistogram(scores, osp.join(self.testDir, 'Histogram'))
        score = sum(scores) / len(scores)
        self.logger.info(f'Test Score : {score}')

    def drawTrees (self, treeList, fileList, path) :
        """
        Convenience function to visualize hierarchies.

        Parameters
        ----------
        treeList : list
            Trees to be drawn.
        fileList : list
            Their corresponding graphics.
        path : str
            Where to save tree plots.
        """
        for tree, file in zip(treeList, fileList) : 
            fname = osp.join(path, osp.splitext(osp.split(file)[1])[0])
            doc = svg.Document(file)
            paths = doc.flatten_all_paths()
            vb = doc.get_viewbox()
            tree.setSVGAttributes(paths, vb)
            matplotlibFigureSaver(treeImageFromGraph(tree.tree), fname)

    def __enter__ (self) : 
        return self

    def __exit__ (self, *args) : 
        saveTrainPath = osp.join(self.exptDir, 'trainCache.pkl')
        saveCvPath    = osp.join(self.exptDir, 'cvCache.pkl')
        saveTestPath  = osp.join(self.exptDir, 'testCache.pkl')
        self.trainCache.save(saveTrainPath)
        self.cvCache.save(saveCvPath)
        self.testCache.save(saveTestPath)
    
def main () :
    torch.multiprocessing.set_start_method('spawn', force=True)
    with open('commonConfig.json') as fd :
        commonConfig = json.load(fd)
    configs = []
    for configFile in os.listdir('./Configs/'):
        configFilePath = osp.join('./Configs/', configFile)
        with open(configFilePath) as fd : 
            configs.append(json.load(fd))
    with Trainer(commonConfig, configs) as trainer : 
        trainer.run()
        trainer.test()

if __name__ == "__main__" :
    main()
