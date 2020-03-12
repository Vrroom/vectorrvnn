import json
import os
import os.path as osp
import datetime
import logging
import Utilities
from Utilities import * 
import torch
from torch import nn
import torch.optim as optim
from torch.autograd import Variable
import matplotlib.pyplot as plt
import multiprocessing as mp

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

        self.exptDir = osp.join(commonConfig['expt_path'], 'Expt_' + str(datetime.date.today()))
        self.modelDir = osp.join(self.exptDir, 'Models')

        self.trainDir = commonConfig['train_directory']
        self.cvDir = commonConfig['cv_directory']
        self.testDir = commonConfig['test_directory']

        self.trainData = None
        self.cvData = None
        self.testData = None

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
        configPath, trainingTreesPath, modelPath = self.createDirectories(i, config)

        logging.info(f'Starting Expt {i}')

        self.setTrainDataLoader(config, trainingTreesPath)
        self.compareWithGroundTruth() 
        self.setModel(config, modelPath)

        encoder, decoder = self.models[-1]

        self.startTrainingLoop(encoder, decoder, config, modelPath)
        self.crossValidate(encoder, decoder, configPath)

    def createDirectories (self, i, config) : 
        configPath = osp.join(self.modelDir, f'config{i}')
        self.makeDir(configPath)

        trainingTreesPath = osp.join(configPath, 'TrainingTrees')
        self.makeDir(trainingTreesPath)

        modelPath = osp.join(configPath, 'Models')
        self.makeDir(configPath)

        return configPath, trainingTreesPath, modelPath

    def setTrainDataLoader (self, config, trainingTreesPath) : 
        functionGetter = lambda x : getattr(Utilities, x) 

        descFunctions = map(functionGetter, config['desc_functions'])
        relationFunctions = map(functionGetter, config['relation_functions'])
        graphClusterAlgo = map(functionGetter, config['graph_cluster_algo'])

        logging.info('Loading Training Data')
        self.trainData = GRASSDataset(
            self.trainDir, 
            trainingTreesPath, 
            graphClusterAlgo=graphClusterAlgo,
            relationFunctions=relationFunctions,
            descFunctions=descFunctions
        )
        self.trainDataLoader = torch.utils.data.DataLoader(
            trainData, 
            batch_size=config['batch_size'], 
            shuffle=True,
            collate_fn=lambda x : x
        )

    def setModel (self, config, modelPath) : 
        deviceNumber = config['gpu']
        torch.cuda.set_device(deviceNumber)

        if config['cuda'] and torch.cuda.is_available() : 
            logging.info('Using CUDA on GPU {deviceNumber}')
        else :
            logging.info('Not using CUDA')

        encoder = GRASSEncoder(config)
        decoder = GRASSDecoder(config)

        if config['cuda'] :
            encoder.cuda()
            decoder.cuda()

        self.models.append(encoder, decoder)

    def saveSnapshots (self, path, encoder, decoder, encName, decName) : 
        logging.info('Saving Models {encName}, {decName}') 
        torch.save(encoder, osp.path.join(path, encName))
        torch.save(decoder, osp.path.join(path, decName))

    def startTrainingLoop(self, encoder, decoder, config, modelPath) :

        def reportStatistics (loss) : 
            elapsedTime = strftime("%H:%M:%S",time.gmtime(time.time()-start))
            donePercent = 100. * (1 + batchIdx + nBatches * epoch) / totalIter
            logging.info(log_template.format(
                elapsedTime, 
                epoch, 
                epochs, 
                1+batch_idx, 
                nBatches, 
                donePercent, 
                loss
            ))

        def trainOneEpoch () :
            for batchIdx, batch in enumerate(self.trainDataLoader):
                losses = map(lambda x : Model.treeLoss(x, encoder, decoder), batch)
                totalLoss = reduce(lambda x, y : x + y, losses)

                encoderOpt.zero_grad()
                decoderOpt.zero_grad()

                totalLoss.backward()

                encoderOpt.step()
                decoderOpt.step()

                if batchIdx % config['show_log_every'] == 0:
                    reportStatistics(totalLoss.data.item())

            return totalLoss.data.item()

        def saveSnapshots () : 
            logging.info('Saving snapshots of the models')
            encoderPath = 'encoder_epoch{}_loss_{:.2f}.pkl'.format(epoch+1, loss)
            decoderPath = 'decoder_epoch{}_loss_{:.2f}.pkl'.format(epoch+1, loss)
            torch.save(encoder, osp.join(modelPath, encoderPath))
            torch.save(decoder, osp.join(modelPath, decoderPath))

        def createAndSaveTrainingPlot () :
            plt.plot(range(epochs), losses) 
            plt.xlabel('Epochs')
            plt.ylabel('Training Loss')
            plt.savefig('TrainingPlot')

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

            if (epoch + 1) % config['save_snapshot_every'] == 0 :
                encName = 'encoder_epoch{}_loss_{:.2f}.pkl'.format(epoch+1, loss)
                decName = 'decoder_epoch{}_loss_{:.2f}.pkl'.format(epoch+1, loss)
                self.saveSnapshots(modelPath, encName, decName)

            losses.append(loss)
        
        self.saveSnapshots(modelPath, encoder, decoder, 'encoder.pkl', 'decoder.pkl')
        createAndSaveTrainingPlot()

    def treeDistHistogram(self, treeDist, path) :
        plt.hist(treeDist)
        plt.xlabel('Tree Edit Distance')
        plt.ylabel('Frequency')
        plt.savefig(path)

    def crossValidate(self, encoder, decoder, configPath) :
        treeDist = []

        for svgFile, gt in self.cvData : 
            netTree = self.findTree(svgFile, encoder, decoder)
            netRoot = netTree.root
            gtRoot = findRoot(gt)
            treeDist.append(match((netRoot, netTree.tree), (gtRoot, gt)))

        self.treeDistHistogram(treeDist, osp.join(configPath, 'CVHistogram'))
        
        score = sum(treeDist) / len(treeDist)
        logging.info(f'Cross Validation Score : {score}')
        self.modelScores.append(score)

    def test(self) : 
        testDir = osp.join(self.exptDir, 'Test')
        self.makeDir(testDir)

        finalTreesDir = osp.join(testDir, 'FinalTrees')
        self.makeDir(finalTreesDir)

        bestEncoder, bestDecoder = self.models[argmax(self.modelScores)]
        self.saveSnapshots(testDir, bestEncoder, bestDecoder, 'bestEnc.pkl', 'bestDec.pkl')
         
        treeDist = []

        for svgFile, gt in self.testData : 
            svgName, ext = osp.splitext(svgFile)

            netTree = self.findTree(svgFile, encoder, decoder)
            netRoot = netTree.root
            gtRoot = findRoot(gt)
            treeDist.append(match((netRoot, netTree.tree), (gtRoot, gt)))

            savePath = osp.join(finalTreesDir, svgName) + '.json'
            GraphReadWrite('tree').write((netTree.tree, netRoot), savePath)

        self.treeDistHistogram(treeDist, osp.join(testDir, 'Histogram'))

        score = sum(treeDist) / len(treeDist)
        logging.info(f'Test Score : {score}')

    def makeDir (self, path) :
        os.mkdir(path)
        logging.info(f'Made directory: {path}')
    
    def findTree (self, svgFile, encoder, decoder) :
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
        svgFile : str
            Path to svgFile
        encoder : GRASSEncoder
            Encoding model
        decoder : GRASSDecoder
            Decoding model

        """
        doc = svg.Document(svgFile)
        paths = svg.Document(svgFile).flatten_all_paths()
        vb = doc.get_viewbox()

        trees = [] 

        # Populate the candidate list with leaves
        for idx, path in enumerate(paths) :
            descriptors = [f(paths[idx].path, vb) for f in DESC_FUNCTIONS]
            flattened = list(more_itertools.collapse(descriptors))

            nxTree = nx.DiGraph()
            nxTree.add_node(idx)

            nxTree.nodes[idx]['desc'] = torch.tensor(flattened).cuda()
            nxTree.nodes[idx]['pathSet'] = [idx]
            nxTree.nodes[idx]['svg'] = getSubsetSvg(paths, [idx], vb)
            
            tree = Tree()
            tree.setTree(nxTree)
            trees.append(tree)

        # Do local greedy search over the space of 
        # candidate trees.
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

        trees[0].relabel()
        return trees[0]

def main () :
    with open('commonConfig.json') as fd :
        commonConfig = json.load(fd)

    configs = []

    for configFile in os.listdir('./Configs/') :
        configFilePath = osp.join('./Configs/', configFile)
        with open(configFilePath) as fd : 
            configs.append(json.load(fd))

    trainer = Trainer(commonConfig, configs)
    trainer.run()
    trainer.test()

if __name__ == "__main__" :
    main()
