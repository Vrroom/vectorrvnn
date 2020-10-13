import json
from tqdm import tqdm
import matplotlib.pyplot as plt
from dictOps import aggregateDict
from Dataset import SVGDataSet
from PathModules import * 
import torch
from torch import nn
import torch.optim as optim

class PathVAE (nn.Module) : 

    def __init__ (self, config) : 
        super(PathVAE, self).__init__() 
        self.sampler = Sampler(config['sampler'])
        self.pathEncoder = MLPPathEncoder(config['pathEncoder'])
        self.pathDecoder = MLPPathDecoder(config['pathDecoder'])
        self.lossWeights = config['lossWeights']
        self.mseLoss = nn.MSELoss()

    def sample (self) : 
        eps = torch.randn(1, self.sampler.input_size)
        pts = self.pathDecoder(eps).detach().numpy()
        return pts.squeeze().T

    def reconstruct (self, x) : 
        e = self.pathEncoder(x)
        root, _ = torch.chunk(self.sampler(e), 2, 1)
        x_ = self.pathDecoder(root)
        return x_

    def forward (self, x) : 
        e = self.pathEncoder(x)
        root, kld = torch.chunk(self.sampler(e), 2, 1)
        kldLoss = -kld.sum()
        x_ = self.pathDecoder(root)
        reconLoss = self.mseLoss(x, x_) 
        losses = {
            'kldLoss': kldLoss * self.lossWeights['kldLoss'],
            'reconLoss': reconLoss * self.lossWeights['reconLoss']
        }
        return losses

def main() :
    with open('commonConfig.json') as fd : 
        commonConfig = json.load(fd)
    with open('./Configs/config.json') as fd: 
        config = json.load(fd)
    # Get all the directory names
    trainDir = commonConfig['train_directory']
    cvDir    = commonConfig['cv_directory']
    testDir  = commonConfig['test_directory']
    # Load all the data
    trainData = SVGDataSet(trainDir, 'adjGraph', 10)
    testData = SVGDataSet(testDir, 'adjGraph', 10)
    cvData = SVGDataSet(cvDir, 'adjGraph', 10)
    trainData.toTensor()
    testData.toTensor()
    cvData.toTensor()
    dataLoader = torch.utils.data.DataLoader(
        trainData, 
        batch_size=config['batch_size'], 
        shuffle=True,
        collate_fn=lambda x : x
    )
    autoencoder = PathVAE(config)

    def trainOneEpoch () :

        for batchIdx, batch in enumerate(dataLoader):
            opt.zero_grad()
            thing = torch.cat([b.descriptors for b in batch])
            losses = autoencoder(thing)
            for k in losses.keys() : 
                losses[k] /= len(batch)
            totalLoss = sum(losses.values())
            totalLoss.backward()
            opt.step()
        return losses

    def createAndSaveTrainingPlot () :
        fig, axes = plt.subplots()
        lossDict = aggregateDict(losses, list)
        for k, v in lossDict.items():
            axes.plot(range(epochs), v, label=k) 
        axes.legend()
        axes.set_xlabel('Epochs')
        axes.set_ylabel('Training Loss')
        fig.savefig('TrainingPlot')
        plt.close(fig)

    nBatches = len(dataLoader)
    epochs = config['epochs']
    opt = optim.Adam(autoencoder.parameters(), lr=config['lr'], weight_decay=config['weight_decay'])
    gamma = 1 / config['lr_decay_by']
    decayEvery = config['lr_decay_every']
    sched = optim.lr_scheduler.StepLR(opt, decayEvery, gamma=gamma)
    totalIter = epochs * nBatches
    losses = []
    for epoch in tqdm(range(epochs)):
        loss = trainOneEpoch()
        sched.step()
        losses.append(loss)
    x = cvData[0].descriptors 
    x_ = autoencoder.reconstruct(x).detach().numpy()
    x = x.detach().numpy()
    for pt1, pt2 in zip(x, x_) :  
        plt.xlim((0, 1))
        plt.ylim((0, 1))
        plt.plot(pt1[0] , pt1[1], c='blue')
        plt.plot(pt2[0] , pt2[1], c='red')
        plt.show()
    createAndSaveTrainingPlot()
    
if __name__ == "__main__" :
    main()
