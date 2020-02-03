import time
import os
from functools import reduce
import os.path as osp
from time import gmtime, strftime
from datetime import datetime
import torch
from torch import nn
from torch.autograd import Variable
import torch.utils.data
# from torchfoldext import FoldExt
from Config import *
from Utilities import *
from Data import GRASSDataset
from Model import GRASSEncoder
from Model import GRASSDecoder
import Model

def setupModelDirectory () :
    if not osp.exists(MODEL_SAVE_PATH):
        os.makedirs(MODEL_SAVE_PATH)
    current_expt_folder = osp.join(MODEL_SAVE_PATH, 'expt_'+strftime("%Y-%m-%d_%H-%M-%S",gmtime()))
    if not os.path.exists(current_expt_folder):
        os.makedirs(current_expt_folder)
    configReadme(osp.join(current_expt_folder, 'README.md'))
    return current_expt_folder

def setupLog () :
    fd_log = open('training_log.log', mode='a')
    fd_log.write('\n\nTraining log at '+datetime.now().strftime('%Y-%m-%d %H:%M:%S'))
    fd_log.write('\n#epoch: {}'.format(EPOCHS))
    fd_log.write('\nbatch_size: {}'.format(BATCH_SIZE))
    fd_log.write('\ncuda: {}'.format(CUDA))
    fd_log.flush()
    return fd_log

def setupModel () :
    torch.cuda.set_device(GPU)

    if CUDA and torch.cuda.is_available():
        print("Using CUDA on GPU ", GPU)
    else:
        print("Not using CUDA.")

    encoder = GRASSEncoder()
    decoder = GRASSDecoder()

    if CUDA:
        encoder.cuda()
        decoder.cuda()

    return encoder, decoder

def setupData () :
    print("Loading data ...... ", end='', flush=True)

    grass_data = GRASSDataset()

    train_iter = torch.utils.data.DataLoader(
        grass_data, 
        batch_size=BATCH_SIZE, 
        shuffle=True, 
        collate_fn=id
    )

    print("DONE")
    return train_iter

def main () :
    encoder, decoder = setupModel()
    train_iter = setupData()
    expt_dir = setupModelDirectory()

    encoder_opt = torch.optim.Adam(encoder.parameters(), lr=1e-3)
    decoder_opt = torch.optim.Adam(decoder.parameters(), lr=1e-3)

    print("Start training ...... ")

    start = time.time()

    if SAVE_LOG:
        fd_log = setupLog()

    total_iter = EPOCHS * len(train_iter)

    header = '     Time    Epoch     Iteration    Progress(%)  TotalLoss'
    log_template = ' '.join('{:>9s},{:>5.0f}/{:<5.0f},{:>5.0f}/{:<5.0f},{:>9.1f}%,{:>10.2f}'.split(','))

    for epoch in range(EPOCHS):
        print(header)
        for batch_idx, batch in enumerate(train_iter):
            import pdb
            pdb.set_trace()
            individual_loss = map(lambda x : treeLoss(x, encoder, decoder), batch)
            total_loss = reduce(lambda x, y : x + y, individual_loss)

            # Do parameter optimization
            encoder_opt.zero_grad()
            decoder_opt.zero_grad()
            total_loss.backward()
            encoder_opt.step()
            decoder_opt.step()

            # Report statistics
            if batch_idx % SHOW_LOG_EVERY == 0:
                elapsedTime = strftime("%H:%M:%S",time.gmtime(time.time()-start))
                donePercent = 100. * (1 + batch_idx + len(train_iter) * epoch) / total_iter
                print(log_template.format(
                    elapsedTime, 
                    epoch, 
                    EPOCHS, 
                    1+batch_idx, 
                    len(train_iter), 
                    donePercent, 
                    total_loss.data.item()
                ))

        # Save snapshots of the models being trained
        if SAVE_SNAPSHOT and (epoch+1) % SAVE_SNAPSHOT_EVERY == 0 :
            print("Saving snapshots of the models ...... ", end='', flush=True)
            encoder_path = 'encoder_epoch{}_loss_{:.2f}.pkl'.format(epoch+1, total_loss.data.item())
            decoder_path = 'decoder_epoch{}_loss_{:.2f}.pkl'.format(epoch+1, total_loss.data.item())
            torch.save(encoder, osp.join(expt_dir, encoder_path))
            torch.save(decoder, osp.join(expt_dir, decoder_path))
            print("DONE")

        # Save training log
        if SAVE_LOG and (epoch+1) % SAVE_LOG_EVERY == 0 :
            fd_log = open('training_log.log', mode='a')
            fd_log.write('\nepoch:{} total_loss:{:.2f}'.format(epoch+1, total_loss.data.item()))
            fd_log.close()

    # Save the final models
    print("Saving final models ...... ", end='', flush=True)
    torch.save(encoder, osp.join(expt_dir, 'encoder.pkl'))
    torch.save(decoder, osp.join(expt_dir, 'decoder.pkl'))
    print("DONE")

if __name__ == "__main__" :
    main()
