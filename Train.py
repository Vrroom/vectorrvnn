import time
import os
from time import gmtime, strftime
from datetime import datetime
import torch
from torch import nn
from torch.autograd import Variable
import torch.utils.data
# from torchfoldext import FoldExt
from Config import *
from Data import GRASSDataset
from Model import GRASSEncoder
from Model import GRASSDecoder
import Model

def main () :
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

    print("Loading data ...... ", end='', flush=True)

    grass_data = None
    grass_data = GRASSDataset()

    def my_collate(batch):
        return batch

    train_iter = torch.utils.data.DataLoader(
        grass_data, 
        batch_size=BATCH_SIZE, 
        shuffle=True, 
        collate_fn=my_collate
    )

    print("DONE")

    encoder_opt = torch.optim.Adam(encoder.parameters(), lr=1e-3)
    decoder_opt = torch.optim.Adam(decoder.parameters(), lr=1e-3)

    print("Start training ...... ")

    start = time.time()

# if config.save_snapshot:
#     if not os.path.exists(config.save_path):
#         os.makedirs(config.save_path)
#     snapshot_folder = os.path.join(config.save_path, 'snapshots_'+strftime("%Y-%m-%d_%H-%M-%S",gmtime()))
#     if not os.path.exists(snapshot_folder):
#         os.makedirs(snapshot_folder)
# 
# if config.save_log:
#     fd_log = open('training_log.log', mode='a')
#     fd_log.write('\n\nTraining log at '+datetime.now().strftime('%Y-%m-%d %H:%M:%S'))
#     fd_log.write('\n#epoch: {}'.format(config.epochs))
#     fd_log.write('\nbatch_size: {}'.format(config.batch_size))
#     fd_log.write('\ncuda: {}'.format(config.cuda))
#     fd_log.flush()
# 
# header = '     Time    Epoch     Iteration    Progress(%)  TotalLoss'
# log_template = ' '.join('{:>9s},{:>5.0f}/{:<5.0f},{:>5.0f}/{:<5.0f},{:>9.1f}%,{:>10.2f}'.split(','))
# 
# total_iter = config.epochs * len(train_iter)
# 
# for epoch in range(config.epochs):
#     print(header)
#     for batch_idx, batch in enumerate(train_iter):
#         total_loss = torch.tensor([0])
#         for example in batch : 
#             import pdb 
#             pdb.set_trace()
#             total_loss += encode_decode(example, encoder, decoder)
#         # Do parameter optimization
#         encoder_opt.zero_grad()
#         decoder_opt.zero_grad()
#         total_loss.backward()
#         encoder_opt.step()
#         decoder_opt.step()
#         # Report statistics
#         if batch_idx % config.show_log_every == 0:
#             print(log_template.format(strftime("%H:%M:%S",time.gmtime(time.time()-start)),
#                 epoch, config.epochs, 1+batch_idx, len(train_iter),
#                 100. * (1+batch_idx+len(train_iter)*epoch) / (len(train_iter)*config.epochs), total_loss.data.item()))
# 
#     # Save snapshots of the models being trained
#     if config.save_snapshot and (epoch+1) % config.save_snapshot_every == 0 :
#         print("Saving snapshots of the models ...... ", end='', flush=True)
#         torch.save(encoder, snapshot_folder+'//auto_encoder_model_epoch_{}_loss_{:.2f}.pkl'.format(epoch+1, total_loss.data.item()))
#         torch.save(decoder, snapshot_folder+'//auto_decoder_model_epoch_{}_loss_{:.2f}.pkl'.format(epoch+1, total_loss.data.item()))
#         print("DONE")
#     # Save training log
#     if config.save_log and (epoch+1) % config.save_log_every == 0 :
#         fd_log = open('training_log.log', mode='a')
#         fd_log.write('\nepoch:{} total_loss:{:.2f}'.format(epoch+1, total_loss.data.item()))
#         fd_log.close()
# 
# # Save the final models
# print("Saving final models ...... ", end='', flush=True)
# torch.save(encoder, config.save_path+'//auto_encoder_model.pkl')
# torch.save(decoder, config.save_path+'//auto_decoder_model.pkl')
# print("DONE")

if __name__ == "__main__" :
    main()
