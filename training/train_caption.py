# Copyright (C) 2019 Computational Science Lab, UPF <http://www.compscience.org/>
# Copying and distribution is allowed under AGPLv3 license
import os
import torch
import torch.nn as nn
from torch.nn.utils.rnn import pack_padded_sequence
from torch.autograd import Variable
import numpy as np
import sys
sys.path.append("..")

from models.models import CNN_Encoder, MolDecoder

from generators import queue_datagen
from keras.utils.data_utils import GeneratorEnqueuer
from tqdm import tqdm
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("-i", "--input", required=True, help="Path to input .smi file.")
parser.add_argument("-o", "--output_dir", required=True, help="Path to model save folder.")
args = vars(parser.parse_args())

cap_loss = 0.
caption_start = 0
batch_size = 128

savedir = args["output_dir"]
os.makedirs(savedir, exist_ok=True)
smiles = np.load(args["input"])

import multiprocessing
multiproc = multiprocessing.Pool(6)
my_gen = queue_datagen(smiles, batch_size=batch_size, mp_pool=multiproc)
mg = GeneratorEnqueuer(my_gen, seed=0)
mg.start()
mt_gen = mg.get()

encoder = CNN_Encoder(5)
decoder = MolDecoder(512, 512, 512)
encoder.cuda()
decoder.cuda()

criterion = nn.CrossEntropyLoss().to('cuda')
encoder_optimizer = torch.optim.Adam(encoder.parameters(), lr=0.001)
decoder_optimizer = torch.optim.Adam(decoder.parameters(), lr=0.001)
 
encoder.train()
decoder.train()

tq_gen = tqdm(enumerate(mt_gen))
log_file = open(os.path.join(savedir, "log.txt"), "w")
epochs = 21000
torch.autograd.set_detect_anomaly(True)

for i, (mol_batch, caption, lengths) in tq_gen:

    in_data = Variable(mol_batch[:, :5])
    in_data = in_data.cuda()
    captions = caption.cuda()
    lengths = [length - 1 for length in lengths]

    features = encoder(in_data)
    outputs, alphas, decode_lengths = decoder(features, captions, lengths)
    targets = pack_padded_sequence(captions[:, 1:],decode_lengths, batch_first=True)[0]
    outputs = pack_padded_sequence(outputs ,decode_lengths, batch_first=True)[0]
    cap_loss = criterion(outputs, targets)
    cap_loss += ((1. - alphas.sum(dim=1)) ** 2).mean()

    encoder_optimizer.zero_grad()
    decoder_optimizer.zero_grad()
    cap_loss.backward()
    encoder_optimizer.step()
    decoder_optimizer.step()

    if (i + 1) % 5000 == 0:
        torch.save(decoder.state_dict(),
                   os.path.join(savedir,
                                'decoder-%d.pkl' % (i + 1)))
        torch.save(encoder.state_dict(),
                   os.path.join(savedir,
                                'encoder-%d.pkl' % (i + 1)))

    if (i + 1) % 10 == 0:
        result = "Step: {}, caption_loss: {:.5f}, " \
                 .format(i + 1,
                 float(cap_loss.data.cpu().numpy()) if type(cap_loss) != float
                 else 0.)
        log_file.write(result + "\n")
        log_file.flush()
        tq_gen.write(result)

    if i == epochs:
        log_file.close()
        del tq_gen
        mt_gen.close()
        multiproc.close()
        sys.exit()

