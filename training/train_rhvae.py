import math
import os
import torch
import torch.nn as nn
from torch.nn.utils.rnn import pack_padded_sequence
import sys
sys.path.append("..")
from torch.autograd import Variable
import numpy as np
from models.rhvae import RH_VAE
from sklearn.model_selection import train_test_split

from generators import queue_datagen
from keras.utils.data_utils import GeneratorEnqueuer
from tqdm import tqdm
import argparse
import wandb
from copy import deepcopy
from rhvae_utils import create_metric, create_inverse_metric

def train(args):

    savedir = args.output_dir
    os.makedirs(savedir, exist_ok=True)
    smiles = np.load(args.input)
    seed = 42
    np.random.seed(seed)
    torch.manual_seed(seed)
    train_smiles, eval_smiles = train_test_split(smiles,
            test_size=0.2, random_state=seed)
    
    import multiprocessing
    multiproc = multiprocessing.Pool(12)
    train_gen = queue_datagen(train_smiles, batch_size=args.batch_size, mp_pool=multiproc)
    mg = GeneratorEnqueuer(train_gen, seed=seed)
    mg.start()
    train_gen = mg.get()

    #eval_gen = queue_datagen(eval_smiles, batch_size=args.batch_size, mp_pool=multiproc)
    #mg = GeneratorEnqueuer(eval_gen, seed=seed)
    #mg.start()
    #eval_gen = mg.get()
    
    vae_model = RH_VAE()
    vae_model.cuda()
    
    vae_optimizer = torch.optim.Adam(vae_model.parameters(), lr=0.0001)
    if args.model_file:
        model_weights, metric_M, metric_centroids = vae_model.load_model(args.model_file)
        vae_model.load_state_dict(model_weights)
        vae_model.M_tens = metric_M
        vae_model.centroids_tens = metric_centroids
        vae_model.G = create_metric(vae_model)
        vae_model.G_inv = create_inverse_metric(vae_model)
    
    train_loader = tqdm(enumerate(train_gen))
    #eval_loader = tqdm(enumerate(eval_gen))
    vae_model.train()

    log_file = open(os.path.join(savedir, "log.txt"), "w")
    n_epochs = 20
    epochs = args.epoch
    wandb.init(project="ligdream", entity="research-team")
    torch.autograd.set_detect_anomaly(True)
    #best_eval_loss = 1e10
    
    for i, (mol_batch, _, _) in train_loader:
        in_data = Variable(mol_batch[:, :5])
        cond_data = Variable(mol_batch[:, 5:])
        in_data = in_data.cuda()
        cond_data = cond_data.cuda()
        vae_optimizer.zero_grad()
        recon_batch, z, vae_loss = vae_model(in_data, cond_data)

        vae_loss.backward()
        wandb.log({"vae_loss": vae_loss})
        vae_optimizer.step()
    
        if (i + 1) % 10 == 0:
            result = "Step: {}, vae_loss: {:.5f}, " \
                     .format(i + 1, vae_loss.item())
            log_file.write(result + "\n")
            log_file.flush()
            train_loader.write(result)

        if (epochs + 1) % 5 == 0:
            vae_model.save(epochs + 1)
        if (i + 1) % int(math.ceil(train_smiles.shape[0]/ args.batch_size)) == 0:
            vae_model._update_metric()
            epochs += 1

        if epochs == n_epochs:
            log_file.close()
            del train_loader
            train_gen.close()
            #del eval_loader
            #eval_gen.close()
            multiproc.close()
            sys.exit()

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("-i", "--input", required=True, help="Path to input .smi file.")
    parser.add_argument("-o", "--output_dir", required=True, help="Path to model save folder.")
    parser.add_argument("--batch_size", type=int, default=16, help="""Batch size for
            training""")
    parser.add_argument("--model_file", default=None, help="""Model Checkpoint file""")
    parser.add_argument("--epoch", type=int, default=0, help="""Epoch from which training
            starts""")
    args = parser.parse_args()
    train(args)

