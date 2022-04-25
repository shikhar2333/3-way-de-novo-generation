import torch
import torch.nn.functional as F
from rdkit import Chem
from rdkit.Chem import AllChem
from moleculekit.smallmol.smallmol import SmallMol
import argparse
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
#def generate_representation(in_smile):
    #"""
    #Makes embeddings of Molecule.
    #"""
    #try:
        #m = Chem.MolFromSmiles(in_smile)
        #mh = Chem.AddHs(m)
        #AllChem.EmbedMolecule(mh)
        #Chem.AllChem.MMFFOptimizeMolecule(mh)
        #m = Chem.RemoveHs(mh)
        #mol = SmallMol(m)
        #return mol
    #except:  # Rarely the conformer generation fails
        #return None

#def get_voxels(smile_input):
    #mol = generate_representation(smile_input)
    #if not mol:
        #return None
    #lig_voxel, lig_C,  lig_N = getVoxelDescriptors(mol, voxelsize=1)
    #lig_voxel = lig_voxel.reshape(lig_N[0], lig_N[1], lig_N[2], lig_voxel.shape[1])
    #return torch.tensor(lig_voxel).permute((3, 0, 1, 2))

from generators import generate_representation, generate_sigmas, voxelize

def get_voxels(smile_input):
    mol = generate_representation(smile_input)
    if mol is None:
        return None
    sigmas, coords, lig_center = generate_sigmas(mol)
    vox = voxelize(sigmas, coords, lig_center)
    vox = torch.tensor(vox)
    return vox

def caption_voxel_beam_search(encoder, decoder, smile_input, \
        vocab_map, vae=None, beam_size = 3, voxel_input=None):

    k = beam_size
    vocab_size = len(vocab_map)
    if voxel_input == None:
        voxel_input = get_voxels(smile_input)
        if voxel_input == None:
            return
    voxel_input = voxel_input.unsqueeze(0).to(device)
    voxel_input, cond_input = voxel_input[:, :5], voxel_input[:, 5:]
    if vae:
        voxel_input, _, _ = vae(voxel_input, cond_input)
        #z, _, _ = vae_encoder(voxel_input)
        #recon_input = vae_decoder(z, cond_input)
        #voxel_input = recon_input

    # (1, grid_size, grid_size, grid_size, channels)
    #encoder_out = encoder(recon_input)
    encoder_out = encoder(voxel_input)
    encoder_grid_size = encoder_out.size(1)
    encoder_dim = encoder_out.size(4)

    # flatten encoder output
    encoder_out = encoder_out.view(1, -1, encoder_dim)
    total_voxels = encoder_out.size(1)
    encoder_out = encoder_out.expand(k, total_voxels, encoder_dim)

    k_prev_words = torch.LongTensor([[vocab_map["start"]]]*k).to(device)
    seqs = k_prev_words

    top_k_scores = torch.zeros(k, 1).to(device)
    seqs_alpha = torch.ones(k, 1, encoder_grid_size, encoder_grid_size,
            encoder_grid_size).to(device)

    complete_seqs = []
    complete_seqs_scores = []
    step = 1
    h,c = decoder.init_hidden_states(encoder_out)

    while True:
        embeddings = decoder.embed_layer(k_prev_words).squeeze(1)
#        print(encoder_out.shape)
        att_weight_encoding, alpha = decoder.attention_block(encoder_out, h)
        alpha = alpha.view(-1, encoder_grid_size, encoder_grid_size,
                encoder_grid_size)
        gate = decoder.sigmoid(decoder.f_beta(h))
        att_weight_encoding = gate * att_weight_encoding
        h, c = decoder.lstm_cell(torch.cat([embeddings, att_weight_encoding],
            dim=1), (h, c))
        pred = decoder.final_layer(h)
        scores = F.log_softmax(pred, dim=1)
        # Add the scores at each step
        scores = top_k_scores.expand_as(scores) + scores

        # get the top k scores at each step
        if step == 1:
            top_k_scores, top_k_words = scores[0].topk(k, dim=0)
        else:
            top_k_scores, top_k_words = scores.view(-1).topk(k, dim=0)


        # get the indices of the previous step corresponding to top k sequences
        prev_word_indices = top_k_words // vocab_size
        # get the indices of the next step corresponding to top k sequences
        next_word_indices = top_k_words % vocab_size

        seqs = torch.cat([seqs[prev_word_indices],
            next_word_indices.unsqueeze(1)], dim=1)
        seqs_alpha = torch.cat([seqs_alpha[prev_word_indices],
            alpha[prev_word_indices].unsqueeze(1)], dim=1)
        
        # check for incomplete sequences
        incomplete_indices = [ind for ind,word in enumerate(next_word_indices)
                if vocab_map["end"] != word]
        # check for complete sequences
        complete_indices = [ind for ind, word in enumerate(next_word_indices)
                if vocab_map["end"] == word]
        if len(complete_indices) > 0:
            complete_seqs.extend(seqs[complete_indices].tolist())
            complete_seqs_scores.extend(top_k_scores[complete_indices])
        k -= len(complete_indices)
        if k == 0:
            break
        seqs = seqs[incomplete_indices]
        seqs_alpha = seqs_alpha[incomplete_indices]
        # filter the indices where sequences are complete
        prev_word_indices = prev_word_indices[incomplete_indices]
        h = h[prev_word_indices]
        c = c[prev_word_indices]
        encoder_out = encoder_out[prev_word_indices]
        top_k_scores = top_k_scores[incomplete_indices].unsqueeze(1)
        k_prev_words = next_word_indices[incomplete_indices].unsqueeze(1)
        step += 1
        if step > 100:
            break
    return complete_seqs

def decode_smiles(smiles_list, idx_to_w):
    smiles = set()
    for smile in smiles_list:
        s = ''
        for c in smile[1:]:
            if c == 2:
                break
            s += idx_to_w[c]
        smiles.add(s)
    return smiles

