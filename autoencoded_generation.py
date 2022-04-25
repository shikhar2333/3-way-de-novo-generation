import torch
import torch.multiprocessing as mp
from rdkit import Chem
from rdkit.Chem import AllChem
from moleculekit.smallmol.smallmol import SmallMol
from rdkit import DataStructs
from caption import caption_voxel_beam_search
from decoding import decode_smiles
from models.models import MolDecoder, CNN_Encoder
from models.rhvae import RH_VAE
from training.rhvae_utils import create_metric, create_inverse_metric
from models.rhconfig import RH_VAE_CONFIG
import argparse

vocab_list = ["pad", "start", "end",
        "C", "c", "N", "n", "S", "s", "P", "O", "o",
        "B", "F", "I",
        "X", "Y", "Z",
        "1", "2", "3", "4", "5", "6",
        "#", "=", "-", "(", ")"
    ]
vocab_i2c_v1 = {i: x for i, x in enumerate(vocab_list)}
vocab_c2i_v1 = {vocab_i2c_v1[i]: i for i in vocab_i2c_v1}

# load saved model wts
def load_checkpoint(args, encoder_checkpoint, decoder_checkpoint):
    encoder = CNN_Encoder(5).to('cuda')
    decoder = MolDecoder(512,512,512).to('cuda')
    vae_model =  RH_VAE(RH_VAE_CONFIG(is_train=False, n_lf_steps=3,
        latent_dim=256)).to('cuda')
    encoder.eval()
    decoder.eval()
    vae_model.eval()
    model_weights, metric_M, metric_centroids = vae_model.load_model(args.vae_wts)
    vae_model.load_state_dict(model_weights)
    vae_model.M_tens = metric_M
    vae_model.centroids_tens = metric_centroids
    vae_model.G = create_metric(vae_model)
    vae_model.G_inv = create_inverse_metric(vae_model)

    encoder.load_state_dict(encoder_checkpoint)
    decoder.load_state_dict(decoder_checkpoint)
    generate(test_smile_file=args.input, beam_size=args.beam_size, vae=vae_model, encoder=encoder, decoder=decoder)


def tanimoto_calc(mol1, mol2):
#    mol1 = Chem.MolFromSmiles(smi1)
#    mol2 = Chem.MolFromSmiles(smi2)
    fp1 = AllChem.GetMorganFingerprintAsBitVect(mol1, 2, nBits=1024)
    fp2 = AllChem.GetMorganFingerprintAsBitVect(mol2, 2, nBits=1024)
    s = round(DataStructs.TanimotoSimilarity(fp1,fp2),3)
    return s

from generators import generate_representation, generate_sigmas, voxelize
def get_mol_voxels(smile_str):
    # Convert smile to 3D structure
    mol = generate_representation(smile_str)
    if mol is None:
        return None,None

    # Generate sigmas
    sigmas, coords, lig_center = generate_sigmas(mol)
    #vox, centers = voxelize(sigmas, coords, lig_center, rotation=False)
    vox = voxelize(sigmas, coords, lig_center, rotation=False)
    vox = torch.Tensor(vox).to('cuda')
    return vox[:5], vox[5:]

def cnt_properties(mol):
    aromatic_rings = Chem.rdMolDescriptors.CalcNumAromaticRings(mol)
    hbond_acceptor_cnt = Chem.rdMolDescriptors.CalcNumHBA(mol)
    hbond_donor_cnt = Chem.rdMolDescriptors.CalcNumHBD(mol)
    return aromatic_rings, hbond_acceptor_cnt, \
            hbond_donor_cnt

def generate(test_smile_file, beam_size, vae, encoder, decoder):
    generated_smiles = open("smiles_gen.smi", "w")
    tanimoto_scores = open("tanimoto_scores.txt", "w")
    mean_scores = open("mean_scores.txt", "w")
    properties  = open("properties.txt", "w")
    valid_smiles = 0
    gen_set = set()
    n_attempts = 5
    autoencoded_valid = 0
    with open(test_smile_file, "r") as fp:
        smile_idx = 0
        for smile in fp:
            smile = smile.strip().split(",")[-1]
            mol1 = Chem.MolFromSmiles(smile)
            if mol1 is None:
                continue
            voxel_input, cond_input = get_mol_voxels(smile)
            if voxel_input == None:
                smile_idx += 1
                continue
            voxel_input = voxel_input.unsqueeze(0).repeat(n_attempts, 1, 1, 1, 1)
            cond_input = cond_input.unsqueeze(0).repeat(n_attempts, 1, 1, 1, 1)
            voxel_input, _, _ = vae(voxel_input, cond_input)
            outputs = []
            for attempt_no in range(n_attempts):
                temp = caption_voxel_beam_search(encoder, decoder, smile,
                        vocab_c2i_v1, beam_size=beam_size,
                        voxel_input=voxel_input[attempt_no])
                if temp is None:
                    continue
                for t in temp:
                    outputs.append(t)
            outputs = decode_smiles(outputs)
            outputs = list(outputs)
            max_tanimoto_score = -1
            sum_scores = 0
            cnt_valid = 0
            output_smile = None
            #top_comps = []
            for idx,gen_smile in enumerate(outputs):
                mol = Chem.MolFromSmiles(gen_smile)
                if mol is not None:
                    tanimoto_score = tanimoto_calc(mol1=mol1, mol2=mol)
                    sum_scores += tanimoto_score
                    cnt_valid += 1
                    #top_comps.append((tanimoto_score, gen_smile))
                    if max_tanimoto_score < tanimoto_score:
                        max_tanimoto_score = tanimoto_score
                        output_smile = gen_smile
                    if gen_smile not in gen_set:
                        gen_set.add(gen_smile)
            #top_comps.sort()
            #generated_smiles.write(str(smile_idx)+',')
            #mean_val = 0
            #total = min(10, len(top_comps)) 
            #for k in range(total):
                #mean_val += top_comps[k][0]
                #if k < total:
                    #generated_smiles.write(top_comps[k][1]+ ',')
                #else:
                    #generated_smiles.write(top_comps[k][1]+str(mean_val/total)+'\n')
            if output_smile:
                mean_score = sum_scores / cnt_valid
                mean_scores.write(str(mean_score)+'\n')
                tanimoto_scores.write(str(max_tanimoto_score)+'\n')
                generated_smiles.write(str(smile_idx)+","+output_smile+'\n')
                a1,a2,a3 = cnt_properties(Chem.MolFromSmiles(output_smile))
                gen_prop = str(a1)+','+str(a2)+','+str(a3)
                a4,a5,a6 = cnt_properties(mol1)
                input_prop = str(a4)+','+str(a5)+','+str(a6)
                properties.write(gen_prop+','+input_prop+'\n')
                valid_smiles += cnt_valid
                autoencoded_valid += 1
            smile_idx += 1
    
    print("Valid smile number: ", valid_smiles)
    print("Autoencoded Valid smile number: ", autoencoded_valid)
    print("Number of unique smiles: ", len(gen_set))
    mean_scores.close()
    generated_smiles.close()
    tanimoto_scores.close()
    properties.close()

def main():
    encoder_wts = "/scratch/shubham/shape_caption_modelweights_zinc_dataset/encoder-20000.pkl"
    decoder_wts = "/scratch/shubham/shape_caption_modelweights_zinc_dataset/decoder-20000.pkl"
    vae_wts = "/scratch/shubham/saved_models/rh_vae_50.pt"
    #test_smile_file = "data/test_smiles.smi"
    test_smile_file = "./data/aa2ar_actives.ism"
    #vae_wts = "/scratch/shubham/modelweights/rhvae_checkpt/rh_vae_50.pt"
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", default=test_smile_file, help="""
            input smiles file""")
    parser.add_argument("--encoder_wts", default=encoder_wts,
            help="""Checkpoint for encoder network""")
    parser.add_argument("--decoder_wts", default=decoder_wts,
            help="""Checkpoint for decoder network""")
    parser.add_argument("--vae_wts", default=vae_wts, help="""\
            Checkpoint for vae network""")
    parser.add_argument("--beam_size", type=int, default=5, help="""\
            beam size for generating smiles""")

    args = parser.parse_args()
    encoder_checkpoint = torch.load(args.encoder_wts, map_location="cpu")
    decoder_checkpoint = torch.load(args.decoder_wts, map_location="cpu")
    load_checkpoint(args, encoder_checkpoint, decoder_checkpoint)

if __name__ == "__main__":
    main()
