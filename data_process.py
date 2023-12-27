import os
import pandas as pd
import numpy as np
import torch
import esm
import traceback
import pickle
from tqdm import tqdm

import sys

sys.path.append('/')


def read_proteins(seq_file):
    pd_file = pd.read_csv(seq_file, sep='\t', header=None)
    names = pd_file[0]
    seqs = pd_file[1]
    # print(names)
    # print(seqs)
    return names, seqs


def read_pairs(pair_file):
    pd_file = pd.read_csv(pair_file, sep='\t', header=None)
    protein_1 = pd_file[0]
    protein_2 = pd_file[1]
    interactions = pd_file[2]
    # print(len(protein_1),len(protein_2),len(interactions))
    return protein_1, protein_2, interactions


def contact_predict(names, seqs, save_path):
    assert len(names) == len(seqs), 'protein names should have the same length as protein seqs.'
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, alphabet = esm.pretrained.esm2_t33_650M_UR50D()
    batch_converter = alphabet.get_batch_converter()
    model.eval()  # disables dropout for deterministic results
    model.to(device)
    contacts = {}
    # data = []
    with torch.no_grad():
        for i in tqdm(range(len(names))):
            name, seq = names[i], seqs[i]
            # print(len(seq))
            save_file = os.path.join(save_path, name + '_contact.npy')
            try:
                if os.path.exists(save_file):
                    contacts[name] = np.load(save_file)
                    continue
                if len(seq) <= 1000:
                    _, _, batch_tokens = batch_converter([(name, seq)])
                    batch_tokens = batch_tokens.to(device)
                    contact = model.predict_contacts(batch_tokens)
                    contacts[name] = contact.to("cpu").numpy()
                    # print(contacts[name].shape, len(seq))
                # if sequence length larger than 1000, it would lead to an "out of memory" error with a 24GB GPU, then concatenate multiple sub seqs.
                elif len(seq) > 1000 and len(seq) <= 2500:
                    _, _, batch_tokens = batch_converter([(name, seq)])
                    model.to("cpu")
                    contact = model.predict_contacts(batch_tokens)
                    contacts[name] = contact.numpy()
                    np.save(save_file, contacts[name])
                    model = model.to(device)
                    # print(contacts[name].shape, len(seq))
                elif len(seq) > 2500:
                    cut_L = 2500
                    step = 2000
                    count = L = len(seq)
                    contact = np.zeros((1, L, L))
                    start = 0
                    # print(L)
                    model.to("cpu")
                    while (count > 0):
                        temp_L = min(cut_L, count)
                        temp_sub_seq = seq[start: start + temp_L]
                        _, _, batch_tokens_temp = batch_converter([(name + "_" + str(i), temp_sub_seq)])
                        contact_temp = model.predict_contacts(batch_tokens_temp)
                        # print("contact_temp:", contact_temp.shape, len(seq))
                        contact[:, start:start + temp_L, start:start + temp_L] = (contact[:, start:start + temp_L,
                                                                                  start:start + temp_L] + contact_temp.numpy()) / 2.0
                        # print(start, start + temp_L)

                        start = start + step
                        count = count - step
                    contacts[name] = contact
                    # print(contacts[name].shape, contact_temp.shape)
                    model.to(device)

                np.save(save_file, contacts[name])

            except:
                traceback.print_exc()
                print('peocess error, protein information:')
                print(save_file)
                print(name)
                print(seq)
                exit(0)
                pass


def read_ppi_pairs_for_DeepTrio(dataset='BioGRID_S', data_process_path='process/benchmarks_DeepTrio'):
    # data_process_path = os.path.join(data_process_path, dataset)
    protein_data_path = os.path.join('data', 'benchmarks_DeepTrio')
    if dataset == 'BioGRID_S':
        print("process BioGRID_S paris ...")
        protein_pair_path = os.path.join(protein_data_path, 'BioGRID multi vaildated physical interaction sets', 'sa',
                                         'third_sa_MV_pair.tsv')
        # pair_save_path=os.path.join(data_process_path,"pairs")
        protein_1, protein_2, interactions = read_pairs(protein_pair_path)
        return protein_1, protein_2, interactions

    elif dataset == 'BioGRID_H':
        print("process BioGRID_H paris ...")
        protein_pair_path = os.path.join(protein_data_path, 'BioGRID multi vaildated physical interaction sets',
                                         'human', 'third_human_MV_pair.tsv')
        # pair_save_path=os.path.join(data_process_path,"pairs")
        protein_1, protein_2, interactions = read_pairs(protein_pair_path)
        return protein_1, protein_2, interactions
    elif dataset == 'multiple_species_01':
        print("process multiple_species paris ...")
        protein_pair_path = os.path.join(protein_data_path, 'multiple species dataset',
                                         'CeleganDrosophilaEcoli.actions.filtered.01.tsv')
        # pair_save_path=os.path.join(data_process_path,"pairs")
        protein_1, protein_2, interactions = read_pairs(protein_pair_path)
        return protein_1, protein_2, interactions
    elif dataset == 'multiple_species_10':
        print("process multiple_species paris ...")
        protein_pair_path = os.path.join(protein_data_path, 'multiple species dataset',
                                         'CeleganDrosophilaEcoli.actions.filtered.10.tsv')
        # pair_save_path=os.path.join(data_process_path,"pairs")
        protein_1, protein_2, interactions = read_pairs(protein_pair_path)
        return protein_1, protein_2, interactions
    elif dataset == 'multiple_species_25':
        print("process multiple_species paris ...")
        protein_pair_path = os.path.join(protein_data_path, 'multiple species dataset',
                                         'CeleganDrosophilaEcoli.actions.filtered.25.tsv')
        # pair_save_path=os.path.join(data_process_path,"pairs")
        protein_1, protein_2, interactions = read_pairs(protein_pair_path)
        return protein_1, protein_2, interactions
    elif dataset == 'multiple_species_40':
        print("process multiple_species paris ...")
        protein_pair_path = os.path.join(protein_data_path, 'multiple species dataset',
                                         'CeleganDrosophilaEcoli.actions.filtered.40.tsv')
        # pair_save_path=os.path.join(data_process_path,"pairs")
        protein_1, protein_2, interactions = read_pairs(protein_pair_path)
        return protein_1, protein_2, interactions
    elif dataset == 'multiple_species_full':
        print("process multiple_species paris ...")
        protein_pair_path = os.path.join(protein_data_path, 'multiple species dataset',
                                         'CeleganDrosophilaEcoli.actions.filtered.full.tsv')
        # pair_save_path=os.path.join(data_process_path,"pairs")
        protein_1, protein_2, interactions = read_pairs(protein_pair_path)
        return protein_1, protein_2, interactions
    elif dataset == 'DeepFE-PPI_core':
        print("process DeepFE-PPI_core paris ...")
        protein_pair_path = os.path.join(protein_data_path, 'yeast core dataset from DeepFE-PPI', 'action_pair.tsv')
        # pair_save_path=os.path.join(data_process_path,"pairs")
        protein_1, protein_2, interactions = read_pairs(protein_pair_path)
        return protein_1, protein_2, interactions
    elif dataset == 'virus-human':
        print("process virus-human paris ...")
        protein_pair_path = os.path.join(protein_data_path, 'virus-human interaction dataset', 'human_virus_pair.tsv')
        # pair_save_path=os.path.join(data_process_path,"pairs")
        protein_1, protein_2, interactions = read_pairs(protein_pair_path)
        return protein_1, protein_2, interactions


def process_proteins_for_DeepTrio(dataset='BioGRID_S', data_process_path='process/benchmarks_DeepTrio'):
    data_process_path_temp = os.path.join(data_process_path, dataset)
    protein_data_path = os.path.join('data', 'benchmarks_DeepTrio')
    if dataset == 'BioGRID_S':
        print("process BioGRID_S dataset ...")
        protein_data_path = os.path.join(protein_data_path, 'BioGRID multi vaildated physical interaction sets', 'sa',
                                         'double_sa_MV_database.tsv')


    elif dataset == 'BioGRID_H':
        print("process BioGRID_H dataset ...")
        protein_data_path = os.path.join(protein_data_path, 'BioGRID multi vaildated physical interaction sets',
                                         'human',
                                         'double_human_MV_database.tsv')

    elif dataset == 'multiple_species_01' or dataset == 'multiple_species_10' or dataset == 'multiple_species_25' or dataset == 'multiple_species_40' or dataset == 'multiple_species_full':
        print("process multiple_species dataset ...")
        data_process_path_temp = os.path.join(data_process_path, 'multiple_species')
        protein_data_path = os.path.join(protein_data_path, 'multiple species dataset',
                                         'CeleganDrosophilaEcoli.dictionary.tsv')


    elif dataset == 'DeepFE-PPI_core':
        print("process DeepFE-PPI_core dataset ...")
        protein_data_path = os.path.join(protein_data_path, 'yeast core dataset from DeepFE-PPI',
                                         'action_dictionary.tsv')

    elif dataset == 'virus-human':
        print("process virus-human dataset ...")
        protein_data_path = os.path.join(protein_data_path, 'virus-human interaction dataset',
                                         'human_virus_database.tsv')

    contact_path = os.path.join(data_process_path_temp, "contacts")
    seq_save_file = os.path.join(data_process_path_temp, "seq.pkl")
    names, seqs = read_proteins(protein_data_path)
    contact_predict(names, seqs, contact_path)
    with open(seq_save_file, "wb") as f:
        pickle.dump([names, seqs], f)


def read_proteins_for_DeepTrio(dataset='BioGRID_S', data_process_path='process/benchmarks_DeepTrio'):
    if dataset == 'multiple_species_01' or dataset == 'multiple_species_10' or dataset == 'multiple_species_25' or dataset == 'multiple_species_40' or dataset == 'multiple_species_full':
        dataset = 'multiple_species'
    data_process_path = os.path.join(data_process_path, dataset)
    contact_path = os.path.join(data_process_path, "contacts")
    print("read contacts ...")
    contacts = {}
    contact_files = os.listdir(contact_path)
    for i in tqdm(range(len(contact_files))):
        file = contact_files[i]
        protein_name = file[:-12]
        # print(file, protein_name)
        contact = np.load(os.path.join(contact_path, file))
        # print(contact.shape)
        contacts[protein_name] = contact.squeeze(0)
    return contacts


if __name__ == '__main__':
    # name = "test"
    # seq = "KALTARQQEVFDLIRDHISQTGMPPTRAEIAQRLGFRSPNAAEEHLKALARKGVIEIVSGASRGIRLLQEEKALTARQQEVFDLIRDHISQTGMPPTRAEIAQRLGFRSPNAAEEHLKALARKGVIEIVSGASRGIRLLQEEKALTARQQEVFDLIRDHISQTGMPPTRAEIAQRLGFRSPNAAEEHLKALARKGVIEIVSGASRGIRLLQEEKALTARQQEVFDLIRDHISQTGMPPTRAEIAQRLGFRSPNAAEEHLKALARKGVIEIVSGASRGIRLLQEEKALTARQQEVFDLIRDHISQTGMPPTRAEIAQRLGFRSPNAAEEHLKALARKGVIEIVSGASRGIRLLQEEKALTARQQEVFDLIRDHISQTGMPPTRAEIAQRLGFRSPNAAEEHLKALARKGVIEIVSGASRGIRLLQEEKALTARQQEVFDLIRDHISQTGMPPTRAEIAQRLGFRSPNAAEEHLKALARKGVIEIVSGASRGIRLLQEEKALTARQQEVFDLIRDHISQTGMPPTRAEIAQRLGFRSPNAAEEHLKALARKGVIEIVSGASRGIRLLQEEKALTARQQEVFDLIRDHISQTGMPPTRAEIAQRLGFRSPNAAEEHLKALARKGVIEIVSGASRGIRLLQEEKALTARQQEVFDLIRDHISQTGMPPTRAEIAQRLGFRSPNAAEEHLKALARKGVIEIVSGASRGIRLLQEEKALTARQQEVFDLIRDHISQTGMPPTRAEIAQRLGFRSPNAAEEHLKALARKGVIEIVSGASRGIRLLQEEKALTARQQEVFDLIRDHISQTGMPPTRAEIAQRLGFRSPNAAEEHLKALARKGVIEIVSGASRGIRLLQEEKALTARQQEVFDLIRDHISQTGMPPTRAEIAQRLGFRSPNAAEEHLKALARKGVIEIVSGASRGIRLLQEEKALTARQQEVFDLIRDHISQTGMPPTRAEIAQRLGFRSPNAAEEHLKALARKGVIEIVSGASRGIRLLQEEKALTARQQEVFDLIRDHISQTGMPPTRAEIAQRLGFRSPNAAEEHLKALARKGVIEIVSGASRGIRLLQEEKALTARQQEVFDLIRDHISQTGMPPTRAEIAQRLGFRSPNAAEEHLKALARKGVIEIVSGASRGIRLLQEEKALTARQQEVFDLIRDHISQTGMPPTRAEIAQRLGFRSPNAAEEHLKALARKGVIEIVSGASRGIRLLQEEKALTARQQEVFDLIRDHISQTGMPPTRAEIAQRLGFRSPNAAEEHLKALARKGVIEIVSGASRGIRLLQEEKALTARQQEVFDLIRDHISQTGMPPTRAEIAQRLGFRSPNAAEEHLKALARKGVIEIVSGASRGIRLLQEEKALTARQQEVFDLIRDHISQTGMPPTRAEIAQRLGFRSPNAAEEHLKALARKGVIEIVSGASRGIRLLQEEKALTARQQEVFDLIRDHISQTGMPPTRAEIAQRLGFRSPNAAEEHLKALARKGVIEIVSGASRGIRLLQEEKALTARQQEVFDLIRDHISQTGMPPTRAEIAQRLGFRSPNAAEEHLKALARKGVIEIVSGASRGIRLLQEEKALTARQQEVFDLIRDHISQTGMPPTRAEIAQRLGFRSPNAAEEHLKALARKGVIEIVSGASRGIRLLQEEKALTARQQEVFDLIRDHISQTGMPPTRAEIAQRLGFRSPNAAEEHLKALARKGVIEIVSGASRGIRLLQEEKALTARQQEVFDLIRDHISQTGMPPTRAEIAQRLGFRSPNAAEEHLKALARKGVIEIVSGASRGIRLLQEE"
    # contact_predict([name], [seq + seq + seq], "test")
    # exit(0)

    import random

    dataset = ['BioGRID_S', 'BioGRID_H', 'multiple_species', 'DeepFE-PPI_core', 'virus-human']
    # dataset.reverse()
    random.shuffle(dataset)
    for set in dataset:
        process_proteins_for_DeepTrio(set)
        read_ppi_pairs_for_DeepTrio(set)
        contacts = read_proteins_for_DeepTrio(set)
        print(len(contacts))
