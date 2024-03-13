from pathlib import Path

import os
import re
from pathlib import Path
from Bio.PDB import PDBParser, Polypeptide, Structure, Model
from Bio.PDB.PDBIO import PDBIO

from collections import defaultdict


def gen_3to1():
    aa_3to1 = {}
    for i in range(20):
        aa_3to1[Polypeptide.index_to_three(i)] = Polypeptide.index_to_one(i)
    return aa_3to1


def parse_pdb_sequence(pdb, aa_3to1=gen_3to1(), chainids='ABCDEFGHIJKLMNOPQRST'):
    type_ = None
    dna_number = -1
    protein_number = -1
    other = -1
    # 读取两个PDB文件
    alias2normal = {'HIS': 'HIS', 'HIE': 'HIS', 'HID': 'HIS', 'HIP': 'HIS',
                    'CYS': 'CYS', 'CYX': 'CYS', 'CYM': 'CYS',
                    'ASP': 'ASP', 'ASH': 'ASP',
                    'GLU': 'GLU', 'GLH': 'GLU',
                    'LYH': 'LYH', 'LYN': 'LYH',
                    'MSE': 'MET'}
    aa_list = set(list(aa_3to1.keys()) + list(alias2normal.keys()))

    dna_list = ['DA', 'DG', 'DC', 'DT', 'DAN', 'DGN', 'DCN', 'DTN', 'DA3', 'DG3', 'DC3', 'DT3', 'DA5', 'DG5', 'DC5',
                'DT5']
    dna_list += ['MPG']  # user defined base AMP-T

    sequences = defaultdict(str)
    resnames = defaultdict(dict)
    start_res_ids = defaultdict(int)
    parser = PDBParser(QUIET=True)
    structure = parser.get_structure('protein', pdb)
    model = structure[0]
    for i, chain in enumerate(model):
        if chain.id == ' ': chain.id = chainids[i]  # 若无链信息，自动按ABCDEFG来补。临时方案
        # sequences[chain.id] = ''
        # resnames[chain.id] = {}
        min_res_id = 1e10
        for idx, res in enumerate(chain):
            min_res_id = min(min_res_id, res.id[1])  # minimum residue id
            if res.resname in aa_list:
                type_ = 'protein_'
                resnames['protein_' + chain.id][res.id[1]] = res.resname
                try:
                    sequences['protein_' + chain.id] += aa_3to1[res.resname]
                except:
                    sequences['protein_' + chain.id] += aa_3to1[alias2normal[res.resname]]
            elif res.resname.strip(' ') in dna_list:
                type_ = 'dna_'
                resnames['dna_' + chain.id][res.id[1]] = res.resname
                sequences['dna_' + chain.id] += res.resname.strip(' ')[1]  # dna碱基命名，碱基都在字符串位置1
            else:
                type_ = 'other_'
                resnames['other_' + chain.id][res.id[1]] = res.resname
                sequences['other_' + chain.id] += (res.resname + '/')

        start_res_ids[type_ + chain.id] = min_res_id  # 每条链第一个残基编号
    # if chain_id: #如果指定了chain，返回该chain对应的信息
    #     return sequences[chain_id], resnames[chain_id], start_res_ids[chain_id]
    dna_list = [i[-1] for n, i in enumerate(sequences.keys()) if 'dna' in i]
    protein_chain_list = [i[-1] for n, i in enumerate(sequences.keys()) if 'protein' in i]
    return sequences, resnames, start_res_ids, dna_list, protein_chain_list


def write_resort_pdb(path):
    sequences, resnames, start_res_ids, dna_order, protein_chain_order = parse_pdb_sequence(path)

    if not isinstance(path, Path):
        path = Path(path)

    dna = []
    protein = []
    other = []

    type_ = None
    with open(path, 'r') as file:
        readline = file.readline()
        while readline:
            if len(readline) > 50:
                if readline[21] in dna_order:
                    dna.append(readline)
                    type_ = 'dna'
                elif readline[21] in protein_chain_order:
                    protein.append(readline)
                    type_ = 'protein'
                else:
                    other.append(readline)
                    type_ = 'other'
            else:
                if type_ == 'dna':
                    dna.append(readline)
                elif type_ == 'protein':
                    protein.append(readline)
                else:
                    other.append(readline)
            readline = file.readline()
    with open(r"C:\Users\mzm\Desktop\project\data\6dt122222222222222222.pdb", 'w') as file:
        for i in protein:
            file.write(i)
        for i in dna:
            file.write(i)
        for i in other:
            file.write(i)
path = r"C:\Users\mzm\Desktop\project\data\6dt1.pdb"
sequences, resnames, start_res_ids, dna_order, protein_chain_order = parse_pdb_sequence(path)
print(sequences)
print(resnames)
print(start_res_ids)
print(dna_order)
print(protein_chain_order)
l_dna = defaultdict(int)
dna_start = 0
l_pro = defaultdict(int)
pro_start = 0
for k, v in sequences.items():
    if 'dna' in k:
        l_dna[k] = dna_start + len(sequences[k])
        dna_start += len(sequences[k])
    elif 'protein' in k:
        l_pro[k] = pro_start + len(sequences[k])
        pro_start += len(sequences[k])

print(l_dna)
print(l_pro)

write_resort_pdb(r"C:\Users\mzm\Desktop\project\data\6dt1.pdb")

import os
def g():
    print(os.getcwd())