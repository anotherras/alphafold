from pathlib import Path
from prettyprinter import pprint
import os
import argparse
import subprocess
from modify_mol2 import gen_fix_dict, run_command_and_check
from utils.command_utils import echo_selection_txt


# reference tutorial
# https://jerkwin.github.io/2017/09/20/GROMACS%E9%9D%9E%E6%A0%87%E5%87%86%E6%AE%8B%E5%9F%BA%E6%95%99%E7%A8%8B2-%E8%8A%8B%E8%9E%BA%E6%AF%92%E7%B4%A0%E5%B0%8F%E8%82%BD%E5%AE%9E%E4%BE%8B/
# generate topology file for special residue using antechamber
# for
atom_rename = {'CNT': '-C',
               'NCT': '+N',
               "O3'P": "-O3'",
               "P5'O": "P5'+"}

def deduplicate_list(lst):
    seen = set()
    result = []
    for item in lst:
        if item not in seen:
            result.append(item)
            seen.add(item)
    return result


def special_res_top(pdb_path, mol2fix=None, atom_type='amber'):
    '''
    Parameters
    ----------
    pdb_path: structure of the special residue. Note to
    mol2fix: file to fix mol2 file bonds, which will influence the topology
    atom_type: bond type for special forcefields

    Returns: generated top file path
    -------
    '''
    if not os.path.exists(pdb_path):
        print('The specified residue structure is not valid!')
        return

    mol_dir = os.path.dirname(pdb_path)
    mol_name = os.path.splitext(os.path.basename(pdb_path))[0]

    mol2_path = mol_dir + '.mol2'
    mod_path = mol_dir + '.mod'
    prm_path = mol_dir + 'prm'
    crd_path = mol_dir + 'crd'

    charge = 0
    if mol2fix:
        charge = gen_fix_dict(mol2fix)[0]['charge']
    command = f'antechamber -i {pdb_path} -fi pdb -o {mol2_path} -fo mol2 -c bcc -nc {charge} -pf y -at {atom_type} -ek "maxcyc=0 qm_theory=\'PM6\'"'
    os.system(command)
    print('!!charge', charge)

    if mol2fix:
        command = f'python modify_mol2.py -f {mol2_path} -fix {mol2fix}'
        os.system(command)

    command = f'parmchk2 -i {mol2_path} -f mol2 -o {mod_path}'
    os.system(command)

    leapin_path = os.path.join(os.path.dirname(pdb_path), 'leap.in')
    with open(leapin_path, 'w') as f:
        f.write(f'source leaprc.protein.ff14SB\n')
        f.write(f'source leaprc.DNA.OL15\n')
        f.write(f'source leaprc.gaff\n')
        f.write(f'loadamberparams {mod_path}\n')
        f.write(f'mol=loadmol2 {mol2_path}\n')
        f.write(f'saveamberparm mol {prm_path} {crd_path}\n')
        f.write(f'quit\n')

    command = f'tleap -f {leapin_path} > output.txt 2>&1'
    os.system(command)
    with open("output.txt", "r") as output_file:
        lines = output_file.readlines()[-1].strip()
    os.remove("output.txt")

    command = f'python acpype.py -p {prm_path} -x {crd_path} -d -b {mol_dir}/{mol_name.upper()}'
    os.system(command)

    print('\nSpecial residue topology completed!!')
    print(lines)

    return f'{mol_dir}/{mol_name.upper()}_GMX.top'


def rename_file(file_path):
    path = Path(file_path)
    parent_dir = path.parent
    new_name = path.name.split('_')[0].lower() + '.rtp'
    new_path = str(parent_dir / new_name)
    # command = f'mv {file_path} {new_path}'
    # run_command_and_check(command)
    return new_path


def write_dict(infos_dict, file_path):
    with open(file_path, 'w') as file:
        for k, infos in infos_dict.items():
            file.write(k + '\n')
            for info in infos:
                file.write(info + '\n')
            file.write('\n')


def extract_info(file_path):
    info_list = ['[ atoms ]', '[ bonds ]', '[ angles ]', '[ dihedrals ] ; propers', '[ dihedrals ] ; improper']

    reading_target_section = False

    section_part = {info: [] for info in info_list}

    with open(file_path, 'r') as file:
        for line in file:
            if any(info in line for info in info_list):
                current_info = next(info for info in info_list if info in line)
                reading_target_section = True
                continue

            elif not line.strip() and reading_target_section:
                reading_target_section = False

            elif reading_target_section:
                section_part[current_info].append(line)

    return section_part


def get_caps(atom_info, i:int=1, j:int=-1):
    n = 0 # deleted atom number
    N = 0 # total atom number
    icap_charge, jcap_charge = 0.0, 0.0
    for line in atom_info[1:]:
        line = line.rstrip()
        N = max(N, int(line[0: 0 + 6]))
        if j==-1: j = N #j=-1 for the last atom

        nr = int(line[0: 0 + 6].strip()) < i or int(line[0: 0 + 6].strip()) > j
        if int(line[0: 0 + 6].strip()) < i:
            n += 1
            icap_charge += float(line[34: 34 + 13].strip())

        if int(line[0: 0 + 6].strip()) > j:
            n += 1
            jcap_charge += float(line[34: 34 + 13].strip())

    # mean_charge = cap_charge / (N - n)
    return N, n, icap_charge, jcap_charge # charge of capping groups will be added to the neareast atom of the residue


def process_atom(atom_info, i, j, add_charge=0.0):
    N, n, qi, qj = get_caps(atom_info, i, j)
    mean_add_charge = add_charge / (N - n)

    # print('mean_charge:', mean_charge)
    cols = ['atoms', 'type', 'charge', 'nrs', ';', 'qtot', 'bond_type']
    start_width = [(23, 6), (6, 5), (34, 13), (0, 6), (60, 2), (62, 5), (67, 10)]
    new_cols = ''
    for idx, ((_, width), col) in enumerate(zip(start_width, cols)):
        if idx == 0:
            new_cols += ';' + col.rjust(width - 1, ' ')
            continue
        new_cols += col.rjust(width, ' ')
    info_list = [new_cols]
    atom_names = ['CNT', 'NCT', "O3'P", "P5'O"]
    atom2nr = {}
    atom_set = set(atom_rename.keys())
    for line in atom_info[1:]:
        line = line.rstrip()
        nr = int(line[0: 0 + 6].strip())
        atom_name = line[start_width[0][0]: start_width[0][0] + start_width[0][1]].strip()
        if atom_name in atom_names:
            atom2nr[atom_name] = nr
        if nr < i or nr > j: continue
        atom_set.add(atom_name)
        data = ''
        for idx, (start, width) in enumerate(start_width):
            if idx == 2 and nr == i:  # charge
                data += str(round(float(line[start: start + width].strip()) + qi + mean_add_charge, 6)).rjust(width, ' ')
                continue
            if idx == 2 and nr == j:  # charge
                data += str(round(float(line[start: start + width].strip()) + qj + mean_add_charge, 6)).rjust(width, ' ')
                continue
            if idx == 2:
                data += str(round(float(line[start: start + width].strip()) + mean_add_charge, 6)).rjust(width, ' ')
                continue
            data += line[start: start + width]
        info_list.append(data)
    return info_list, atom2nr, atom_set


def process_bond(bond_info, atom2nr, atom_set, i, j):
    cols = ['atomi', 'atomj', 'r', 'k', '']
    start_width = [(47, 7), (54, 9), (17, 14), (31, 14), (45, 2)]
    new_cols = ''
    for idx, ((_, width), col) in enumerate(zip(start_width, cols)):
        if idx == 0:
            new_cols += ';' + col.rjust(width - 1, ' ')
            continue
        new_cols += col.rjust(width, ' ')
    info_list = [new_cols]
    bond_info = deduplicate_list(bond_info)
    for line in bond_info[1:]:
        line = line.rstrip()
        ai = int(line[0: 0 + 6].strip()) < i
        aj = int(line[6: 6 + 7].strip()) > j
        bond_l = line[start_width[0][0]: start_width[0][0] + start_width[0][1]].strip() not in atom_set
        bond_r = line[start_width[1][0]: start_width[1][0] + start_width[1][1]].replace('-', ' ').strip() not in atom_set
        if bond_l or bond_r:
            continue
        data = ''
        for idx, (start, width) in enumerate(start_width):
            if idx in [0, 1]:
                atom = line[start: start + width].replace('-', ' ').strip()
                if atom in atom_rename:
                    atom = atom_rename[atom]
                data += atom.rjust(width, ' ')
                continue
            data += line[start: start + width]
        info_list.append(data)
    return info_list


def process_angle(angle_info, i, j):
    cols = ['atomi', 'atomj', 'atomk', 'theta', 'cth', '']
    start_width = [(57, 7), (64, 8), (72, 12), (27, 14), (41, 14), (55, 2)]
    new_cols = ''
    for idx, ((_, width), col) in enumerate(zip(start_width, cols)):
        if idx == 0:
            new_cols += ';' + col.rjust(width - 1, ' ')
            continue
        new_cols += col.rjust(width, ' ')
    info_list = [new_cols]
    angle_info = deduplicate_list(angle_info)
    for line in angle_info[1:]:
        line = line.rstrip()
        ai = int(line[0: 0 + 6].strip()) < i or int(line[0: 0 + 6].strip()) > j
        aj = int(line[6: 6 + 7].strip()) < i or int(line[6: 6 + 7].strip()) > j
        ak = int(line[13: 13 + 7].strip()) < i or int(line[13: 13 + 7].strip()) > j
        if ai or aj or ak:
            continue
        data = ''
        for idx, (start, width) in enumerate(start_width):
            if idx in [1, 2]:
                data += line[start: start + width].replace('-', ' ').strip().rjust(width, ' ')
                continue
            data += line[start: start + width]
        info_list.append(data)
    return info_list


def process_dihedral(dihedral_info, i, j):
    cols = ['atomi', 'atomj', 'atomk', 'atoml', 'phase', 'kd', 'pn', '']
    start_width = [(59, 8), (67, 7), (74, 7), (81, 6), (34, 9), (43, 10), (53, 4), (57, 2)]
    new_cols = ''
    for idx, ((_, width), col) in enumerate(zip(start_width, cols)):
        if idx == 0:
            new_cols += ';' + col.rjust(width - 1, ' ')
            continue
        new_cols += col.rjust(width, ' ')
    info_list = [dihedral_info[0].rstrip()]
    info_list.append(new_cols)
    dihedral_info = deduplicate_list(dihedral_info)
    for line in dihedral_info[2:]:
        line = line.rstrip()
        ai = int(line[0: 0 + 6].strip()) < i or int(line[0: 0 + 6].strip()) > j
        aj = int(line[6: 6 + 7].strip()) < i or int(line[6: 6 + 7].strip()) > j
        ak = int(line[13: 13 + 7].strip()) < i or int(line[13: 13 + 7].strip()) > j
        al = int(line[20: 20 + 7].strip()) < i or int(line[20: 20 + 7].strip()) > j
        if ai or aj or ak or al:
            continue
        data = ''
        for idx, (start, width) in enumerate(start_width):
            if idx in [0, 1, 2]:
                data += line[start: start + width].replace('-', ' ').strip().rjust(width, ' ')
                continue
            data += line[start: start + width]
        info_list.append(data)
    return info_list


def process_improper(improper_info, atom_set, i, j):
    cols = ['atomi', 'atomj', 'atomk', 'atoml', 'phase', 'kd', 'pn', '']
    start_width = [(59, 8), (67, 7), (74, 7), (81, 6), (34, 9), (43, 10), (53, 4), (57, 2)]
    new_cols = ''
    for idx, ((_, width), col) in enumerate(zip(start_width, cols)):
        if idx == 0:
            new_cols += ';' + col.rjust(width - 1, ' ')
            continue
        new_cols += col.rjust(width, ' ')
    info_list = [improper_info[0].rstrip()]
    info_list.append(new_cols)
    improper_info = deduplicate_list(improper_info)
    for line in improper_info[2:]:
        line = line.rstrip()
        if line.startswith(';'):
            info_list.append(line)
            continue
        ai = int(line[0: 0 + 6].strip()) < i
        al = int(line[20: 20 + 7].strip()) > j
        is_delete = False
        for start, width in start_width[:4]:
            is_delete |= line[start: start + width].replace('-', ' ').strip() not in atom_set
        # if (ai or al) and (is_delete):
        if is_delete:
            continue
        data = ''
        for idx, (start, width) in enumerate(start_width):
            if idx in list(range(4)):
                atom = line[start: start + width].replace('-', ' ').strip()
                if atom in atom_rename:
                    atom = atom_rename[atom]
                data += atom.rjust(width, ' ')
                continue
            data += line[start: start + width]
        info_list.append(data)
    return info_list


def top2rtp(top_file_path, i=1, j=-1, add_charge=0.):
    rtp_file_path = rename_file(top_file_path)
    infos = extract_info(top_file_path)
    new_infos = {}

    N, _, _, _ = get_caps(infos['[ atoms ]'], i, j)
    if j == -1:
        j = N
    new_infos['[ bondedtypes ]'] = ['   1  1  9  4  1  3  1  0']
    mol_name = top_file_path.split('/')[-1].split('_')[0]
    new_infos[f'[ {mol_name} ]'] = []
    new_infos['[ atoms ]'], atom2nr, atom_set = process_atom(infos['[ atoms ]'], i, j, add_charge=add_charge)
    new_infos['[ bonds ]'] = process_bond(infos['[ bonds ]'], atom2nr, atom_set, i, j)
    new_infos['[ angles ]'] = process_angle(infos['[ angles ]'], i, j)
    new_infos['[ dihedrals ] ; propers'] = process_dihedral(infos['[ dihedrals ] ; propers'], i, j)
    new_infos['[ impropers ] ; '] = process_improper(infos['[ dihedrals ] ; improper'], atom_set, i, j)

    write_dict(new_infos, rtp_file_path)
    return rtp_file_path


if __name__ == '__main__':
    file = '../data/6dt1/c3a.pdb'
    fix = '../data/6dt1/mol2fix_c3a.txt'

    parser = argparse.ArgumentParser('Generate topology file for special residue', add_help=False)
    # Common parameters
    parser.add_argument('-f', '--pdb_file', default=file, type=str, help='Structure file for residue')
    parser.add_argument('-fix', '--fix', default=fix, type=str, help='File for mol2 fixing')
    parser.add_argument('-at', '--atom_type', default='amber', type=str, help='Atom type for force field')

    # The whole PDB file contains structure of the special residue, but the residue might be capped between start and end capping groups.
    # So, atoms before the beginning index=i, or those after the endding index=j, shoud not be contained in the final rtp file.
    # That's to say, only atoms in [i, j] will be kept. Here we use j=-1 for the last atom (j=N).

    # The start group and end group could be some other residue or a special chemical group like -NH2.

    # FOR AMBER FORCEFIELD, the story is much more simple for C terminus or .
    # If j not the last atom, then nothing special is needed,
    # If i>1, we ask the

    parser.add_argument('-i', '--start_idx', default=1, type=int, help='Start atom index of the special residue in PDB file')
    parser.add_argument('-j', '--end_idx', default=-1, type=int, help='End atom index of the special residue in PDB file')
    parser.add_argument('-c', '--charge', default=0., type=float, help='Forcefield/User added charge: -0.3079 for 5-ter nt, -0.6921 for 3-ter nt, 0.0 for other nt')


    args = parser.parse_args()

    top_file = special_res_top(pdb_path=args.pdb_file, mol2fix=args.fix, atom_type=args.atom_type)
    rtp_file = top2rtp(top_file, i=args.start_idx, j=args.end_idx, add_charge=args.charge)
    print(rtp_file)