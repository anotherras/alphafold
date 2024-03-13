import collections
import itertools
import dataclasses
import sys

from typing import Dict, Iterable, List, Optional, Sequence, Tuple, Set

DeletionMatrix = Sequence[Sequence[int]]


def _keep_line(line: str, seqnames) -> bool:
    """Function to decide which lines to keep."""
    if not line.strip():
        return True
    if line.strip() == '//':  # End tag
        return True
    if line.startswith('# STOCKHOLM'):  # Start tag
        return True
    if line.startswith('#=GC RF'):  # Reference Annotation Line
        return True
    if line[:4] == '#=GS':  # Description lines - keep if sequence in list.
        _, seqname, _ = line.split(maxsplit=2)
        return seqname in seqnames
    elif line.startswith('#'):  # Other markup - filter out
        return False
    else:  # Alignment data - keep if sequence in list.
        seqname = line.partition(' ')[0]
        return seqname in seqnames


def deduplicate_stockholm_msa(stockholm_msa: str) -> str:
    """Remove duplicate sequences (ignoring insertions wrt query)."""
    sequence_dict = collections.defaultdict(str)

    # First we must extract all sequences from the MSA.
    for line in stockholm_msa.splitlines():
        # Only consider the alignments - ignore reference annotation, empty lines,
        # descriptions or markup.
        if line.strip() and not line.startswith(('#', '//')):
            line = line.strip()
            seqname, alignment = line.split()
            sequence_dict[seqname] += alignment

    # sequence_dict = { 'name' : seq }
    seen_sequences = set()
    seqnames = set()
    # First alignment is the query.
    query_align = next(iter(sequence_dict.values()))  # 第一个为query  'RKDD'
    mask = [c != '-' for c in query_align]  # Mask is False for insertions. [True,True,True,True ]
    for seqname, alignment in sequence_dict.items():
        # Apply mask to remove all insertions from the string.
        masked_alignment = ''.join(itertools.compress(alignment, mask))  # 对齐后为-的地方删除
        if masked_alignment in seen_sequences:
            continue
        else:
            seen_sequences.add(masked_alignment)
            seqnames.add(seqname)

    filtered_lines = []
    for line in stockholm_msa.splitlines():
        if _keep_line(line, seqnames):
            filtered_lines.append(line)

    return '\n'.join(filtered_lines) + '\n'


def remove_empty_columns_from_stockholm_msa(stockholm_msa: str) -> str:
    """Removes empty columns (dashes-only) from a Stockholm MSA."""
    processed_lines = {}
    unprocessed_lines = {}
    for i, line in enumerate(stockholm_msa.splitlines()):
        if line.startswith('#=GC RF'):
            reference_annotation_i = i
            reference_annotation_line = line
            # Reached the end of this chunk of the alignment. Process chunk.
            _, _, first_alignment = line.rpartition(' ')
            mask = []
            for j in range(len(first_alignment)):
                for _, unprocessed_line in unprocessed_lines.items():
                    prefix, _, alignment = unprocessed_line.rpartition(' ')
                    if alignment[j] != '-':
                        mask.append(True)
                        break
                    else:  # Every row contained a hyphen - empty column.
                        mask.append(False)
            # Add reference annotation for processing with mask.
            unprocessed_lines[reference_annotation_i] = reference_annotation_line

            if not any(mask):  # All columns were empty. Output empty lines for chunk.
                for line_index in unprocessed_lines:
                    processed_lines[line_index] = ''
            else:
                for line_index, unprocessed_line in unprocessed_lines.items():
                    prefix, _, alignment = unprocessed_line.rpartition(' ')
                    masked_alignment = ''.join(itertools.compress(alignment, mask))
                    processed_lines[line_index] = f'{prefix} {masked_alignment}'

            # Clear raw_alignments.
            unprocessed_lines = {}
        elif line.strip() and not line.startswith(('#', '//')):
            unprocessed_lines[i] = line
        else:
            processed_lines[i] = line
    return '\n'.join((processed_lines[i] for i in range(len(processed_lines))))


def _convert_sto_seq_to_a3m(
        query_non_gaps, sto_seq: str):
    for is_query_res_non_gap, sequence_res in zip(query_non_gaps, sto_seq):
        if is_query_res_non_gap:
            yield sequence_res
        elif sequence_res != '-':
            yield sequence_res.lower()


def convert_stockholm_to_a3m(stockholm_format: str,
                             max_sequences=None,
                             remove_first_row_gaps: bool = True) -> str:
    """Converts MSA in Stockholm format to the A3M format."""
    descriptions = {}
    sequences = {}
    reached_max_sequences = False

    for line in stockholm_format.splitlines():
        reached_max_sequences = max_sequences and len(sequences) >= max_sequences
        if line.strip() and not line.startswith(('#', '//')):
            # Ignore blank lines, markup and end symbols - remainder are alignment
            # sequence parts.
            seqname, aligned_seq = line.split(maxsplit=1)
            if seqname not in sequences:
                if reached_max_sequences:
                    continue
                sequences[seqname] = ''
            sequences[seqname] += aligned_seq

    for line in stockholm_format.splitlines():
        if line[:4] == '#=GS':
            # Description row - example format is:
            # #=GS UniRef90_Q9H5Z4/4-78            DE [subseq from] cDNA: FLJ22755 ...
            columns = line.split(maxsplit=3)
            seqname, feature = columns[1:3]
            value = columns[3] if len(columns) == 4 else ''
            if feature != 'DE':
                continue
            if reached_max_sequences and seqname not in sequences:
                continue
            descriptions[seqname] = value
            if len(descriptions) == len(sequences):
                break

    # Convert sto format to a3m line by line
    a3m_sequences = {}
    if remove_first_row_gaps:
        # query_sequence is assumed to be the first sequence
        query_sequence = next(iter(sequences.values()))
        query_non_gaps = [res != '-' for res in query_sequence]
    for seqname, sto_sequence in sequences.items():
        # Dots are optional in a3m format and are commonly removed.
        out_sequence = sto_sequence.replace('.', '')
        if remove_first_row_gaps:
            out_sequence = ''.join(
                _convert_sto_seq_to_a3m(query_non_gaps, out_sequence))
        a3m_sequences[seqname] = out_sequence

    fasta_chunks = (f">{k} {descriptions.get(k, '')}\n{a3m_sequences[k]}"
                    for k in a3m_sequences)
    return '\n'.join(fasta_chunks) + '\n'  # Include terminating newline.


def parse_stockholm(stockholm_string: str):
    """Parses sequences and deletion matrix from stockholm format alignment.

    Args:
      stockholm_string: The string contents of a stockholm file. The first
        sequence in the file should be the query sequence.

    Returns:
      A tuple of:
        * A list of sequences that have been aligned to the query. These
          might contain duplicates.
        * The deletion matrix for the alignment as a list of lists. The element
          at `deletion_matrix[i][j]` is the number of residues deleted from
          the aligned sequence i at residue position j.
        * The names of the targets matched, including the jackhmmer subsequence
          suffix.
    """
    name_to_sequence = collections.OrderedDict()
    for line in stockholm_string.splitlines():
        line = line.strip()
        if not line or line.startswith(('#', '//')):
            continue
        name, sequence = line.split()
        if name not in name_to_sequence:
            name_to_sequence[name] = ''
        name_to_sequence[name] += sequence

    msa = []
    deletion_matrix = []

    query = ''
    keep_columns = []
    for seq_index, sequence in enumerate(name_to_sequence.values()):
        if seq_index == 0:
            # Gather the columns with gaps from the query
            query = sequence
            keep_columns = [i for i, res in enumerate(query) if res != '-']

        # Remove the columns with gaps in the query from all sequences.
        aligned_sequence = ''.join([sequence[c] for c in keep_columns])

        msa.append(aligned_sequence)

        # Count the number of deletions w.r.t. query.
        deletion_vec = []
        deletion_count = 0
        for seq_res, query_res in zip(sequence, query):
            if seq_res != '-' or query_res != '-':
                if query_res == '-':
                    deletion_count += 1
                else:
                    deletion_vec.append(deletion_count)
                    deletion_count = 0
        deletion_matrix.append(deletion_vec)

    return Msa(sequences=msa,
               deletion_matrix=deletion_matrix,
               descriptions=list(name_to_sequence.keys()))


@dataclasses.dataclass(frozen=True)
class Msa:
    """Class representing a parsed MSA file."""
    sequences: Sequence[str]
    deletion_matrix: DeletionMatrix
    descriptions: Sequence[str]

    def __post_init__(self):
        if not (len(self.sequences) ==
                len(self.deletion_matrix) ==
                len(self.descriptions)):
            raise ValueError(
                'All fields for an MSA must have the same length. '
                f'Got {len(self.sequences)} sequences, '
                f'{len(self.deletion_matrix)} rows in the deletion matrix and '
                f'{len(self.descriptions)} descriptions.')

    def __len__(self):
        return len(self.sequences)

    def truncate(self, max_seqs: int):
        return Msa(sequences=self.sequences[:max_seqs],
                   deletion_matrix=self.deletion_matrix[:max_seqs],
                   descriptions=self.descriptions[:max_seqs])


# jackhmmer_uniref90_result['sto']
# with open(
#         r'C:\Users\mzm\pythonProject\linux\project\Alphafold-main\alphafold-main\test_alphafold2\test_data\jackhmmer_output.sto') as stockholm_msa:
#     x = stockholm_msa.read()
# y = parse_stockholm(x)
#
# print(y)
#
#
# x = deduplicate_stockholm_msa(x)
#
# x = remove_empty_columns_from_stockholm_msa(x)
# #
# with open(
#         r'C:\Users\mzm\pythonProject\linux\project\Alphafold-main\alphafold-main\test_alphafold2\test_data\clear_from_jackhmmer.sto',
#         'w') as file:
#     file.write(x)
#
# x = convert_stockholm_to_a3m(x)
#
# with open(r'C:\Users\mzm\pythonProject\linux\project\Alphafold-main\alphafold-main\test_alphafold2\test_data\convert2a3m.a3m','w') as file:
#     file.write(x)


#
# y = parse_stockholm(x)
# print(y)
#
# import sys
#
# sys.exit(0)
# x = convert_stockholm_to_a3m(x)
# with open(
#         r'C:\Users\mzm\pythonProject\linux\project\Alphafold-main\alphafold-main\test_alphafold2\data\change2a3m.a3m',
#         'w') as file:
#     file.write(x)
# with open(r'C:\Users\mzm\pythonProject\linux\project\Alphafold-main\alphafold-main\test_alphafold2\data\test.sto','r') as file:
#     x = file.read()
# y = parse_stockholm(x)
# print(y)

from alphafold.data.parsers import parse_hhr, parse_fasta
from alphafold.data.templates import HhsearchHitFeaturizer

# from alphafold.data.templates import HhsearchHitFeaturizer, _process_single_hit
#

with open(
        r'C:\Users\mzm\pythonProject\linux\project\Alphafold-main\alphafold-main\test_alphafold2\test_data\hhsearch_output.hhr') as file:
    pdb_template_hits = file.read()

with open(
        r'C:\Users\mzm\pythonProject\linux\project\Alphafold-main\alphafold-main\test_alphafold2\test_data\input.fasta') as f:
    input_fasta_str = f.read()
input_seqs, input_descs = parse_fasta(input_fasta_str)

input_sequence = input_seqs[0]

x = parse_hhr(pdb_template_hits)
print(x)
print('++++++++++++++++')
h = HhsearchHitFeaturizer(
    mmcif_dir=r'C:\Users\mzm\pythonProject\linux\project\Alphafold-main\alphafold-main\test_alphafold2\test_data',
    max_template_date='2021-2-10',
    max_hits=10,
    kalign_binary_path=None,
    release_dates_path=None,
    obsolete_pdbs_path=None)
h.get_templates(input_sequence, x)
