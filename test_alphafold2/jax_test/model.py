import torch

model_data = torch.load(r"C:\Users\mzm\Desktop\esm_msa1b_t12_100M_UR50S.pt")

print(type(model_data))
print(model_data.keys())
print(type(model_data['args']))

print(type(model_data["args"].arch))

print(model_data["args"].arch)

proteinseq_toks = {
    'toks': ['L', 'A', 'G', 'V', 'S', 'E', 'R', 'T', 'I', 'D', 'P', 'K', 'Q', 'N', 'F', 'Y', 'M', 'H', 'W', 'C', 'X',
             'B', 'U', 'Z', 'O', '.', '-']
}
name = ''
if name in ("MSA Transformer", "msa_transformer"):
    standard_toks = proteinseq_toks["toks"]
    prepend_toks = ("<cls>", "<pad>", "<eos>", "<unk>")
    append_toks = ("<mask>",)
    prepend_bos = True
    append_eos = False
    use_msa = True

xxx

import esm

esm.pretrained.esm_msa1b_t12_100M_UR50S()
