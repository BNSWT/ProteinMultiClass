import lmdb
import os, sys
import pandas as pd
import torch

seq_dir = "/shanjunjie/ProteinMultiClass/scan/result/csv/domain"
repr_dir = "/shanjunjie/ProteinMultiClass/visual/repr"
lmdb_dir = "/shanjunjie/ProteinMultiClass/visual/lmdb/repr.lmdb"

env = lmdb.open(lmdb_dir, map_size=1099511627776)

for file in os.listdir(seq_dir):
    if file.endswith(".csv"):
        seq_file = os.path.join(seq_dir, file)
        repr_file = os.path.join(repr_dir, file).replace(".csv", ".pt")
        seqs = pd.read_csv(seq_file)['domain'].values.tolist()
        reprs = torch.load(repr_file)
        for i, seq in enumerate(seqs):
            with env.begin(write=True) as txn:
                txn.put(seq.encode(), reprs[i].cpu().numpy().tobytes())

