import os
import pandas as pd
from Bio import SeqIO

result = pd.DataFrame(columns=['id', 'seq', 'annotation'])
fasta_dir = "/shanjunjie/protein/fasta/"

for file in os.listdir(fasta_dir):
    if file.endswith(".faa"):
        fasta_file = os.path.join(fasta_dir, file)
        seqs = SeqIO.parse(fasta_file, "fasta")
        for seq in seqs:
            annotations = seq.description.split("[")[1].split("]")[0]
            result = pd.concat([result, pd.DataFrame({'id': seq.id, 'seq': str(seq.seq), 'annotation': annotations}, index=[0])], ignore_index=True)
    
result.to_csv("/shanjunjie/ProteinMultiClass/data/annotation.csv", index=False)