import torch
import pandas as pd

classes = torch.load("./data/result/labelid2label.pt")
seqs = torch.load("./data/result/seqs.pt")
logit = torch.load("./data/result/logits.pt").cpu()
label = torch.load("./data/result/labels.pt").cpu()
df = pd.DataFrame(logit.numpy(), columns=classes)

print(f"seqs: {len(seqs)}")
print(f"logit: {logit.shape}")

part = pd.DataFrame(columns=["sequence", "pred", "score", "label"])
for i, seq in enumerate(seqs):
    part.loc[i] = [seq, df.iloc[i].idxmax(), df.iloc[i].max(), classes[int(label[i])]]
part.to_csv("part_result.csv", index=False)

whole = pd.DataFrame(columns=["sequence", "pred", "score", "label"]+list(classes))
for i, seq in enumerate(seqs):
    whole.loc[i] = [seq, df.iloc[i].idxmax(), df.iloc[i].max(), classes[int(label[i])]] + df.iloc[i].tolist()
df.to_csv("full_result.csv", index=False)

