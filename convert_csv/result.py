import argparse
import torch
import pandas as pd

def main(args):
    classes = torch.load("./labelid2label.pt")
    seqs = torch.load("./seqs.pt")
    seqs = [s for seq in seqs for s in tuple(seq)]
    logit = torch.load("./logits.pt").cpu()
    label = torch.load("./labels.pt").cpu()
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
    whole.to_csv("full_result.csv", index=False)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--seqs", type=str, default="./seqs.pt")
    parser.add_argument("--logits", type=str, default="./logits.pt")