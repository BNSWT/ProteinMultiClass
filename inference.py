import torch
import os
os.environ['OPENBLAS_NUM_THREADS'] = '1'
from torch.utils.data import Dataset, DataLoader
from torch.optim.lr_scheduler import LambdaLR
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.callbacks.early_stopping import EarlyStopping
import pytorch_lightning as pl
import numpy as np
import pandas as pd
from torch.optim.lr_scheduler import StepLR
from transformers import EsmTokenizer, EsmForSequenceClassification
model_path = "./esm2_t33_650M_UR50D"

# add parameters
import argparse
# overasmpling_size, finetune_layer, lr, weight_decay, batch_size, sechdule_gamma
parser = argparse.ArgumentParser()
parser.add_argument('--oversampling_size', type=int, default=150)
parser.add_argument('--finetune_layer', type=int, default=5)
parser.add_argument('--lr', type=float, default=5e-5)
parser.add_argument('--weight_decay', type=float, default=0.01)
parser.add_argument('--batch_size', type=int, default=16)
parser.add_argument('--sechdule_gamma', type=float, default=0.9)
parser.add_argument('--device', type=int, default=0)
args = parser.parse_args()


class ProSeqDataset(Dataset):
    def __init__(self, df, tokenizer):
        self.df = df.copy()
        self.tokenizer = tokenizer
        self.max_len = df["sequence"].apply(lambda x: len(x)).max()
        self.seqs = list(self.df['sequence'])
        self.inputs = self.tokenizer(list(self.df['sequence']), return_tensors="pt", padding="max_length", truncation=True, max_length=self.max_len)
        self.mapped_label, self.labelid2label = df['label'].factorize()
        self.num_labels = df['label'].nunique()
                
    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        label = self.mapped_label[idx]
        seqs = self.seqs[idx]
        input_ids = self.inputs["input_ids"][idx].squeeze()
        attention_mask = self.inputs["attention_mask"][idx].squeeze()
        return seqs, input_ids, attention_mask, label
    
class ProFunCla(pl.LightningModule):
    def __init__(self, model, lr, weight_decay, finetune_layer, gama, oversampling, **kwargs):
        super().__init__()
        self.save_hyperparameters(ignore=["model"])
        self.model = model
        for n, p in self.model.named_parameters():
            if n.startswith("esm.encoder.layer"):
                if int(n.split(".")[3]) <= 33-finetune_layer:
                    p.requires_grad = False
        self.lr = lr
        self.weight_decay = weight_decay
        self.gama = gama
        self.oversampling = oversampling

    def forward(self, input_ids, attention_mask, labels=None):
        output = self.model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
        return output
    
    def training_step(self, batch, batch_idx):
        seqs, input_ids, attention_mask, labels = batch
        outputs = self(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
        loss = outputs.loss
        return loss
    
    # def training_epoch_end(self, outputs):
    #     loss = torch.stack([x for x in outputs]).mean()
    #     self.log("train_loss", loss)
    #     return loss
    
    def validation_step(self, batch, batch_idx):
        seqs, input_ids, attention_mask, labels = batch
        outputs = self(input_ids=input_ids, attention_mask=attention_mask)
        return seqs, outputs, labels
    
    
    def validation_epoch_end(self, outputs):
        logits = torch.cat([x[1].logits for x in outputs], dim=0)
        labels = torch.cat([x[2] for x in outputs], dim=0)
        acc = logits.argmax(dim=1).eq(labels).sum().item() / labels.size(0)
        self.log("val_acc", acc)
        return acc
    
    def test_step(self, batch, batch_idx):
        seqs, input_ids, attention_mask, labels = batch
        outputs = self(input_ids=input_ids, attention_mask=attention_mask)
        return seqs, outputs, labels
    
    
    def test_epoch_end(self, outputs):
        seqs = torch.cat([x[0] for x in outputs], dim=0)
        torch.save(seqs, "/shanjunjie/ProteinMultiClass/data/result/seqs.pt")
        logits = torch.cat([x[1].logits for x in outputs], dim=0)
        torch.save(logits, "/shanjunjie/ProteinMultiClass/data/result/logits.pt")
        labels = torch.cat([x[2] for x in outputs], dim=0)
        torch.save(labels, "/shanjunjie/ProteinMultiClass/data/result/labels.pt")
        acc = logits.argmax(dim=1).eq(labels).sum().item() / labels.size(0)
        self.log("val_acc", acc)
        return acc
        
    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(params=self.parameters(), lr=self.lr, weight_decay=self.weight_decay)
        # optimizer = torch.optim.AdamW(params=self.trainer.model.parameters(), lr=self.lr, weight_decay=self.weight_decay)
        scheduler = StepLR(optimizer, step_size=1, gamma=self.gama)      
        return [optimizer], [scheduler]
        # return optimizer

def preprocess(df, max_len=500):
    if "Name" in df.columns:
        df = df.rename(columns={"Name": "name"})
    if "Sequence full length" in df.columns:
        df = df.rename(columns={"Sequence full length": "sequence"})
    if "Label" in df.columns:
        df = df.rename(columns={"Label": "label"})
    
    all_names = df["name"].values.tolist()
    all_names = [n.strip() for n in all_names]
    all_names = [n[n.find("(")+1:n.rfind(")")] if n.find("(")+1 != n.rfind(")") else " " for n in all_names]
    all_names = [n[:n.find("/")] if n.find("/") != -1 else n for n in all_names]
    all_names = [n.lower() for n in all_names]
    all_names = pd.Series(all_names).value_counts()
    all_names = all_names[all_names >= 8]
    df['label'] = df['name'].apply(lambda x: x[x.find("(")+1:x.find(")")] if x.find("(")+1 != x.find(")") else " ")
    df['label'] = df['label'].apply(lambda x: x[:x.find("/")] if x.find("/") != -1 else x)
    df['label'] = df['label'].apply(lambda x: x.lower())
    df = df[df['label'] != " "]
    df = df[df['label'].isin(all_names.index)]
    # reset index
    df = df.reset_index(drop=True)
    # if the seq len > 1000, drop the row
    df = df[df["sequence"].apply(lambda x: len(x)) < max_len]
    print(list(set(df['label'])))
    return df


def train_eval_split(df, val_ratio, random_state):
    train_df = []
    val_df = []
    for label in df['label'].unique():
        label_df = df[df['label'] == label]
        val_label_data = label_df.sample(frac=val_ratio, random_state=random_state)
        train_label_data = label_df.drop(val_label_data.index)
        train_df.append(train_label_data)
        val_df.append(val_label_data)
    train_df = pd.concat(train_df)
    val_df = pd.concat(val_df)
    return train_df, val_df

if __name__ == "__main__":
    random_seed = 3407
    max_seq_len = 1000
    val_ratio = 0.2
    pl.seed_everything(random_seed)
    df = pd.read_csv('/shanjunjie/ProteinMultiClass/data/AIdataset230810.csv')
    df = preprocess(df, max_len=max_seq_len)
    train_df, val_df = train_eval_split(df, val_ratio, random_state=random_seed)
    train_df.to_csv("train.csv", index=False)
    val_df.to_csv("test.csv", index=False)
    oversampling_size = args.oversampling_size
    for label in train_df['label'].unique():
        label_df = train_df[train_df['label'] == label]
        if len(label_df) < oversampling_size:
            train_df = pd.concat([train_df, label_df.sample(n=oversampling_size-len(label_df), replace=True, random_state=random_seed)])


    tokenizer = EsmTokenizer.from_pretrained(model_path)
    train_dataset = ProSeqDataset(train_df, tokenizer)
    val_dataset = ProSeqDataset(val_df, tokenizer)
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=64)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, num_workers=64)

    num_labels = train_dataset.num_labels
    esm_model = EsmForSequenceClassification.from_pretrained(pretrained_model_name_or_path = model_path, num_labels = num_labels)

    # model = ProFunCla(model=esm_model, lr=args.lr, weight_decay=args.weight_decay, finetune_layer=args.finetune_layer, gama=args.sechdule_gamma, oversampling=args.oversampling_size)
    model = ProFunCla.load_from_checkpoint(model = esm_model,checkpoint_path="/shanjunjie/ProteinMultiClass/checkpoint/epoch=4-val_acc=0.8778-loss=0.0000.ckpt")
    # save the checkpoint with name as oversampling_size and finetune_layer and val_acc
    # result = model(val_dataset.inputs["input_ids"], val_dataset.inputs["attention_mask"])
    
    # print(len(result))