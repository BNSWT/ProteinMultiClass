import pandas as pd
from transformers import EsmTokenizer
from thundersvm import SVC

max_len = 500
oversampling_size = 150
model_path = "/shanjunjie/ProteinMultiClass/esm2_t33_650M_UR50D"


def preprocess(df):
    # only keep those more than 8
    all_names = df['label'].values.tolist()
    all_names = pd.Series(all_names).value_counts()
    all_names = all_names[all_names >= 8]
    
    # filter out those longer than max_len
    df = df[df['label'].isin(all_names.index)]
    df = df[df["sequence"].apply(lambda x: len(x)) < max_len]
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

def oversample(df):
    for label in df['label'].unique():
        label_df = df[df['label'] == label]
        if len(label_df) < oversampling_size:
            df = pd.concat([df, label_df.sample(n=oversampling_size-len(label_df), replace=True)])
    return df

def get_data(df):
    tokenizer = EsmTokenizer.from_pretrained(model_path)
    seqs = df['Seq-SVM'].tolist()
    data = tokenizer(seqs, padding=True, truncation=True, return_tensors="pt")
    
    labels = df['Label'].tolist()
    return data, labels


df = pd.read_csv('/shanjunjie/ProteinMultiClass/data/AIdataset230810-forSVMcomparison.csv')
df = preprocess(df)
num_classes = len(df['label'].unique())

train_df, val_df = train_eval_split(df, 0.2, 42)
train_df = oversample(train_df)

train_data, train_labels = get_data(train_df)
val_data, val_labels = get_data(val_df)

clf = SVC(kernel='linear', C=num_classes)
clf.fit(train_data['input_ids'], train_labels)
clf.save_to_file('/shanjunjie/ProteinMultiClass/svm_model')

preds = clf.predict(val_data['input_ids'])
print("Accuracy: ", (preds == val_labels).mean())