import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import argparse
from mpl_toolkits.mplot3d import Axes3D  
from sklearn.decomposition import PCA  
from sklearn.metrics import confusion_matrix

def confusion(pred, labels):
    label_set = set(labels)
    label_to_int = {label: i for i, label in enumerate(label_set)}

    pred_int = [label_to_int[label] for label in pred]
    label_int = [label_to_int[label] for label in labels]

    confusion = confusion_matrix(label_int, pred_int)

    plt.figure(figsize=(10, 8))
    sns.heatmap(confusion, cmap='Blues')

    labels = [label for label, _ in sorted(label_to_int.items(), key=lambda x: x[1])]
    plt.xticks(np.arange(len(labels)), labels, rotation=45)
    plt.yticks(np.arange(len(labels)), labels)
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.title('Confusion Matrix Heatmap')

    # plt.show()
    plt.savefig("confusion.png")

def scatter(df, labels):
    classes = df.columns
    print(len(classes))
    if args.highlight_class not in classes:
        print(f"No class {args.highlight_class} in data")
        return 1
    
    labels = np.array(labels)
    highlight_index = np.where(labels == args.highlight_class)
    normal_index = np.where(labels != args.highlight_class)

    data = df.values

    pca = PCA(n_components=3)
    data_3d = pca.fit_transform(data)

    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    x = np.array(data_3d[:, 0])
    y = np.array(data_3d[:, 1])
    z = np.array(data_3d[:, 2])

    highlight = ax.scatter(x[highlight_index], y[highlight_index], z[highlight_index], color='red', label=args.highlight_class)
    others = ax.scatter(x[normal_index], y[normal_index], z[normal_index], color='blue', label="others")

    ax.set_xlabel('Component 1')
    ax.set_ylabel('Component 2')
    ax.set_zlabel('Component 3')

    ax.legend(handles=[highlight,others])
    # plt.show()
    plt.savefig("scatter.png")

def distribution(df):
    class_counts = df['pred'].value_counts()
    plt.figure(figsize=(10, 6))
    class_counts.plot(kind='bar', color='skyblue')
    plt.xlabel('Class')
    plt.ylabel('Count')
    plt.title('Class Distribution in Multiclass Results')
    plt.xticks(rotation=45)

    # plt.show()
    plt.savefig("distribution.png")




def main(args):
    seq_df = pd.read_csv("part_result.csv")
    seqs = seq_df['sequence'].values.tolist()
    pred = seq_df['pred'].values.tolist()
    labels = seq_df['label'].values.tolist()

    df = pd.read_csv("full_result.csv")

    distribution(seq_df)
    confusion(pred, labels)
    scatter(df, labels)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--highlight_class", type=str, default="cys")
    args = parser.parse_args()
    main(args)