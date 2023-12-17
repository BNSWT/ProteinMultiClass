import torch
import os
from tsnecuda import TSNE
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from tqdm import tqdm
import matplotlib.pyplot as plt
import numpy as np
from collections import Counter
from matplotlib.colors import ListedColormap

tsne = TSNE(n_components=2, perplexity=15, learning_rate=10)

labels = None
repr = None
max_len = 233
seq_data = []
label_data = []

files = os.listdir("/shanjunjie/ProteinMultiClass/visual/repr")
for file in tqdm(files, desc="loading data", unit="file"):
    if file.endswith(".pt"):
        path = os.path.join("/shanjunjie/ProteinMultiClass/visual/repr", file)
        result_path = os.path.join("/shanjunjie/ProteinMultiClass/visual/data-001", file[:-3] + ".part.csv")
        part_labels = np.array(pd.read_csv(result_path)["pred"].values.tolist())
        part_repr = np.array(torch.load(path).cpu())
        seqlen = part_repr.shape[1]
        padding_rows = max_len - seqlen
        part_repr = np.pad(part_repr, ((0, 0), (0, padding_rows), (0, 0)), "constant", constant_values=0)
        if labels is None:
            labels = part_labels
        else:
            labels = np.concatenate((labels, part_labels))
        # if repr is None:
        #     repr = part_repr
        # else:
        #     repr = np.concatenate((repr, part_repr))
        # print(repr.shape, labels.shape)
        part_repr = part_repr.reshape(part_repr.shape[0], -1)
        part_embedded = tsne.fit_transform(part_repr)
        seq_data.append(part_embedded)
        label_data.append(part_labels)



label_counts = Counter(labels)

# 选择出现次数大于150的label
selected_labels = [label for label, count in label_counts.items() if count > 150]

tick_label = ["other"]+[label for label in selected_labels]

viridis_colors = plt.cm.viridis(np.linspace(0, 1, len(selected_labels)))
# 其余label都用同一个颜色表示

new_color = np.array([0, 0, 0, 1])
combined_colors = np.vstack([new_color, viridis_colors])

cmap = ListedColormap(combined_colors)

# 为每个出现次数大于150的label分配一个颜色
# colors = [selected_labels.index(label) + 1 if label in selected_labels else 0 for label in labels]

for i, seq in enumerate(seq_data):
    label_list = label_data[i]

    # 生成示例数据
    x = seq[:, 0]
    y = seq[:, 1]
    colors = [selected_labels.index(label) + 1 if label in selected_labels else 0 for label in label_list]

    # 绘制散点图
    scatter = plt.scatter(x, y, c=colors, cmap=cmap)

# 添加标题和标签
plt.title('Scatter Plot of different domains')
plt.xlabel('X-axis')
plt.ylabel('Y-axis')

# 显示颜色条
cbar = plt.colorbar(scatter, 
             label='Label Category (Count > 150)')
cbar.set_ticklabels(tick_label)

# 显示图形
plt.savefig("tsne.png")


