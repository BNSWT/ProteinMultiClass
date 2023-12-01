import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

df = pd.read_csv("/shanjunjie/ProteinMultiClass/visual/data/bacteria.nonredundant_protein.1.protein.full.csv")
type = ["cys","val","gly","leu"]
columns = ["pred","cys","val","gly","leu"]
df = df[columns]
df = df[df["pred"].isin(type)]
sns.pairplot(df, diag_kind='kde', hue='pred')

# diag_kind='kde'用于在对角线上显示核密度估计图
# hue='species'用于根据不同类别着色

plt.savefig("pairplot.png")