# Filter most variable genes in input and ground truth 
import pandas as pd

var_threshold = 10.6
path = "../demo_data/GRN_inference/input/500_ChIP-seq_hESC"

input = pd.read_csv(path + "/data.csv", index_col = 0)
label = pd.read_csv(path + "/label.csv")

keepcols = input.var(0) > var_threshold #threshold to keep ~ 100 most variable genes
print("Number of genes kept =", keepcols.sum())
input = input[input.columns[keepcols]]
input.to_csv(path + "/data2.csv")
print("Saving filtered dataset to " + path + "/data2.csv")

#filter ground truth to keep only genes present in expression data
gene_feats = set(input.columns)
keeprow = label["Gene1"].isin(gene_feats) & label["Gene2"].isin(gene_feats)
label = label[keeprow]
print("Number of ground truth edges =", keeprow.sum())
label.to_csv(path + "/label2.csv")
print("Saving filtered label to " + path + "/label2.csv")