from sklearn.metrics import average_precision_score, precision_recall_curve
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import networkx as nx
import argparse
import os 

parser = argparse.ArgumentParser()
parser.add_argument('--pred_dir', type=str, help = "Path to pandas file with prediction or to directory with multiple predictions if --multiple=True")
parser.add_argument('--label_file', type=str, help='Path to label')
parser.add_argument('--draw_graph', action='store_true', help="Whether to draw the true and predicted graphs")
parser.add_argument('--no_graph', dest='draw_graph', action='store_false')
parser.add_argument('--multiple', action='store_true', help="Whether to evaluate a single prediction or several")
parser.add_argument('--single', dest='multiple', action='store_false')
parser.set_defaults(multiple=True)
opt = parser.parse_args()

col1 = "TF"
col2 = "Target"

def compute_epr(pred, label):
    output = pred.copy()
    output['EdgeWeight'] = abs(output['EdgeWeight'])
    output = output.sort_values('EdgeWeight',ascending=False)
    
    TFs = set(label['Gene1'])
    Genes = set(label['Gene1'])| set(label['Gene2'])
    output = output[output[col1].apply(lambda x: x in TFs)]
    output = output[output[col2].apply(lambda x: x in Genes)]
    label_set = set(label['Gene1']+'|'+label['Gene2'])
    output = output.iloc[:len(label_set)]
    early_TPs = len(set(output[col1]+'|'+output[col2]) & label_set)
    den = (len(label_set)**2/(len(TFs)*len(Genes)-len(TFs)))

    EP = early_TPs/len(label_set)
    EPR = early_TPs/den

    #print("true positives among top k scored =", early_TPs)
    #print("true positives among top k random scored =", den)
    #print("AUPRC ratio = ", early_TPs / den)
    return EP, EPR 

def compute_prc(pred, label):
    output = pred.copy()
    output['EdgeWeight'] = abs(output['EdgeWeight'])
    output = output.sort_values('EdgeWeight',ascending=False)
    
    TFs = set(label['Gene1'])
    Genes = set(label['Gene1'])| set(label['Gene2'])
    #print("number or target genes in output :", len(output['Target'].unique()))
    #print("number or TFs in output :", len(output['TF'].unique()))
    output = output[output[col1].apply(lambda x: x in TFs)]
    output = output[output[col2].apply(lambda x: x in Genes)]
    label_set = set(label['Gene1']+'|'+label['Gene2'])
    res_d = {}
    l = []
    p= []
    for item in (output.to_dict('records')):
            res_d[item[col1] + '|' + item[col2]] = item['EdgeWeight']
    for item in (set(label['Gene1'])):
            for item2 in  set(label['Gene1'])| set(label['Gene2']):
                if item+ '|' + item2 in label_set:
                    l.append(1)
                else:
                    l.append(0)
                if item + '|' + item2 in res_d:
                    p.append(res_d[item + '|' + item2])
                else:
                    p.append(-1)
    score = average_precision_score(l,p)
    precision, recall, thresholds = precision_recall_curve(l, p)
    random_score = len(label)/(len(TFs)*(len(Genes) - 1))
    return precision, recall, score, random_score

def plot_graphs(pred, label):
    output = pred.copy()
    
    TFs = set(label['Gene1'])
    Genes = set(label['Gene1'])| set(label['Gene2'])

    output = output[output[col1].apply(lambda x: x in TFs)]
    output = output[output[col2].apply(lambda x: x in Genes)] #filter out edges out of ground truth domain
    output = output.sort_values('EdgeWeight', ascending=False)
    output = output[:len(label)]

    Gt = nx.DiGraph()
    Gt.add_edges_from([tuple(label.iloc[i]) for i in range(len(label))])

    Gp = nx.DiGraph()
    Gp.add_weighted_edges_from([tuple(output.iloc[i]) for i in range(len(output))])

    print(f"Acyclicity of ground truth: {nx.is_directed_acyclic_graph(Gt)}")
    print(f"Acyclicity of prediction: {nx.is_directed_acyclic_graph(Gp)}")
    print(f"Number of edges in common: {len(Gt.edges & Gp.edges)}")
    print(f"Number of edges in ground truth: {len(Gt.edges)}")

    fig, ax = plt.subplots(1, 2, figsize=(16,8))
    allnodes = list(Gt.nodes) + list(Gp.nodes)
    pos = nx.spring_layout(allnodes, seed=123)  # positions of all nodes
    options = {'node_size': 1000, 'alpha': 1}
    nx.draw_networkx_nodes(Gp, pos, nodelist=set(output[col2]), node_color='tab:blue', ax = ax[1], **options)
    nx.draw_networkx_nodes(Gp, pos, nodelist=TFs, node_color='tab:orange', ax = ax[1], **options)
    nx.draw_networkx_edges(Gp,pos, edgelist = Gp.edges - (Gt.edges & Gp.edges), ax = ax[1], **options)
    nx.draw_networkx_edges(Gp, edgelist = Gt.edges & Gp.edges, pos = pos, ax = ax[1], edge_color = "tab:orange", **options)
    nx.draw_networkx_labels(Gp, pos, ax = ax[1])
    ax[1].set_title("Prediction")

    nx.draw_networkx_nodes(Gt, pos, nodelist=Genes, node_color='tab:blue', ax = ax[0], **options)
    nx.draw_networkx_nodes(Gt, pos, nodelist=TFs & Gt.nodes, node_color='tab:orange', ax = ax[0], **options)
    nx.draw_networkx_edges(Gt, pos, ax = ax[0], **options)
    nx.draw_networkx_labels(Gt, pos, ax=ax[0])
    ax[0].set_title("Ground truth")
    plt.tight_layout()
    plt.savefig("graph_out.png")
    plt.show()
    return Gt, Gp

label = pd.read_csv(opt.label_file, index_col = 0)

TFs = set(label['Gene1'])
Genes = set(label['Gene1'])| set(label['Gene2'])


if opt.multiple: 
    preds = []

    for root, dirs, files in os.walk(opt.pred_dir):
        for filename in files:
            preds.append(pd.read_csv(os.path.join(root, filename), sep = '\t'))

    n_preds = len(preds)
    eps = []
    auprcs = []
    for output in preds: 
        ep, epr = compute_epr(output, label)
        precision, recall, score, random_score = compute_prc(output, label)
        auprcs.append(score)
        random = ep/epr
        eps.append(ep)
    plt.figure()
    plt.boxplot([eps, auprcs], labels=['Early Precision', 'AUPRC'])
    plt.plot([0.9, 1.1], [random, random], label = "class balance", c = 'tab:blue')
    plt.plot([1.9, 2.1], [random_score, random_score], c = 'tab:blue')
    plt.legend()
    plt.title("Boxplots of metrics values across 10 runs")
    plt.savefig("boxplot_2_metrics.png")

else:
    #output = pd.read_csv("/home/jel-khoury/Differentiable-DAG-Sampling/src/notebooks/pred2.tsv", sep = '\t')
    output = pd.read_csv(opt.pred_dir,sep='\t')
    #output format: pandas dataframe with 'TF' column, 'Target' column and 'EdgeWeight' column

    #Compute and print precision of top-k scored
    ep, epr = compute_epr(output, label)
    print(f"Early precision = {ep}")
    print(f"Early precision rate = {epr}")

    #Compute and plot precision-recall curve
    precision, recall, score, rd_score = compute_prc(output, label)
    plt.plot(recall, precision)
    plt.plot(recall, rd_score*np.ones_like(recall))
    plt.xlabel("recall")
    plt.ylabel("precision")
    plt.ylim(0, 1)
    plt.savefig("PRC.png")
    plt.show()

    plot_graphs(output, label)