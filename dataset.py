import networkx
import obonet
import numpy as np
import pandas as pd
import random, time
from itertools import product
from srlearn import Database
from srlearn.rdn import BoostedRDNClassifier
from srlearn import Background
from sklearn.metrics import roc_auc_score, average_precision_score, precision_score, recall_score
from os import path, makedirs

def build_dataset(graph, frame, taxonomy, target='is_a', neg_pos_ratio=2):
    relations = ['part_of', 'regulates', 'is_a', 'positively_regulates', 'negatively_regulates']
    terms = list(graph.nodes)
    facts = []
    for _, row in frame.iterrows():
        first = row["geneID"].lower()
        second = row["term"].replace(":", "_").lower()
        kind = row["aspect"]
        # The three GO ontologies are is a disjoint, meaning that no is a relations operate between terms from the different ontologies. 
        # However, other relationships such as part of and regulates do operate between the GO ontologies
        # Molecular Function: the molecular activities of individual gene products
        # Cellular Component: where the gene products are active
        # Biological Process: the pathways and larger processes to which that gene product’s activity contributes

        relation_name = { "MFO": "molecular_function", "CCO": "cellular_component", "BPO": "biological_process"}[kind]
        facts.append(f"{relation_name}({first},{second}).")
    
    for _, row in taxonomy.iterrows():
        first = row["geneID"].lower()
        second = f"taxonomy_{row['taxonomyID']}"
        
        facts.append(f"taxonomy({first},{second}).")
        
    positives = []
    for first, second, relation_name in graph.edges(keys = True):
        first = first.replace(":", "_").lower()
        second = second.replace(":", "_").lower()
        if relation_name == target:
            positives.append(f"{relation_name}({first},{second}).")
        else:
            facts.append(f"{relation_name}({first},{second}).")
        

    negatives = []
    while len(negatives) < neg_pos_ratio*len(positives):
        [first, second] = random.sample(terms, k=2)
        relation_name = target 

        if not graph.has_edge(first, second, key=relation_name):
            first = first.replace(":", "_").lower()
            second = second.replace(":", "_").lower()
            negatives.append(f"{relation_name}({first},{second}).")

    db = Database()
    db.pos = positives
    db.neg = negatives
    db.facts = list(set(facts))
    return db




def save_database(db, base, prefix):
    with open(path.join(base, f"{prefix}_pos.txt"), "w") as f:
        f.write("\n".join(db.pos))

    with open(path.join(base, f"{prefix}_neg.txt"), "w") as f:
        f.write("\n".join(db.neg))

    with open(path.join(base, f"{prefix}_facts.txt"), "w") as f:
        f.write("\n".join(db.facts))
    
    with open(path.join(base, f"{prefix}_bk.txt"), "w") as f:
        f.write("\n".join(db.modes))


def split_dataset(graph, frame, taxonomy):
    
    nodes = list(graph.nodes)
    random.shuffle(nodes)
    N = len(nodes)
    index = int(0.8*N)
    train_nodes, test_nodes = nodes[0:index], nodes[index:]
    
    train_frame = frame[frame.term.isin(train_nodes)]
    test_frame = frame[frame.term.isin(test_nodes)]

    train_taxonomy = taxonomy[taxonomy.geneID.isin(train_frame.geneID.unique())]
    test_taxonomy = taxonomy[taxonomy.geneID.isin(test_frame.geneID.unique())]

    train_graph = networkx.subgraph(graph, train_nodes)
    test_graph = networkx.subgraph(graph, test_nodes)
    
    return (train_graph, train_frame, train_taxonomy), (test_graph, test_frame, test_taxonomy)

def get_k_folds(graph, frame, taxonomy, k=5):
    nodes = list(graph.nodes)
    random.shuffle(nodes)
    N = len(nodes)
    for i in range(0, N, N//k):
        test_nodes = nodes[i:i+N//k]
        train_nodes = nodes[0:i] + nodes[i+N//k:]

        train_frame = frame[frame.term.isin(train_nodes)]
        test_frame = frame[frame.term.isin(test_nodes)]

        train_taxonomy = taxonomy[taxonomy.geneID.isin(train_frame.geneID.unique())]
        test_taxonomy = taxonomy[taxonomy.geneID.isin(test_frame.geneID.unique())]

        train_graph = networkx.subgraph(graph, train_nodes)
        test_graph = networkx.subgraph(graph, test_nodes)
        
        yield (train_graph, train_frame, train_taxonomy), (test_graph, test_frame, test_taxonomy)
         


if __name__ == "__main__":
    random.seed(0)
    graph = obonet.read_obo("go-basic.obo")
    frame = pd.read_csv("terms.tsv", sep="\t")
    taxonomy = pd.read_csv("taxonomy.tsv", sep="\t")

    train_split, test_split = split_dataset(graph, frame, taxonomy)
    train, test = build_dataset(*train_split), build_dataset(*test_split)
    ts = str(int(time.time()))[-1]
    train_base = path.join(f"go_{ts}", "train")
    test_base = path.join(f"go_{ts}", "test")

    makedirs(train_base, exist_ok=True)
    makedirs(test_base, exist_ok=True)

    save_database(train, train_base, "train")
    save_database(test, test_base, "test")
    


