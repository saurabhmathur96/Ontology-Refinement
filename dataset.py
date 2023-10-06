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

def build_dataset(graph, terms, frame, taxonomy, neg_pos_ratio=2):
    relations = ['part_of', 'regulates', 'is_a', 'positively_regulates', 'negatively_regulates']

    facts = []
    for _, row in frame.iterrows():
        first = row["geneID"].lower()
        second = row["term"].replace(":", "_").lower()
        kind = row["aspect"]
        
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
        positives.append(f"{relation_name}({first},{second}).")

    negatives = []
    while len(negatives) < neg_pos_ratio*len(positives):
        [first, second] = random.sample(terms, k=2)
        relation = random.choice(relations)

        if not graph.has_edge(first, second, key=relation):
            first = first.replace(":", "_").lower()
            second = second.replace(":", "_").lower()
            negatives.append(f"{relation}({first},{second}).")

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

if __name__ == "__main__":
    random.seed(0)
    graph = obonet.read_obo("go-basic.obo")
    nodes = list(graph.nodes)
    random.shuffle(nodes)
    N = len(nodes)
    index = int(0.8*N)
    train_nodes, test_nodes = nodes[0:index], nodes[index:]
    frame = pd.read_csv("terms.tsv", sep="\t")
    taxonomy = pd.read_csv("taxonomy.tsv", sep="\t")
    train_frame = frame[frame.term.isin(train_nodes)]
    test_frame = frame[frame.term.isin(test_nodes)]

    train_taxonomy = taxonomy[taxonomy.geneID.isin(train_frame.geneID.unique())]
    test_taxonomy = taxonomy[taxonomy.geneID.isin(test_frame.geneID.unique())]


    train_graph = networkx.subgraph(graph, train_nodes)
    test_graph = networkx.subgraph(graph, test_nodes)

    modes = [
        "setParam: maxTreeDepth=3.",
        "setParam: nodeSize=2.",

        "mode: part_of(-T, +T).",
        "mode: regulates(-T, +T).",
        "mode: is_a(-T, +T).",
        "mode: positively_regulates(-T, +T).",
        "mode: negatively_regulates(-T, +T).",

        "mode: part_of(+T, -T).",
        "mode: regulates(+T, -T).",
        "mode: is_a(+T, -T).",
        "mode: positively_regulates(+T, -T).",
        "mode: negatively_regulates(+T, -T).",
        
        "mode: part_of(+T, +T).",
        "mode: regulates(+T, +T).",
        "mode: is_a(+T, +T).",
        "mode: positively_regulates(+T, +T).",
        "mode: negatively_regulates(+T, +T).",
        
        "mode: molecular_function(-G, +T).",
        "mode: cellular_component(-G, +T).",
        "mode: biological_process(-G, +T).",

        "mode: molecular_function(+G, +T).",
        "mode: cellular_component(+G, +T).",
        "mode: biological_process(+G, +T).",

        "mode: taxonomy(+G, -O).",
        "mode: taxonomy(+G, +O).",
    ]

    train, test = build_dataset(train_graph, train_nodes, train_frame, train_taxonomy), build_dataset(test_graph, test_nodes, test_frame, test_taxonomy)
    train.modes = modes
    test.modes = modes
    ts = str(int(time.time()))[-1]
    train_base = path.join(f"go_{ts}", "train")
    test_base = path.join(f"go_{ts}", "test")
    makedirs(train_base, exist_ok=True)
    makedirs(test_base, exist_ok=True)

    save_database(train, train_base, "train")
    save_database(test, test_base, "test")
    
    relations = ['part_of', 'regulates', 'is_a', 'positively_regulates', 'negatively_regulates']
    target = ",".join(relations)
    print (target)
