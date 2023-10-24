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


def create_database(pos, neg, facts):
    db = Database()
    db.pos = list(pos) 
    db.neg = list(neg)
    db.facts = list(facts)
    return db

def train_test_split(db, test_size=0.2):
    pos = list(db.pos)
    neg = list(db.neg)

    random.shuffle(pos)
    random.shuffle(neg)

    i = int(len(pos)*test_size)
    test_pos = pos[0:i]
    train_pos = pos[i:]

    i = int(len(neg)*test_size)
    test_neg = neg[0:i]
    train_neg = neg[i:]

    train = create_database(train_pos, train_neg, db.facts)
    test = create_database(test_pos, test_neg, db.facts)

    return train, test 

def kfold_split(db, k = 5):
    pos = list(db.pos)
    neg = list(db.neg)

    N_pos, N_neg = len(pos), len(neg)
    for i, j in zip(range(0, N_pos, N_pos//k), range(0, N_neg, N_neg//k)):
        test_pos = pos[i:i+N_pos//k]
        train_pos = pos[0:i] + pos[i+N_pos//k:]

        test_neg = neg[j:j+N_neg//k]
        train_neg = neg[0:j] + neg[j+N_neg//k:]

        train = create_database(train_pos, train_neg, db.facts)
        test = create_database(test_pos, test_neg, db.facts)
        yield (train, test)



import os, pathlib

random.seed(0)
graph = obonet.read_obo("go-basic.obo")
frame = pd.read_csv("terms.tsv", sep="\t")
taxonomy = pd.read_csv("taxonomy.tsv", sep="\t")

target = 'is_a'
db = build_dataset(graph, frame, taxonomy, target=target)

os.makedirs(target, exist_ok=True)
train, test = train_test_split(db, test_size=0.2)
for i, (train_, test_) in enumerate(kfold_split(db)):
    fold_path = path.join(target, f"fold{i}")
    os.makedirs(fold_path, exist_ok=True)

    train_path = pathlib.Path(path.join(fold_path, "train"))
    test_path = pathlib.Path(path.join(fold_path, "test"))

    os.makedirs(train_path, exist_ok=True)
    os.makedirs(test_path, exist_ok=True)

    train_.write(filename=f"train", location = train_path)
    test_.write(filename=f"test", location = test_path)

test_path = pathlib.Path(path.join(target, "test"))
os.makedirs(test_path, exist_ok=True)
test.write(filename=f"test", location = test_path)


