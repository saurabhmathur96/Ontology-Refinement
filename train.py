from sklearn.metrics import roc_auc_score
from srlearn.rdn import BoostedRDNClassifier
from srlearn import Background
import numpy as np 
import pandas as pd
import random, obonet
from dataset import split_dataset, build_dataset, get_k_folds

modes = [
    "part_of(-T, +T).",
    "regulates(-T, +T).",
    "is_a(-T, +T).",
    "positively_regulates(-T, +T).",
    "negatively_regulates(-T, +T).",

    "part_of(+T, -T).",
    "regulates(+T, -T).",
    "is_a(+T, -T).",
    "positively_regulates(+T, -T).",
    "negatively_regulates(+T, -T).",

    "part_of(+T, +T).",
    "regulates(+T, +T).",
    "is_a(+T, +T).",
    "positively_regulates(+T, +T).",
    "negatively_regulates(+T, +T).",

    "molecular_function(-G, +T).",
    "cellular_component(-G, +T).",
    "biological_process(-G, +T).",

    "molecular_function(+G, +T).",
    "cellular_component(+G, +T).",
    "biological_process(+G, +T).",

    "taxonomy(+G, -O).",
    "taxonomy(+G, +O).",
]


random.seed(0)

from srlearn.system_manager import FileSystem
from os import getcwd

FileSystem.boostsrl_data_directory = getcwd()

systems = []
graph = obonet.read_obo("go-basic.obo")
frame = pd.read_csv("terms.tsv", sep="\t")
taxonomy = pd.read_csv("taxonomy.tsv", sep="\t")


train_split, test_split = split_dataset(graph, frame, taxonomy)
# train, test = build_dataset(*train_split, target='is_a'), build_dataset(*test_split, target='is_a')

bk = Background(modes=modes)


max_trees = 2
scores = []
for fold_train, fold_valid in get_k_folds(*train_split):
    systems.append(FileSystem())
    clf = BoostedRDNClassifier(
        background=bk,
        target="is_a",
        max_tree_depth=2,
        node_size=2,
        n_estimators=max_trees,
    )
    train = build_dataset(*fold_train, target='is_a')
    valid = build_dataset(*fold_valid, target='is_a')
    clf.fit(train)

    fold_scores = []
    for n_trees in range(1, max_trees+1):
        clf.n_estimators = n_trees
        probs = clf.predict_proba(valid)
        auc_roc = roc_auc_score(clf.classes_, probs)
        fold_scores.append(auc_roc)
    scores.append(np.array(fold_scores))
scores = np.array(scores)
mean,std = np.mean(scores, axis=0),np.std(scores, axis=0) 
for n_trees in range(1, max_trees+1):
    print (f"{n_trees:2} {mean[n_trees-1]:.4f} ± {std[n_trees-1]:.4f}")
