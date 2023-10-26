from srlearn.rdn import BoostedRDNClassifier
from srlearn import Background, Database
import random

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

from os import path
import os

random.seed(0)

from srlearn.system_manager import FileSystem
from os import getcwd
import config

FileSystem.boostsrl_data_directory = getcwd()
bk = Background(modes=modes)

scores = []


base = path.join(config.target, "models")
os.makedirs(base, exist_ok=True)

for i in range(config.k):
    train_path = path.join(config.target, f"fold{i}", "train")
    train = Database.from_files(path.join(train_path, "train_pos.txt"),  path.join(train_path, "train_neg.txt"),  
                            path.join(train_path, "train_facts.txt"), lazy_load=False)
    test_path = path.join(config.target, f"fold{i}", "test")
    test = Database.from_files(path.join(test_path, "test_pos.txt"),  path.join(test_path, "test_neg.txt"),  
                            path.join(test_path, "test_facts.txt"), lazy_load=False)
    clf = BoostedRDNClassifier(
        background=bk,
        target=config.target,
        max_tree_depth=config.max_depth,
        node_size=config.node_size,
        n_estimators=config.max_trees,
    )
    clf.fit(train)
    clf.to_json(path.join(base, f"fold{i}.json"))
