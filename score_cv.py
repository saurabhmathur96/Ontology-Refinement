from sklearn.metrics import roc_auc_score
from srlearn.rdn import BoostedRDNClassifier
from srlearn import Database
import numpy as np 

from os import path
import config


from srlearn.system_manager import FileSystem
from os import getcwd
import numpy as np
np.loadtxt = np.genfromtxt
FileSystem.boostsrl_data_directory = getcwd()

scores = []
base = path.join(config.target, "models")
for i in range(config.k):
    test_path = path.join(config.target, f"fold{i}", "test")
    test = Database.from_files(path.join(test_path, "test_pos.txt"),  path.join(test_path, "test_neg.txt"),  
                            path.join(test_path, "test_facts.txt"), lazy_load=False)
    clf = BoostedRDNClassifier()
    clf.from_json(file_name = path.join(base, f"fold{i}.json"))

    fold_scores = []
    for n_trees in range(1, config.max_trees+1):
        clf.n_estimators = n_trees
        probs = clf.predict_proba(test)
        auc_roc = roc_auc_score(clf.classes_, probs)
        fold_scores.append(auc_roc)
    scores.append(np.array(fold_scores))
    
scores = np.array(scores)
mean, std = np.mean(scores, axis=0), np.std(scores, axis=0) 
for n_trees in range(1, config.max_trees+1):
    print (f"{n_trees:2} {mean[n_trees-1]:.4f} ± {std[n_trees-1]:.4f}")   
