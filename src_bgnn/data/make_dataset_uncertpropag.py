from src.data import utils as ut
from src.data import config
from src.features import build_features as bf
import pickle
import networkx as nx
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
import random
from scipy.stats import rankdata
import numpy as np
import pandas as pd
from sklearn import preprocessing, feature_extraction, model_selection
from stellargraph.core.graph import StellarGraph
from src_bgnn.data import config as cn

## load
filepath = cn.datapath + "\\ppi\ppi-G.csv"

g = ut.get_graphfromdf(filepath, 'source', 'target')

mapper = {}

for count, nodeid in enumerate(g.nodes):
    mapper[nodeid] = count

g = nx.relabel_nodes(g, mapper)

filepath = cn.datapath + "\\ppi\\ppi-class_map.json"

targets = ut.get_jsondata(filepath)

newdict={}
for count,(key,value) in enumerate(targets.items()):
    newdict[mapper[int(key)]] = value
    # print(key, value)
    # if count>3:
    #     break

filepath = cn.datapath + "\\ppi\\ppi-feats.npy"

features = np.load(filepath)

targets = pd.DataFrame.from_dict(newdict, orient='index')

train_subjects, test_subjects = model_selection.train_test_split(targets, test_size=0.7)

test_subjects = test_subjects.iloc[0:1000]

train_targets = np.array(train_subjects)
test_targets = np.array(test_subjects)

rev_mapper = {v: k for k, v in mapper.items()}

for node_id, node_data in g.nodes(data=True):
    # node_data["feature"] = [g.degree(node_id), nx.average_neighbor_degree(g, nodes=[node_id])[node_id], 1, 1,1]
    node_data["feature"] = features[rev_mapper[node_id],:]

G = StellarGraph.from_networkx(g, node_features="feature")
print(G.info())



