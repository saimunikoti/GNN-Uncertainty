import tensorflow as tf
config = tf.compat.v1.ConfigProto(gpu_options =
                         tf.compat.v1.GPUOptions(per_process_gpu_memory_fraction=0.8)
# device_count = {'GPU': 1}
)
config.gpu_options.allow_growth = True
session = tf.compat.v1.Session(config=config)
tf.compat.v1.keras.backend.set_session(session)
import networkx as nx
import pandas as pd
import os
import stellargraph as sg
from stellargraph.core.graph import StellarGraph
from stellargraph.mapper.sampled_node_generators_bayesian import GraphSAGENodeGenerator
from stellargraph.layer.graphsage_bayesian import GraphSAGE, MaxPoolingAggregator, MeanAggregator, MeanAggregatorvariance

from tensorflow.keras import layers, optimizers, losses, metrics, Model
from sklearn import preprocessing, feature_extraction, model_selection
from stellargraph import datasets
from IPython.display import display, HTML
import matplotlib.pyplot as plt
from src.data import config
from tensorflow.keras.callbacks import ModelCheckpoint
import numpy as np
import random

## ======================build graph ###############################################

dataset = datasets.Cora()
display(HTML(dataset.description))
G, node_subjects = dataset.load()

print(G.info())

## generate probability of edges
g = G.to_networkx() # multi
g = nx.Graph(g) # simple graph

for (cn1, cn2) in g.edges:
    g[cn1][cn2]['weight'] = np.round(random.uniform(0.5, 1), 3)

G = StellarGraph.from_networkx(g, node_features="feature")
print(G.info())

train_subjects, test_subjects = model_selection.train_test_split(
    node_subjects, train_size=0.30, test_size=None, stratify=node_subjects, random_state=42
)

train_subjects = train_subjects[0:800]
test_subjects = test_subjects[0:1896]

target_encoding = preprocessing.LabelBinarizer()
train_targets = target_encoding.fit_transform(train_subjects)

test_targets = target_encoding.transform(test_subjects)

batch_size = 40
# number of nodes to consider for each hop
num_samples = [15, 10, 5]
#
generator = GraphSAGENodeGenerator(G, batch_size, num_samples)

train_gen = generator.flow(train_subjects.index, train_targets, shuffle=True)  # train_subjects.index for selecting training nodes
test_gen = generator.flow(test_subjects.index, test_targets)

## model building
# graphsage_model = GraphSAGE(layer_sizes=[32, 32, 16], generator=generator, bias=True, dropout=0.5)
graphsage_model = GraphSAGE(layer_sizes=[32, 32, 16], generator=generator,
                            bias=True, aggregator= MeanAggregator,  dropout=0.1)

x_inp, x_out = graphsage_model.in_out_tensors()

prediction = layers.Dense(units=train_targets.shape[1], activation="softmax")(x_out)

model = Model(inputs=x_inp, outputs=prediction)
model.summary()

## %% ##################################### Model training #######################################################

model.compile( optimizer=optimizers.Adam(lr=0.001), loss=losses.categorical_crossentropy, metrics=["acc"])

fileext = "Graphsage_cora2.h5"
filepath = config.Uncertprop_path + fileext
mcp = ModelCheckpoint(filepath, save_best_only=True, monitor='val_loss', mode='min')

history = model.fit(train_gen, epochs=20, validation_data=test_gen, callbacks=[mcp], verbose=2, shuffle=False)

## testing evaluation
test_metrics = model.evaluate(test_gen)
print("\nTest Set Metrics:")
for name, val in zip(model.metrics_names, test_metrics):
    print("\t{}: {:0.4f}".format(name, val))

all_nodes = node_subjects.index
all_mapper = generator.flow(all_nodes)
all_predictions = model.predict(all_mapper)

node_predictions = target_encoding.inverse_transform(all_predictions)

df = pd.DataFrame({"Predicted": node_predictions, "True": node_subjects})
df.head(10)
