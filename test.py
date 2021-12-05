import numpy as np
import os
import pickle
from classes import *

dir = './datasets'

names_test_sets = ['sequential.pkl', 'branch_loop.pkl']

# first create the 'sequential' set
length = 30
# generate nodes with id and random sizes
sizes_random = np.random.uniform(0, 25, [length])
nodes = [Graph_Node(i, sizes_random[i]) for i in range(length)]
# set termination nodes
nodes.append(Graph_Node(-1, None))
# set links
for i in range(length):
    nodes[i].branches[0] = nodes[i+1]

# pack all nodes as a graph to save
graphs_full = [nodes]
# no branch, so all 0
branch_samples = [[np.zeros([length], dtype=int)]]

dataset = {'graphs_full':graphs_full, 'branch_samples': branch_samples}
with open(os.path.join(dir, names_test_sets[0]), 'wb') as f:
    pickle.dump(dataset, f)

#--------------------------------------------------------------
