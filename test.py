import numpy as np
import os
import pickle
from classes import *

dir = './datasets'

names_test_sets = ['sequential.pkl', 'branch_loop.pkl']

#======================================================
# # first create the 'sequential' set
# length = 30
# # generate nodes with id and random sizes
# sizes_random = np.random.uniform(0, 25, [length])
# nodes = [Graph_Node(i, sizes_random[i]) for i in range(length)]
# # set termination nodes
# nodes.append(Graph_Node(-1, None))
# # set links
# for i in range(length):
#     nodes[i].branches[0] = nodes[i+1]
#
# # pack all nodes as a graph to save
# graphs_full = [nodes]
# # no branch, so all 0
# branch_samples = [[np.zeros([length], dtype=int)]]
#
# dataset = {'graphs_full':graphs_full, 'branch_samples': branch_samples}
# with open(os.path.join(dir, names_test_sets[0]), 'wb') as f:
#     pickle.dump(dataset, f)

#=====================================================

# then we create 'branch' set
# 0->1->2->3->4->5->-1, 4->3, 5->2
length = 6
sizes_random = np.random.uniform(0, 25, [length])
nodes = [Graph_Node(i, sizes_random[i]) for i in range(length)]
nodes.append(Graph_Node(-1, None))
for i in range(length):
    nodes[i].branches[0] = nodes[i+1]
nodes[5].branches[1] = nodes[2]
nodes[4].branches[1] = nodes[3]
graphs_full = [nodes]
branch_samples = [[]]

n_samples = 1000
for n in range(n_samples):
    n_large_loop = np.random.randint(1,10)
    n_small_loops = np.random.randint(1, 10, [n_large_loop])
    cur_path = [0,0]
    for n_small_loop in n_small_loops:
        cur_path = cur_path + [0] + [0, 1] * (n_small_loop-1) + [0, 0, 1]
    cur_path = cur_path + [0,0,0,0]
    branch_samples[0].append(np.array(cur_path))
#     # try to debug
#     print('n_large_loop', n_large_loop)
#     print('n_small_loops', n_small_loops)
#
# # try to debug
# cur_node = nodes[0]
# path = branch_samples[0][0]
# index = 0
# print(len(path))
# while cur_node.id >= 0:
#     print(cur_node.id, '->', end='')
#     cur_node = cur_node.branches[path[index]]
#     index += 1

dataset = {'graphs_full':graphs_full, 'branch_samples': branch_samples}
with open(os.path.join(dir, names_test_sets[1]), 'wb') as f:
    pickle.dump(dataset, f)
