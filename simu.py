"""
Code for 15-740 project, simulation part. author Yige Hong

To-do list:
#1 generate data set
#2 write save and load
#3 reduce the frequency of generating random variables
#4 debugging
#5 other optimizations
#6 fix random seeds
7 read data and generate datasets
8 debug branches
"""

import numpy as np
from numba import jit
import matplotlib.pyplot as plt
from collections import namedtuple
import pickle
import os
import logging
from functools import partial
from time import time
from classes import *
logging.basicConfig(level=logging.ERROR)


EPS = 1e-7

# hyper parameter section, global var or static object
Settings = namedtuple('Settings', ['n_skip', 'M', 'alpha', 'beta'])

# class Jobs:
#     def __init__(self):
#         self.type = 0 # which graph
#         self.path = []
#         self.node = Graph_Node()
#         self.index = n # which node
#
#     def reinit(self):
#         pass
#
#     def move(self):
#         self.index += 1
#         self.node = self.node[self.graph[self.index]]
#
#     def is_finish(self):
#         return self.node.id < 0


# loading inputs: annotated graphs, list of branch decisions

def load_pkl_data(data_path):
    """
    :param data_path: the path for dataset
    :return: graphs = [] * n_progs, branch_samples = [[]*n_paths] * n_progs
    """
    with open(data_path, 'rb') as f:
        dataset = pickle.load(f)

    graphs_full = dataset['graphs_full']
    branch_samples = dataset['branch_samples']
    # number of programs, and number of sample paths collected for each program
    n_progs = len(graphs_full)
    n_paths = len(branch_samples[0])
    # taking the head node of each graph
    graphs = [nodes[0] for nodes in graphs_full]

    ## make the sample paths non-IMMUTABLE here
    for i in range(n_progs):
        for j in range(n_paths):
            branch_samples[i][j].flags.writeable = False

    data = {'graphs':graphs, 'branch_samples':branch_samples, 'graphs_full': graphs_full}
    return data

def load_str_data(data_path):
    """
    :param data_path: here data_path is a directory, with subdirs data_path/Graphs and data_path/Samples
    :return: data
    """
    graph_dir = os.path.join(data_path, 'Graphs')
    sample_dir = os.path.join(data_path, 'Samples')
    # look for all subdirs
    graph_files_names = os.listdir(graph_dir)
    graph_names = [name[:-4] for name in graph_files_names if name.endswith('.txt')]
    graphs_full = []
    branch_samples = []
    for name in graph_names:
        with open(name, 'r') as f:
            line = f.readline()
            graph_dict = eval(line)
            n_nodes = len(graph_dict)
            nodes = [Graph_Node(i, graph_dict[i][0]) for i in range(n_nodes)]
            for i in range(n_nodes):
                nodes[i].branches = graph_dict[i][1]
        with open(name + '_sample', 'r') as f:
            lines = f.readlines()
            samples = [np.array(eval(line)) for line in lines]
            for sample in samples:
                sample.flags.writeable = False
        graphs_full.append(nodes)
        branch_samples.append(samples)
    graphs = [nodes[0] for nodes in graphs_full] ## assume id-0 is the head node
    return {'graphs':graphs, 'branch_samples':branch_samples, 'graphs_full': graphs_full}

# main simulation section
# recover and save
def simulate(num_stp, settings, data_path, policy):
    """
    :param num_stp: number of iterations
    :param settings: the Settings object, contain hyperparameters of the model
    :param data: dictionary, {'graphs': graphs, 'branch_samples': branch_samples}
    :param policy: function handle of the policy
    :return:
    """
    timer_debug = 0
    M = settings.M
    n_skip = settings.n_skip
    beta = settings.beta
    alpha = settings.alpha
    data = load_pkl_data(data_path)
    graphs = data['graphs']
    branch_samples = data['branch_samples']
    n_progs = len(graphs)
    n_paths = len(branch_samples[0])

    # maintain a global time
    t_glob = 0
    # maintain a counter
    count_comp = 0
    # initialize:
    # type of jobs in the head of line, nodes in the graph, sample path,
    # index in the sample path, size, next job in the queue,
    # memory allocation decisions
    ### make them objects, except job size being numpy array
    job_types = np.random.randint(0, n_progs, [n_skip])
    path_inds = np.random.randint(0, n_paths, [n_skip])
    job_paths = [branch_samples[job_types[i]][path_inds[i]] for i in range(n_skip)]
    job_nodes = [graphs[job_types[i]] for i in range(n_skip)]
    job_indices = np.zeros([n_skip], dtype=int)
    job_sizes = np.array([job_nodes[i].size for i in range(n_skip)])
    ### restructure to a queue, with head and tail
    #next_arrs = np.roll(np.arange(0, n_skip), -1) # 2, 3, ..., 1
    #prev_arrs = np.roll(np.arange(0, n_skip), 1)
    #earl_arr = 0
    job_orders = [Order_Node(i) for i in range(n_skip)]
    for i in range(n_skip):
        if i > 0:
            job_orders[i].prev = job_orders[i-1]
        if i < n_skip-1:
            job_orders[i].next = job_orders[i+1]
    head_node = job_orders[0]
    tail_node = job_orders[-1]
    cur_mem = np.zeros([n_skip])

    # need some burn-in periods...
    for cur_stp in range(num_stp):
        # update memory allocation
        cur_mem = policy(M, settings, job_sizes, [job_orders, head_node, tail_node]) # this is specific to each policy
        ## logging lines for debug
        # if cur_stp > 0:
        #     order_str = traverse_ids(head_node, n_skip)
        #     logging.info('cur_stp={}, last event={}, \n cur_order={}, \n job_sizes={}, \n cur_mem={}'.format(cur_stp, i, order_str, job_sizes, cur_mem))
        # compute rate
        speedups = np.clip(cur_mem-alpha * job_sizes, EPS,  (beta-alpha)*job_sizes)
        scales = job_sizes / speedups
        # sample next event
        ev_times = np.random.exponential(1, [n_skip]) * scales # generate exponential on a less frequent bases
        # getting the event i
        i = np.argmin(ev_times)
        t_incre = np.min(ev_times)
        #logging.info('cur_stp={}, i={}'.format(cur_stp, i))
        t_glob += t_incre
        # update corresponding job
        #logging.info('node id={}, index={}'.format(job_nodes[i].id, job_indices[i]))
        job_nodes[i] = job_nodes[i].branches[job_paths[i][job_indices[i]]] # this needs optimize; maybe make it a linked list
        job_indices[i] += 1
        if job_nodes[i].id >= 0:
            job_sizes[i] = job_nodes[i].size
            #logging.warning('job_size={}, {}'.format(job_sizes[i], job_nodes[i].size))
            # if i == 0:
            #     logging.warning('job 0, node_id={}, job_sizes={}'.format(job_nodes[i].id, job_nodes[i].size))
        else:
            # the job has finished, counter+1, regenerate
            count_comp += 1
            # update job type
            new_type = np.random.randint(0, n_progs)
            job_types[i] = new_type
            job_paths[i] = branch_samples[new_type][np.random.randint(0, n_paths)]
            new_node = graphs[new_type]
            job_nodes[i] = new_node
            job_sizes[i] = new_node.size
            # restart index
            job_indices[i] = 0
            # update the linked list or arrival orders
            order_node = job_orders[i]
            if order_node == head_node:
                head_node = order_node.next
                tail_node.next = order_node
                order_node.prev = tail_node
                order_node.next = None
                tail_node = order_node
            elif order_node == tail_node:
                pass
            else:
                order_node.next.prev = order_node.prev
                order_node.prev.next = order_node.next
                tail_node.next = order_node
                order_node.prev = tail_node
                order_node.next = None
                tail_node = order_node
            ### logging lines for debug
            # logging.warning('restart job {}, counter = {}, t={}'.format(i, count_comp, t_glob))
            # order_str = traverse_ids(head_node, n_skip)
            # logging.warning('current order {}'.format(order_str))
    return count_comp, t_glob

def policy_best_fit(M, settings, job_sizes, arrival_orders, fix=-1):
    """
    :param M: total capacity
    :param settings: settings
    :param job_sizes: np array of the sizes of current phases of the jobs
    :param arrival_orders: [job_orders, head_node, tail_node]
    :return: mem: np array of the cache allocation
    """
    (job_orders, head_node, tail_node) = arrival_orders
    n_skip = job_sizes.shape[0]
    sort_indices = np.argsort(job_sizes)
    #logging.info('sort_indices={}'.format(sort_indices))
    mem = np.zeros([n_skip])
    M_remain = M
    for i in range(n_skip):
        # allocate from large to small
        index = sort_indices[-i]
        if fix > 0:
            alloc = fix
        else:
            alloc = settings.beta * job_sizes[index]
        if M_remain < alloc:
        # if not enough, allocate the remaining
            alloc = M_remain
            mem[index] = alloc
            break
        else:
            mem[index] = alloc
            M_remain -= alloc
    return mem

def policy_fcfs(M, settings, job_sizes, arrival_orders, fix=-1):
    """
    :param M: total capacity
    :param settings: settings
    :param job_sizes: np array of the sizes of current phases of the jobs
    :param arrival_orders: [job_orders, head_node, tail_node]
    :return: mem: np array of the cache allocation
    """
    (job_orders, head_node, tail_node) = arrival_orders
    n_skip = job_sizes.shape[0]
    mem = np.zeros([n_skip])
    M_remain = M
    cur_node = head_node
    while True:
        index = cur_node.id
        if fix > 0:
            alloc = fix
        else:
            alloc = settings.beta * job_sizes[index]
        if M_remain < alloc:
            alloc = M_remain
            mem[index] = alloc
            break
        else:
            mem[index] = alloc
            M_remain -= alloc
        cur_node = cur_node.next
        if cur_node is None:
            break
    return mem


if __name__ == '__main__':
    #np.random.seed(0)
    settings1 = Settings(n_skip=30, M = 100, alpha=0.5, beta=1.5)

    static_best_fit = partial(policy_best_fit, fix=20)
    static_fcfs = partial(policy_fcfs, fix=20)

    dataset_dir = './datasets'
    dataset_name = 'sequential.pkl'
    data_path = os.path.join(dataset_dir, dataset_name)

    count_comp, t_glob = simulate(3*10**5, settings1, data_path, static_fcfs)
    print(count_comp / t_glob)

    # timing analysis: n_skip=30, M = 100, alpha=0.5, beta=1.5, sequential 30 phase
    # 1e4 iterations, around 5~9 sec, static_fcfs
    # generating random variables takes 1.4% of the time
    # policy takes 0.6% of the time
    # restarting job takes 0.2% of the time
    # updatig node takes 0.5% of the time
    # argmin and min takes 2% of the time
    # generating rates takes 5%
    # preprocessing takes 0.02%
    # logging takes > 90% of the time!
    # after commenting out logging, ~0.5 sec
    # policy takes 8% time
    # computing rates, generate r.v., take min takes 80%-90% of the time
