import os
from gunfolds.utils import zickle as zkl
import pickle
import numpy as np
import time, socket
import scipy
from gunfolds.solvers.clingo_rasl import drasl
import argparse
from gunfolds.utils import graphkit as gk
import distutils.util
from gunfolds.utils.calc_procs import get_process_count
from gunfolds.viz import gtool as gt
from gunfolds.utils import bfutils

TIMEOUT = 6 * 60 * 60  # seconds = 6 hours
POSTFIX = 'D_RASL'
CLINGO_LIMIT = 64
PNUM = int(min(CLINGO_LIMIT, get_process_count(1)))

parser = argparse.ArgumentParser(description='Run settings.')
parser.add_argument("-c", "--CAPSIZE", default=0, help="stop traversing after growing equivalence class to this size.",
                    type=int)
parser.add_argument("-n", "--NODE", default=10, help="number of nodes.", type=int)
parser.add_argument("-u", "--UNDERSAMPLING", default="2,3,4", help="number of undersampling. e.g. -u=2,3,4", type=str)
parser.add_argument("-d", "--DENITIES", default="0.2,0.25,0.3", help="densities to be ran. e.g. -d=0.2,0.25,0.3",
                    type=str)
parser.add_argument("-b", "--BATCH", default=1, help="slurm batch.", type=int)
parser.add_argument("-m", "--MODEDOUBLE", default="f",
                    help="whether or not use two input for d_rasl. Use y for true and n for false.", type=str)
parser.add_argument("-s", "--SCC", default="f", help="whether or not use scc for d_rasl.Use y for true and n for false",
                    type=str)
parser.add_argument("-a", "--ARRAY", default=1, help="number of total batches.", type=int)
parser.add_argument("-p", "--PNUM", default=PNUM, help="number of CPUs in machine.", type=int)
args = parser.parse_args()
MODEDOUBLE = bool(distutils.util.strtobool(args.MODEDOUBLE))
SCCMODE = bool(distutils.util.strtobool(args.SCC))


# dens_list = args.DENITIES.split(',')
# dens_list = [float(item) for item in dens_list]
# u_list = args.UNDERSAMPLING.split(',')
# u_list = [int(item) for item in u_list]


def clingo_caller(g):
    startTime = int(round(time.time() * 1000))
    c = drasl(g, capsize=args.CAPSIZE, urate=min(30, (3 * len(g[0]) + 1)), timeout=TIMEOUT, scc=SCCMODE, pnum=args.PNUM,
              weighted=False, configuration="jumpy")
    endTime = int(round(time.time() * 1000))
    sat_time = endTime - startTime
    return c, sat_time


def fan_wrapper(graph_true, g1_dens, rate, g1_deg):
    output = {}
    try:
        try:
            # graphs = [bfutils.undersample(graph_true, rate)]
            g = {1: {}, 2: {1: 1}, 3: {4: 1}, 4: {1: 1, 2: 1}, 5: {2: 1, 5: 1}}  # gk.bp_mean_degree_graph(5, 1.2)
            g3 = {1: {2: 2, 5: 2}, 2: {1: 2, 5: 2}, 3: {1: 1}, 4: {}, 5: {1: 3, 2: 3, 5: 1}}
            g3 = bfutils.all_undersamples(g)[2]
            graphs = [g3]
            if MODEDOUBLE:
                graphs.append(bfutils.undersample(graph_true, rate + 1))
            if not np.prod([bfutils.g2num(x) for x in graphs]):
                print("input graph is empty. Moving on to next graph")
                return output
            else:
                c, sat_time = clingo_caller(graphs)

        except TimeoutError:
            c = None
            sat_time = None
            print("Time Out. {:10} seconds have passed. moving to next graph".format(TIMEOUT))

    except MemoryError:
        c = None
        sat_time = None
        print('memory error... retrying')

    if sat_time is not None:
        print("{:8} : {:4}  {:10} seconds". \
              format(round(g1_dens, 3), len(c),
                     round(sat_time / 1000., 3)))
    if c is not None:
        output = {'gt': graph_true,
                  'solutions': {'eq_size': len(c), 'ms': sat_time},
                  'u': rate, 'dens': g1_dens, 'deg': g1_deg}

    return output


for nodes in [args.NODE]:  # np.sort(densities.keys())[0:]:
    print(nodes, ': ----')
    print('')
    # if SCCMODE:
    all_graphs = zkl.load('datasets/graphs_node_' + str(nodes) + '_gcd1_.zkl')
    # else:
    #     all_graphs = zkl.load('datasets/graphs_node_' + str(nodes) + '.zkl')
    B = int(len(all_graphs) / args.ARRAY)
    batch_graph = all_graphs[(int(args.BATCH) - 1) * B:(int(args.BATCH)) * B]
    density_distibution = np.zeros(11)
    print("{:2}: {:8} : {:10}  {:10}".format('id', 'density', 'eq class', 'time'))
    if not os.path.exists('res_CAP_' + str(args.CAPSIZE) + '_'):
        os.makedirs('res_CAP_' + str(args.CAPSIZE) + '_')
    for item in batch_graph:
        if not bfutils.g2num(item['gt']) == 0:
            eqclasses = fan_wrapper(graph_true=item['gt'],
                                              g1_dens=item['dens'], rate=item['u'], g1_deg=item['deg'])
            if not len(eqclasses) == 0:
                filename = 'res_CAP_' + str(args.CAPSIZE) + '_/' + \
                           socket.gethostname().split('.')[0] + \
                           '_nodes_' + str(nodes) + '_batch_' + str(args.BATCH) + '_' + \
                           POSTFIX + '_CAPSIZE_' + str(args.CAPSIZE) + '_SCC_' + str(SCCMODE) + '_doubleMode_' + str(
                    MODEDOUBLE)
                with open(filename, mode='a+b') as fp:
                    pickle.dump(eqclasses, fp)

