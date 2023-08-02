import os
from gunfolds.utils import bfutils
from gunfolds.utils import zickle as zkl
import pickle
import numpy as np
import time, socket
from os import listdir
from gunfolds.solvers.clingo_rasl import drasl
import argparse
from gunfolds.utils import graphkit as gk
import distutils.util
from gunfolds.utils.calc_procs import get_process_count

TIMEOUT=5 * 24 * 60 * 60 # seconds = 132 hours
POSTFIX='D_RASL_after_optim'
CLINGO_LIMIT= 64
PNUM = int(min(CLINGO_LIMIT, get_process_count(1)))

parser = argparse.ArgumentParser(description='Run settings.')
parser.add_argument("-c", "--CAPSIZE",default=0, help="stop traversing after growing equivalence class to this size.", type=int)
parser.add_argument("-n", "--NODE", default=6,help="number of nodes.", type=int)
parser.add_argument("-b", "--BATCH",default=1, help="slurm batch.", type=int)
parser.add_argument("-s", "--SCC",default="f", help="whether or not use scc for d_rasl.Use y for true and n for false", type=str)
parser.add_argument("-p", "--PNUM",default=PNUM, help="number of CPUs in machine.", type=int)
parser.add_argument("-t", "--TIMEOUT", default=TIMEOUT, help="timeout in hours", type=int)
args = parser.parse_args()
SCCMODE = bool(distutils.util.strtobool(args.SCC))



def clingo_caller(g):
    startTime = int(round(time.time() * 1000))
    c = drasl(g, capsize=args.CAPSIZE, urate=min(15,(3*len(g[0])+1)), timeout=args.TIMEOUT, scc=SCCMODE, pnum=args.PNUM)
    endTime = int(round(time.time() * 1000))
    sat_time = endTime-startTime
    return c, sat_time


def fan_wrapper(graph_true,rate):
    output = {}
    try:
        try:
            graphs = [bfutils.undersample(graph_true, rate)]
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
        print ('memory error... retrying')

    if sat_time is not None:
        print("eq size: {0:}, time: {1:}".format(len(c), round((sat_time / 60000), 3)))
    if c is not None:
        output = {'g_opt': graph_true, 'solutions': {'eq': c, 'ms': sat_time}, 'u': rate}

    return output


for nodes in [args.NODE]:
    print (nodes, ': ----')
    print ('')
    list_sizes = listdir('./results/weighted_zero_cap/' + str(nodes) + ' nodes/')
    list_sizes.sort()
    if list_sizes[0].startswith('.'):
        list_sizes.pop(0)
    res = zkl.load('./results/weighted_zero_cap/' + str(nodes) + ' nodes/' + list_sizes[args.BATCH-1])
    print("batch : {0:},\nerror : {1:} \nNum nodes  "
          ": {2:},\nprevious time : {3:} mis\nprevious results : {4:}".format(args.BATCH,
                                                                 res['normed_errors'],
                                                                 args.NODE,
                                                                 round(((res['ms'])/60000), 3),
                                                                 res['results']))
    if not os.path.exists('res_drasl_after_optim/'+str(args.NODE)):
        os.makedirs('res_drasl_after_optim/'+str(args.NODE))
    eqclasses = fan_wrapper(graph_true=res['opt_g'], rate=res['u'])
    if not len(eqclasses) == 0:
        min_err = {'total': (0, 0)}
        min_norm_err = {'total': (0, 0)}
        min_val = 1000000
        min_error_graph = {}
        for answer in eqclasses['solutions']['eq']:
            curr_errors = gk.OCE(bfutils.num2CG(answer[0], len(res['g'])), res['g'], undirected=False, normalized=False)
            '''compute the normalized/relative error and save it'''
            curr_normed_errors = gk.OCE(bfutils.num2CG(answer[0], len(res['g'])), res['g'],
                                        normalized=True,
                                        undirected=False)
            if 0.5 * (curr_errors['total'][0] + curr_errors['total'][1]) < min_val:
                min_err = curr_errors
                min_norm_err = curr_normed_errors['total']
                min_val = 0.5 * (curr_errors['total'][0] + curr_errors['total'][1])
                min_error_graph = answer
                if min_val < 0.001:
                    break
        new_results ={'solutions': eqclasses['solutions'],
                      'min_norm_err': min_norm_err,
                      'min_error_graph': min_error_graph}
        filename = 'res_drasl_after_optim/'+str(args.NODE) + '/' + \
               'nodes_' + str(nodes) + '_batch_' + str(args.BATCH) + '_' + \
               POSTFIX + '_CAPSIZE_' + str(args.CAPSIZE) + '_SCC_'+str(SCCMODE)
        zkl.save([res, new_results], filename + '.zkl')

