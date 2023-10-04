import os
import random
import gunfolds.utils.graphkit as gk
from gunfolds.utils import bfutils as bfu
import gunfolds.solvers.clingo_rasl as rsl
import time
import gunfolds.utils.zickle as zkl
import argparse
from gunfolds.utils.calc_procs import get_process_count
from gunfolds.viz.gtool import plotg

CLINGO_LIMIT= 64
PNUM = int(min(CLINGO_LIMIT, get_process_count(1)))

parser = argparse.ArgumentParser(description='Run settings.')
parser.add_argument("-b", "--BATCH", default=1, help="slurm batch.", type=int)
parser.add_argument("-t", "--TIMEOUT", default=24, help="timeout in hours", type=int)
parser.add_argument("-n", "--NODE", default=5, help="number of nodes.", type=int)
parser.add_argument("-d", "--DEG", default=1.5, help="degree of graph.", type=float)
parser.add_argument("-u", "--UNDERSAMPLING", default=3, help="undersampling of the graph", type=int)
parser.add_argument("-x", "--MAXU",default=20, help="maximum number of undersampling to look for solution.", type=int)
parser.add_argument("-c", "--CAPSIZE",default=0, help="stop traversing after growing equivalence class to this size."
                    , type=int)
parser.add_argument("-p", "--PNUM",default=PNUM, help="number of CPUs in machine.", type=int)
args = parser.parse_args()
# generate a random graph with required sctructure and density
num_edges =int(args.DEG * args.NODE)
x = []
while args.UNDERSAMPLING > len(x):
    g = gk.ringmore(args.NODE, max(1,num_edges-args.NODE))
# undersample
    x = bfu.all_undersamples(g)
print('------------------------------')
print('there are ', len(x), ' unique undersampled versions of g')
# select an undersampled version
if args.UNDERSAMPLING <= len(x):
    g_broken = x[args.UNDERSAMPLING-1].copy()

    # break (replace) one edge randomly
    node = random.randint(1, len(g_broken))
    child = random.randint(1, len(g_broken))
    if child in g_broken[node]:
        choices = [1, 2, 3]
        choices.remove(g_broken[node][child])
        g_broken[node][child] = random.choice(choices)
    else:
        g_broken[node][child] = random.randint(1, 4)
    # run a weighted (optimization) search (document the time it took)
    print("nodes : {0:}, undersampling : {1:}, Degree: {2:}, Batch : {3:}".format(
        args.NODE, args.UNDERSAMPLING, args.DEG, args.BATCH))
    startTime = int(round(time.time() * 1000))
    r = rsl.drasl([g_broken], args.CAPSIZE, weighted=True, urate=min(args.MAXU,(3*len(g_broken)+1)),
                  timeout=60 * 60 * args.TIMEOUT, pnum=args.PNUM)
    endTime = int(round(time.time() * 1000))
    sat_time = endTime - startTime

    min_err = {'directed': (0, 0), 'bidirected': (0, 0), 'total': (0, 0)}
    min_norm_err = {'directed': (0, 0), 'bidirected': (0, 0), 'total': (0, 0)}
    min_val = 1000000
    print('number of optimal solutions is', len(r))
    for answer in r:  # compute the absolute error and save it
        curr_errors = gk.OCE(bfu.num2CG(answer[0][0], args.NODE),g)
        # compute the normalized/relative error and save it
        curr_normed_errors = gk.OCE(bfu.num2CG(answer[0][0], args.NODE), g, normalized=True)
        if 0.5*(curr_errors['total'][0]+curr_errors['total'][1]) < min_val:
            min_err = curr_errors
            min_norm_err = curr_normed_errors
            min_val = 0.5*(curr_errors['total'][0]+curr_errors['total'][1])


# compute the absolute error and save it
#     errors = gk.OCE(bfu.num2CG(r[0][0], args.NODE), g)
#     # compute the normalized/relative error and save it
#     normed_errors = gk.OCE(bfu.num2CG(r[0][0], args.NODE), g, normalized=True)



    results = {'errors': min_err,
               'normed_errors': min_norm_err,
               'g': g,
               'density': args.DEG,
               'u': args.UNDERSAMPLING,
               'g_broken': g_broken,
               'ms': sat_time,
               'results': r}

    filename = 'res_CAP_' + str(args.CAPSIZE) + '_/' + 'optimization_nodes_' + str(args.NODE) + '_u_'\
               + str(args.UNDERSAMPLING) + '_degree_' + str(args.DEG) \
               + '_batch_' + str(args.BATCH) + '_TimeOut_' + str(args.TIMEOUT) + 'hrs_MaxU_' + str(args.MAXU) \
               + '_CAP_' + str(args.CAPSIZE) + '.zkl'

    if not os.path.exists('res_CAP_' + str(args.CAPSIZE) + '_'):
        os.makedirs('res_CAP_' + str(args.CAPSIZE) + '_')

    zkl.save(results, filename)
    print('It took {0:} seconds to compute with norm_err {1:}, {2:}'.format(sat_time/1000,
                                                                            round(min_norm_err['directed'][0], 2),
                                                                            round(min_norm_err['directed'][1], 2)))
    print('------------------------------')