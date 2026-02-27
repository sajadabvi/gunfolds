import os
import random
import gunfolds.tools.graphkit as gk
from gunfolds.tools import bfutils as bfu
import gunfolds.tools.clingo_rasl as rsl
import time
import gunfolds.tools.zickle as zkl
import argparse
from gunfolds.tools.calc_procs import get_process_count
import distutils.util

CLINGO_LIMIT= 64
PNUM = int(min(CLINGO_LIMIT, get_process_count(1)))

parser = argparse.ArgumentParser(description='Run settings.')
parser.add_argument("-b", "--BATCH", default=1, help="slurm batch.", type=int)
parser.add_argument("-t", "--TIMEOUT", default=24, help="timeout in hours", type=int)
parser.add_argument("-n", "--NODE", default=6, help="number of nodes.", type=int)
parser.add_argument("-d", "--DEG", default=2.5, help="degree of graph.", type=float)
parser.add_argument("-u", "--UNDERSAMPLING", default=7, help="undersampling of the graph", type=int)
parser.add_argument("-s", "--SCC",default="f", help="whether or not use scc for d_rasl.Use y for true and n for false", type=str)
parser.add_argument("-x", "--MAXU",default=20, help="maximum number of undersampling to look for solution.", type=int)
parser.add_argument("-c", "--CAPSIZE",default=10000, help="stop traversing after growing equivalence class to this size."
                    , type=int)
parser.add_argument("-p", "--PNUM",default=PNUM, help="number of CPUs in machine.", type=int)
args = parser.parse_args()
SCCMODE=bool(distutils.util.strtobool(args.SCC))

# generate a random graph with required sctructure and density
num_edges =int(args.DEG * args.NODE)
g = gk.ringmore(args.NODE, max(1,num_edges-args.NODE))
# undersample
x = bfu.all_undersamples(g)
count =0
while (not args.UNDERSAMPLING <= len(x)) and count< 50:
    count +=1
    g = gk.ringmore(args.NODE, max(1, num_edges - args.NODE))
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
                  timeout=60 * 60 * args.TIMEOUT, pnum=args.PNUM,scc=SCCMODE)
    endTime = int(round(time.time() * 1000))
    G1_opt = bfu.num2CG(r[0],args.NODE)
    Gu_opt = bfu.undersample(G1_opt,r[1][0])
    sat_time = endTime - startTime
    init_errors = gk.OCE(g, bfu.num2CG(r[0], args.NODE))
    # compute the normalized/relative error and save it
    init_normed_errors = gk.OCE(g, bfu.num2CG(r[0], args.NODE), normalized=True)
    print('1. It took {0:} seconds to compute with norm_err {1:}, {2:}'.format(sat_time / 1000,
                                                                           round(init_normed_errors['directed'][0], 2),
                                                                           round(init_normed_errors['directed'][1], 2)))
    startTime2 = int(round(time.time() * 1000))
    c = rsl.drasl(glist=[Gu_opt],capsize=args.CAPSIZE, weighted=False, urate=min(args.MAXU, (3 * len(Gu_opt) + 1)),
                  timeout=60 * 60 * args.TIMEOUT, scc=SCCMODE, pnum=args.PNUM)
    endTime2 = int(round(time.time() * 1000))
    sat_time2 = endTime2 - startTime2
    min_err = {'directed': (0, 0), 'bidirected': (0, 0), 'total': (0, 0)}
    min_norm_err = {'directed': (0, 0), 'bidirected': (0, 0), 'total': (0, 0)}
    min_val = 1000000
    print('cap size is', len(c))
    for answer in c:  # compute the absolute error and save it
        curr_errors = gk.OCE(g, bfu.num2CG(answer[0], args.NODE))
        # compute the normalized/relative error and save it
        curr_normed_errors = gk.OCE(g, bfu.num2CG(answer[0], args.NODE), normalized=True)
        if 0.5*(curr_errors['total'][0]+curr_errors['total'][1]) < min_val:
            min_err = curr_errors
            min_norm_err = curr_normed_errors
            min_val = 0.5*(curr_errors['total'][0]+curr_errors['total'][1])
    results = {'errors': min_err,
               'normed_errors': min_norm_err,
               'g': g,
               'density': args.DEG,
               'u': args.UNDERSAMPLING,
               'g_broken': g_broken,
               'G1_opt': bfu.num2CG(r[0], args.NODE),
               'ms': sat_time,
               'eq_class': c}
    filename = 'res_CAP_' + str(args.CAPSIZE) + '_/' + 'optimization_nodes_' + str(args.NODE) + '_u_'\
               + str(args.UNDERSAMPLING) + '_degree_' + str(args.DEG) \
               + '_batch_' + str(args.BATCH) + '_TimeOut_' + str(args.TIMEOUT) + 'hrs_MaxU_' + str(args.MAXU) \
               + '_CAP_' + str(args.CAPSIZE) + '.zkl'
    if not os.path.exists('res_CAP_' + str(args.CAPSIZE) + '_'):
        os.makedirs('res_CAP_' + str(args.CAPSIZE) + '_')
    zkl.save(results, filename)
    print('2. It took {0:} seconds to compute with norm_err {1:}, {2:}'.format((sat_time + sat_time2)/1000,
                                                                            round(min_norm_err['directed'][0], 2),
                                                                            round(min_norm_err['directed'][1], 2)))
    print('------------------------------')
else:
    print('G1 cannot be undersampled by ',args.UNDERSAMPLING, 'after 50 tries change setting')
