from gunfolds.utils import bfutils
from gunfolds.utils import graphkit as gk
from gunfolds.utils import zickle as zkl
import argparse
import random
from gunfolds.viz import dbn2latex as latex
from gunfolds.viz.dbn2latex import output_graph_figure

REPEATS = 50

parser = argparse.ArgumentParser(description='Run settings.')
parser.add_argument("-n", "--NODE", default=6, help="number of nodes in each scc.", type=int)
parser.add_argument("-z", "--NUMSCC", default=9, help="number of SCCs in the graph", type=int)
parser.add_argument("-u", "--UNDERSAMPLING", default="2,3,4",help="number of undersampling. e.g. -u=2,3,4", type=str)
parser.add_argument("-d", "--DENITIES", default="0.2,0.25,0.3", help="densities to be ran. e.g. -d=0.2,0.25,0.3", type=str)
parser.add_argument("-g", "--DEGREE", default="1.5", help="average degree to be ran. e.g. -g=0.9,2,3,5", type=str)
args = parser.parse_args()
dens_list = args.DENITIES.split(',')
dens_list = [float(item) for item in dens_list]
u_list = args.UNDERSAMPLING.split(',')
u_list = [int(item) for item in u_list]
deg_list = args.DEGREE.split(',')
deg_list = [float(item) for item in deg_list]


for nodes in [args.NODE]:
    print (nodes, ': ----')
    print ('')
    graphs = []
    devisables = list(filter(lambda x: (nodes % x == 0), [i + 1 for i in range(1, nodes)]))
    devisables = devisables[:-1]
    for i in [args.NUMSCC]:
        count = 0
        try:
            while True:
                for deg in deg_list:
                    num_edges =(deg * nodes)
                    desired_dens = num_edges/((nodes)**2)
                    if desired_dens > 1:
                        continue
                    g = gk.ring_sccs(nodes,i, dens=desired_dens, degree=1, max_cross_connections=2)
                    # g = gk.ensure_gcd1(gk.ringmore(nodes, max(int(desired_dens * nodes**2)-nodes,1)))
                    while not 21 < len(bfutils.all_undersamples(g)):
                        g = gk.ring_sccs(nodes, i, dens=desired_dens, degree=1, max_cross_connections=2)
                        # gk.ensure_gcd1(gk.ringmore(nodes, max(int(desired_dens * nodes ** 2) - nodes, 1)))
                    dens = gk.density(g)
                    latex.output_graph_figure(g)
                    for u in u_list:
                        tup ={}
                        tup['gt'],tup['dens'],tup['u'], tup['deg'], tup['num_scc']= g,dens,u,deg,i
                        graphs.append(tup)
                        count += 1
                        print(count)
                        if count >= (3*REPEATS):
                            raise StopIteration
        except StopIteration:
            pass
    random.shuffle(graphs)
    zkl.save(graphs, 'datasets/graph_multi_scc_size_'+str(nodes)+'_numSCC_'+str(args.NUMSCC)+'_connected.zkl')
