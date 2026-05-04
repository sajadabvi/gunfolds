""" This module contains graph format conversion functions """
from __future__ import print_function
from networkx.algorithms.components import condensation, strongly_connected_components
import networkx as nx
import numpy as np
import igraph
import sys

################### Start of Internal Conversions ###################

def nodenum(edgepairs):
    """
    Returns the number of nodes in the graph

    :param edgepairs: list of edge pairs
    :type edgepairs: list
    
    :returns: number of nodes in the graph
    :rtype: integer
    """
    nodes = 0
    for e in edgepairs:
        nodes = np.max([nodes, int(e[0]), int(e[1])])
    return nodes

def g2num(g):
    """ 
    Convert a graph into a long int 

    :param g: ``gunfolds`` graph
    :type g: dictionary (``gunfolds`` graphs)
    
    :returns: unique number for each graph considering only directed edges
    :rtype: long integer
    """
    n = len(g)
    num = ['0']*n*n
    for v in range(1, n + 1):
        idx = (v-1)*n
        for w in g[v]:
            num[idx + (w-1)] = '1'

    return int(''.join(num), 2)


def ug2num(g):
    """
    Convert non-empty edges into a tuple of (directed, bidriected) in
    binary format
    
    :param g: ``gunfolds`` graph
    :type g: dictionary (``gunfolds`` graphs)
    
    :returns: unique number for each graph considering directed and bidirected in binary format
    :rtype: a tuple of binary integer
    """
    n = len(g)
    n2 = n ** 2 + n
    num = 0
    mask = 0
    num2 = 0
    for v in g:
        for w in g[v]:
            if g[v][w] in (1, 3):
                mask = (1 << (n2 - v * n - w))
                num |= mask
            if g[v][w] in (2, 3):
                num2 |= mask
    return num, num2


def bg2num(g):
    """
    Convert bidirected edges into a binary format
    
    :param g: ``gunfolds`` graph
    :type g: dictionary (``gunfolds`` graphs)
    
    :returns: unique number for each graph considering bidirected in binary format
    :rtype: a tuple of binary integer
    """
    n = len(g)
    n2 = n ** 2 + n
    num = 0
    for v in g:
        for w in g[v]:
            if g[v][w] in (2, 3):
                num = num | (1 << (n2 - v * n - w))
    return num

def num2CG(num, n):
    """
    Converts a number  whose binary representaion encodes edge
    presence/absence into a compressed graph representaion
    
    :param num: unique graph representation in numbers
    :type num: integer
    
    :param n: number of nodes
    :type n: integer
    
    :returns: ``gunfolds`` graph 
    :rtype: dictionary (``gunfolds`` graphs)
    """
    s = bin(num)[2:].zfill(n*n)
    g = {i+1: {} for i in range(n)}
    for v in g:
        for w in range(n):
            if s[(v-1)*n:(v-1)*n+n][w] == '1':
                g[v][w+1] = 1
    return g


def dict_format_converter(H):
    """ Convert a graph from the set style dictionary format to the integer style
    
        :param H: set style dictionary format
        :type H: dictionary 
        
        :returns: ``gunfolds`` graph 
        :rtype: dictionary (``gunfolds`` graphs)

        >>> test = {'1': {'1': {(0, 1)},
        ...   '2': {(0, 1), (2, 0)},
        ...   '3': {(0, 1), (2, 0)},
        ...   '4': {(2, 0)},
        ...   '5': {(0, 1)}},
        ...  '2': {'1': {(2, 0)}, '2': {(0, 1)}, '5': {(0, 1), (2, 0)}},
        ...  '3': {'1': {(0, 1), (2, 0)}, '2': {(0, 1)}, '5': {(0, 1)}},
        ...  '4': {'1': {(2, 0)},
        ...   '2': {(0, 1)},
        ...   '3': {(0, 1)},
        ...   '4': {(0, 1)},
        ...   '5': {(0, 1)}},
        ...  '5': {'1': {(0, 1)}, '2': {(0, 1), (2, 0)}, '5': {(0, 1)}}}
        >>> dict_format_converter(test)
        {1: {1: 1, 2: 3, 3: 3, 4: 2, 5: 1}, 2: {1: 2, 2: 1, 5: 3}, 3: {1: 3, 2: 1, 5: 1}, 4: {1: 2, 2: 1, 3: 1, 4: 1, 5: 1}, 5: {1: 1, 2: 3, 5: 1}}
        >>>
    """
    H_new = {}
    for vert_a in H:
        H_new[int(vert_a)] = {}
        for vert_b in H[vert_a]:
            edge_val = 0
            if (0, 1) in H[vert_a][vert_b]:
                edge_val = 1
            if (2, 0) in H[vert_a][vert_b]:
                edge_val = 2 if edge_val == 0 else 3
            if edge_val:
                H_new[int(vert_a)][int(vert_b)] = edge_val
    return H_new


def g2ian(g):
    """
    Convert a set-based graph dictionary to its integer-encoded form.

    The input graph uses sets of edge tuples ``(0, 1)`` and ``(2, 0)`` to
    indicate directed and bidirected edges respectively.  The returned graph
    stores these edges as integers ``1`` (directed), ``2`` (bidirected) or
    ``3`` (both).

    :param g: graph in set-based format
    :type g: dictionary

    :returns: graph with integer-encoded edges
    :rtype: dictionary
    """
    return dict_format_converter(g)


def ian2g(g):
    """
    Convert an integer-encoded graph dictionary to its set-based form.

    Each edge in the input is encoded as ``1`` (directed), ``2`` (bidirected)
    or ``3`` (both).  The output graph represents edges using sets of tuples
    ``(0, 1)`` and ``(2, 0)`` to mark the corresponding edge types.

    :param g: graph with integer-encoded edges
    :type g: dictionary

    :returns: graph in set-based format
    :rtype: dictionary
    """
    c = {1: {(0, 1)}, 2: {(2, 0)}, 3: {(0, 1), (2, 0)}}
    gg = {}
    for w in g:
        gg[str(w)] = {}
        for v in g[w]:
            gg[str(w)][str(v)] = c[g[w][v]]
    return gg

def edgepairs2g(edgepairs):
    """
    Converts edge pairs to a ``gunfolds`` graph

    :param edgepairs: list of edge pairs
    :type edgepairs: list
    
    :returns: ``gunfolds`` graph
    :rtype: dictionary (``gunfolds`` graph)
    """
    n = nodenum(edgepairs)
    g = {x+1: {} for x in range(n)}
    for e in edgepairs:
        g[int(e[0])][int(e[1])] = 1
    return g

# Adjacency matrix functions

def graph2adj(G):
    """ 
    Convert the directed edges to an adjacency matrix 
    
    :param G: ``gunfolds`` format graph
    :type G: dictionary (``gunfolds`` graphs)
    
    :returns: graph adjacency matrix for directed edges
    :rtype: numpy matrix
    """
    n = len(G)
    A = np.zeros((n, n), dtype=np.int8)
    for v in G:
        A[int(v) - 1, [int(w)-1 for w in G[v] if G[v][w] in (1, 3)]] = 1
    return A


def graph2badj(G):
    """ 
    Convert the bidirected edges to an adjacency matrix 
    
    :param G: ``gunfolds`` format graph
    :type G: dictionary (``gunfolds`` graphs)
    
    :returns: graph adjacency matrix for bidirected edges
    :rtype: numpy matrix
    """
    n = len(G)
    A = np.zeros((n, n), dtype=np.int8)
    for v in G:
        A[int(v) - 1, [int(w)-1 for w in G[v] if G[v][w] in (2, 3)]] = 1
    return A


def adjs2graph(directed, bidirected):
    """ 
    Convert an adjacency matrix of directed and bidirected edges to a graph
    
    :param directed: graph adjacency matrix for directed edges
    :type directed: numpy matrix

    :param bidirected: graph adjacency matrix for bidirected edges
    :type bidirected: numpy matrix
    
    :returns: ``gunfolds`` format graph
    :rtype: dictionary (``gunfolds`` graphs)
    """
    G = {i: {} for i in range(1, directed.shape[0] + 1)}
    for i in range(directed.shape[0]):
        for j in np.where(directed[i, :] == 1)[0] + 1:
            G[i + 1][j] = 1

    for i in range(bidirected.shape[0]):
        for j in range(bidirected.shape[1]):
            if bidirected[i, j] and j != i:
                if j + 1 in G[i + 1]:
                    G[i + 1][j + 1] = 3
                else:
                    G[i + 1][j + 1] = 2
    return G


def g2vec(g):
    """
    Converts ``gunfolds`` graph to a vector

    :param g: ``gunfolds`` graph
    :type g: dictionary (``gunfolds`` graphs)
    
    :returns: a vector representing a ``gunfolds`` graph 
    :rtype: numpy vector
    """
    A = graph2adj(g)
    B = graph2badj(g)
    return np.r_[A.flatten(), B[np.triu_indices(B.shape[0], k=1)]]


def vec2adj(v, n):
    """
    Converts a vector representation to adjacency matrix

    :param v: vector representation of ``gunfolds`` graph
    :type v: numpy vector
    
    :param n: number of nodes
    :type n: integer
    
    :returns: a tuple of Adjacency matrices
    :rtype: a tuple of numpy matrices
    """
    A = np.zeros((n, n))
    B = np.zeros((n, n))
    A[:] = v[:n ** 2].reshape(n, n)
    B[np.triu_indices(n, k=1)] = v[n ** 2:]
    B = B + B.T
    return A, B


def vec2g(v, n):
    """
    Converts a vector representation to ``gunfolds`` graph 

    :param v: vector representation of ``gunfolds`` graph
    :type v: numpy vector
    
    :param n: number of nodes
    :type n: integer
    
    :returns: ``gunfolds`` format graph
    :rtype: dictionary (``gunfolds`` graphs)
    """
    A, B = vec2adj(v, n)
    return adjs2graph(A, B)

def Glag2CG(results):
    """Convert tigramite PCMCI / PCMCIplus results to gunfolds graph format.

    Tigramite convention: ``graph[i, j, tau]`` describes the link from
    variable *i* at time *t - tau* to variable *j* at time *t*, i.e.
    ``graph[source, target, lag]``.  This matches the row=source,
    col=target layout expected by :func:`adjs2graph`, so **no transpose
    is needed**.

    Handles all edge types produced by ``run_pcmci`` and
    ``run_pcmciplus``:

    * ``'-->'`` — directed (lagged *and* contemporaneous from PCMCIplus)
    * ``'o-o'`` — undirected contemporaneous (from ``run_pcmci``)
    * ``'x-x'`` — conflicting contemporaneous orientation (PCMCIplus)
    * ``'o->'`` / ``'<-o'`` — partially oriented contemporaneous

    All contemporaneous link types are mapped to *bidirected* edges in
    the gunfolds graph (edge value 2, or 3 when a directed edge is also
    present).

    Args:
        results (dict): Tigramite results dictionary with at least
            ``'graph'`` (str array ``[N, N, tau_max+1]``) and
            ``'val_matrix'`` (float array, same shape).

    Returns:
        tuple: ``(graph_dict, A_matrix, B_matrix)``

            * *graph_dict* — gunfolds CG (1-indexed node dict)
            * *A_matrix*   — ``val_matrix[:, :, 1]`` (lag-1 weights)
            * *B_matrix*   — ``val_matrix[:, :, 0]`` (contemporaneous weights)
    """
    graph_array = results['graph']

    # Lag-1 directed edges
    directed_edges = np.where(graph_array[:, :, 1] == '-->', 1, 0).astype(int)

    # Contemporaneous edges — treat all link types as bidirected
    contemp = graph_array[:, :, 0]
    bidirected_edges = np.zeros_like(directed_edges)
    for marker in ('o-o', '-->', 'x-x', 'o->', '<-o'):
        bidirected_edges = np.clip(
            bidirected_edges + np.where(contemp == marker, 1, 0).astype(int),
            0, 1,
        )

    graph_dict = adjs2graph(directed_edges, bidirected_edges)
    A_matrix = results['val_matrix'][:, :, 1]
    B_matrix = results['val_matrix'][:, :, 0]

    return graph_dict, A_matrix, B_matrix

def rate(u):
    """
    Converts under sampling rate to ``clingo`` predicate

    :param u: maximum under sampling rate
    :type u: integer
    
    :returns: ``clingo`` predicate for under sampling rate
    :rtype: string
    """
    s = "u(1.."+str(u)+")."
    return s


def clingo_preamble(g):
    """
    Converts number of nodes into a ``clingo`` predicate

    :param g: ``gunfolds`` graph
    :type g: dictionary (``gunfolds`` graphs)
    
    :returns: ``clingo`` predicate
    :rtype: string 
    """
    s = ''
    n = len(g)
    s += '#const n = '+str(n)+'. '
    s += 'node(1..n). '
    return s


def g2clingo(g, directed='hdirected', bidirected='hbidirected', both_bidirected=False, preamble=True):
    """ Convert a graph to a string of grounded terms for clingo
    
        :param g: ``gunfolds`` graph
        :type g: dictionary (``gunfolds`` graphs)
        
        :param directed: name of the variable for directed edges in the observed graph
        :type directed: string

        :param bidirected: name of the variable for  bidirected edges in the observed graph
        :type bidirected: string
        
        :param both_bidirected: (Ask)
        :type both_bidirected: boolean
        
        :param preamble: (Ask)
        :type preamble: boolean
        
        :returns: ``clingo`` predicate
        :rtype: string 
        
        .. code-block:: 
        
           Example: {1:{3:1,4:2,5:3}}
           "1": node 1 has an edge with node 3 => edge(1,3).
           "2": node 1 has an conf with node 4 => conf(1,4).
           "3": node 1 has both edge and conf with node 5 => edge(1,5). conf(1,5).
         
     """
    s = ''
    if preamble:
        s += clingo_preamble(g)
    for v in g:
        for w in g[v]:
            if both_bidirected:
                direction = True
            else:
                direction = v < w
            if g[v][w] & 1:
                s += directed+'('+str(v)+','+str(w)+'). '
            if g[v][w] & 2 and direction:
                s += bidirected+'('+str(v)+','+str(w)+'). '
    return s


def numbered_g2clingo(g, n, directed='hdirected', bidirected='hbidirected'):
    """ Convert a graph to a string of grounded terms for clingo
    
        :param g: ``gunfolds`` graph
        :type g: dictionary (``gunfolds`` graphs)
        
        :param n: number of nodes
        :type n: integer
        
        :param directed: name of the variable for directed edges in the observed graph
        :type directed: string

        :param bidirected: name of the variable for  bidirected edges in the observed graph
        :type bidirected: string
        
        :returns: ``clingo`` predicate
        :rtype: string 
    
         .. code-block:: 

               Example: {1:{3:1,4:2,5:3}}
               "1": node 1 has an edge with node 3 => edge(1,3).
               "2": node 1 has an conf with node 4 => conf(1,4).
               "3": node 1 has both edge and conf with node 5 => edge(1,5). conf(1,5).
           
     """
    s = ''
    for v in g:
        for w in g[v]:
            if g[v][w] & 1:
                s += directed+'('+str(v)+','+str(w)+','+str(n)+'). '
            if g[v][w] & 2 and v < w:
                s += bidirected+'('+str(v)+','+str(w)+','+str(n)+'). '
    return s


def clingo_wedge(x, y, w, n, name='edge'):
    """
    Returns ``clingo`` predicate for weighted edge

    :param x: outgoing edge
    :type x: integer
    
    :param y: incoming edge
    :type y: integer
    
    :param w: weight
    :type w: integer
    
    :param n: number of nodes
    :type n: integer

    :param name: name of the variable for ``clingo``
    :type name: string
    
    :returns: ``clingo`` predicate for weighted edge
    :rtype: string
    """
    edge = name+'('+str(x)+', '+str(y)+', '+str(w)+', '+str(n)+'). '
    return edge


def numbered_g2wclingo(g, num, directed_weights_matrix=None, bidirected_weights_matrix=None,
                       directed='hdirected', bidirected='hbidirected'):
    """
    Convert a graph to a string of grounded terms for ``clingo``
    
    :param g: ``gunfolds`` graph
    :type g: dictionary (``gunfolds`` graphs)

    :param num: index of the graph in the resulting clingo command
    :type num: integer
    
    :param directed_weights_matrix: directed weight matrix
    :type directed_weights_matrix: numpy matrices 
    
    :param bidirected_weights_matrix: bidirected weight matrix
    :type bidirected_weights_matrix: numpy matrices

    :param directed: name of the directed edges in the observed graph
    :type directed: string

    :param bidirected: name of the bidirected edges in the observed
        graph
    :type bidirected: string
    
    :returns: ``clingo`` predicate
    :rtype: string  
    """
    s = ''
    n = len(g)

    if directed_weights_matrix is None:
        directed_weights_matrix = np.ones((n, n))
    directed_weights_matrix = directed_weights_matrix.astype('int')

    if bidirected_weights_matrix is None:
        bidirected_weights_matrix = np.ones((n, n))
    bidirected_weights_matrix = bidirected_weights_matrix.astype('int')

    assert directed_weights_matrix.shape[0] == n
    assert directed_weights_matrix.shape[1] == n
    assert bidirected_weights_matrix.shape[0] == n
    assert bidirected_weights_matrix.shape[1] == n

    for v in range(1, n+1):
        for w in range(1, n+1):
            i = v-1
            j = w-1
            missing = [clingo_wedge(v, w, bidirected_weights_matrix[i, j], num, name='no_'+bidirected),
                       clingo_wedge(v, w, directed_weights_matrix[i, j], num, name='no_'+directed)]
            if w in g[v]:
                if g[v][w] & 1:
                    s += clingo_wedge(v, w, directed_weights_matrix[i, j], num, name=directed)
                    missing = [missing[0], '']
                if g[v][w] & 2:
                    s += clingo_wedge(v, w, bidirected_weights_matrix[i, j], num, name=bidirected)
                    missing = ['', missing[1]]
            s += ' '.join(missing)
    return s


def g2wclingo(g):
    """
    Convert a graph to a string of grounded terms for ``clingo``
    
    :param g: ``gunfolds`` graph
    :type g: dictionary (``gunfolds`` graphs)
    
    :returns: ``clingo`` predicate
    :rtype: string  
    """
    s = ''
    n = len(g)
    s += 'node(1..'+str(n)+'). '
    for v in range(1, n+1):
        for w in range(1, n+1):
            missing = ['no_confh('+str(v)+','+str(w)+', 1).',
                       'no_edgeh('+str(v)+','+str(w)+', 1).']
            if w in g[v]:
                if g[v][w] & 1:
                    s += 'edgeh('+str(v)+','+str(w)+', 1). '
                    missing = [missing[0]]
                if g[v][w] & 2:
                    s += 'confh('+str(v)+','+str(w)+', 1). '
                    missing = [missing[1]]
            s += ' '.join(missing)
    return s


def clingo2num(value):
    """
    Converts the output of ``clingo`` into list of edges and under sampling rates for ``drasl``

    :param value: output of ``clingo``
    :type value: string
    
    :returns: list of edges and under sampling rates
    :rtype: a tuple of lists
    """
    a2edgetuple(value)


def rasl_a2edgetuple(answer):
    """
    Returns list of edges and the under sampling rate for ``rasl``

    :param answer: output of ``clingo``
    :type answer: string
    
    :returns: list of edges and the under sampling rate
    :rtype: a tuple of list and an integer
    """
    edges = [x for x in answer if 'edge' in x]
    u = [x for x in answer if 'min' in x]
    if not u:
        u = [x for x in answer if 'trueu' in x]
    u = u[0].split('(')[1].split(')')[0]
    return edges, int(u)


def a2edgetuple(answer):
    """
    Converts the output of ``clingo`` into list of edges and under sampling rates for ``drasl``
 
    :param answer: output of ``clingo``
    :type answer: string
    
    :returns: list of edges and under sampling rates
    :rtype: a tuple of lists
    """
    edges = [x for x in answer if 'edge1' in x]
    u = [x for x in answer if x[0] == 'u']
    return edges, u


def rasl_c2edgepairs(clist):
    """
    Converts ``clingo`` predicates to edge pairs for ``rasl``

    :param clist: ``clingo`` predicates
    :type clist: list of strings
    
    :returns: list of edge pairs
    :rtype: list
    """
    return [x[5:-1].split(',') for x in clist]


def c2edgepairs(clist):
    """
    Converts ``clingo`` predicates to edge pairs for ``drasl``

    :param clist: ``clingo`` predicates
    :type clist: list of strings
    
    :returns: list of edge pairs
    :rtype: list
    """
    return [x.strip(' ')[6:-1].split(',') for x in clist]


def msl_jclingo2g(output_g):
    """
    Converts the output of ``clingo`` to ``gunfolds`` graph for ``rasl_msl``

    :param output_g: the output of ``clingo`` for ``rasl_msl``
    :type output_g: string
    
    :returns: ``gunfolds`` graph
    :rtype: dictionary (``gunfolds`` graph)
    """
    l = a2edgetuple(output_g)
    l = (c2edgepairs(l[0]), l[1][0])
    l = (g2num(edgepairs2g(l[0])), int(l[1][2:-1]))
    return l


def rasl_jclingo2g(output_g):
    """
    Converts the output of ``clingo`` to ``gunfolds`` graph for ``rasl``

    :param output_g: the output of ``clingo`` for ``rasl``
    :type output_g: string
    
    :returns: ``gunfolds`` graph
    :rtype: dictionary (``gunfolds`` graph)
    """
    l = rasl_a2edgetuple(output_g)
    l = (rasl_c2edgepairs(l[0]), l[1])
    l = (g2num(edgepairs2g(l[0])), l[1])
    return l


def drasl_jclingo2g(output_g):
    """
    Converts the output of ``clingo`` to ``gunfolds`` graph for ``drasl``

    :param output_g: the output of ``clingo`` for ``drasl``
    :type output_g: string
    
    :returns: ``gunfolds`` graph
    :rtype: dictionary (``gunfolds`` graph)
    """
    l = a2edgetuple(output_g)
    l = (c2edgepairs(l[0]), tuple(np.sort([int(x.split(',')[0][2:]) for x in l[1]])))
    l = (g2num(edgepairs2g(l[0])), l[1])
    return l


def old_g2clingo(g, file=sys.stdout):
    """
    (Ask)

    :param g: ``gunfolds`` graph
    :type g: dictionary (``gunfolds`` graphs)
    
    :param file: (Ask)
    :type file:
    """
    n = len(g)
    print('node(1..'+str(n)+').', file=file)
    for v in g:
        for w in g[v]:
            if g[v][w] == 1:
                print('edgeu('+str(v)+','+str(w)+').', file=file)
            if g[v][w] == 2:
                print('confu('+str(v)+','+str(w)+').', file=file)
            if g[v][w] == 3:
                print('edgeu('+str(v)+','+str(w)+').', file=file)
                print('confu('+str(v)+','+str(w)+').', file=file)

def encode_sccs(g, idx, components=True, SCCS=None, quotient_edges=None):
    """
    Encodes strongly connected components of ``gunfolds`` graph to ``clingo`` predicates.

    Emits three families of facts:

    - ``scc_edge(K, L, idx)`` — there is at least one edge from a node in
      group ``K`` to a node in group ``L`` in the measured graph ``g``.  This
      is the edge relation of the quotient graph ``G / SCCS``.
    - ``scc(node, K)`` — node membership in group ``K`` (only when
      ``components=True``).
    - ``sccsize(K, Z)`` — group ``K`` has ``Z`` nodes (only when
      ``components=True``).

    .. note::

       The caller is responsible for passing a partition whose quotient is
       acyclic, OR providing the ``quotient_edges`` override.  Callers that
       go through :func:`encode_list_sccs` get this for free — that function
       computes an acyclic edge set via :func:`_acyclic_quotient_edges`
       (dropping back-edges within any cyclic SCC of the quotient) and
       passes the result via ``quotient_edges``, so the partition itself is
       preserved.

       Direct callers who omit ``quotient_edges`` and pass an arbitrary
       partition through ``SCCS`` will produce a cyclic ``scc_edge``
       relation via NetworkX's :func:`condensation`, which makes the SCC
       integrity constraints in :func:`encode_list_sccs` unsound.  Either
       supply ``quotient_edges`` or pass ``SCCS=None`` to use the true SCCs
       of ``g``.

    :param quotient_edges: optional precomputed list of ``(K, L)`` quotient
        edges to emit as ``scc_edge(K, L, idx)``.  When ``None``, the edge
        set is computed via :func:`condensation` (which may produce cycles
        for arbitrary partitions — see the note above).
    :type quotient_edges: list of (int, int) pairs, or None

    :param g: ``gunfolds`` graph
    :type g: dictionary (``gunfolds`` graphs)

    :param idx: index of the graph
    :type idx: integer

    :param components: If True, encodes SCC components and memberships to ``clingo`` predicates
    :type components: boolean

    :param SCCS: SCC membership of nodes (or any node partition)
    :type SCCS: list

    :returns: ``clingo`` predicates
    :rtype: string
    """
    G = graph2nx(g)
    if SCCS is None:
        SCCS = strongly_connected_components(G)
    s = ''
    if quotient_edges is not None:
        for (v, w) in quotient_edges:
            s += 'scc_edge(' + str(v) + ', ' + str(w) + ', ' + str(idx) + '). '
    else:
        CG = condensation(G, scc=SCCS)
        for v in CG:
            for w in CG[v]:
                s += 'scc_edge(' + str(v) + ', ' + str(w) + ', ' + str(idx) + '). '
    if not components:
        return s
    for c, component in enumerate(SCCS):
        cl = len(component)
        if cl == 1:
            x = [x for x in component][0]
            if x in G[x]:
                cl = 2
        s += 'sccsize(' + str(c) + ', ' + str(cl) + '). '
        for node in component:
            s += 'scc(' + str(node) + ', ' + str(c) + '). '
    return s


def _acyclic_quotient_edges(glist, partition, dm=None):
    """
    Compute the per-graph quotient edges for ``partition`` over ``glist``,
    with cycles broken to keep the ``scc_edge/3`` relation acyclic *without*
    merging any classes.

    Why we drop edges instead of merging classes
    --------------------------------------------
    The DRASL encoding's SCC integrity constraints are technically sound
    only when ``partition`` is a coarsening of the measured graph's actual
    SCC partition (so the quotient is a DAG by construction).  When the
    caller supplies a partition that *splits* a real SCC across multiple
    classes — e.g. ``--scc_strategy=domain`` grouping by NeuroMark domain,
    where two domains routinely exchange signals in both directions — the
    quotient gains cycles.  The mathematically clean response is to merge
    any classes within the same quotient-SCC; the side-effect on fMRI is
    that *every* domain pair has bidirectional evidence at PCMCI
    alpha=0.05, so the partition collapses to a single SCC and all SCC
    pruning is lost.

    This helper takes the *opposite* trade-off: keep every input class as
    its own SCC in the encoding, and drop only the back-edges in the
    quotient that close cycles.  The resulting ``scc_edge/3`` relation is
    acyclic, the constraints fire on every cross-class arrow, and per-SCC
    decomposition (e.g. solving each class as an independent sub-problem)
    remains meaningful.

    The cost is *technical unsoundness*: a valid causal graph whose
    cross-class arrows happen to go in a dropped (back) direction will be
    rejected.  This is an explicit speed-vs-correctness lever.

    Algorithm
    ---------
    1. Build the union digraph ``H`` over ``glist`` (directed edges only;
       bidirected edges do not contribute to the quotient).
    2. Build the quotient ``Q`` over ``H`` using ``partition``.
    3. Find SCCs of ``Q``.  Edges between distinct SCCs of ``Q`` are
       always one-directional (otherwise the two SCCs would be a single
       SCC), so they are kept verbatim.  Edges within a non-trivial SCC
       of ``Q`` are filtered: we pick a back-edge set for removal as
       described below.
    4. Apply the same per-class forward filter to each graph in
       ``glist`` independently to produce the final ``(K, L, idx)``
       triples.

    Back-edge selection
    -------------------
    Two strategies, picked at runtime based on ``dm``:

    - **Weighted MFAS (preferred, ``dm`` provided).**  For each cyclic SCC
      of ``Q``, run an exact integer-programming Minimum Feedback Arc Set
      with edge weights derived from the PCMCI evidence.  For class pair
      ``(K, L)`` the drop cost is

      ``w(K -> L) = pos(K -> L) + neg(L -> K)``

      where ``pos(K -> L)`` is the total ``hdirected`` weight of node-pair
      arrows from class K to class L (sum of ``dm[g_idx][X-1, Y-1]`` over
      ``X in K, Y in L`` with ``g[X][Y] in {1, 3}``), and ``neg(L -> K)``
      is the total ``no_hdirected`` weight in the *reverse* direction
      (sum over ``Y in L, X in K`` with ``g[Y][X] not in {1, 3}``).  This
      uses both signals the user identified — evidence supporting the
      arrow we'd drop and evidence against the arrow we'd keep instead —
      and minimises the total evidence cost of the drops.  See
      ``gunfolds/scripts/papers/scc_quotient_edge_dropping_research.md``
      for the literature review and the rationale for this choice over
      Bayesian-ratio alternatives (logged in ``todo.md`` for future).
    - **Class-index ordering (fallback, ``dm`` is None).**  Sort the
      classes within each cyclic SCC by their integer class index and
      keep only forward edges.  Used when callers do not supply ``dm``,
      preserving backward compatibility.

    :param glist: list of measured graphs (gunfolds 1-indexed dicts).
    :param partition: list of node-id sets — one per class.
    :param dm: optional list of NxN integer matrices, one per graph in
        ``glist``, holding the directed-edge PCMCI weights (``DD`` from
        the standard recipe).  When provided, weighted MFAS is used to
        pick back-edges; otherwise class-index fallback is used.
    :type dm: list of numpy.ndarray, or None

    :returns: a tuple ``(triples, n_dropped, weight_dropped)``:
        ``triples`` is a deduplicated list of ``(K, L, idx)`` triples to
        emit as ``scc_edge`` facts (idx is 1-based);
        ``n_dropped`` is the number of distinct ``(K, L)`` quotient edges
        that were dropped to break cycles (0 means the partition's
        quotient was already a DAG);
        ``weight_dropped`` is the total MFAS weight cost of the drops
        (0 in the unweighted fallback path).
    """
    H = nx.DiGraph()
    for g in glist:
        for u in g:
            H.add_node(u)
            for v in g[u]:
                # gunfolds edge codes: 1=directed, 2=bidirected, 3=both.
                if g[u][v] in (1, 3):
                    H.add_edge(u, v)

    node_to_class = {}
    for idx, members in enumerate(partition):
        for n in members:
            node_to_class[n] = idx

    # Quotient over the union — used to detect within-SCC back-edges.
    Q = nx.DiGraph()
    Q.add_nodes_from(range(len(partition)))
    for u, v in H.edges():
        ku = node_to_class.get(u)
        kv = node_to_class.get(v)
        if ku is None or kv is None or ku == kv:
            continue
        Q.add_edge(ku, kv)

    # Identify SCCs of Q.  Singleton SCCs play no role (no internal
    # back-edges possible).  Non-singleton SCCs are exactly the cycles
    # we need to break.
    class_qscc = {}
    qscc_members = {}
    for q_idx, comp in enumerate(strongly_connected_components(Q)):
        members = set(comp)
        qscc_members[q_idx] = members
        for cls in members:
            class_qscc[cls] = q_idx

    # Decide which back-edges to drop, using weighted MFAS when we have
    # PCMCI weights and class-index fallback otherwise.
    dropped_edges = set()
    weight_dropped = 0

    if dm is not None:
        # Weighted MFAS via igraph's exact_ip.  Pre-compute pos/neg per
        # class pair (only for pairs we may need — i.e. those inside a
        # non-trivial SCC of Q).
        cyclic_classes = set()
        for q_idx, members in qscc_members.items():
            if len(members) > 1:
                cyclic_classes |= members
        if cyclic_classes:
            pos_evidence = {}  # (K, L) -> int
            neg_evidence = {}  # (K, L) -> int
            for K in cyclic_classes:
                for L in cyclic_classes:
                    if K == L:
                        continue
                    p, n_ev = 0, 0
                    for g_idx, g in enumerate(glist):
                        M = dm[g_idx]
                        for x in partition[K]:
                            row_x = M[x - 1]
                            adj_x = g.get(x, {})
                            for y in partition[L]:
                                w = int(row_x[y - 1])
                                if adj_x.get(y, 0) in (1, 3):
                                    p += w
                                else:
                                    n_ev += w
                    pos_evidence[(K, L)] = p
                    neg_evidence[(K, L)] = n_ev

            for q_idx, members in qscc_members.items():
                if len(members) <= 1:
                    continue
                # Collect internal edges of this cyclic SCC.
                internal = [(u, v) for u, v in Q.edges()
                            if u in members and v in members]
                if not internal:
                    continue
                # Local class indexing for the igraph subgraph.
                cls_list = sorted(members)
                cls_to_local = {c: i for i, c in enumerate(cls_list)}
                local_edges = [(cls_to_local[u], cls_to_local[v])
                               for (u, v) in internal]
                weights = [pos_evidence[(u, v)] + neg_evidence[(v, u)]
                           for (u, v) in internal]
                ig_g = igraph.Graph(n=len(cls_list), edges=local_edges,
                                    directed=True)
                ig_g.es['weight'] = weights
                # Exact ILP — fast at our scale (<= 7 classes per SCC).
                fas_local = ig_g.feedback_arc_set(weights='weight',
                                                  method='exact_ip')
                for li in fas_local:
                    dropped_edges.add(internal[li])
                    weight_dropped += weights[li]
    else:
        # Class-index fallback: deterministic within-SCC ordering.
        within_qscc_pos = {}
        for members in qscc_members.values():
            for pos, cls in enumerate(sorted(members)):
                within_qscc_pos[cls] = pos
        for (u, v) in Q.edges():
            if class_qscc[u] != class_qscc[v]:
                continue  # different SCCs -> already forward
            if within_qscc_pos[u] >= within_qscc_pos[v]:
                dropped_edges.add((u, v))

    n_dropped = len(dropped_edges)

    # Per-graph emission: drop any cross-class arrow whose quotient edge
    # is in the dropped set; emit the rest.
    triples_seen = set()
    triples = []
    for g_idx, g in enumerate(glist):
        for u in g:
            ku = node_to_class.get(u)
            if ku is None:
                continue
            for v in g[u]:
                if g[u][v] not in (1, 3):
                    continue
                kv = node_to_class.get(v)
                if kv is None or ku == kv:
                    continue
                if (ku, kv) in dropped_edges:
                    continue
                key = (ku, kv, g_idx + 1)
                if key not in triples_seen:
                    triples_seen.add(key)
                    triples.append(key)
    return triples, n_dropped, weight_dropped


def encode_list_sccs(glist, scc_members=None, dm=None):
    """
    Encodes strongly connected components of a list of ``gunfolds`` graphs to
    ``clingo`` predicates and the three integrity constraints that use them.

    Predicates emitted (per graph in ``glist``):

    - ``scc_edge(K, L, idx)`` — the measured graph ``idx`` has at least one
      edge from a node in group ``K`` to a node in group ``L``.  Always
      acyclic across all idx values.  When ``scc_members`` is supplied and
      its quotient would be cyclic, the cycle-creating back-edges are
      dropped via :func:`_acyclic_quotient_edges` — every input class is
      preserved as its own SCC in the encoding, but the constraint may
      reject some valid causal graphs whose arrows go in dropped
      directions.  This is an explicit speed-vs-soundness trade-off: it
      keeps the per-class pruning power that lets the SCC integrity
      constraints below actually fire, at the cost of technical
      unsoundness.  The alternative (merging classes within each cyclic
      quotient-SCC) is theoretically sounder but collapses the partition
      to one class on typical fMRI data, eliminating all pruning.
    - ``scc(node, K)``, ``sccsize(K, Z)`` — emitted only for the first graph
      (the partition is graph-independent).

    Constraints emitted:

    1. No ``edge1(X, Y)`` may cross from group ``K`` to a non-singleton group
       ``L`` unless the measured graph witnessed at least one direct quotient
       edge ``K -> L`` (in any idx).
    2. No ``2``-cycle between two distinct groups at the same undersampling
       rate (would imply they are a single SCC).
    3. Per-undersampling version of constraint (1): a ``U``-step directed
       path crossing into a non-singleton group must be witnessed at the
       matching ``u(U, idx)`` in the corresponding measured graph.

    :param glist: a list of graphs that are under-sampled versions of
        the same system
    :type glist: list of dictionaries (``gunfolds`` graphs)

    :param scc_members: a list of node-id sets describing the SCC partition
        used to interpret ``glist``.  If ``None``, the actual strongly
        connected components of ``glist[0]`` are computed and used.
    :type scc_members: list of sets, or None

    :returns: ``clingo`` predicates
    :rtype: string
    """
    s = ''
    if scc_members is not None:
        SCCS = scc_members
        # Precompute the acyclic quotient edge set: drop back-edges in any
        # cyclic SCC of the quotient, but keep every input class as its own
        # SCC in the encoding (no class merging).  Trade-off: the encoding
        # may technically reject some valid causal graphs whose cross-class
        # arrows go in dropped directions; in exchange the SCC integrity
        # constraints retain pruning power and per-SCC decomposition is
        # meaningful.  See ``_acyclic_quotient_edges`` for the algorithm —
        # weighted MFAS via igraph when ``dm`` is provided, class-index
        # fallback otherwise.
        triples, n_dropped, weight_dropped = _acyclic_quotient_edges(
            glist, SCCS, dm=dm)
        if n_dropped > 0:
            method = "weighted MFAS" if dm is not None else "class-index"
            extra = (f", total evidence cost: {weight_dropped}"
                     if dm is not None else "")
            print(f"  [encode_list_sccs] supplied partition had a cyclic "
                  f"quotient; kept all {len(SCCS)} classes and dropped "
                  f"{n_dropped} back-edge(s) via {method}{extra}.")
        # Bucket triples by graph idx for per-graph emission.
        edges_by_idx = {}
        for (k, l, gi) in triples:
            edges_by_idx.setdefault(gi, []).append((k, l))
    else:
        SCCS = None  # will be computed from glist[0] inside encode_sccs
        edges_by_idx = None  # condensation path inside encode_sccs

    first_graph = True
    for i, g in enumerate(glist):
        if first_graph and SCCS is None:
            SCCS = [c for c in strongly_connected_components(graph2nx(g))]
        qe = (edges_by_idx.get(i + 1, []) if edges_by_idx is not None else None)
        s += encode_sccs(g, i + 1, components=first_graph, SCCS=SCCS,
                         quotient_edges=qe)
        first_graph = False
    # If the generating (causal-scale) graph proposes a cross-group edge to a
    # non-singleton group L that was *never observed* in any measured graph
    # (i.e. no scc_edge(K,L,_) fact for any graph idx), reject. ``scc_edge``
    # here is the existence relation on quotient edges of the measured graph
    # — it is not required to be a DAG; see ``encode_sccs`` for the rationale.
    s += ':- edge1(X,Y), scc(X,K), scc(Y,L), K != L, sccsize(L,Z), Z > 1, not scc_edge(K,L,_). '
    # If the produced graph has a 2-cycle between two distinct groups at the
    # measurement undersampling rate, reject (would imply a single SCC).
    s += ':- directed(X,Y,M), directed(Y,X,N), scc(X, K), scc(Y,L), K != L, M<=U, N<=U, M<=N, u(U,_).'
    # Same check as the first constraint but at the per-undersampling level:
    # forbid a U-step directed path crossing into a non-singleton group L
    # unless the corresponding measured graph (idx N) directly witnessed a
    # K -> L quotient edge.
    s += ':- directed(X,Y,U), scc(X,K), scc(Y,L), K != L, sccsize(L,Z), Z > 1, not scc_edge(K,L,N), u(U,N).'
    return s


################### Add only new functions to clingo conversions above #############
################### End of Clingo Conversions ########################

# Dont remove this fake function for automating sphinx build.
def sphinx_automation_fake():
    return

################### Start of External Conversions ###################

def graph2nx(G):
    """
    Convert a ``gunfolds`` graph to NetworkX format ignoring bidirected edges
    
    :param G: ``gunfolds`` format graph
    :type G: dictionary (``gunfolds`` graphs)
    
    :returns: NetworkX format graph
    :rtype: NetworkX graph
    """
    g = nx.DiGraph()
    for v in G:
        edges = [(v, x) for x in G[v] if G[v][x] in (1, 3)]
        if edges:
            g.add_edges_from(edges)
        else:
            g.add_node(v)
    return g


def graph2dot(g, filename):
    """
    Save the graph structure of `g` to a graphviz format dot file with the name `filename`

    :param g: ``gunfolds`` graph
    :type g: dictionary (``gunfolds`` graphs)

    :param filename: name of the file
    :type filename: string
    """
    G = graph2nx(g)
    nx.drawing.nx_pydot.write_dot(G, filename)


def nx2graph(G):
    """
    Convert NetworkX format graph to ``gunfolds`` graph ignoring bidirected edges
    
    :param G: ``gunfolds`` format graph
    :type G: dictionary (``gunfolds`` graphs)
    
    :returns: ``gunfolds`` graph
    :rtype: dictionary (``gunfolds`` graphs)
    """
    g = {n: {} for n in G}
    for n in G:
        g[n] = {x: 1 for x in G[n]}
    return g

def nxbp2graph(G):
    """
    Ask 

    :param G: ``gunfolds`` format graph
    :type G: dictionary (``gunfolds`` graphs)
    
    :returns: Ask
    :rtype: 
    """
    nodesnum = len(G)//2
    g = {n+1: {} for n in range(nodesnum)}
    for n in g:
        g[n] = {(x % nodesnum+1): 1 for x in G[n-1]}
    return g

def g2ig(g):
    """
    Converts our graph representation to an igraph for plotting
    
    :param g: ``gunfolds`` graph
    :type g: dictionary (``gunfolds`` graphs)
    
    :returns: igraph representation of ``gunfolds`` graph
    :rtype: igraph
    """
    t = np.where(graph2adj(g) == 1)
    l = zip(t[0], t[1])
    ig = igraph.Graph(l, directed=True)
    ig.vs["name"] = np.sort([u for u in g])
    ig.vs["label"] = ig.vs["name"]
    return ig

################### Add only new functions to external conversions above #############
################### End of External Conversions ########################