"""This module contains clingo interaction functions"""
from __future__ import print_function
from string import Template
from gunfolds.utils.clingo import clingo
from gunfolds.utils.calc_procs import get_process_count
from gunfolds.conversions import g2clingo, rate, rasl_jclingo2g,\
     drasl_jclingo2g, clingo_preamble,\
     numbered_g2clingo, numbered_g2wclingo, encode_list_sccs

CLINGO_LIMIT = 64
PNUM = min(CLINGO_LIMIT, get_process_count(1))
CAPSIZE = 1

all_u_rasl_program = """
{edge(X,Y)} :- node(X), node(Y).
directed(X,Y,1) :- edge(X,Y).
directed(X,Y,L) :- directed(X,Z,L-1), edge(Z,Y), L <= U, u(U).
bidirected(X,Y,U) :- directed(Z,X,L), directed(Z,Y,L), node(X;Y;Z), X < Y, L < U, u(U).
countdirh(C):- C = #count { hdirected(X, Y): hdirected(X, Y), node(X), node(Y)}.
countbidirh(C):- C = #count { hbidirected(X, Y): hbidirected(X, Y), node(X), node(Y)}.
equald(L):- { directed(X,Y,L): hdirected(X,Y), node(X), node(Y) } == C, countdirh(C),u(L).
equalb(L):- { bidirected(X,Y,L): hbidirected(X,Y), node(X), node(Y) } == C, countbidirh(C),u(L).
equal(L) :- equald(L), equalb(L).
{trueu(L)} :- equal(L).
equaltest(M) :- 1 < {equal(_)}, equal(M).
min(M):- #min {MM:equaltest(MM)}=M, equaltest(_).
repeat(N):- min(M), equal(N), M<N.
:- directed(X, Y, L), not hdirected(X, Y), node(X), node(Y), trueu(L).
:- not directed(X, Y, L) , hdirected(X, Y), trueu(L).
:- bidirected(X, Y, L), not hbidirected(X, Y), node(X), node(Y), X < Y, trueu(L).
:- not bidirected(X, Y, L), hbidirected(X, Y), X < Y, trueu(L).
:- not trueu(_).
:- min(M), trueu(N), M<N.
    """

# The ASP formulation that does the heavylifting of the RASL encoding
# -------------------------------------------------------------------
# Generate powerset of all possible edges and for given edges produce directed
# edges up to the current undersampling and bidirected edges at the current
# undersampling.
drasl_program = """
    {edge1(X,Y)} :- node(X), node(Y).
    directed(X, Y, 1) :- edge1(X, Y).
    directed(X, Y, L) :- directed(X, Z, L-1), edge1(Z, Y), L <= U, u(U, _).
    bidirected(X, Y, U) :- directed(Z, X, L), directed(Z, Y, L), node(X;Y;Z), X < Y, L < U, u(U, _).
    """
# A set of no-go rules that compare the produces undersampled graph from the
# current element of the edge powerset to the input measurement timescae
# graph(s) and get rid of solutions that have a mismatch.
drasl_program += """
    :- directed(X, Y, L), not hdirected(X, Y, K), node(X;Y), u(L, K).
    :- bidirected(X, Y, L), not hbidirected(X, Y, K), node(X;Y), u(L, K), X < Y.
    :- not directed(X, Y, L), hdirected(X, Y, K), node(X;Y), u(L, K).
    :- not bidirected(X, Y, L), hbidirected(X, Y, K), node(X;Y), u(L, K), X < Y.
    """
# Filer out graphs that have already converged
# ----------------------------------------
# The next two sections are connected and are there to make sure that we do not
# count graphs more than once. Withoutthese sections if a converged or
# oscillating graph matches the measured graph H at multiple undersampling
# rates, all of them will be listed as unique solutions. The main idea is to
# first define a notequal operator, that works by checking individual edges,
# and then filter out solutions where this operator does not hold for at least
# one pair of undersampling rates.

# Turns out that the generating rules above produce all directed edges for each
# undersampling rate from 1 to u in each answer set, but only one set of
# bidirected edges that correspond to the current u in this answer. Often, it
# suffices to check repeatition of all directed edges to determine that a graph
# has converged. The main reason is persistence of bidirected edges that stay
# present once they appear at some undersampling rate. However, in some cases
# the directed edges converge before the bidirected edges do. This happens when
# the final set of directed edges generates at least one bidirected edge at the
# next step.

# The following two lines take advantage of the present history of all directed
# edges across undersampling rates up to u, and build a workaround for the
# absent history of bidirected edges. The first line is true if there was a
# fork with X and Y at the ends before the last step of undersampling. This
# would mean that the bidirected adge, if it exists at the current
# undersampling step, was already there at the last step as well. The second
# line sets notequal to true if at the current undersampling rate L a new
# bidirected edge appeared.
drasl_program += """
    pastfork(X,Y,L) :- directed(Z, X, K), directed(Z, Y, K), node(X;Y;Z), X < Y, K < L-1, uk(K), u(L, _).
    notequal(L-1,L) :- bidirected(X,Y,L), not pastfork(X,Y,L), node(X;Y), X < Y, u(L, _).
    """

# The following lines set the notequal condition for the cases when there is a
# mismatch in directed edges between rate K lower than L and L - the current
# undersampling rate.
drasl_program += """
    notequal(K,L) :- directed(X, Y, K), not directed(X, Y, L), node(X;Y), K<L, uk(K), u(L,_).
    notequal(K,L) :- not directed(X, Y, K), directed(X, Y, L), node(X;Y), K<L, uk(K), u(L,_).
    """

# This line filters out all solutions for which there is a K lower than the
# current undersampling rate and the graph at rate K equals to the one at rate
# L. This is a way to bypass our inability to specify the foroll quantifier
# directly in clingo.
drasl_program += """
    :- not notequal(K,L), K<L, uk(K), u(L,_).
    """

# The following section refuces to handle graphs if their compressed
# representation is a DAGs withuot forks. Too many options that are
# uninformative anyways and thre is no need to waste computation on them.
drasl_program += """
    nonempty(L) :- directed(X, Y, L), u(L,_).
    nonempty(L) :- bidirected(X, Y, L), u(L,_).
    :- not nonempty(L), u(L,_).
    """


def weighted_drasl_program(directed, bidirected, no_directed, no_bidirected):
    """
    Adjusts the optimization code based on the directed and bidirected priority

    :param directed: priority of directed edges in optimization
    :type directed: integer

    :param bidirected: priority of bidirected edges in optimization
        graph
    :type bidirected: integer

    :returns: optimization part of the ``clingo`` code
    :rtype: string
    """
    # The term tuple appended to each weak constraint is the dedup key:
    # clingo counts two ground instances as the same cost element when their
    # (weight, priority, tuple) triple is identical, and only adds the weight
    # once.  Without a type tag, a directed-mismatch penalty at (X,Y) with the
    # same weight as a co-firing bidirected penalty at the same (X,Y) would be
    # silently dropped.  The trailing constant (1..4) makes every source
    # distinct.  K disambiguates across multi-subject inputs.
    t = Template("""
    {edge1(X,Y)} :- node(X), node(Y).
    directed(X, Y, 1) :- edge1(X, Y).
    directed(X, Y, L) :- directed(X, Z, L-1), edge1(Z, Y), L <= U, u(U, _).
    bidirected(X, Y, U) :- directed(Z, X, L), directed(Z, Y, L), node(X;Y;Z), X < Y, L < U, u(U, _).

    :~ directed(X, Y, L),    no_hdirected(X, Y, W, K),   node(X;Y), u(L, K).         [W@$directed,X,Y,K,1]
    :~ bidirected(X, Y, L),  no_hbidirected(X, Y, W, K), node(X;Y), u(L, K), X < Y.  [W@$bidirected,X,Y,K,2]
    :~ not directed(X, Y, L),   hdirected(X, Y, W, K),   node(X;Y), u(L, K).         [W@$no_directed,X,Y,K,3]
    :~ not bidirected(X, Y, L), hbidirected(X, Y, W, K), node(X;Y), u(L, K), X < Y.  [W@$no_bidirected,X,Y,K,4]

    """)

    return t.substitute(directed=directed, bidirected=bidirected,no_directed=no_directed,no_bidirected=no_bidirected)


def rate(u, uname='u'):
    """
    Adds under sampling rate to ``clingo`` code

    :param u: maximum under sampling rate
    :type u: integer

    :param uname: name of the parameter
    :type uname: string

    :returns: predicate for under sampling rate
    :rtype: string
    """
    s = "1 {" + uname + "(1.."+str(u)+")} 1."
    return s


def drate(u, gnum, weighted=False):
    """
    Replaces ``rate`` if there are multiple under sampled inputs

    :param u: maximum under sampling rate
    :type u: integer

    :param gnum: number of under sampled inputs
    :type gnum: integer

    :param weighted: whether the input graphs are weighted or
        precize.  If `True` but no weight matrices are provided -
        all weights are set to `1`
    :type weighted: boolean

    :returns: ``clingo`` code for under sampling with multiple under sampled inputs
    :rtype: string
    """
    s = f"1 {{u({int(weighted)+1}..{u}, {gnum})}} 1."
    return s


def rasl_command(g, urate=0):
    """
    Given a graph generates ``clingo`` code

    :param g: ``gunfolds`` graph
    :type g: dictionary (``gunfolds`` graphs)

    :param urate: maximum undersampling rate to consider
    :type urate: integer

    :returns: ``clingo`` code
    :rtype: string
    """
    if not urate:
        urate = 1+3*len(g)
    command = g2clingo(g) + ' ' + rate(urate) + ' '
    command += '{edge(X,Y)} :- node(X), node(Y). ' + all_u_rasl_program + ' '
    command += "#show edge/2. "
    command += "#show trueu/1. "
    command += "#show min/1."
    command = command.encode().replace(b"\n", b" ")
    return command


def glist2str(g_list, weighted=False, dm=None, bdm=None):
    """
    Converts list of graphs into ``clingo`` predicates

    :param g_list: a list of graphs that are undersampled versions of
        the same system
    :type g_list: list of dictionaries (``gunfolds`` graphs)

    :param weighted: whether the input graphs are weighted or
        precize.  If `True` but no weight matrices are provided -
        all weights are set to `1`
    :type weighted: boolean

    :param dm: a list of n-by-n 2-d square matrix of the weights for
        directed edges of each input n-node graph
    :type dm: list of numpy arrays

    :param bdm: a list of *symmetric* n-by-n 2-d square matrix of the
        weights for bidirected edges of each input n-node graph
    :type bdm: list of numpy arrays

    :returns: ``clingo`` predicates as a string
    :rtype: string
    """
    if dm is None:
        dm = [None]*len(g_list)
    else:
        dm = [nd.astype('int') for nd in dm]
    if bdm is None:
        bdm = [None]*len(g_list)
    else:
        bdm = [nd.astype('int') for nd in bdm]
    s = ''
    for count, (g, D, B) in enumerate(zip(g_list, dm, bdm)):
        if weighted:
            s += numbered_g2wclingo(g, count+1, directed_weights_matrix=D, bidirected_weights_matrix=B) + ' '
        else:
            s += numbered_g2clingo(g, count+1) + ' '
    return s


def _compute_directed_density_pct(g):
    """
    Compute the directed-edge density of a gunfolds graph as an integer
    percentage (0-100).  Counts every (i, j) with g[i][j] in {1, 3} and
    divides by N² (matches the ASP-side ``hypoth_density`` definition,
    which counts ``edge1`` over ``n*n``).
    """
    n = len(g)
    if n == 0:
        return 0
    n_dir = sum(1 for src in g for tgt, val in g[src].items() if val in (1, 3))
    return int(round(100.0 * n_dir / (n * n)))


def drasl_command(g_list, max_urate=0, weighted=False, scc=False, scc_members=None, dm=None, bdm=None, edge_weights=[1, 1, 1, 1, 1], GT_density=None, selfloop=False, density_weight=50, density_mode='soft', tol=5, tol_low=None, tol_high=None):
    """
    Given a list of graphs generates ``clingo`` codes

    :param g_list: a list of graphs that are undersampled versions of
        the same system
    :type g_list: list of dictionaries (``gunfolds`` graphs)

    :param max_urate: maximum under sampling rate
    :type max_urate: integer

    :param weighted: whether the input graphs are weighted or
        precize.  If ``True`` but no weight matrices are provided -
        all weights are set to ``1``
    :type weighted: boolean

    :param scc: whether to assume that each SCC in the input graph is
        either a singleton or have ``gcd=1``.  If `True` a much more
        efficient algorithm is employed.
    :type scc: (GUESS)boolean

    :param scc_members: a list of sets for nodes in each SCC
    :type scc_members: list

    :param dm: a list of n-by-n 2-d square matrix of the weights for
        directed edges of each input n-node graph
    :type dm: list of numpy arrays

    :param bdm: a list of *symmetric* n-by-n 2-d square matrix of the
        weights for bidirected edges of each input n-node graph
    :type bdm: list of numpy arrays

    :param edge_weights: priority levels for the four weak constraint types
        (directed false-positive, bidirected false-positive,
        directed false-negative, bidirected false-negative).  Index 4,
        if present, is ignored (density is controlled by ``density_weight``).
    :type edge_weights: list of integers (length 4 or 5)

    :param GT_density: desired density of the ground truth at causal
        time-scale, expressed as density × 100 (e.g. 35 means 35 %).
        Internally quantised to 50 two-percent bins so that ``d``
        inside the ASP program equals ``GT_density // 2``.
    :type GT_density: integer

    :param density_weight: weight per density-bin deviation in the soft
        density constraint.  Each bin is 2 percentage points.  A value of
        50 means one bin of density error costs 50, making density more
        important than a single edge mismatch (MAXCOST = 20).
    :type density_weight: integer

    :param density_mode: how to encode the density target.  One of
        ``'soft'`` (legacy), ``'hard'``, ``'hard_soft0'``,
        ``'hard_soft1'``, or ``'none'``.  See :func:`drasl` for the
        full description (``'adaptive'`` is handled at the ``drasl``
        level and should not be passed here directly).
    :type density_mode: string

    :param tol: symmetric density tolerance (in percentage points) for
        the hard cardinality bounds (used by all ``'hard*'`` modes).
        When ``tol_low`` or ``tol_high`` is supplied separately, it
        overrides ``tol`` in that direction.
    :type tol: integer

    :param tol_low: downward tolerance in percentage points
        (``d_lo = (GT − tol_low) · N² / 100``).  If ``None`` the
        symmetric ``tol`` is used.  PCMCI density tends to overestimate
        the causal-scale density, so the production default uses a
        wider downward tolerance than upward.
    :type tol_low: integer or None

    :param tol_high: upward tolerance in percentage points
        (``d_hi = (GT + tol_high) · N² / 100``).  If ``None`` the
        symmetric ``tol`` is used.
    :type tol_high: integer or None

    :returns: clingo code as a string
    :rtype: string
    """
    if dm is not None:
        dm = [nd.astype('int') for nd in dm]
    if bdm is not None:
        bdm = [nd.astype('int') for nd in bdm]

    assert len({len(g) for g in g_list}) == 1, "Input graphs have variable number of nodes!"

    if not max_urate:
        max_urate = 1+3*len(g_list[0])
    n = len(g_list)
    command = clingo_preamble(g_list[0])

    # Per-subject GT_density auto-computation.  When the caller does not
    # supply GT_density explicitly we derive it from the directed-edge
    # density of the first input graph, treating PCMCI's measurement
    # density as the prior on the causal-scale density.  This avoids the
    # population-level mismatch where a fixed GT_density excludes the
    # actual optimal region for sparse subjects.
    if GT_density is None and density_mode != 'none':
        GT_density = _compute_directed_density_pct(g_list[0])

    if density_mode not in ('soft', 'hard', 'hard_soft0', 'hard_soft1', 'none'):
        raise ValueError(
            f"density_mode must be one of "
            f"'soft', 'hard', 'hard_soft0', 'hard_soft1', 'none'; "
            f"got {density_mode!r}"
        )

    if GT_density is not None and density_mode != 'none':
        # Asymmetric tolerance.  ``tol_low`` and ``tol_high`` (when provided)
        # specify the downward / upward percentage-point relaxation around
        # GT_density.  Falling back to the symmetric ``tol`` preserves
        # backward compatibility with callers that did not split the bound.
        # Asymmetric defaults make sense because PCMCI's measurement-graph
        # density systematically *overestimates* the causal-scale density
        # (every length-u path becomes an observed edge), so a wider
        # downward window is the right prior.
        eff_tol_low  = tol if tol_low  is None else tol_low
        eff_tol_high = tol if tol_high is None else tol_high

        # Convert GT_density (density × 100) to 50-level bin index (each bin = 2 %).
        # e.g. GT_density=35 → d=17  (17 bins × 2 % = 34 %, nearest even percent)
        #      GT_density=22 → d=11  (11 bins × 2 % = 22 %, exact)
        d_bins = GT_density // 2
        n_nodes = len(g_list[0])
        n_sq = n_nodes * n_nodes
        d_lo_edges = max(0, int((GT_density - eff_tol_low) * n_sq / 100))
        d_hi_edges = min(n_sq, int((GT_density + eff_tol_high) * n_sq / 100) + 1)

        command += f"#const d = {d_bins}. "
        command += 'countedge1(C):- C = #count { edge1(X, Y): edge1(X, Y), node(X), node(Y)}. '
        command += 'countfull(C):- C = n*n. '
        # Scale density to 0-50 bins (50 * edges / N²).
        command += 'hypoth_density(D) :- D = 50*X/Y,  countfull(Y), countedge1(X). '
        command += 'abs_diff(Diff) :- hypoth_density(D), Diff = |D - d|. '

        # Hard cardinality bounds (for any mode that uses them).
        if density_mode in ('hard', 'hard_soft0', 'hard_soft1'):
            command += f"#const d_lo = {d_lo_edges}. "
            command += f"#const d_hi = {d_hi_edges}. "
            command += ":- countedge1(K), K < d_lo. "
            command += ":- countedge1(K), K > d_hi. "

        # Soft penalty (with priority chosen by mode).
        if density_mode == 'soft':
            # Original behaviour: soft penalty at @1 mixed with edge cost.
            command += f':~ abs_diff(Diff). [Diff*{density_weight}@1] '
        elif density_mode == 'hard_soft0':
            # Lex below edge matching: density only breaks ties.
            command += f':~ abs_diff(Diff). [Diff*{density_weight}@0] '
        elif density_mode == 'hard_soft1':
            # Density still mixed with edges, but constrained to a window.
            command += f':~ abs_diff(Diff). [Diff*{density_weight}@1] '
        # density_mode == 'hard' adds no soft term.
    if scc:
        command += encode_list_sccs(g_list, scc_members)
        print("edit this function later to adjust")
    command += f"dagl({len(g_list[0])-1}). "
    command += glist2str(g_list, weighted=weighted, dm=dm, bdm=bdm) + ' '   # generate all graphs
    command += 'uk(1..'+str(max_urate)+').' + ' '
    command += ' '.join([drate(max_urate, i+1, weighted=weighted) for i in range(n)]) + ' '
    command += weighted_drasl_program(edge_weights[0], edge_weights[1],edge_weights[2], edge_weights[3]) if weighted else drasl_program
    # command += f":- M = N, {{u(M, 1..{n}); u(N, 1..{n})}} == 2, u(M, _), u(N, _). "
    if selfloop is not None:
        if selfloop:
            command += ":- not edge1(X, X), node(X)."
        else:
            command += ":-  edge1(X, X), node(X)."
    command += ":- u(L,A), u(T,B), not T=L, A<B. "
    command += ":- not edge1(_, _)."
    command += "#show edge1/2. "
    command += "#show u/2."
    command = command.encode().replace(b"\n", b" ")
    return command


def drasl(glist, capsize=CAPSIZE, timeout=0, urate=0, weighted=False, scc=False, scc_members=None, dm=None,
          bdm=None, pnum=PNUM, GT_density=None, edge_weights=[1, 1, 1, 1, 1], configuration="crafty", optim='optN',
          multi_individual=False, selfloop=False, density_weight=50,
          density_mode='adaptive', tol=None, tol_low=15, tol_high=5, tol_widen=10,
          verbose=True):
    """
    Compute all candidate causal time-scale graphs that could have
    generated all undersampled graphs at all possible undersampling
    rates up to ``urate`` in ``glist`` each at an unknown undersampling
    rate.

    :param glist: a list of graphs that are undersampled versions of
        the same system
    :type glist: list of dictionaries (``gunfolds`` graphs)

    :param capsize: maximum number of candidates to return
    :type capsize: integer

    :param timeout: timeout in seconds after which to interrupt
        computation (0 - no limit)
    :type timeout: integer

    :param urate: maximum undersampling rate to consider
    :type urate: integer

    :param weighted: whether the input graphs are weighted or
        imprecize.  If ``True`` but no weight matrices are provided -
        all weights are set to ``1``
    :type weighted: boolean

    :param scc: whether to assume that each SCC in the input graph is
        either a singleton or have ``gcd=1``.  If ``True`` a much more
        efficient algorithm is employed.
    :type scc: boolean

    :param scc_members: a list of sets for nodes in each SCC
    :type scc_members: list

    :param dm: a list of n-by-n 2-d square matrix of the weights for
        directed edges of each input n-node graph
    :type dm: list of numpy arrays

    :param bdm: a list of *symmetric* n-by-n 2-d square matrix of the
        weights for bidirected edges of each input n-node graph
    :type bdm: list of numpy arrays

    :param pnum: number of parallel threads to run ``clingo`` on
    :type pnum: integer

    :param GT_density: desired density of the ground truth at causal
        time-scale, expressed as density × 100 (e.g. 35 means 35 %).
        Converted to 50-level bins internally (``GT_density // 2``).
        If ``None`` (the default), GT_density is **auto-computed per
        subject** from the directed-edge density of ``glist[0]``,
        making the prior follow the input rather than a fixed
        population value.
    :type GT_density: integer or None

    :param density_mode: density encoding strategy.

        - ``adaptive`` *(default, production)* : Try ``hard_soft0`` with
          ``tol``, then with ``tol_widen``, then fall back to ``soft``
          if both are UNSAT.  Combines pruning speedup with robustness
          on subjects whose true causal density falls outside the
          tight window.
        - ``soft`` : single soft penalty at @1 (legacy behaviour).
        - ``hard`` : hard cardinality bounds [GT±tol], no soft term.
        - ``hard_soft0`` : hard bounds + soft density at @0 (lex below
          edge matching).  Best primary-cost quality when bounds fit.
        - ``hard_soft1`` : hard bounds + soft density at @1.  Same
          objective shape as ``soft`` but with pruning.
        - ``none`` : no density encoding at all (use only when caller
          will append density code separately).
    :type density_mode: string

    :param tol: legacy symmetric tolerance override.  When not ``None``,
        sets both ``tol_low`` and ``tol_high`` to ``tol``.  Use
        ``tol_low``/``tol_high`` for the production asymmetric default.
    :type tol: integer or None

    :param tol_low: downward tolerance in percentage points
        (``d_lo = (GT − tol_low) · N² / 100``).  Default ``15``: a
        wide downward window because PCMCI's measurement density tends
        to overestimate the causal-scale density.
    :type tol_low: integer

    :param tol_high: upward tolerance in percentage points
        (``d_hi = (GT + tol_high) · N² / 100``).  Default ``5``: a
        narrow upward window because the causal density is rarely
        higher than the measurement density.
    :type tol_high: integer

    :param tol_widen: amount (in percentage points) added to *both*
        ``tol_low`` and ``tol_high`` for the second adaptive attempt
        before falling back to soft-only.  Default ``10``.
    :type tol_widen: integer

    :param verbose: print progress lines for each adaptive attempt.
    :type verbose: boolean

    :param edge_weights: priority levels for the four weak constraint types.
        Index 4, if present, is ignored; density is controlled by
        ``density_weight``.
    :type edge_weights: list of integers (length 4 or 5)

    :param density_weight: weight per density-bin deviation (each bin = 2 %).
        Default 50 makes one bin of density error cost 50 units, which
        dominates a single edge mismatch (MAXCOST = 20).
    :type density_weight: integer

    :param configuration: Select configuration based on problem type

        - ``frumpy`` : Use conservative defaults
        - ``jumpy`` : Use aggressive defaults
        - ``tweety`` : Use defaults geared towards asp problems
        - ``handy`` : Use defaults geared towards large problems
        - ``crafty`` : Use defaults geared towards crafted problems
        - ``trendy`` : Use defaults geared towards industrial problems
    :type configuration: string

    :param optim: a comma separated string containing configuration for optimization algorithm and optionally a bound [<arg>[, <bound>]]
        
        - <arg> : <mode {opt|enum|optN|ignore}>
            - ``opt`` : Find optimal model
            - ``enum`` : Find models with costs <= <bound>
            - ``optN`` : Find optimum, then enumerate optimal models
            - ``ignore`` : Ignore optimize statements
        - <bound> : Set initial bound for objective function(s)
    :type optim: string

    :param multi_individual: if True, can pass multiple estimated graphs from several individuals
        and optimize for making them similar and get a single graph output
    :type multi_individual: boolean


    :returns: results of parsed equivalent class
    :rtype: dictionary
    """
    if dm is not None:
        dm = [nd.astype('int') for nd in dm]
    if bdm is not None:
        bdm = [nd.astype('int') for nd in bdm]
    if not isinstance(glist, list):
        glist = [glist]

    # Resolve per-subject GT_density once so subsequent retries with
    # different tolerances reuse the same derived density.
    effective_GT_density = GT_density
    if effective_GT_density is None:
        effective_GT_density = _compute_directed_density_pct(glist[0])
        if verbose:
            print(f"[drasl] auto-computed GT_density = {effective_GT_density}% "
                  f"(from input graph density)", flush=True)

    # Resolve symmetric vs. asymmetric tolerance.  ``tol`` (when not None)
    # is the legacy symmetric override that sets both directions; passing
    # ``tol_low`` / ``tol_high`` keeps the asymmetric defaults.
    if tol is not None:
        eff_tol_low  = tol
        eff_tol_high = tol
    else:
        eff_tol_low  = tol_low
        eff_tol_high = tol_high

    def _run(mode, t_low, t_high):
        cmd = drasl_command(
            glist, max_urate=urate, weighted=weighted,
            scc=scc, scc_members=scc_members, dm=dm, bdm=bdm,
            edge_weights=edge_weights, GT_density=effective_GT_density,
            selfloop=selfloop, density_weight=density_weight,
            density_mode=mode, tol_low=t_low, tol_high=t_high,
        )
        return clingo(cmd, capsize=capsize, convert=drasl_jclingo2g,
                      configuration=configuration, timeout=timeout,
                      exact=not weighted, pnum=pnum, optim=optim)

    if density_mode == 'adaptive':
        # Production fallback ladder.  Both attempts use asymmetric tolerance
        # (downward window wider than upward) because PCMCI's measurement
        # density overestimates the causal-scale density.
        # 1. hard_soft0 with (tol_low, tol_high)                     (production tight)
        # 2. hard_soft0 with (tol_low+tol_widen, tol_high+tol_widen) (relaxed)
        # 3. soft                                                    (no bounds)
        attempts = [
            ('hard_soft0', eff_tol_low,              eff_tol_high),
            ('hard_soft0', eff_tol_low + tol_widen,  eff_tol_high + tol_widen),
        ]
        for attempt, (mode, t_low, t_high) in enumerate(attempts, start=1):
            if verbose:
                print(f"[drasl] adaptive attempt {attempt}: "
                      f"mode={mode} tol_low=−{t_low}% tol_high=+{t_high}% "
                      f"(GT={effective_GT_density}%)", flush=True)
            result = _run(mode, t_low, t_high)
            if result:
                if verbose:
                    print(f"[drasl] adaptive attempt {attempt}: "
                          f"SUCCESS — {len(result)} solution(s)", flush=True)
                return result
            if verbose:
                print(f"[drasl] adaptive attempt {attempt}: "
                      f"UNSAT — falling back", flush=True)

        # Final fallback: original soft-only encoding (no hard bounds at all).
        if verbose:
            print(f"[drasl] adaptive fallback: mode=soft (no hard bounds)",
                  flush=True)
        return _run('soft', eff_tol_low, eff_tol_high)

    return _run(density_mode, eff_tol_low, eff_tol_high)


def rasl(g, capsize, timeout=0, urate=0, pnum=None, configuration="tweety"):
    """
    :param g: ``gunfolds`` graph
    :type g: dictionary (``gunfolds`` graphs)

    :param capsize: maximum number of candidates to return
    :type capsize: integer

    :param timeout: timeout in seconds after which to interrupt
        computation (0 - no limit)
    :type timeout: integer

    :param urate: maximum undersampling rate to consider
    :type urate: integer

    :param pnum: number of parallel threads to run ``clingo`` on
    :type pnum: integer

    :param configuration: Select configuration based on problem type

        - ``frumpy`` : Use conservative defaults
        - ``jumpy`` : Use aggressive defaults
        - ``tweety`` : Use defaults geared towards asp problems
        - ``handy`` : Use defaults geared towards large problems
        - ``crafty`` : Use defaults geared towards crafted problems
        - ``trendy`` : Use defaults geared towards industrial problems
    :type configuration: string

    :returns: results of parsed equivalent class
    :rtype: dictionary
    """
    return clingo(rasl_command(g, urate=urate), capsize=capsize, configuration=configuration, convert=rasl_jclingo2g, timeout=timeout, pnum=pnum)
