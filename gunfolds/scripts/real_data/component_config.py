"""
NeuroMark ICA component configuration for fMRI experiments.

Defines component subsets (N=10, 20, 53), domain mappings based on the
NeuroMark paper (Du et al., 2020), and SCC-grouping strategies (domain-based
and correlation-based) for use with RASL/DRASL.

0-based indices correspond to row order in ICN_coordinates.csv and to the
feature axis of fbirn_sz_data.npz (shape [n_subjects, T, 53]).
"""

import numpy as np
from scipy.cluster.hierarchy import linkage, fcluster
from collections import defaultdict

# ---------------------------------------------------------------------------
# Domain definitions (0-based row indices in the 53-component ordering)
# ---------------------------------------------------------------------------
DOMAINS = {
    "SC": list(range(0, 5)),    # Subcortical         (5 ICNs)
    "AU": list(range(5, 7)),    # Auditory             (2 ICNs)
    "SM": list(range(7, 16)),   # Sensorimotor         (9 ICNs)
    "VI": list(range(16, 25)),  # Visual               (9 ICNs)
    "CC": list(range(25, 42)),  # Cognitive Control    (17 ICNs)
    "DM": list(range(42, 49)),  # Default Mode          (7 ICNs)
    "CB": list(range(49, 53)),  # Cerebellar            (4 ICNs)
}

DOMAIN_ORDER = ["SC", "AU", "SM", "VI", "CC", "DM", "CB"]

# Reverse lookup: 0-based index -> domain name
INDEX_TO_DOMAIN = {}
for _dom, _idxs in DOMAINS.items():
    for _i in _idxs:
        INDEX_TO_DOMAIN[_i] = _dom

# ---------------------------------------------------------------------------
# Short labels for all 53 components (from ICN_coordinates.csv)
# ---------------------------------------------------------------------------
COMP_LABELS_53 = [
    # SC (0-4)
    "Caudate",    "Subthalamus", "Putamen",  "Caudate2",   "Thalamus",
    # AU (5-6)
    "STG",        "MTG_au",
    # SM (7-15)
    "PoCG",       "L_PoCG",      "ParaCL",   "R_PoCG",     "SPL",
    "ParaCL2",    "PreCG",       "SPL2",     "PoCG2",
    # VI (16-24)
    "CalcarineG", "MOG",         "MTG_vi",   "Cuneus",     "R_MOG",
    "Fusiform",   "IOG",         "LingualG", "MTG_vi2",
    # CC (25-41)
    "IPL",        "Insula",      "SMFG",     "IFG",        "R_IFG",
    "MiFG",       "IPL2",        "R_IPL",    "SMA",        "SFG",
    "MiFG2",      "HiPP",        "L_IPL",    "MCC",        "IFG2",
    "MiFG3",      "HiPP2",
    # DM (42-48)
    "Precuneus",  "Precuneus2",  "ACC",      "PCC",        "ACC2",
    "Precuneus3", "PCC2",
    # CB (49-52)
    "CB",         "CB2",         "CB3",      "CB4",
]

# ---------------------------------------------------------------------------
# Predefined component subsets
# ---------------------------------------------------------------------------
# N=10: 1-2 per domain, emphasising regions with known SZ alterations
COMP_SET_10 = [
    0,   # SC  Caudate
    4,   # SC  Thalamus
    5,   # AU  STG
    7,   # SM  PoCG
    16,  # VI  CalcarineG
    25,  # CC  IPL
    26,  # CC  Insula
    44,  # DM  ACC
    45,  # DM  PCC
    49,  # CB  Cerebellum
]

# N=20: 2-4 per domain
COMP_SET_20 = [
    # SC (4)
    0, 1, 2, 4,
    # AU (2)
    5, 6,
    # SM (3)
    7, 9, 13,
    # VI (3)
    16, 17, 18,
    # CC (4)
    25, 26, 27, 35,
    # DM (3)
    42, 44, 45,
    # CB (1)
    49,
]

# N=53: all components
COMP_SET_53 = list(range(53))

COMP_SETS = {
    10: COMP_SET_10,
    20: COMP_SET_20,
    53: COMP_SET_53,
}


def get_comp_indices(n_components):
    """Return the list of 0-based component indices for a given size."""
    if n_components not in COMP_SETS:
        raise ValueError(
            f"n_components must be one of {list(COMP_SETS.keys())}, got {n_components}"
        )
    return COMP_SETS[n_components]


def get_comp_names(comp_indices):
    """Return short labels for a list of 0-based component indices."""
    return [COMP_LABELS_53[i] for i in comp_indices]


# ---------------------------------------------------------------------------
# SCC strategies
# ---------------------------------------------------------------------------

def get_domain_sccs(comp_indices):
    """
    Group selected components into SCCs by NeuroMark functional domain.

    Parameters
    ----------
    comp_indices : list[int]
        0-based indices of selected components.

    Returns
    -------
    list[set[int]]
        Each set contains 1-based node IDs (matching gunfolds convention)
        for one SCC.  Singleton domains are included.
    """
    pos_to_node = {idx: pos + 1 for pos, idx in enumerate(comp_indices)}

    domain_groups = defaultdict(set)
    for idx in comp_indices:
        dom = INDEX_TO_DOMAIN[idx]
        domain_groups[dom].add(pos_to_node[idx])

    return list(domain_groups.values())


def get_correlation_sccs(ts_2d, n_clusters=None, max_cluster_size=8):
    """
    Data-driven SCC grouping via hierarchical clustering of the absolute
    correlation matrix.

    Parameters
    ----------
    ts_2d : ndarray, shape [T, N]
        Time series for one subject (columns = selected components).
    n_clusters : int or None
        Fixed number of clusters.  If None, automatically choose a cut
        that keeps max cluster size <= max_cluster_size.
    max_cluster_size : int
        Upper bound on cluster size when n_clusters is None.

    Returns
    -------
    list[set[int]]
        Each set contains 1-based node IDs for one SCC.
    """
    N = ts_2d.shape[1]
    if N <= max_cluster_size and n_clusters is None:
        return [set(range(1, N + 1))]

    corr = np.abs(np.corrcoef(ts_2d.T))
    np.fill_diagonal(corr, 0)
    dist = 1.0 - corr
    # Condensed distance matrix for linkage
    from scipy.spatial.distance import squareform
    dist_condensed = squareform(dist, checks=False)
    Z = linkage(dist_condensed, method="ward")

    if n_clusters is not None:
        labels = fcluster(Z, t=n_clusters, criterion="maxclust")
    else:
        # Binary search for the fewest clusters where max size <= threshold
        lo, hi = 2, N
        best_k = N
        while lo <= hi:
            mid = (lo + hi) // 2
            lab = fcluster(Z, t=mid, criterion="maxclust")
            sizes = np.bincount(lab)
            if sizes.max() <= max_cluster_size:
                best_k = mid
                hi = mid - 1
            else:
                lo = mid + 1
        labels = fcluster(Z, t=best_k, criterion="maxclust")

    groups = defaultdict(set)
    for node_0based, cl in enumerate(labels):
        groups[cl].add(node_0based + 1)
    return list(groups.values())


def get_scc_members(strategy, comp_indices, ts_2d=None, **kwargs):
    """
    Dispatch to the appropriate SCC strategy.

    Parameters
    ----------
    strategy : str
        One of 'domain', 'correlation', 'estimated', 'none'.
    comp_indices : list[int]
        0-based component indices.
    ts_2d : ndarray or None
        Time series [T, N], required for 'correlation'.
    **kwargs : dict
        Forwarded to the strategy function (e.g. n_clusters, max_cluster_size).

    Returns
    -------
    list[set[int]] or None
        SCC members (1-based node IDs), or None for 'none'/'estimated'
        (estimated is handled in the caller from the PCMCI graph).
    """
    if strategy == "domain":
        return get_domain_sccs(comp_indices)
    elif strategy == "correlation":
        if ts_2d is None:
            raise ValueError("ts_2d required for correlation SCC strategy")
        return get_correlation_sccs(ts_2d, **kwargs)
    elif strategy in ("estimated", "none"):
        return None
    else:
        raise ValueError(
            f"Unknown SCC strategy '{strategy}'. "
            "Use 'domain', 'correlation', 'estimated', or 'none'."
        )


def build_fully_connected_gt(n_nodes):
    """
    Build a fully-connected ground truth graph (1-based node keys,
    all directed edges with weight 1, no self-loops).
    """
    gt = {}
    for i in range(1, n_nodes + 1):
        gt[i] = {}
        for j in range(1, n_nodes + 1):
            if i != j:
                gt[i][j] = 1
    return gt
