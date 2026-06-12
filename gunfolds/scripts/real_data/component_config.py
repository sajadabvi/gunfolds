"""
NeuroMark ICA component configuration for fMRI experiments.

Defines component subsets (N=10, 13, 14, 20, 53), domain mappings based on the
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

# N=13: COMP_SET_10 plus three prefrontal association hubs (fronto-limbic
# extension).  Each added region is mapped to its NeuroMark ICN by anatomy +
# hemisphere (X-sign) from ICN_coordinates.csv / Du et al. 2020 Table 2:
#   IFG        -> idx 28 "IFG"   (IC 70, MNI -48.5, 34.5,  -0.5; left)    -> CC
#                 left inferior frontal gyrus (Broca / language), the canonical
#                 NeuroMark IFG; complements the STG language node in N=10.
#   rDLPFC     -> idx 35 "MiFG2" (IC 88, MNI  30.5, 41.5,  28.5; right)   -> CC
#                 right dorsolateral PFC = right middle frontal gyrus (BA 9/46);
#                 the only right-hemisphere dorsal MiFG in the parcellation.
#   mPFC/VMPFC -> idx 46 "ACC2"  (IC 17, MNI  -9.5, 46.5, -10.5; ventral) -> DM
#                 ventromedial PFC = ventral anterior medial cingulate; the
#                 ventral (Z<0) anterior DM node, distinct from N=10's ACC (44).
# Domain composition: SC2 AU1 SM1 VI1 CC4 DM3 CB1.  Largest domain SCC = 4 (CC),
# well under get_correlation_sccs' max_cluster_size=8.  Anatomical NeuroMark
# labels are kept in COMP_LABELS_53 (shared with N=20/53); the rDLPFC / VMPFC
# functional aliases are documented here only.
COMP_SET_13 = [
    0,   # SC  Caudate
    4,   # SC  Thalamus
    5,   # AU  STG
    7,   # SM  PoCG
    16,  # VI  CalcarineG
    25,  # CC  IPL
    26,  # CC  Insula
    28,  # CC  IFG        (added: left inferior frontal gyrus / language)
    35,  # CC  rDLPFC     (added: right MiFG, dorsolateral PFC)
    44,  # DM  ACC
    45,  # DM  PCC
    46,  # DM  mPFC/VMPFC (added: ventromedial PFC / ventral ACC)
    49,  # CB  Cerebellum
]

# N=14: exactly 2 per domain (7 domains × 2). Superset of COMP_SET_10,
# subset of COMP_SET_20 — chosen so any sweep across N=10/14/20 only adds
# components, never swaps them.
COMP_SET_14 = [
    # SC (2): Caudate, Thalamus  — same as N=10
    0, 4,
    # AU (2): STG, MTG_au        — N=10 had 1, add MTG_au from N=20
    5, 6,
    # SM (2): PoCG, +1 from N=20 (index 9)
    7, 9,
    # VI (2): CalcarineG, +1 from N=20 (index 17)
    16, 17,
    # CC (2): IPL, Insula        — same as N=10
    25, 26,
    # DM (2): ACC, PCC           — same as N=10
    44, 45,
    # CB (2): Cerebellum, +1 from CB pool (index 50)
    49, 50,
]

# N=15: COMP_SET_13 plus the 2 highest-priority circuit completers, giving the
# clean nesting N=10 ⊂ N=13 ⊂ N=15 ⊂ N=20 (N=15 ⊂ N=20 since N=20 also adds
# 13 & 36).  Indices from ICN_coordinates.csv / Du et al. 2020 Table 2
# (row = 0-based index):
#   13 PreCG (SM)  PreCG(66), MNI −42.5,−7.5,46.5 — primary motor; the other half
#                  of the thalamo-sensorimotor finding (N=13 had only somato-
#                  sensory PoCG; motor is the Woodward/Cheng counterpart).
#   36 HiPP  (CC)  HiPP(48), MNI 23.5,−9.5,−16.5 (right, anterior — the SZ-
#                  relevant CA1 region; closes the hippocampal gap the
#                  comparison doc flagged).  NeuroMark has no limbic domain, so
#                  hippocampus is filed under CC and groups with CC under the
#                  domain SCC strategy (not a limbic block).
# Domain composition: SC2 AU1 SM2 VI1 CC5 DM3 CB1.  Largest domain SCC = 5 (CC),
# under get_correlation_sccs' max_cluster_size=8.
COMP_SET_15 = [
    0,    # SC  Caudate      (N=13)
    4,    # SC  Thalamus     (N=13)
    5,    # AU  STG          (N=13)
    7,    # SM  PoCG         (N=13)
    13,   # SM  PreCG        (added: primary motor)
    16,   # VI  CalcarineG   (N=13)
    25,   # CC  IPL          (N=13)
    26,   # CC  Insula       (N=13)
    28,   # CC  IFG          (N=13)
    35,   # CC  MiFG2/rDLPFC (N=13)
    36,   # CC  HiPP         (added: anterior hippocampus — CC domain in NeuroMark)
    44,   # DM  ACC          (N=13)
    45,   # DM  PCC          (N=13)
    46,   # DM  ACC2/VMPFC   (N=13)
    49,   # CB  Cerebellum   (N=13)
]

# N=20: COMP_SET_13 plus 7 regions that complete the major SZ circuits the
# N=13 set could only partially touch.  N=10 ⊂ N=13 ⊂ N=20 (each a strict
# superset).  Indices resolved by anatomy + hemisphere from ICN_coordinates.csv
# / Du et al. 2020 Table 2 (row = 0-based index).  The 7 additions (and why):
#   2  Putamen   (SC)  Putamen(98), MNI −26.5,1.5,−0.5  — dorsal striatum;
#                      completes the salience triad (putamen + insula + ACC).
#   6  MTG_au    (AU)  MTG(56), MNI −42.5,−6.5,10.5 (left) — the *auditory*-
#                      domain MTG; rounds out the auditory-language loop behind
#                      the STG→DMN finding.  (Distinct from the VI-domain MTGs.)
#   13 PreCG     (SM)  PreCG(66), MNI −42.5,−7.5,46.5 — primary motor; the other
#                      half of the thalamo-sensorimotor finding (N=13 had only
#                      somatosensory PoCG).
#   17 MOG       (VI)  MOG(5), MNI −23.5,−93.5,−0.5 (left) — visual association
#                      (V2/V3), directly up-hierarchy from CalcarineG(16); tests
#                      whether the cortex→thalamus reversal generalises.
#   33 SMA       (CC)  SMA(84), MNI −6.5,13.5,64.5 — supplementary motor area.
#                      NOTE: NeuroMark files SMA under Cognitive Control, NOT
#                      Sensorimotor, so under --scc_strategy domain it groups
#                      with CC.  (Chosen over left DLPFC per user, 2026-06-08.)
#   36 HiPP      (CC)  HiPP(48), MNI 23.5,−9.5,−16.5 (right, anterior — the
#                      SZ-relevant CA1 region; limbic dysconnectivity +
#                      treatment-response literature).  NeuroMark has no limbic
#                      domain — hippocampus is filed under CC, so it groups with
#                      CC under the domain SCC strategy (not a limbic block).
#   42 Precuneus (DM)  Precuneus(32), MNI −8.5,−66.5,35.5 — second posteromedial
#                      DMN hub; confirms the DMN effect is not PCC-specific.
# Domain composition: SC3 AU2 SM2 VI2 CC6 DM4 CB1.  Largest domain SCC = 6 (CC),
# under get_correlation_sccs' max_cluster_size=8.
# (This REDEFINES the old N=20 set, which was NOT a superset of N=13 — prior
# N=20 results are on a different component set and are not comparable.)
COMP_SET_20 = [
    # SC (3)
    0,    # Caudate     (N=13)
    2,    # Putamen     (added: dorsal striatum / salience)
    4,    # Thalamus    (N=13)
    # AU (2)
    5,    # STG         (N=13)
    6,    # MTG_au      (added: auditory-language MTG)
    # SM (2)
    7,    # PoCG        (N=13)
    13,   # PreCG       (added: primary motor)
    # VI (2)
    16,   # CalcarineG  (N=13)
    17,   # MOG         (added: visual association)
    # CC (6)
    25,   # IPL         (N=13)
    26,   # Insula      (N=13)
    28,   # IFG         (N=13)
    33,   # SMA         (added: supplementary motor — CC domain in NeuroMark)
    35,   # MiFG2/rDLPFC (N=13)
    36,   # HiPP        (added: anterior hippocampus — CC domain in NeuroMark)
    # DM (4)
    42,   # Precuneus   (added: 2nd posteromedial DMN hub)
    44,   # ACC         (N=13)
    45,   # PCC         (N=13)
    46,   # ACC2/VMPFC  (N=13)
    # CB (1)
    49,   # Cerebellum  (N=13)
]

# N=53: all components
COMP_SET_53 = list(range(53))

COMP_SETS = {
    10: COMP_SET_10,
    13: COMP_SET_13,
    14: COMP_SET_14,
    15: COMP_SET_15,
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
