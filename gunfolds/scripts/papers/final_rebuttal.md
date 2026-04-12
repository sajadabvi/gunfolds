# Final Rebuttal: Real-world noisy RASL (RnR)

---

## Part 1 — Reviewer jQK7

We want to sincerely thank reviewer for their constructive feedback and many suggestions.

W1 & Q1:
>it would be more convincing to see whether edge-level frequency in the solution set tracks with correctness

We have conducted the requested analysis. Across 100 simulation runs (5 Sanchez-Romero networks × 2 undersampling rates × 10 batches) with known ground truth, we computed each edge's frequency across all near-optimal solutions and measured precision (fraction correct) per frequency bin. We added a new section in Appendix D, with a new figure exploring this analysis.

The results confirm that solution-set agreement tracks with correctness:

Unanimous edges (freq ≈ 1.0): precision ≈ 0.85 (n=569), the largest bin and strongest signal.
Low-agreement edges (freq 0.1–0.3): precision ≈ 0.10–0.25, overwhelmingly spurious.
The trend is monotonically increasing from low to high frequency (excluding the 0.0–0.1 bin, which contains ground-truth edges absent from all solutions — a recall-ceiling artifact, not a confidence signal).
This validates the real-data finding in Section 4.5: the VMPFC→rFIC connection (89.7% agreement) can be interpreted as a calibrated confidence score, not merely a descriptive frequency. No prior ASP-based causal discovery method provides this per-edge uncertainty quantification.

We acknowledge that edge frequency is an approximate signal and does not guarantee identification of the single correct graph. However, the design objective of RnR is not to recover a point estimate but rather to produce a solution set that is minimal in size yet maximal in reliability, enabling a domain expert to make informed selections from a tractable set of candidates. Edge-level frequency serves as the best available tool for ranking edges within this framework, and the calibration results above confirm that it provides a meaningful and actionable confidence signal for this purpose.

W2 & Q2:
>"What does the solution set look like after stage 1 alone, and how does it change through stages 2 and 3?"

We have conducted the requested ablation by modifying the ASP solver to show changes to solutions through stages. We added subsection C.2 to the appendix with a new figure. We ran three configurations on the same data across multiple graph sizes, densities, and undersampling rates:

The results confirm that the lexicographic stages progressively narrow the search space while improving accuracy:

Stage 1 (Density only): Constraining only graph density yields a very large solution set (~168,920 graphs on average), essentially every graph matching the ground-truth density and low orientation F1 = 0.211. Density deviation is near zero (0.000), confirming this stage achieves its intended goal.
Stage 2 (+ Bidirected): Adding bidirected-edge constraints reduces the solution set by two orders of magnitude to ~1,752 graphs while improving both Orientation F1 to 0.350. Density deviation rises slightly (0.012) as the solver balances competing objectives.
Stage 3 (Full pipeline): Enabling all stages further compresses the set to ~23 graphs, a tractable number for expert review with Orientation F1 = 0.488. Each stage monotonically improves F1.
This demonstrates that each optimization stage performs its intended structural role: density matching anchors the global sparsity of the search space, bidirected resolution dramatically prunes structurally inconsistent candidates, and directed-edge orientation fine-tunes the final set to a small, high-quality collection of graphs.

Q3
>"It would be helpful to see how the runtime grows as you increase the number of nodes, even just to 15 or 20."

We apologize for not including a more systematic runtime analysis. For the graph sizes used in this paper (≤10 nodes), computation time was on the order of seconds to minutes, which we did not consider a bottleneck worth reporting in detail. However, the underlying problem is super-exponential, and runtime grows accordingly with graph size. Since scaling to larger graphs is outside the scope of the current work, we deferred this to future work.

For the reviewer's reference: in our experiments, graphs of size 10 are solved in seconds; graphs of size 20 require on the order of hours, typically 10 to 24 hours depending on graph density. Importantly, these times are for the general case without strongly connected component (SCC) decomposition. When SCC constraints are available, computation is substantially faster (up to two orders of magnitude). We will add a runtime discussion to the revision.

Q4:
>"Figure 3 orders nodes 6, 7, 9, 8 on the x-axis. Is there a specific reason for this ordering?"

This experiment is a direct comparison to the prior work we build upon, and is specifically designed to demonstrate improvement under identical conditions. We therefore use the same node counts, density settings, and ordering as Abavisani et al. (2023) to ensure a fair and controlled comparison.

---

## Part 2 — Reviewer L7F5


> The notation part in the Background sections needs improvement. 

We thank the reviewer for pointing out the brevity notation in Background Sec. We have revised and improved notation. 


> I found the claimed contributions to be marginal...

We thank the reviewer for this close reading. We see now that our framing overstated the novelty of each point taken alone. These five moves form a single system. Their value lies in the whole, not the parts. The core contribution is a method that works on real fMRI data where prior methods fail, as shown in Figures 4 and 5.

We clarify each point briefly:
The reviewer is right that the feasible set is simple. What we failed to stress is why it matters. A single graph gives the practitioner no way to judge its reliability. A small feasible set lets the expert see which structures hold across solutions and choose one with the aid of domain knowledge.
Right again, this is one added constraint. What we want to report is that this particular constraint has a large effect on runtime of an otherwise NP-hard problem. Which constraint helps, and by how much, is worth knowing.
We searched the literature for meta-solver use in realistic undersampled settings and found none. We welcome any pointers. Figures 4 and 5 show the gains.
Lexicographic optimization is established but, to our knowledge, not yet applied in this setting. We agree the main text relied too much on Appendix A and will bring the key reasoning into the paper. See also our response to R-jQK7 for a step-by-step breakdown.
The reviewer is right that adaptive weighting is not new. We claim it only as part of the full system, where it works with the other four moves to yield clear gains on real data.

We will revise the Methods section to better frame each point and its role in the whole.

> the paper states that PCMCI can produce contemporaneous links.

We thank the reviewer for this insightful distinction. While variants such as PCMCI+ and LPCMCI are indeed required to orient contemporaneous edges or explicitly model latent confounders, the standard PCMCI algorithm (Runge et al., 2019, Sec. Materials and Methods) does detect undirected contemporaneous dependencies by testing for conditional independence at time delay τ=0. In our framework, we assume causal sufficiency and allow for undersampling, but crucially make no assumption about the undersampling rate — thereby permitting it to be arbitrarily fine, such that no physical process occurs truly contemporaneously. Under this assumption, an undirected contemporaneous link detected by PCMCI is interpretable as an unobserved common cause induced by the sampling process (temporal aggregation), rather than by the masking of a latent variable. This is why we treat such links as bidirected edges encoding potential confounding due to undersampling, and we will clarify this reasoning in the revision.

>  method assume causal stationarity? 

Yes. RnR assumes both causal stationarity (the causal structure is time-invariant) and causal sufficiency (no unmeasured confounders beyond those induced by undersampling). We will state these assumptions explicitly in the revision.

> *"when the initial method does not account for under-sampling, then the initial graph H may contain spurious patterns such as a two-node cycle"*

What we mean here is that an isochronal bidirected edge denotes a confounder that naturally arises in the undersampling case. If an algorithm is not modeling such situations, it tends to add both edges at once: A->B, B->A.

> *"How critical is the proxy nature of BOLD signal when learning causal mechanisms with fMRI data?"*

The approximate nature of the BOLD signal is in fact a central motivation for this work. RnR was designed specifically to recover causal structure from noisy data in the presence of artifacts introduced by both temporal undersampling and the convolution properties of the hemodynamic response. Our experimental progression demonstrates the increasing difficulty as these factors compound: Figure 3 shows the idealized edge-breaking setting, where a single modification renders the true graph unreachable; Figure 5 introduces realistic VAR-generated signals, adding process noise and dynamic complexity; and Figure 6 further applies BOLD convolution to those VAR signals, simulating the full measurement pipeline. Performance degrades progressively across these three settings, directly illustrating the impact of measurement noise, undersampling, and hemodynamic variability on structure learning. We appreciate the reviewer's suggestion and have added a dedicated discussion section addressing the proxy nature of BOLD and its implications for causal inference.

---

## Part 3 — Reviewer rp9v

*(To be added)*

---

## Part 4 — Reviewer cuMG

*(To be added)*
