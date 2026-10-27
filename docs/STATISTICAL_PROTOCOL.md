# Statistical protocol — decision recorded 2026-09-13

The **training objective is lane-weighted MSE**. It averages squared errors
across valid series–time observations, without first averaging within each
state. With equal observation counts per series, a state with 200 series
contributes twice as much as one with 100. Value and weight retain equal
channel weights: `(value squared error + weight squared error) / 2`.

The primary reported mean and paired comparisons use the same observation
weights. State clustering changes the uncertainty calculation, not these
weights. Comparisons report Student-t uncertainty with seven degrees of
freedom and wild-cluster p-values, adjusted over the declared comparison
family using Holm (and BH as exploratory screening). Equal-state differences
and state-rank Friedman/Nemenyi outputs are labeled **sensitivity analyses**.
Eight states and shared national shocks limit interpretation; these corrected
reruns remain retrospective rather than confirmation on an untouched holdout.

S4 state size is 64 across the family. Exp 2 contains 57 arms × three seeds
(171 runs); publication consumers require the complete source-validated roster,
checkpoint-bound completion, and identical prediction identities before analysis.
