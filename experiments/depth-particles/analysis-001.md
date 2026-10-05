# Q0: carrier checks pass; equal-mass representation loses multimodal depth

Protocol23ebde8 precedes implementation1fe0566 and both-host CPU execution.
This is an exploratory synthetic feasibility experiment. It is not a native
teacher, KITTI, trained detector or positive geometry-specific result.

All twelve source distributions, seven physical conditions and64 repeated
cells per distribution are retained on each host. Four complete artifacts and
10752 cell records/168 source-condition statistics pass local verification,
including received-only scalar pool-adjacent-violators replay, literal physical
array accounting and a separate scalar integral for every histogram W1.
Maximum independent scalar W1 disagreement is4.13e-13m; cross-host raw-array
maximum difference2.85e-14. These are numerical agreements, not identical hashes.

The fixed anchor carries its own charged coordinate. Each cell uses two complex
symbols and energy2, Es1; every condition has1536 actual symbols/energy1536.
Noiseless inverse error is1.12e-16; energy error<=4.45e-16. Thirty-two independent
constrained-optimizer comparisons agree within1.39e-13. The pointwise noise bound
passes9152 perturbation cases per host, and finite-difference gradients pass for
positive histograms away from branch changes and the carrier. This does not
establish bounded histogram-to-quantile gradients near an empty density valley.

Source approximation error remains when the channel is noiseless:

| Synthetic source | Three-atom W1 (m) |
|---|---:|
| Narrow near | 0.066667 |
| Narrow middle | 0.066667 |
| Narrow far | 0.066667 |
| Uniform | 4.800000 |
| Broad middle | 1.716448 |
| Broad far | 2.179094 |
| Two modes, 50/50 | 8.000000 |
| Two modes, 35/65 | 0.918681 |
| Distant mode, 5% | 2.449123 |
| Distant mode, 10% | 4.829630 |
| Distant mode, 20% | 6.433333 |
| Distant mode, 40% | 3.277778 |

For the two narrow modes separated by48m with equal probability, three atoms
cannot split equal one-third masses evenly between the modes. At least one
sixth of the probability must effectively cross the gap. W1 is8m here even at
the optimal equal-mass source quantiles. Increasing SNR cannot fix that intrinsic
approximation. The identity, AWGN18 and Rayleigh18 complete histogram W1 means
are8.0000,10.0356 and11.2596m for this source. Report every condition rather
than selecting one noise level as evidence of a full method advantage.

The bounded decoder's useful property is narrower. For q in the ordered cube,
x=T(q), x4>=a=1/sqrt(2), d=max(x4+n4,a), and |d-x4|<=|n4|. Before projection,
the coordinate error is (n[0:3]+q*(x4-d))/d. Its norm is bounded by
sqrt(2)*(||n[0:3]||+sqrt(3)*|n4|)<=2sqrt(2)||n||. Euclidean projection is
nonexpansive and leaves q fixed. For ordered equal-mass laws, W1 equals mean
absolute particle displacement in metric depth. This only bounds their coded
transport, not the information removed by source approximation or detection AP.
Rayleigh heavy-tailed equalization precludes inferring finite average noise
second moments from this pointwise inequality; bounded outputs still hold.

Do not promote this equal-three-atom representation directly to main detector
training. The result motivates mode-mass flexibility and a density-valley
stability diagnostic. A two-node distribution with transmitted mass fits three
independent coordinates and can represent unequal two-mode probabilities, but
its depth/weight error sensitivity, unimodal approximation, receiver adaptation
and matched controls are untested. That is a following hypothesis, not a chosen
solution. Any next representation must retain meaningful native teacher/task
information and match generic latent/teacher/receiver/compute/resource exposure.
The carrier is standard geometric analog coding, and W1 quantile approximation
is not claimed as the novel contribution.

Complete evidence: `data/engineering/depth-particles-Q0-{local,sheng}-CPU-001/`,
`data/provenance/depth-particles-Q0-complete-local-verification-001.json`.
