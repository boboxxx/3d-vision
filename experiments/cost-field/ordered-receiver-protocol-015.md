# Received-only ordered geometry and conditional channel uncertainty

Prelock before execution. This is a new receiver intervention, not a change to
the frozen004 source/training or the existing primary84 endpoints. Scientific
question: under a declared ordered finite-grid prior, do exact conditional
receiver uncertainty and order preserve latent coordinate/field squared error
better than independent clipped phases, independent posteriors, weighted
isotonic point projection, and a fading-blind posterior?

Freeze the derivation, receiver code and checks. Fixed production grid289,
no new parameters or source symbols:19 complex symbols/Es19 per pooled site.
Use only noisy ZF observations, public mean SNR/axis and actual perfect CSI
already assumed for the main Rayleigh contract. Preserve source slot identities;
never sort received features or use source probabilities/norms/means/GT. Reuse
unchanged004 sphere feature projection, learned feature/appearance decoders and
interpolation. Identity bypass is bit-identical frozen decoding. Additional
receiver compute/memory/latency remains an actual measurement requirement.

1. Independent complete enumeration of165 ordered triples on a9-point grid
for32 fixed observations (one uninformative,31 PCG1715 Gaussian likelihood
arrays with scale30). Compare all posterior marginals/normalization to exact
enumeration at absolute5e-12; posterior means must be ordered to5e-12.
Compare128 weighted isotonic controls to independent enumeration of all four
contiguous partitions at5e-12. Check32 actual ZF/baseband likelihood expansions
at absolute5e-10 after subtracting each likelihood's max.
2. Synthetic untrained G/P/S feature codecs, source cost32x72x8x12 and app
32x8x12, seed17, no dataset/detector. Three channels(identity/AWGN10/Rayleigh10)
and four modes(ordered/independent/isotonic/ordered_blind),36 full received-only
decoder cases. Require finite correct shapes, appearance exact frozen, all
identity outputs exact frozen, received-clone deterministic, unchanged states
and source packet, exactly114 charged complex uses for six pooled sites, and
private clean_mu keyword rejection in every arm.
3. Full declared-prior experiment:33-point unit grid, uniform over each6545
inclusive ordered index triples,65536 independent trials per condition,
batch4096. AWGN every integer6..18 and iid perfect-CSI Rayleigh every even6..18:
all20 conditions,1310720 physical triples/3932160 complex position symbols.
PCG1715+1000*(Rayleigh)+SNR; proper complex noise varianceq=10^(-SNR/10),
CN fading variance1, no fade clip. Position-only synthetic test charges three
symbols per triple; it is not a measurement of the full19-symbol system AP.

For each condition retain paired per-trial coordinate and fixed-coefficient
field errors; output means/standard errors and all paired differences. Field
gridnine uniform points, sigma1/3, coefficients(.8,-.3,.5), basis exp/K.
Comparators: hard clipped phase, independent conditional posterior, weighted
isotonic phase coordinates, ordered fading-blind posterior; additionally field
basis evaluated at ordered posterior mean. Exact ordered conditional posterior
must not be worse in declared-prior mean squared risk than each comparator
beyond fixed8 paired standard errors+1e-12 numerical allowance, separately for
every condition and metric. Positive means within allowance are recorded, not
rewritten as improvements. Report normalized-coordinate and latent57.6m-range
MSE; it is not object depth error. No SNR or threshold selection after results.

Execute exact same lock/checks on local and sheng CPU; source/hash/runtime/full
condition outputs and actual exits recorded, all failures retained. Independent
enumeration provides algorithm evidence; the declared finite prior provides
the risk theorem's scope. No actual KITTI distribution calibration, all stored
float32 source-node ordering,289-grid continuous approximation accuracy,
detector AP, new Bayesian theorem, RD theorem, dynamic source rate allocation
or publication novelty is claimed. Original-paper/full validation/resource/
manuscript requirements remain. A later separately prelocked full validation
receiver intervention must use fixed weights and complete3769 scenes, same
physical samples/source packet, all main channels/SNRs and measured resource
cost; preserve the original primary endpoint and all existing comparisons.
