# Q0: bounded three-particle depth communication feasibility

Exploratory representation feasibility after negative F9, not a selected
new full detector or a claim of novel Wasserstein/analog sphere coding.
The stereo distribution-loss and geometric analog-coding prior-art notes
already constrain those claims. Existing original-scenario main jobs and
formal SRCNN keep their priority. This CPU experiment changes none of them.

Represent a bounded metric-depth histogram by three equally weighted ordered
particles at inverse-CDF levels1/6,1/2,5/6. For a piecewise-uniform histogram,
interpolate inside the selected bin, rather than use a hard bin index with
zero training gradient. These levels minimize one-dimensional W1 among three
ordered equal-mass atoms: each atom is the median of the corresponding third
of the source quantile function. This is a basic approximation property,
not novelty or a statement that three atoms preserve detection information.

Use fixed public depth bounds to scale the particles to q in[-1,1]^3 with
q1<=q2<=q3. Encode four real values
T(q)=sqrt(2)*(q1,q2,q3,1)/sqrt(1+||q||^2), occupying exactly two complex
symbols and total energy2 per cell. The fourth coordinate is a charged
anchor, not an untransmitted normalization factor. Each symbol pair has mean
Es1; do not renormalize after concatenation with future appearance content.
Receiver consumes only four received real values and the public bounds.
Decode qhat=P_C(r[0:3]/max(r[3],1/sqrt(2))), where C is the ordered cube
and P_C its Euclidean projection. No divide-by-near-zero or source-side scale.
This is a robust bounded decoder, not an ML/MMSE optimum.

For r=T(q)+n, noiseless anchor>=1/sqrt(2) and projection nonexpansiveness give
||qhat-q||<=2sqrt(2)||n|| pointwise. Consequently, transportation between the
source and decoded three-atom laws is their mean absolute metric-depth change.
It is bounded by the normalized coordinate error times the fixed depth scale.
Full histogram error additionally contains intrinsic three-atom approximation
error. Neither this inequality nor a CPU experiment bounds detection AP,
establishes a rate-distortion frontier, posterior calibration or information
sufficiency. Rayleigh perfect-CSI equalization has heavy-tailed noise; the
pointwise bound does not imply a finite average equalized-noise second moment.

Before results, fix the following checks and measurements:

- Float64 noiseless inversion/energy and within-bin quantiles agree with a
  separate NumPy reference to1e-12; PyTorch finite-difference gradient checks
  test soft positive histograms away from bin-boundary changes.
- Exact ordered-cube projection enumerates the four contiguous partitions and
  is compared against a separate constrained SciPy optimizer on32 seeded cases,
  absolute difference<=2e-7. Check pointwise bound on all286 sorted triples from
  eleven evenly spaced coordinates, with32 Gaussian perturbations each.
- Twelve fixed synthetic72-bin distributions on edges2.0:0.8:59.6 cover narrow
  near/mid/far peaks, uniform/broad distributions, two modes and low-mass distant
  modes. Report exact piecewise-uniform-to-three-atom W1 for every distribution.
  These synthetic bounds resemble one inspected coarse support but do not
  substitute for any native teacher's72/288 support or calibration.
- Repeat each distribution64 times in identity and AWGN/Rayleigh6,10,18dB.
  Noise has real variance10^(-SNR/10)/2 at Es1. Rayleigh uses two independent
  CN(0,1) fades, perfect received CSI and unclipped zero forcing; no artificial
  fade floor. Fixed NumPyPCG64 seed17, equal randomness across source cases.
  Retain all transmitted/received/noise/fade/decoded arrays and actual energy,
  symbol counts and every-case distortion. No learned parameter fitting.

Record representation error, channel-only three-law W1 and full histogram W1
separately; no synthetic result can satisfy a native train-only sufficiency gate.
Before any full task trial, compare equal-capacity generic ordered latent
content under the same four-real carrier, receiver capacity, teacher exposure,
training steps and actual symbol/energy/compute budgets. A learned feature
adapter to the existing multi-channel LIGA input is required. Three particles
may erase minority modes/tails; retain such failures rather than selecting
favorable depth distributions or noise conditions.
