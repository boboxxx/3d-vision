# Exact coordinate phase mechanism: numerical verification 009

Confirmatory verification of an analytic mechanism, locked before execution. This
does not change training, channel simulation, checkpoints, validation selection,
the 19-complex-use packet budget, or the AP primary endpoint.

For a unit-energy position symbol x=exp(i theta), theta in [-pi/2,pi/2], let
q=10^(-SNR/10), CN(0,q) mean real/imaginary variance q/2, and L=57.6m.
AWGN has y=x+n; iid Rayleigh with perfect CSI and no fade clipping has
y=x+n/h, independent h~CN(0,1). The continuum decoder is
mu_hat=2+L/2+(L/pi)*clip(arg(y),-pi/2,pi/2).
Its squared displacement is bounded by L^2. Frozen codec004 uses a float32
near-zero fallback as well; the continuum expression excludes this event.
A uniform density bound gives P(fallback)<=4*eps32^2/(pi*q), so changing this
event can change the MSE by at most L^2 times this probability. This does not
bound all floating-point errors or prove CUDA/CPU numerical identity.

Verify the derived densities and clipped, wrapped moment integral described in
position-phase-derivation-009.md. Use all integer SNR6..18 for BOTH channels for
mechanism analysis (the primary Rayleigh AP grid remains every2dB). Fixed theta
values [-pi/2,-pi/3,0,pi/3,pi/2], not a fitted distribution of learned positions.
Report all130 combinations and all failures; do not choose favorable angles.

Independent checks, fixed before execution:

- Phase PDF nonnegative on4097 evenly spaced phases and mass error<=2e-10;
  exact even symmetry to<=2e-12, radial integral comparison at13 evenly spaced
  phases for each of26 physical conditions, error<=2e-10+2e-9*abs(reference).
- Split adaptive quadrature at0, all clipping and wrapping breakpoints;
  epsabs=epsrel=2e-11, limit=300. Confirm MSE in[0,L^2], probability in[0,1],
  positive/negative theta MSE and clipping probability difference<=2e-9.
- Independent direct Gaussian simulation, PCG64 seed2027100509, 1048576 draws
  per physical condition, generated noise first then fading, reused across the
  five source angles within that condition. Explicit h*x+n followed by complex
  division by h, with no fading clipping. Independent np.angle decoder, including
  the actual eps32 near-zero fallback. Check each MSE against the integral using
  abs difference<=8*sample standard error+1e-8 m^2, and clipping probability
  <=8*sqrt(p*(1-p)/N)+8/N. Sample SE is descriptive, not a detector significance
  claim; the conservative8SE threshold is an engineering admission criterion.
- Keep generated theory/MC JSON and130-row CSV, source/protocol/derivation/codec
  hashes, versions and RNG description. Exclusive output creation. Log failure
  and retain its outputs; changes require a separately locked repair.

Execution is CPU-only local, bounded memory, no KITTI/GT/model access. A second
host may rerun the same source/protocol in its existing CPU environment; no
install or GPU training interruption. A graph of exact continuum RMSE and MC
checks will be made only from complete accepted rows, with all source angles
disclosed. No linearized Rayleigh average, high-SNR theorem, fundamental new
Gaussian-ratio distribution, Shannon rate-distortion bound, detector error bound,
task uncertainty calibration, or AP gain is claimed.
