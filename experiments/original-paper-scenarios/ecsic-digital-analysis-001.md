# ECSIC complete source containers through actual digital PHY

The prelocked 20 one-attempt packets finished on Artemis CPU job11424223 with
all Slurm steps COMPLETED0:0. A separately prepared, independently implemented
full replay on CPU job11424237 also completed0:0. All20 packet audits pass;
11 pairs are received and9 are erased. This closes fixed-source physical-channel
engineering, not neural reconstruction, KITTI AP or original10/30/50 operating points.

Protocol728df8d precedes implementationd73015c; local fixturesf171c34 precede
native execution. Independent auditor3e9121c and audit allocationcc090b3 precede
its execution. Historical digital/entropy sources were unchanged. Complete
original and replay terminal records include raw UTC sacct, empty squeue and
fresh artifact hashes. Native PHY arrays remain on Artemis; transferred metadata,
auditor, logs and both terminal bindings pass local verification.

| Source / code | Inner bytes | Outer wire bytes | Blocks | Padding bits | Complex uses / attempt | Actual energy / attempt |
|---|---:|---:|---:|---:|---:|---:|
| synthetic32x64 / 2/3 64QAM |813|833|6|1112|1944|1726.6666667|
| synthetic32x64 / 1/2 256QAM |813|833|7|140|1701|1610.7882353|
| training000000 / 2/3 64QAM |44824|44844|277|240|89748|89517.3333333|
| training000000 / 1/2 256QAM |44824|44844|370|888|89910|89813.2|

The166-byte source header/CRC is already inside the complete ECSIC container;
20 additional outer bytes traverse the same actual LDPC/QAM chain. All codewords,
rate matching/filler/interleaving, Gray QAM, APP LLRs, saved BP outputs, dedicated
noise/fading draws, received samples, energies and double-container validation
are independently checked. Receiver length comes only from actual decoded
information blocks and their received header. No source-length argument or image
fallback is allowed. Whole-pair erasures retain full attempted symbols/energy.

All four identity packets are exact. All eight6dB AWGN/Rayleigh attempts erase
at header validation. Seven of eight18dB attempts are received; the native
1/2+256QAM Rayleigh18 attempt erases because decoded padding is nonzero. Do not
recover it by using source bytes/length, changing padding rules or retrying.
Each condition has one fixed draw per source/code; these counts are not a BER,
BLER or reliability estimate. Perfect instantaneous CSI, per-symbol independent
Rayleigh/ZF, ideal frame boundaries and public code configuration are declared.
CRC acceptance is error detection and does not guarantee undetected-error absence.

The independently recomputed148 local fixtures include4 valid and144 malformed
cases. Installed PHY/dependency identities match before/after/replay. The original
run asserted unchanged global RNG per packet but did not serialize those states;
only dedicated per-packet RNG can be independently replayed from complete saved
states. The auditor separately proves its own full replay preserves global RNG.

Evidence: `data/provenance/ecsic-digital-CPU-001-audit.json`,
`artemis-ecsic-digital-CPU-001-terminal.json`,
`artemis-ecsic-digital-audit-terminal-001.json`, and
`ecsic-digital-transferred-verification-001.json`. Next: separately lock causal
neural reception/cropping on the fixed sheng ECSIC CPU runtime, then prepare
KITTI adaptation/rate points and detector-facing shared received-image caches.
