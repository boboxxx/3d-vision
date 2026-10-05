# Independent uniform arithmetic repair, original limits retained

All full-proof-protocol-005.md checks and numerical limits remain unchanged.
Keep original005 code, successful native256 proof and failed local005 proof.

The local NumPy2.4.4 Generator.uniform(6,18) differs from frozen native1.26.4
by one FP64 ULP for some SNR values. Independent PCG64 integer channel draws
agree on all11136 steps. Independent rng.random() followed by explicit Python
FP64 arithmetic 6.+12.*u agrees exactly with every native SNR. This is the
two-operation uniform transformation; no numerical acceptance bound is relaxed.

Isolated006 replaces only independent regeneration of uniform with that
explicit expression. Exact channel and SNR equality is still required for the
complete schedule and each saved row. It changes no training004 code, schedule,
source lock, model, experimental endpoint or live job. Relock before rerunning
all256 saved records natively/locally. Prefix remains engineering only.

Diagnostic live report rsync returned23 while its source was rewritten. Its
parsed bytes are preserved as diagnostic data, not a validated report identity.
Use the independently checked per-row hashes for comparing prefix records.
