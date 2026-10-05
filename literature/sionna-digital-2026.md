# Pinned Sionna digital comparator

NVIDIA official implementation, tagv2.1.0,
revision6498239a72267ee25edb600e5b826b0531971d7f,
https://github.com/NVlabs/sionna/tree/v2.1.0.
Official LDPC documentation:
https://nvlabs.github.io/sionna/v2.1.0/phy/api/fec/ldpc/index.html
Official mapping:
https://nvlabs.github.io/sionna/v2.1.0/phy/api/mapping/index.html

Read official pyproject, LDPC encoding/decoding and Gray mapping source. This
release's PHY is PyTorch and package requiresPython>=3.11,torch>=2.9.1,
numpy>=2.2.6; it cannot silently be installed over sheng's3.10/2.5 orArtemis
original2.7/1.26 environment. Independent CPU PHY-only setup is now verified.
RT is explicitly omitted and sole missingdependency message retained; do not
claim full Sionna all-module environment or RTX operator portability from it.

Encoder implementsNR38.212 with basegraph/lifting/filler shortening/puncturing,
RV0 rate matching and optional modulation-dependent output interleaving.
LDPC5GDecoder must be linked to the encoder; compatible LLR sign is logp1/p0.
The original Cao paper only specifiesrates2/3+64QAM and1/2+256QAM; source does
not establish authors usedNR,n1944,Gray label or20-iteration BP. Our choices
are explicit ambiguity resolutions, not original author-code recovery.

Actual original-scenario digital engineering is recorded in
[the result note](../experiments/original-paper-scenarios/digital-analysis-001.md).
Eight actual packets/12 checks/16 artifact audits passed, including native
zero-noise pairedJP2 bytes/CRC.10dB synthetic failures are retained; no noise
curve/detection AP or comparative communication gain has been established.
