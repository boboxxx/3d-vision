# Position symbols have a measurable channel precision cost

The fixed G/seed17 engineering prefix contains256 updates, all1198080 position symbols.
All saved records were checked again and match the previously sealed complete-prefix ledger.
The measured displacement compares the noiseless and received packet coordinates; it is not object
depth error, posterior calibration, detector localization error or a validation/AP result.

| SNR interval, dB | AWGN frames | AWGN coordinate RMS, m | Rayleigh frames | Rayleigh coordinate RMS, m |
| --- | ---: | ---: | ---: | ---: |
| [6, 8) | 23 | 5.903 | 20 | 10.684 |
| [8, 10) | 26 | 4.795 | 32 | 8.879 |
| [10, 12) | 17 | 3.672 | 19 | 7.719 |
| [12, 14) | 20 | 2.876 | 9 | 6.244 |
| [14, 16) | 21 | 2.302 | 19 | 5.218 |
| [16, 18) | 23 | 1.824 | 27 | 4.331 |

![Packet coordinate displacement](/Users/chen/Documents/ChatGPT/paper6/to_human/packet-position-channel-006.png)

The phase coordinate is mu=lo+span/2+span*theta/pi. For an interior node and small
complex AWGN, delta_theta has approximate variance N0/(2*Eposition). Thus the
coordinate variance is approximately (span/pi)^2*N0/(2*Eposition). At unit position
energy, span57.6m and10dB this gives about4.10m RMS. The plotted dashed curve uses
the uniform SNR interval average of N0, not a fitted parameter. Finite-SNR wrapping
and boundary clipping make this a local approximation rather than an exact law.

Conditioned on Rayleigh fading, the linearized variance also divides by|h|^2.
Its fading average diverges, while the actual bounded decoder error is finite and
squared displacement cannot exceed span^2. The observation supports retaining the
real deep fades in evaluation; it does not justify clipping the physical channel.
At[10,12)dB the decoder clips5.39% of Rayleigh position symbols versus0.48% forAWGN.

This result shows an explicit-coordinate reliability cost. It does not establish
that the conditional field is inferior or superior: the feature vectors, learned
kernel, spatial interpolation and frozen task network also determine predictions.
G learned bandwidths decline from19.2m initialization to14.80/10.45m at epochs1/2
forseed17 and12.02/9.61m forseed23; these are training-state observations, not
validation-selected settings. Full finalepoch3 and primary AWGN10 P-B stay fixed.

P-B compares complete structured and dense codec architectures. Their source/receiver
normalization groups differ, despite identical total symbols/energy. P-G/P-S use the
same architecture and normalization and isolate prior/organization effects. A positive
P-B alone would not prove that the stereo prior is the cause, nor prove a Shannon
rate-distortion bound, geometric sufficiency or low-cost edge deployment.

Evidence: data/provenance/cost-field-position-channel-006.json; all12 predefined
intervals reported. Current single-seed-scope007 requires44544 training updates, all3769
validation and original complete radio/matched-rate controls. Historical133632 was the
three-seed plan; it is superseded by the user-authorized single-seed scope.
