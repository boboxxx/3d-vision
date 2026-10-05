# Staged reproduction: verified specification and remaining gates

Source: [Cao et al., Section IV and Appendix B](https://arxiv.org/pdf/2502.12735).
The paper specifies Adam and the following stages. These stages have **not**
been executed here; the existing probes verify structure and gradients only.

`stages.py` now implements every phase's parameter/LR scope and losses.
The CPU phase probe executes actual optimizer updates at all eight phase
boundaries, verifies finite active gradients and unchanged frozen states, and
checks that Adam state is absent before flow/semantic unfreezing. Stage1
has562 then622 active tensors; stage2 has16, stage3 has12, stage4 has638;
stage5 has80 channel-only then718 joint. These synthetic weights are discarded.
The full native370×1224 GPU sanity also passes stage1 and joint stage5 with
562/718 finite gradients; peak torch allocations are5.76/7.07GB (decimal).
One native frame is a memory/gradient gate, not full-fold training or AP.

Choices to lock for the formal trainer: full native RGB, batch1, four spawn
workers, no crop/augmentation, Adam default betas/epsilon and weight_decay0,
FP32, elementwise native-pixel Charbonnier epsilon1e-3, fixed stage4 switch
after epoch5, clipping10/nonfinite rejection. Training SNR uniform6–18dB is
a variant choice based on the evaluated range; the primary text does not
specify that training distribution. New optimizer between stages, retained
optimizer state across within-stage unfreezing, complete predecessor weights.
An erased stage5 frame must consume a scheduled sample but skip the optimizer
update and be counted explicitly. Native data loader, saved checkpoint/raw
scalar audit and complete staged training are still to be implemented.

| Stage | Trainable components | Epochs | Learning rates |
|---|---|---:|---|
| 1 | Global encoder/decoder, fusion; SpyNet frozen initially | 2 frozen-flow + 10 unfrozen-flow | Global 2e-4, fusion 1e-4, flow 2.5e-5 |
| 2 | Key encoder/decoder; frozen YOLO | 10 | Encoder 1e-4, decoder 2e-4 |
| 3 | Fusion; frozen global/key | 6 | 1e-4 |
| 4 | Semantic codec; frozen YOLO | 10 | Global 1e-4, key/fusion 2e-5, flow 1.25e-5 |
| 5 | Channel codec, then joint semantic/channel | 25 + 20 | Channel/global 1e-4, key/fusion 2e-5, flow 1.25e-5 |

Losses: Charbonnier in stages1/2; masked plus0.5×global Charbonnier in stage3;
hybrid then global Charbonnier in stage4. In stage5, the first25 epochs train
only the channel codec using MSE on frozen semantic-encoder outputs; the next20
jointly train the semantic/channel codec using RGB Charbonnier (SectionIV,
paragraph spanning pages7–8). Stage1 uses zero recovered-key
inputs. The paper does not resolve every augmentation, batch, epsilon,
loss-switch epoch, parameter-sharing or ROI selection setting. Lock explicit
variant choices before training, and retain actual parameter/compute counts.

Fixed sensor ROI cache: COCO-pretrained YOLOv5n v7.0, car/bus/truck, confidence
0.25, NMS IoU0.45; native image coordinates, no GT boxes. Train3340 and holdout372
folds are fixed; author pretrained detector exposure remains separately disclosed.
Changing ROI settings or architecture requires a new variant and identity.

Full372-pair ROI/mask audit passed. Average union area is7.2296%; idealized
global36×/key4× source-element compression would be21.8094×, ignoring padding,
convolution support, quantization and control. This is a diagnostic estimate,
not measured bitstream compression or channel use. Do not label it30×.

Before wireless comparisons, implement/audit actual sparse key support,
TableIX channel codec, symbol normalization/packing, quantization where used,
serialized boxes and protected control cost. Then lock matched physical symbol
and energy budgets. Native Stereo-RCNN and a receiver-matched RGB comparison
answer different questions; report them separately. Final AP requires trained
weights, complete prediction files and independent native evaluator audit.
