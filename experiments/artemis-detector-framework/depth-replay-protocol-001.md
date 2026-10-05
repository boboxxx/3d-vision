# Exact native arithmetic replay, separate from failed FP64 tolerance

Local identity audit003 covers every saved value. Original FP64 mean test
fails 3e-5 and remains failed. Before new execution, lock a separate diagnostic:
load both complete unchanged logits and their288-element depth axis from the
existing full forward artifacts, verify original file/array hashes, and on an
actual RTX PRO6000 with the same immutable torch2.7.1+cu128 runtime compute
torch.softmax(logits[:,0],dim=1), then torch.sum(prob*axis,dim=1), exactly the
author's native FP32 depth equation. Require bit equality for every one of
798720 depth outputs. No tolerance is fitted to observed discrepancies.

Save all replayed outputs, native/source/runtime identities and actual terminal.
This tests deterministic reproduction of native FP32 arithmetic from all saved
logits; it is not an independent GPU algorithm or a pass of the original FP64
criterion. Retain all prior failures and unchanged artifact bytes. No detector
construction, weight load, training, optimizer, teacher, codec or AP claim.
