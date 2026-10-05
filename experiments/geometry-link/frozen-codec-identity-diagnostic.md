# F3 codec isolation, locked after negative10dB result and before evaluation

F3 fixed3340-update final checkpoint SHA
cb50b4d4073fb2e5be31533d949a2141a46389436a5080429753de8b038aef7c
has audited10dB Car3D AP_R40 E/M/H0.26152/0.06349/0.03881% on the fixed372
train-only holdout. All519 initial states stayed fixed except three allowed
training buffers; all28 link optimizer states have3340 updates. This result
does not support resource-allocation gain claims.

Diagnostic: keep exactly the same checkpoint, encoder/decoder, pooling,
normalization, token count, uniform power and complete sensor-only detector.
Change only physical channel fromAWGN toidentity (received symbols equal
transmitted symbols, no noise). Public snr_db stays10 for manifest continuity
but has no effect. Use seed17, batch1/four workers, all372 IDs, zero training
updates. No checkpoint/epoch selection or main-validation access.

Prediction/GT/native AP audit and all372 communication records are required.
Use the explicitly labelled --identity-codec-diagnostic audit; never disable
the codec or label the result an AWGN/high-SNR measurement. Physical symbols
and energy are hypothetical counted codec slots in this diagnostic.

Compare with F3's first strict10dB result and F2's first strict uncompressed
diagnosis (42.31946% Moderate). If identity also fails, noise removal alone
is insufficient; this does not identify pooling, normalization, decoder
capacity or optimization as a unique cause. If identity succeeds, investigate
noise robustness under matched training before any allocation experiment.
