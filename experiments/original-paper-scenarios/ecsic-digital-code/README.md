# ECSIC counted digital framing

This directory adds P6SB v2/codec3 around the complete fixed P6EC container.
It imports unchanged sealed entropy parsing and unchanged historical Sionna PHY.
No neural model, source image, GT or source-side length is available to the
receiver. Missing or changed dependencies are fatal; malformed received frames
are erased. Both inner166-byte and outer20-byte overheads remain in source bits.

Local engineering only:

```sh
python experiments/original-paper-scenarios/ecsic-digital-code/check_local.py \
  --output data/engineering/ecsic-digital-local-CPU-001.json
```

The independent byte parser in check_local.py validates the two actual source
containers and malformed received-block fixtures without calling the production
P6EC parser. This is framing evidence, not new entropy/neural/physical evidence.

Prepared Artemis job: engineering-001.sbatch, CPU4/16GiB/1h. It is intentionally
not submitted by local checks. Before an authorized launch, transfer this whole
directory, its protocol, unchanged historical dependency files, complete local
gate+fixtures, the pinned stageB manifest/audit/two source files, and the existing
runtime manifest. Check destination SHA and source identity before submitting.
The runner additionally checks every installed PHY Python/CSV and pip inventory.
It writes exactly20 packet attempts, preserving partial evidence on errors and
refusing existing paths. It never resumes/retries packets automatically.

The runner's first transceiver return is discarded. Only decoded information
blocks/public k enter receive_container. Received P6EC files are intended for
later reconstruction on the same sheng CPU runtime as stageB. These outputs
do not by themselves prove neural RGB equivalence or close the physical audit.
Independent raw PHY/framing replay and actual terminal Slurm evidence are still
required; any subsequent neural reconstruction must use received files only.

No source-only identity comparison against the current Cityscapes lambda0.01
model should be called a nominal30x KITTI baseline. Main rate selection, crop/
dtype contract and detector evaluation require separate locked execution work.
