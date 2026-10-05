# Offline compute-node build repair003

Locked before staging/retry. CPU Slurm11424736 passed all357/361 frozen input
checks and retained the original runtime, then downloaded 25 PyPI wheels and
received HTTP403 for the next archive (official torchvision CDN). It ended
FAILED1:0 without installation or compilation. Preserve every downloaded file,
report and log. This is archive access failure, not a compatibility result.

Stage the identical 48 locked official archives from the Artemis login node,
where dependency resolution already read the official CDN. Copy previously
downloaded matching archives into a new immutable archive-cache003; download
the remainder without package installation. Verify every complete SHA256 and
retain URL/size/digest manifest. No login-node compilation or GPU operation.
If any archive fails, retain the attempt rather than change versions or hashes.

Build003 uses only these local archives, independent overlay003 and actual
CPU Slurm; no build-stage network access. Archive source URL/version/hash,
MMCV source, sm120/GNU12 settings and import assertions are unchanged. No
existing runtime, previous overlay or frozen source is modified. Full GPU
detector execution and sparse teacher/training remain separate future gates.
