# Artemis RTX PRO6000 bootstrap 001

User explicitly authorized using Artemis and requested RTX PRO6000 first.
Observed2026-10-04: alias artemis-ood, wc296, accountengdes, normalQoS;
enginf allows fs-hpc_engdes (account's actual group); artemis-rtx-03 idle,
GRES gpu:RTX:2, node featuresgpu/amd_zen5/rtx/highmem. Scheduler does not
expose exact product name. Use observed enginf + gpu:RTX:1, one task8CPU,
32GiB RAM,30min maximum to verify actual allocated hardware and environment.
Do not invent a gpu:6000pro resource type or substitute A40/A6000 silently.

Before any research training, an srun step must record its Slurm identity,
CUDA_VISIBLE_DEVICES, actual nvidia-smi products/driver/memory, and exact
allocated logical GPU. Require RTX PRO6000 product string with at least90GiB
VRAM. Other products are an explicit bootstrap failure; no training is run.
For Blackwell require a compatible separate Python/PyTorch/CUDA environment,
never copy the sm89/CUDA11.8 sheng binaries or modify unrelated conda envs.
Inspect available modules/environments read-only before creating dependencies.

Bootstrap script writes exclusive machine evidence under new dedicated
/mnt/nfs2/engdes/wc296/paper6 paths. Allocated GPU tests, if the existing runtime
supports its capability, are small finite matrix/conv forward-backward checks,
not AP or training. No model/dataset migration in this first job. It exits after
inventory/checks and releases the allocation; no idle GPU reservation or
unbounded sleep. Keep jobID, sbatch source SHA, scheduler terminal state,
stdout/stderr and machine-readable results. A finished inventory job is not a
trained research experiment. Any later resource job has a concrete workload.

No email/admin message is authorized or sent. sheng training and its frozen
native-validation queues continue unchanged. The original-paper scenario lock
and remaining method/scientific gates apply on both compute hosts.
