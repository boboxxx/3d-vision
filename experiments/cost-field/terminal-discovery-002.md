# Completed Slurm job aged out of targeted squeue

Closure001 stops at squeue -j11424889 returning Invalid job id specified.
Earlier actual sacct confirms COMPLETED0:0 for the job and its .0 step. This
is queue-discovery behavior after a completed job leaves live Slurm state,
not a method failure or authorization to relaunch it.

Closure002 queries the user's full live queue and requires no row for the
exact job; independently require both accounting recordsCOMPLETED0:0.
All array/state/source/physical checks remain unchanged. Keep the first helper
and failure log; no experiment rerun or source change.
