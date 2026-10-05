# Framing engineering repair 001

2026-10-04, after local attempt001, before any repaired or Artemis observations.
Attempt001's68 receiver assertions finished, but final artifact serialization
failed because a relative --output path was compared with absolute ROOT.
Its manifest remains statefailed and its emitted codec/decoded fixtures remain
unchanged. It is not passed evidence. Fix only Path(output).resolve() in the
CPU checker; receiver/code/channel/protocol are unchanged. Repeat locally under
unique002 output. Artemis has not started; its unique001 paths remain unused.
Require the same68 checks and source/runtime identities; preserve all failures.
