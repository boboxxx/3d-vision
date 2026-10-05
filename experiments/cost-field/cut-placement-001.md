# Receiver appearance cut placement

Before implementation: the original backbone computes sem_neck and rpn_feature
before its last stereo hourglass. A cut inserted later would leave clean
appearance in those output fields. The isolated new forward must therefore
move the unchanged appearance-output block after the cost/appearance channel
and before depth prediction. The identity cut must verify the original math.
No author source file is edited. Receiver depth/sem/rpn/voxel fields must all
derive from corrupted received tensors. Original images remain in the sensor
batch only as the common public size/preprocessing metadata consumed downstream;
check actual downstream modules do not extract clean image content. This is
required placement for protocol001, not an exception to its no-bypass rule.
