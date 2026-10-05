# GraftNet: related feature/receiver alignment evidence

Biyang Liu, Huimin Yu, Guodong Qi, CVPR2022. Primary sources:
[CVF paper](https://openaccess.thecvf.com/content/CVPR2022/papers/Liu_GraftNet_Towards_Domain_Generalized_Stereo_Matching_With_a_Broad-Spectrum_and_CVPR_2022_paper.pdf),
[author preprint](https://arxiv.org/abs/2204.00179),
[author code](https://github.com/SpadeLiu/Graft-PSMNet).

The method uses cosine-similarity cost volumes to connect frozen features from
a broadly pretrained image model with a stereo aggregation network. A shallow
feature adaptor recovers task-specific information, and the aggregation network
is retrained. It evaluates stereo domain generalization, not wireless 3D detection.
The primary method text directly discusses feature representation and matching
space when replacing the feature source.

Relevance to our project is limited but concrete: replacing or compressing a
feature source can change the representation expected by a frozen receiver.
Consequently, sender/receiver adaptation and cost-space construction require
controlled comparison. That is our inference from the method and our F4
negative results; it is not a reported GraftNet communication result. Generic
feature adaptation, cosine cost or frozen-image-feature grafting cannot alone
be claimed as our novelty.

No implementation adaptation, reproduced GraftNet result or complete
comparison/accounting audit has been done. This note reads the primary method
content returned by the CVF source; it does not certify every experiment or
prove our proposed method's novelty/non-overlap.
