# Artemis complete KITTI deployment

Original native GPU model probe11423908 passed and terminated with discarded
weights. Next prepare complete data for both main3712/3769 experiments and
future detector integration. This is data deployment only, not model selection.

Use unchanged project scripts/download_kitti.py and prepare_kitti_splits.py;
their exact SHA is recorded before/after. Download the same five publicly
readable official S3 archives already verified on sheng: left/right images,
labels,calibration,velodyne. Verify entire archive bytes/SHA256 against the
existing data/kitti-download-final.json; extract only training data with ZIP
CRC/size checks and require7481files for each component. Verify member-identity
SHA against the sealed sheng manifest. Standard split train3712,val3769 must
have original pinned SHA b6417a1d9b18c8fdb085128e633d28ff321b7674a6d1b3841b8f43d865b281cb
and657ac4bcc1e156e5b106a4ca18e1f88e012787ea1d2b5d0adeea97fee903fa86.

New destination /mnt/nfs2/engdes/wc296/paper6/data/kitti, independent of the
small assets fixture and other projects. Require>=100GiB physical free before
download, as the existing downloader does. Source archives retained on shared
project storage; no local large-data relay or uncharged GPU reserved.
CPU-only short2h job,8CPU/16GiB,2archive workers and4range connections per
archive. Exact ETag/range/prefix protection and bounded retries remain unchanged.
No alteration of user shell config/shared environments; no overwriting a
previously different dataset/split. Preserve partial/failure evidence and use
new evidence IDs for any later recovery. A passed deployment manifest means
available verified data, not main validation AP, detector-operator portability
or completed scientific experiment. No formal training launched automatically.
