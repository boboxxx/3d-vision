# Official torchvision CDN acceptance and build input lock, 003

Locked before installation/build. Resolution002 completed pip dependency
selection (47 additional wheel archives), then rejected the official
`download-r2.pytorch.org` URL because the original guard allowed only
`download.pytorch.org`. The exact selected URL occurs in the saved official
torchvision index; no package was installed. Preserve both previous attempts.

Accept only the exact torchvision0.22.1+cu128 cp312 Linux wheel URL selected
from that saved official index, with its recorded SHA256. All other archives
must be hosted on files.pythonhosted.org. Revalidate the full saved pip report
against the base freeze and reject any existing-package replacement. Save
its report/index hashes in build-inputs001 along with all exact archive
versions/URLs/SHA256, framework Python sources, existing native binaries and
operator gate records. This validation is a build-input preparation; it is
not a rerun, installation, import or compatibility result.

Execute unchanged MMCV compilation and real registry imports as specified in
protocol001, with the fire wheel repair from protocol002. The package overlay
must not alter the frozen runtime or repo. No detector forward, training or
GPU allocation in this build stage. Retain any compile/import failure.
