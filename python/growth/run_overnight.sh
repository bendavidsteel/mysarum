#!/bin/bash
# ~12 h: QD illumination (10.5 h on both GPUs), then CLIP-guided imitation of a few organisms.
set -x
cd "$(dirname "$0")"
export HF_HUB_OFFLINE=1
PY=.venv/bin/python
RUN=outputs/qd_overnight
$PY qd.py out=$RUN max_hours=10.5 > $RUN.log 2>&1
(CUDA_VISIBLE_DEVICES=0 $PY imitate.py "a brain coral" --qd $RUN --gens 20 && \
 CUDA_VISIBLE_DEVICES=0 $PY imitate.py "an oyster mushroom" --qd $RUN --gens 20) > outputs/imitate_gpu0.log 2>&1 &
(CUDA_VISIBLE_DEVICES=1 $PY imitate.py "a nudibranch sea slug" --qd $RUN --gens 20 && \
 CUDA_VISIBLE_DEVICES=1 $PY imitate.py "a lettuce" --qd $RUN --gens 20) > outputs/imitate_gpu1.log 2>&1 &
wait
