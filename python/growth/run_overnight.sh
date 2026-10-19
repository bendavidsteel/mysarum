#!/bin/bash
# QD illumination on both GPUs, then CLIP-guided imitation of TARGETS split across them.
#   [QD_HOURS=10.5 GENS=20 POP=24 REPS=3 MAX_VERTS=40000] ./run_overnight.sh [param|nca]
#   SKIP_QD=1 RUN=outputs/qd_morph OUT=outputs/imitate_morph TARGETS="a staghorn coral;a sea fan" ./run_overnight.sh nca
set -x
cd "$(dirname "$0")"
export HF_HUB_OFFLINE=1
PY=.venv/bin/python
GENOME=${1:-param}
QD_HOURS=${QD_HOURS:-10.5}
RUN=${RUN:-outputs/qd_${GENOME}}
OUT=${OUT:-outputs/imitate_${GENOME}}
IM="--qd $RUN --gens ${GENS:-20} --popsize ${POP:-24} --reps ${REPS:-3} --genome $GENOME --max_verts ${MAX_VERTS:-40000}"
if [ -n "$TARGETS" ]; then
  IFS=';' read -ra TARGETS <<< "$TARGETS"
else
  TARGETS=(
    "a brain coral" "a staghorn coral" "a lettuce coral" "a sea fan" "a table coral" "a mushroom coral"
    "an oyster mushroom" "a morel mushroom" "a chanterelle mushroom" "a turkey tail fungus" "a coral fungus"
    "a nudibranch sea slug" "a lettuce"
  )
fi
[ -z "$SKIP_QD" ] && $PY qd.py out=$RUN genome=$GENOME max_hours=$QD_HOURS > $RUN.log 2>&1
mkdir -p $OUT
for gpu in 0 1; do
  (for ((i = gpu; i < ${#TARGETS[@]}; i += 2)); do
     t=${TARGETS[$i]}
     CUDA_VISIBLE_DEVICES=$gpu $PY imitate.py "$t" $IM --out "$OUT/$(echo ${t#a* } | tr ' ' _)"
   done) > $OUT.gpu$gpu.log 2>&1 &
done
wait
