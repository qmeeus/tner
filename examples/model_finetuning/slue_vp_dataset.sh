#!/bin/bash

set -euo pipefail

model_development () {
  DATA=${1}
  MODEL=${2}
  BATCH=${3}
  GRAD_1=${4}
  GRAD_2=${5}
  export MODEL_ALIAS="${MODEL##*/}"
  tner-train-search -m "${MODEL}" -c "tner_ckpt/${DATA}_${MODEL_ALIAS//-/_}" -d "${DATA}" -e 15 --epoch-partial 5 --n-max-config 3 -b "${BATCH}" -g "${GRAD_1}" "${GRAD_2}" --lr 1e-6 1e-5 --crf 0 1 --max-grad-norm 0 10 --weight-decay 0 1e-7
  tner-evaluate -m "tner_ckpt/${DATA}_${MODEL_ALIAS//-/_}/best_model" -e "tner_ckpt/${DATA}_${MODEL_ALIAS//-/_}/best_model/eval/metric.json" -d "${DATA}" -b "${BATCH}" --return-ci
  tner-evaluate -m "tner_ckpt/${DATA}_${MODEL_ALIAS//-/_}/best_model" -e "tner_ckpt/${DATA}_${MODEL_ALIAS//-/_}/best_model/eval/metric_span.json" -d "${DATA}" -b "${BATCH}" --return-ci --span-detection-mode
  tner-push-to-hub -o "tner" -m "tner_ckpt/${DATA}_${MODEL_ALIAS//-/_}/best_model" -a "${MODEL_ALIAS}-${DATA//_/-}"
}


model_development "slue-voxpopuli-dataset" "roberta-large" 64 1 2
