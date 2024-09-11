#!/bin/bash

. scripts/env.sh

[ "$DEBUG" ] && PYTHON=ipdb || PYTHON=python

CUDA_VISIBLE_DEVICES=$(echo $CUDA_VISIBLE_DEVICES | sed "s:CUDA::g") $PYTHON $@

