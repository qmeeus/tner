#!/bin/bash

export HOME=/users/spraak/qmeeus
. $HOME/.profile
export MAMBA_EXE='/users/spraak/qmeeus/.local/bin/micromamba';
export MAMBA_ROOT_PREFIX='/users/spraak/qmeeus/micromamba';
__mamba_setup="$("$MAMBA_EXE" shell hook --shell bash --root-prefix "$MAMBA_ROOT_PREFIX" 2> /dev/null)"
if [ $? -eq 0 ]; then
    eval "$__mamba_setup"
else
    alias micromamba="$MAMBA_EXE"  # Fallback on help from mamba activate
fi
unset __mamba_setup

micromamba activate torch-cu121
export PYTHONPATH=${PWD}${PYTHONPATH:+:$PYTHONPATH}
