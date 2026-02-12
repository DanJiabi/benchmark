#!/bin/bash

export PYTORCH_ENABLE_MPS_FALLBACK=1

python benchmark.py "$@"
