#!/bin/bash
cd "$(dirname "$0")"
export PYTHONPATH="$(pwd)/src:$(pwd)/external/PnLCalib:$(pwd)/external/no_bells_just_whistles"
python3 src/main.py "$@"
