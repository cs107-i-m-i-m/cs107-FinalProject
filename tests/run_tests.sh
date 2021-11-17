#!/usr/bin/env bash

export PYTHONPATH="$(pwd -P)/../src":${PYTHONPATH}

python -m pytest test_forward_mode.py
