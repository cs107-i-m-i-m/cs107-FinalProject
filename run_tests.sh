#!/usr/bin/env bash

coverage run -m pytest tests/test_forward_mode.py
coverage report -m

coverage run -m pytest tests/test_reverse_mode.py
# run the code coverage
coverage report -m
