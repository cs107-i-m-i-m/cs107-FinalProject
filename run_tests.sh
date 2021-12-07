#!/usr/bin/env bash

coverage run -m pytest tests/test_forward_mode.py
# run the code coverage
coverage report -m
