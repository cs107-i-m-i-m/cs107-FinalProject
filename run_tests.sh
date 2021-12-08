#!/usr/bin/env bash

# Run test files
coverage run -m pytest tests/test_forward_mode.py test_reverse_mode.py

# Run the code coverage on both
coverage report -m
