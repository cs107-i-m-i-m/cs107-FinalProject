#!/usr/bin/env bash

pip install coverage

# Run test files
coverage run -m pytest tests/

# Run the code coverage
coverage html
