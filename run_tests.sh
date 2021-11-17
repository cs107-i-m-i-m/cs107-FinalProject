#!/usr/bin/env bash

tests=(tests/test_forward_mode.py)

# decide what driver to use (depending on arguments given)
unit='-m unittest'
if [[ $# -gt 0 && ${1} == 'coverage' ]]; then
  driver="${@} ${unit}"
elif [[ $# -gt 0 && ${1} == 'pytest'* ]]; then
  driver="${@}"
else
  driver="python ${@} ${unit}"
fi

export PYTHONPATH="$(pwd -P)/":${PYTHONPATH}

# run the tests
${driver} ${tests[@]}
