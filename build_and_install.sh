#!/bin/bash

source build_flags.sh

"$PYTHON_BIN_PATH" setup.py bdist_wheel

# Note: this MUST not be done in the horovod home directory
cd dist
"$PIP_COMMAND" uninstall -y horovod
"$PIP_COMMAND" install --user --no-cache-dir horovod*.whl

