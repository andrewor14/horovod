#!/bin/bash

source build_flags.sh

"$PYTHON_BIN_PATH" setup.py bdist_wheel
"$PIP_COMMAND" uninstall -y horovod
"$PIP_COMMAND" install --user --no-cache-dir dist/horovod*.whl

