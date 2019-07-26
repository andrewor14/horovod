#!/bin/bash

export ENVIRONMENT="$(hostname | awk -F '[.-]' '{print $1}' | sed 's/[0-9]//g')"

# Common flags
export HOROVOD_WITH_TENSORFLOW=1
export HOROVOD_WITHOUT_PYTORCH=1
export HOROVOD_WITHOUT_MXNET=1

# Environment-dependent flags and libraries
if [[ "$ENVIRONMENT" == "tigergpu" ]]; then
  export PYTHON_BIN_PATH="/usr/licensed/anaconda3/5.2.0/bin/python"
  export PIP_COMMAND="pip"
  # Note: Do NOT use Anaconda 5.3.0, which uses Python 3.7, otherwise you'll
  # run into this issue https://github.com/tensorflow/tensorflow/pull/21202 
  module load anaconda3/5.2.0
elif [[ "$ENVIRONMENT" == "visiongpu" ]]; then
  export PYTHON_BIN_PATH="/usr/bin/python3"
  export PIP_COMMAND="pip"
elif [[ "$ENVIRONMENT" == "ns" ]]; then
  export PYTHON_BIN_PATH="/usr/licensed/anaconda3/5.2.0/bin/python3.6"
  export PIP_COMMAND="/usr/licensed/anaconda3/5.2.0/bin/pip"
elif [[ "$ENVIRONMENT" == "snsgpu" ]]; then
  export PYTHON_BIN_PATH="/usr/bin/python3"
  export PIP_COMMAND="pip3"
elif [[ -n "$IN_DOCKER_CONTAINER" ]]; then
  export PYTHON_BIN_PATH="/usr/bin/python3"
  export PIP_COMMAND="pip3"
else
  echo "ERROR: Unknown environment '$ENVIRONMENT'"
  exit 1
fi

