#!/bin/bash

# Author: Salvatore Mandra (salvatore.mandra@nasa.gov)
#
# Copyright © 2021, United States Government, as represented by the Administrator
# of the National Aeronautics and Space Administration. All rights reserved.
#
# The HybridQ: A Hybrid Simulator for Quantum Circuits platform is licensed under
# the Apache License, Version 2.0 (the "License"); you may not use this file
# except in compliance with the License. You may obtain a copy of the License at
# http://www.apache.org/licenses/LICENSE-2.0.
#
# Unless required by applicable law or agreed to in writing, software distributed
# under the License is distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR
# CONDITIONS OF ANY KIND, either express or implied. See the License for the
# specific language governing permissions and limitations under the License.

# Check yapf version
YAPF_VERSION=0.31.0
if [[ $(yapf --version | awk '{print$2}') != $YAPF_VERSION ]]; then
  echo "'yapf==${YAPF_VERSION}' is needed."
  exit 1
fi

# Check clang-format version
CLANG_VERSION=13.0.0
if [[ $(clang-format --version | awk '{print$3}') != $CLANG_VERSION ]]; then
  echo "'clang-format==${CLANG_VERSION}' is needed."
  exit 1
fi

# Check Python files
PY_FAILED=""
for file in $(git ls-files $(git rev-parse --show-toplevel) | \grep '\.py$'); do
  echo -ne "[......] $file" >&2
  if [[ $(yapf --style=google -d $file | wc -l) -ne 0 ]]; then
    PY_FAILED="$file $PY_FAILED"
    echo -e "\r[FAILED]" >&2
  else
    echo -e "\r[PASSED]" >&2
  fi
done

# Check C++ files
CPP_FAILED=""
for file in $(git ls-files $(git rev-parse --show-toplevel) | \grep -E '\.cpp$|\.h$'); do
  echo -ne "[......] $file" >&2
  if [[ $(clang-format --style=google --output-replacements-xml $file | wc -l) -gt 3 ]]; then
    CPP_FAILED="$file $CPP_FAILED"
    echo -e "\r[FAILED]" >&2
  else
    echo -e "\r[PASSED]" >&2
  fi
done

if [[ $PY_FAILED != "" ]]; then
  echo -e "\nFormat of some python files must be fixed. Please use:" >&2
  echo -e "\n\t'yapf --style=google -i ${PY_FAILED::-1}'" >&2
  echo -e '\n' >&2
fi

if [[ $CPP_FAILED != "" ]]; then
  echo -e "\nFormat of some C++ files must be fixed. Please use:" >&2
  echo -e "\n\t'clang-format --style=google -i ${CPP_FAILED::-1}'" >&2
  echo -e '\n' >&2
fi

if [[ $CPP_FAILED != "" || $PY_FAILED != "" ]]; then
  exit 1
fi
