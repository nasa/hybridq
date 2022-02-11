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

function get_files() {
  git ls-files \
    $(git rev-parse --show-toplevel) -- \
      ':!:*.ipynb' \
      ':!:docs/' \
      ':!:.github/' \
      ':!:packages/'
}

function check_cpp_files() {
  cat < /dev/stdin | xargs -P8 -l bash -c '\
    TYPE=$(file $0 | grep -i c++ | wc -l)
    if [[ $TYPE != 0 ]]; then
      STATUS=$(clang-format --style=google --output-replacements-xml $0 | wc -l)
      if [[ $STATUS -gt 3 ]]; then
        echo $(tput bold setaf 1)[FAILED]$(tput sgr 0) $0 >&2
        echo $0
      else
        echo $(tput bold setaf 2)[PASSED]$(tput sgr 0) $0 >&2
      fi
    fi' | sort
}

function check_py_files() {
  cat < /dev/stdin | xargs -P8 -l bash -c '\
    TYPE=$(file $0 | grep -i python | wc -l)
    if [[ $TYPE != 0 ]]; then
      STATUS=$(yapf -d --style=google $0 2>&1 >/dev/null)
      ERR=$?
      if [[ $ERR != 0 ]]; then
        if [[ x$STATUS == "x" ]]; then
          echo $(tput bold setaf 1)[FAILED]$(tput sgr 0) $0 >&2
          echo $0
        fi
      else
        echo $(tput bold setaf 2)[PASSED]$(tput sgr 0) $0 >&2
      fi
    fi' | sort
}

# Check yapf version
YAPF_VERSION=0.32.0
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

# If no filenames are provided, collect filenames from git tree
if [[ $# == 0 ]]; then
  CPP_FAILED=$(get_files | check_cpp_files | tr '\n' ' ')
  PY_FAILED=$(get_files | check_py_files | tr '\n' ' ')

# Otherwise, check provided filenames
else
  CPP_FAILED=$(for var in "$@"; do echo $var; done | check_cpp_files | tr '\n' ' ')
  PY_FAILED=$(for var in "$@"; do echo $var; done | check_py_files | tr '\n' ' ')
fi

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
