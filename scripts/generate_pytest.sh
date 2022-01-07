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

# Specify Python versions
PYTHON_VERSIONS="[cp37-cp37m, cp38-cp38, cp39-cp39]"

# Get all tests names
TEST_NAMES=$(cat tests/tests.py | grep ^'def test' | sed -e 's/def test_//g' -e 's/__.*//g' | sort -g | uniq)

# For each test, generate yml
for name in $TEST_NAMES; do
  echo "Generating test for $name." >&2
  cat .github/python-pytest.yml.__base__ | \
    sed -e "s/\[\[:TESTNAME:\]\]/${name}/g" \
        -e "s/\[\[:PYTHON_VERSIONS:\]\]/${PYTHON_VERSIONS}/g" > ".github/workflows/python-pytest_${name}.yml"
done
