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

DOCKER=${DOCKER:-docker}
COMPOSER=${COMPOSER:-${DOCKER}-compose}

# Build using the following Python's versions
PYTHON_VERSIONS=${PYTHON_VERSIONS:-"cp38-cp38 cp39-cp39 cp310-cp310"}

for VERSION in $PYTHON_VERSIONS; do
  echo "# Building ($VERSION)" >&2

  # Build baseline
  $COMPOSER build --build-arg PYTHON_VERSION=$VERSION hybridq-baseline

  # Re-tag baseline
  $DOCKER tag hybridq-baseline:latest hybridq-baseline:$VERSION

  # Build HybridQ
  $COMPOSER build --build-arg BASELINE=hybridq-baseline:$VERSION hybridq

  # Get HybridQ version
  HVERSION=$($DOCKER run -ti --rm hybridq:latest bash -c "pip list frozen | grep hybridq | awk '{print\$2}'" | sed \$'s/[^[:print:]\t]//g')

  # Re-tag HybridQ
  $DOCKER tag hybridq:latest hybridq:${HVERSION}-${VERSION}

  # Remove latest
  $DOCKER rmi hybridq-baseline:latest
  $DOCKER rmi hybridq:latest

done
