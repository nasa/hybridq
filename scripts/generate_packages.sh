#!/bin/bash

DOCKER=${DOCKER:-docker}
COMPOSER=${COMPOSER:-docker-compose}

# Create and copy packages
true && \
  ${COMPOSER} -f packages/docker-compose.yml build
  ${DOCKER} run --rm -v $(pwd)/packages:/packages hybridq-manylinux2010_x86_64 bash -c 'cp /hybridq*.whl /packages' && \
  ${DOCKER} run --rm -v $(pwd)/packages:/packages hybridq-manylinux2014_x86_64 bash -c 'cp /hybridq*.whl /packages'
