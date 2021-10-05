#!/bin/bash

git ls-files -cmo --exclude-standard -- ':!:docs/*' | \
  xargs -l bash -c 'echo $0 $(file $0 | grep C\+\+ | wc -l)' | \
  awk '$2 != 0 {print $1}' | sort | uniq
