#!/bin/bash

# Get all tests names
TEST_NAMES=$(cat tests/tests.py | egrep 'def test_[[:alpha:]]*__' | sed -e 's/__.*//g' -e 's/.*def //' | sort | uniq)

# For each test, generate yml
for name in $TEST_NAMES; do
  echo "Generating test for $name." >&2
  cat .github/python-pytest.yml.__base__ | sed "s/\[\[:TESTNAME:\]\]/${name}/g" > ".github/workflows/python-pytest_${name}.yml"
done
