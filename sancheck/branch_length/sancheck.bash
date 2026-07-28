#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")"
"${CXX:-g++}" -std=c++20 -O2 branch_length_test.cpp -o branch_length_test
./branch_length_test
rm -f branch_length_test
