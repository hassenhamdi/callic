#!/bin/bash
set -euo pipefail
# Correctness backpressure: masks, parity, round-trip, param budgets, no hard-coded bpsp.
cd "$(dirname "$0")/.."
/usr/bin/python3 -m pytest tests/ -q 2>&1 | tail -n 40
# Forbid hard-coded paper numbers as fake results
if grep -rn "2\.54\|1\.74\|2\.74\|2\.46\|2\.30" callic/ tools/ 2>/dev/null | grep -v -i "target\|paper\|comment\|#"; then
  echo "CHECKS_FAIL hard-coded bpsp suspected"
  exit 1
fi
echo "CHECKS_OK"
