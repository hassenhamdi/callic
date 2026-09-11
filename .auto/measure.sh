#!/bin/bash
set -euo pipefail
# Fast pre-check + honest smoke bpsp. Keeps every iteration <30s locally; full runs live in Colab.
# Outputs METRIC name=value lines parsed by run_experiment.
cd "$(dirname "$0")/.."
export PYTHONPATH=.:$PYTHONPATH
/usr/bin/python3 -m py_compile callic/*.py tools/*.py tests/*.py 2>&1 | head -n 20
if [ -f tools/eval_smoke.py ]; then
  /usr/bin/python3 tools/eval_smoke.py
else
  echo "SMOKE missing_implementation bpsp=inf"
  echo "METRIC kodak_bpsp=99.0"
  echo "METRIC params=0"
  echo "METRIC mergeable_params=0"
  echo "METRIC enc_time_s=0"
fi
