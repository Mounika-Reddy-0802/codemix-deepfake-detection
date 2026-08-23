#!/usr/bin/env bash
# Stage-3 LoRA adaptation on the code-mixed adaptation pool. Owner M.
#
#   sh scripts/06_train_lora.sh            # full run
#   sh scripts/06_train_lora.sh --smoke    # 1% subset, verifies the loop learns
#
# The anti-leakage suite runs first: adapting on speakers that also appear in the
# eval column would make the gap-closure number meaningless, and that is exactly
# the failure this gate exists to catch.
set -euo pipefail
python -m pytest tests/test_splits.py tests/test_lora.py -q
python -m src.training.train --config configs/train_lora_codemix.yaml "$@"
