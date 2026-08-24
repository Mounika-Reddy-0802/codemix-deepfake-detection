#!/usr/bin/env bash
# Install the repo's git hooks. Run once per clone.
#   sh scripts/install_hooks.sh
#
# Points core.hooksPath at the tracked githooks/ directory rather than copying
# into .git/hooks, so the hooks cannot silently fall out of date — and a fresh
# clone is one command away from being protected.
set -euo pipefail
repo_root="$(git rev-parse --show-toplevel)"
chmod +x "$repo_root/githooks/commit-msg" 2>/dev/null || true
git -C "$repo_root" config core.hooksPath githooks
echo "Installed hooks -> core.hooksPath=githooks"
echo "Verify with: git -C '$repo_root' config core.hooksPath"
