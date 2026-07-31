#!/usr/bin/env bash
# Install the repo's git hooks into .git/hooks. Run once per clone.
#   sh scripts/install_hooks.sh
set -euo pipefail
repo_root="$(git rev-parse --show-toplevel)"
cp "$repo_root/githooks/commit-msg" "$repo_root/.git/hooks/commit-msg"
chmod +x "$repo_root/.git/hooks/commit-msg"
echo "Installed commit-msg hook -> .git/hooks/commit-msg"
