#!/usr/bin/env bash
set -euo pipefail
PROJECT_ROOT="${1:-/home/server/v2x_projects/OPV2V}"
mkdir -p "${PROJECT_ROOT}/scripts"
cp "$(dirname "$0")/count_test_collaborators.py" "${PROJECT_ROOT}/scripts/count_test_collaborators.py"
chmod +x "${PROJECT_ROOT}/scripts/count_test_collaborators.py"
echo "Installed to ${PROJECT_ROOT}/scripts/count_test_collaborators.py"
