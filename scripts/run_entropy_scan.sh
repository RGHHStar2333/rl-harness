#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."

echo "=============================="
echo "RL Harness Entropy Scan"
echo "Time: $(date)"
echo "Project: $(pwd)"
echo "=============================="

python3 scripts/entropy_scan.py --config configs/pipeline.yaml

echo ""
echo "Report preview:"
if [ -f reports/entropy_report.json ]; then
  python3 -m json.tool reports/entropy_report.json | head -n 80
else
  echo "reports/entropy_report.json not found"
fi

if [ -f scripts/git_auto_commit.py ]; then
  python3 scripts/git_auto_commit.py --level ENTROPY --reason "scheduled entropy scan" || true
fi

echo ""
echo "✅ Entropy scan finished at $(date)"
