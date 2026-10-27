#!/bin/bash
# Claude가 .py 파일을 수정한 직후 ruff로 포맷·import 정렬을 적용한다.
set -euo pipefail

file_path=$(jq -r '.tool_input.file_path // empty')

case "$file_path" in
  *.py) ;;
  *) exit 0 ;;
esac

[ -f "$file_path" ] || exit 0

cd "$CLAUDE_PROJECT_DIR"
uv run --quiet ruff check --select I --fix --quiet "$file_path" || true
uv run --quiet ruff format --quiet "$file_path" || true
