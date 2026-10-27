#!/bin/bash
# 클라우드 세션 시작 시 의존성을 설치해 Claude가 바로 코드를 실행·검증할 수 있게 한다.
set -euo pipefail

# 로컬 환경에서는 사용자가 직접 관리하므로 건너뛴다.
if [ "${CLAUDE_CODE_REMOTE:-}" != "true" ]; then
  exit 0
fi

cd "$CLAUDE_PROJECT_DIR"
uv sync
