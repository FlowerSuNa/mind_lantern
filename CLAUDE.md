# mind_lantern

법륜스님 [즉문즉설] 유튜브 자막을 근거로 고민 상담 답변을 생성하는 RAG 챗봇. Hugging Face Spaces(Gradio)에 배포된다. 비공식·비상업 연구용 프로젝트다.

## 명령어

```bash
uv sync                          # 의존성 설치 (Python 3.12 고정)
uv run python app.py             # Gradio 앱 실행
uv run ruff check .              # 린트
uv run ruff format .             # 포맷
uv run python data-collection.py <playlist_id>     # 자막 수집 → Google Drive 업로드
uv run python data-preprocessing.py <playlist_id>  # 자막 요약 가공
```

- 테스트 스위트는 아직 없다. 변경 후에는 최소한 `uv run ruff check <파일>`과 `uv run python -c "import <모듈>"`로 확인한다.
- `data-*.py`는 실행 시 `GoogleAuth().LocalWebserverAuth()`로 브라우저 인증을 띄운다. 클라우드 세션에서는 실행하지 말 것.

## 구조

데이터 파이프라인은 번호 순서대로 진행된다. 노트북에서 실험하고, 굳은 로직을 `.py`로 옮기는 방식이다.

| 단계 | 파일 | 역할 |
|---|---|---|
| 0 수집 | `0-data-collection*.ipynb`, `data-collection.py` | yt-dlp + youtube-transcript-api로 플레이리스트 자막 수집 |
| 0 가공 | `data-preprocessing.py`, `prompts.py` | Gemini로 자막을 `질문자:/스님:` 대화체로 요약 |
| 1 인덱싱 | `1-document-indexing.ipynb`(Chroma), `1-document-indexing-pinecone.ipynb` | OpenAI 임베딩으로 벡터 스토어 구축 |
| 2 RAG | `2-basic.ipynb`, `3-retriever-test.ipynb` | 검색·프롬프트 실험 |
| 앱 | `app.py`, `style.css` | Gradio UI, MMR 검색 → Gemini 스트리밍 답변 |

- 중간 데이터(JSON)는 Google Drive 폴더에 저장한다. 폴더 ID는 `.env`의 `GOOGLE_DRIVE_VTT_ID`, `GOOGLE_DRIVE_TXT_ID`, `GOOGLE_DRIVE_RT_ID`.
- `chroma_db/`는 배포용 벡터 스토어로 git에 커밋되어 있다(컬렉션 `content-250623`). 직접 수정하지 말고 인덱싱 노트북으로 다시 만든다.
- 모델: 임베딩 `text-embedding-3-small`(1536차원), 생성 `gemini-2.5-flash`, 요약 `gemini-2.5-pro`. 임베딩 모델·차원을 바꾸면 `chroma_db` 전체를 재인덱싱해야 한다.

## 환경 변수 (`.env`, 커밋 금지)

`OPENAI_API_KEY`, `GOOGLE_API_KEY`, `GOOGLE_DRIVE_*_ID`, (Pinecone 사용 시) `PINECONE_API_KEY`. pydrive 인증 파일(`client_secrets.json`, `credentials.json`, `settings.yaml`)도 커밋하지 않는다.

## 코드 규칙

- 주석, docstring, 사용자에게 보이는 문구, 프롬프트는 한국어로 쓴다. 기존 코드처럼 Google 스타일 docstring(`Args:`, `Returns:`)과 타입 힌트를 쓴다.
- 프롬프트는 `prompts.py`에 상수로 모은다.
- `.py` 파일을 수정하면 PostToolUse 훅이 ruff 포맷과 import 정렬을 자동 적용한다(`.claude/hooks/format-python.sh`). 기존 파일은 아직 ruff로 정리되지 않아 처음 수정할 때 포맷 diff가 함께 생길 수 있다.
- 노트북(`.ipynb`)은 ruff 대상에서 제외한다. 노트북의 실행 결과는 그대로 둔다.
- 답변이 실제 법륜스님의 발언처럼 보이지 않도록 하는 README·앱의 안내 문구(비공식·비상업)는 유지한다.

## 커밋

- 커밋은 반드시 `/git-commit` 스킬로 한다. Conventional Commits 형식, 타입 접두어는 영어, 제목과 본문은 한국어. 예: `feat(app): 대화 기록 초기화 버튼 추가`
- 변경을 논리 단위로 나눠 커밋한다. 데이터 파일(`*.json`, `*.txt`, `*.vtt`)과 `.env`는 커밋하지 않는다.

## 알려진 이슈 (아직 수정 안 함)

- `data-preprocessing.py` 마지막의 `upload_to_drive(...)` 호출이 `utils.upload_to_drive(drive, google_drive_id, filename, data)` 시그니처와 맞지 않는다(`DRIVE` 누락, 인자 순서 틀림). 요약 루프도 아직 없다.
- `app.py`의 안내 `gr.Textbox`에 `dedent(...).strip`이 호출 없이 넘어간다(`.strip()`이어야 함).
- `app.py`는 `chroma_db/`가 없으면 `chroma_db.zip`을 풀지만 zip 파일은 저장소에 없다.
- `ruff check .` 기준으로 미사용 import(F401)와 import 정렬(I001) 경고가 남아 있다.
