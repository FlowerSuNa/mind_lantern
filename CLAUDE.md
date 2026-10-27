# mind_lantern

법륜스님 [즉문즉설] 유튜브 자막을 근거로 고민 상담 답변을 생성하는 RAG 챗봇. Hugging Face Spaces(Gradio)에 배포된다. 비공식·비상업 연구용 프로젝트다.

## 작업 원칙

### 확인 팝업 (필수)

애매하거나 사용자 확인이 필요한 부분은 추측으로 진행하지 않는다. `AskUserQuestion` 팝업으로 확인받은 뒤 이어서 작업한다.

- 팝업을 띄우는 경우:
  - 요구사항 해석이 둘 이상으로 갈릴 때
  - 방향성·범위가 바뀔 때(기능 추가·제거, 의존성 교체, 데이터·인덱스 변경, 모델 교체)
  - 되돌리기 어렵거나 외부로 나가는 작업일 때(배포, 데이터 삭제, 권한·설정 변경)
  - 작업 중 버그나 개선점을 발견했는데 요청 범위 밖일 때
- 팝업에 반드시 담을 내용:
  1. 확인할 부분: 무엇을 결정해야 하는지 한 문장으로
  2. 근거: 현재 상태, 확인한 사실, 수치. 사용자가 대화를 거슬러 읽지 않아도 판단할 수 있게
  3. 제안 또는 받고 싶은 피드백: 선택지마다 결과와 트레이드오프. 권장안은 첫 번째에 두고 `(Recommended)`를 표시
- 관련된 질문은 한 팝업(최대 4개)에 묶는다. 답이 기존 규칙으로 정해지는 사안은 묻지 않는다.
- 답변이 선택지 밖이면 그 내용을 그대로 따르고, 다시 애매하면 다시 묻는다.

### 스킬·서브에이전트 생성 규율

새 스킬이나 서브에이전트는 아래를 모두 만족할 때만 만든다. 만들기 전에 팝업으로 승인받는다.

- 같은 절차가 반복될 것이 확실하고(3회 이상 예상), 매번 다시 파악하는 비용이 크다
- 기존 스킬·이 문서·`.claude/rules/`로 해결되지 않는다
- 입력·출력·완료 조건을 한 문단으로 정의할 수 있다

서브에이전트 일회성 호출(만들지 않고 쓰기)은 서로 독립적인 조사를 병렬로 돌리거나, 큰 출력을 메인 맥락에서 격리할 때만 쓴다. 결과는 메인에서 검증한 뒤 반영한다.

## 명령어

```bash
uv sync                          # 의존성 설치 (Python 3.12 고정)
uv sync --group chroma           # 보관용 chroma_db/ 조회가 필요할 때만
uv run python app.py             # Gradio 앱 실행 (http://127.0.0.1:7860)
uv run ruff check . && uv run ruff format --check .   # 린트·포맷 확인
uv run python data-collection.py <playlist_id>        # 자막 수집 → Google Drive 업로드
uv run python data-preprocessing.py <playlist_id>     # 자막 요약 가공
```

- `data-*.py`는 import 시점에 `GoogleAuth().LocalWebserverAuth()`로 브라우저 인증을 띄운다. 클라우드 세션에서는 실행하지 말고 `py_compile`로만 확인한다.

## 구조

데이터 파이프라인은 번호 순서대로 진행된다. 노트북에서 실험하고, 굳은 로직을 `.py`로 옮긴다.

| 단계 | 파일 | 역할 |
|---|---|---|
| 0 수집 | `data-collection.py` | yt-dlp + youtube-transcript-api로 플레이리스트 자막 수집 |
| 0 가공 | `data-preprocessing.py`, `prompts.py` | Gemini로 자막을 `질문자:/스님:` 대화체로 요약 |
| 1 인덱싱 | `1-document-indexing-pinecone.ipynb` | OpenAI 임베딩으로 Pinecone 인덱스 구축 |
| 앱 | `app.py`, `prompts.py`, `style.css` | Gradio UI, Pinecone MMR 검색 → Gemini 스트리밍 답변 |

- 벡터 스토어는 Pinecone 인덱스 `mind-lantern`(237건, 1536차원)이다. 텍스트는 메타데이터 `text` 키에 있다(langchain-pinecone 기본값).
- `chroma_db/`는 Pinecone 이전 전의 보관본이다(chromadb 1.0 스키마). 앱은 쓰지 않으며 수정하지 않는다. chromadb 1.5로 열면 파일이 마이그레이션되므로 복사본으로 연다.
- 노트북(`*.ipynb`)은 LangChain 0.3 시절 실험 기록이다. 1.x에서 `langchain.chains`·`langchain.vectorstores`가 사라져 1·2번 노트북은 그대로 실행되지 않는다. 의도적으로 수정하지 않았다.
- 모델: 임베딩 `text-embedding-3-small`(1536차원), 답변 `CHAT_MODEL`, 요약 `SUMMARY_MODEL`. 임베딩 모델·차원을 바꾸면 인덱스 전체를 다시 만들어야 한다.
- UI 프레임워크 비교와 디자인 개편 방향은 `docs/ui-framework-review.md`에 있다.

## 환경 변수

목록과 기본값은 `.env.example`에 있다. `.env`와 pydrive 인증 파일(`client_secrets.json`, `credentials.json`, `settings.yaml`)은 커밋하지 않는다.

- 클라우드 세션에서는 `.env` 대신 환경 설정(세션 제목 표시줄의 클라우드 환경 메뉴 → Edit)에 등록한다.
  - 환경 변수: `OPENAI_API_KEY`, `GOOGLE_API_KEY`, `PINECONE_API_KEY`
  - Network access 허용 도메인: `api.openai.com`, `*.pinecone.io` (Gemini는 기본 허용)
- 세 키가 없거나 호스트가 막혀 있으면 앱은 뜨지만 질문에 `⚠︎ 지금은 답변을 만들 수 없어요` 안내만 나온다.

## 코드 규칙

- 어체·아이콘·주석 원칙은 `.claude/rules/tone.md`를 따른다(매 세션 자동 로드).
- 프롬프트는 `prompts.py`에 상수로 모은다. 문구는 답변 품질에 직결되므로 공백 하나도 임의로 바꾸지 않는다.
- `.py` 파일을 수정하면 PostToolUse 훅이 ruff 포맷과 import 정렬을 자동 적용한다(`.claude/hooks/format-python.sh`).
- 노트북(`.ipynb`)은 ruff 대상에서 제외하고 실행 결과도 그대로 둔다.
- README·앱의 비공식·비상업 안내 문구는 유지한다.

## 검증

테스트 스위트는 없다. 변경 후 다음을 확인한다.

1. `uv run ruff check . && uv run ruff format --check .`
2. `uv run python -m py_compile *.py`
3. 앱 흐름: 가짜 임베딩(`DeterministicFakeEmbedding(size=1536)`)·`InMemoryVectorStore`·가짜 채팅 모델로 `get_retriever`/`get_llm`을 바꿔 끼운 뒤 `get_response`를 끝까지 돌린다. UI 변경이면 같은 구성으로 앱을 띄워 Playwright로 질문·스트리밍·출처 링크·입력창 복구를 확인한다.
4. 키와 네트워크가 있으면 실제 Pinecone 검색과 Gemini 답변을 한 번 호출해 본다.

## 커밋

- 커밋은 반드시 `/git-commit` 스킬로 한다. Conventional Commits 형식, 타입 접두어는 영어, 제목과 본문은 한국어. 예: `feat(app): 대화 기록 초기화 버튼 추가`
- 변경을 논리 단위로 나눠 커밋한다. 대량 포맷 변경과 로직 변경은 분리한다.
- 데이터 파일(`*.json`, `*.txt`, `*.vtt`)과 `.env`는 커밋하지 않는다.

## 배포 (보류 중)

- HF Space는 이 저장소와 따로 관리되고 있다. Space 쪽 `requirements.txt`와 README YAML(`sdk_version`)이 이 저장소에 없다.
- 이 저장소의 앱은 Gradio 6.30을 쓴다. Space에 반영하려면 `sdk_version`, 의존성, `PINECONE_API_KEY` 시크릿을 함께 바꿔야 한다.
- HF는 새 Gradio·Docker Space 생성에 유료 플랜을 요구한다(2026-10 기준, `docs/ui-framework-review.md`). 기존 Space를 지우고 다시 만들지 않는다.
- 같은 저장소로 통합할지는 사용자가 검토 중이다. 결정 전에는 배포 파일을 추가하지 않는다.

## 알려진 이슈 (아직 수정 안 함)

- `data-preprocessing.py` 마지막의 `upload_to_drive(...)` 호출이 `utils.upload_to_drive(drive, google_drive_id, filename, data)` 시그니처와 맞지 않는다(`DRIVE` 누락, 인자 순서 틀림). 요약 루프도 아직 없다.
- `data-collection.py`가 자막 조회(`YouTubeTranscriptApi().fetch`)를 `try` 밖에서 해서, 자막 없는 영상 하나에 전체 수집이 멈춘다.
- langchain-pinecone(0.2.13, 2025-11 이후 릴리스 없음)이 pinecone SDK를 7.x로 묶어 둔다(최신 10.x).
- 예제 질문 버튼 높이가 100px로 고정돼 긴 문구가 잘린다(`style.css` `.same-height`). 디자인 개편 때 함께 다룬다.
