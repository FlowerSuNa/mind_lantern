# mind_lantern 챗봇 UI 프레임워크 비교 (2026-10-27 기준)

## 먼저 볼 것: HF Spaces 정책이 바뀌었다
- HF 공식 문서(hub-docs `main`)에 따르면 이제 **Gradio·Docker Space를 새로 만들려면 유료 플랜(PRO, Team/Enterprise)이 필요**하다. CPU Basic 하드웨어(2 vCPU, 16GB) 자체는 시간당 무료다. 무료 개인 계정은 ZeroGPU Gradio Space 2개까지만 쓸 수 있고, Static Space는 누구나 무료다.
- 정책 시점은 미확인이다. 포럼에 2026년 7월경부터 보고가 올라왔다.
- 기존 무료 CPU Basic Space가 계속 돌아가는지, SDK를 `gradio`에서 `docker`로 바꿀 수 있는지는 **미확인**이다. 이 점 때문에 **현재 Space의 `sdk: gradio`를 유지하는 쪽이 가장 안전하다.**
- Streamlit 내장 SDK는 2025-04-30에 deprecated됐다. 지금은 Docker SDK의 Streamlit 템플릿(포트 8501)으로만 만든다.
- 무료 CPU Basic Space는 48시간 동안 접속이 없으면 잠들고, 다음 방문 때 다시 시작된다. Docker 이미지가 크면(특히 Node 빌드 포함) 빌드와 기동이 더 오래 걸린다(소요 시간 미확인).
- 시크릿 처리:
  - Gradio Space: Settings에 넣은 secret이 실행 시 환경 변수로 들어온다.
  - Docker Space: 실행 시에는 똑같이 환경 변수다. 빌드 시점에 쓰려면 `RUN --mount=type=secret`이 필요하다.

## 비교표

| 프레임워크 (최신 PyPI) | HF 배포 방식 | 스트리밍 | 디자인 자유도 | 마이그레이션 비용 | 비고 |
|---|---|---|---|---|---|
| **Gradio 6.30.0** `gr.Blocks`+`gr.Chatbot` | 내장 Gradio SDK | async 제너레이터 yield, SSE 큐 | 중: 테마 12종, `css`/`js`/`head`, `gr.HTML` 템플릿 | **최저**: 메시지 포맷과 파라미터만 수정 | 현재 구조를 그대로 유지 |
| Gradio 6 `gr.ChatInterface` | 내장 Gradio SDK | 같음 | 중하: 레이아웃이 고정됨 | 중 | `additional_inputs_accordion`, `examples`, `save_history` 제공 |
| **Gradio `gr.Server`** (6.10~) + 직접 만든 프론트 | 내장 Gradio SDK | `@server.api` 제너레이터를 SSE로, JS는 `@gradio/client` | **상**: HTML/React를 자유롭게 | 중상: 프론트를 새로 작성 | FastAPI를 상속해서 `GET /`을 직접 서빙할 수 있다 |
| Streamlit 1.66.1 | Docker 템플릿만 가능 | `st.write_stream`, async는 1.64부터 지원 | 중하: `config.toml` 테마, CSS는 비공식 hack, Custom Components v2 | 중: 매번 스크립트를 다시 실행하는 모델이라 구조를 다시 짜야 함 | 유료 생성 정책 영향을 받는다 |
| Chainlit 2.12.0 | Docker | 네이티브 토큰 스트리밍, Step | 중상: `theme.json`(shadcn 변수), CSS/JS 주입, React client | 중 | **아카이브됨, 유지보수·보안 패치 없음.** 2.12.0에서 CVSS 9.8 취약점을 패치한 뒤 중단 |
| FastAPI 0.162.0 + React/Svelte | Docker | `fastapi.sse` 네이티브 지원(0.135~) | **최상** | 상: 백엔드와 프론트를 모두 새로 작성 | 유료 생성 정책 영향, Dockerfile은 UID 1000으로 실행 |
| Mesop 1.3.7 | Docker | 지원 | 중 | 상 | **2026-09-30 유지보수 종료** |
| NiceGUI 3.18.0 / Reflex 0.10.1 | Docker | WebSocket 기반 | 중상 / 상 | 상 | Reflex는 Node 빌드가 무겁고, CPU Basic에서 기동이 느릴 수 있다(미확인) |

- 모든 패키지가 `requires_python` 기준으로 Python 3.12와 호환된다. Chainlit은 `<3.14`, Reflex는 `>=3.11,<4`다.

## 현재 코드 기준 이전 포인트 (Gradio 6)
- **tuples 포맷이 삭제됐다.** `parsing_history`와 `get_response`가 쓰는 `(user, bot)` 튜플을 `{"role", "content"}` 메시지 형식으로 바꿔야 한다.
- `gr.Chatbot(show_copy_button=True)`는 6.x에서 `buttons=["copy"]`로 바뀌었다.
- `theme`/`css`/`js`/`head`는 6.0에서 `launch()`로 옮겨졌다. 6.1에서 `Blocks()` 생성자 인자도 다시 허용됐지만, `launch(css=...)` 쪽을 권장한다.
- 상태 메시지("⌕ 참고 자료를 찾고 있어요…")는 지금처럼 HTML span을 yield하는 방식이 계속 동작한다. 6.0부터 `allow_tags=True`가 기본값이다.
- 대안으로 `ChatMessage(metadata={"title": ..., "status": "pending"})`을 쓰면 스피너가 달린 접이식 "thought" 메시지로 보여줄 수 있다.
- 출처 링크는 지금처럼 최종 답변 끝에 Markdown으로 붙이면 된다. 변경할 필요가 없다.
- `gr.State`에 담긴 retriever/llm, API 키 Accordion, `gr.update`는 그대로 쓸 수 있다.
  - ChatInterface로 옮긴다면 키 입력란은 `additional_inputs` + `additional_inputs_accordion="⚿ 개인 키 등록하기"`로 옮길 수 있다.
  - 다만 "인증하기" 버튼 흐름과 좌우 2단 레이아웃은 잃는다. 그래서 이득이 적다.
- **gr.Server로 갈 때:**
  - API 키는 요청마다 받거나 세션 단위로 관리해야 한다.
  - history는 프론트가 들고 있다가 매 요청에 함께 보내야 한다.
  - 상태 메시지는 스트림 이벤트 타입(status/token/sources)으로 나누는 설계가 필요하다.

## 권고
**(a) 지금: Gradio 6.30.0 + `gr.Blocks`/`gr.Chatbot`을 유지한다.**
- SDK를 바꾸지 않아도 되므로 유료 생성 정책을 피할 수 있다. 기존 Space가 그대로 유지되는지는 미확인이지만, 위험이 가장 낮은 경로다.
- 변경 범위가 메시지 포맷, `buttons`, `launch(css=...)` 정도로 작다.
- 토큰 스트리밍, async, 시크릿 처리가 모두 이미 동작한다.
- 이번 이전에서 `gr.themes.Base` 기반의 커스텀 테마와 `style.css`를 정리해 두면 디자이너에게 넘기기가 쉬워진다.

**(b) 디자이너가 맡을 때: `gr.Server` + 정적 프론트를 쓴다.**
- 백엔드는 LangChain 로직을 `@server.api(name="chat", concurrency_limit=None)` 제너레이터로 옮긴다. 외부 API를 부르는 앱이므로 `concurrency_limit`을 풀어 둔다.
- 프론트는 디자이너가 HTML/CSS/JS나 React/Svelte 빌드 결과물로 만든다. `GET /`에서 서빙하고 `@gradio/client`로 스트림을 받는다.
- Space는 여전히 `sdk: gradio`라서 Docker 전환이 필요 없다. 그러면서도 FastAPI + 커스텀 프론트와 거의 같은 디자인 자유도를 얻는다.
- Gradio SDK 빌드에서 Node를 쓸 수 있는지는 미확인이다. 빌드 산출물(`dist/`)을 커밋하거나 CDN ESM으로 불러오는 방식을 전제로 한다.
- 중간 단계로는 Blocks를 유지한 채 `gr.HTML` 커스텀 템플릿(`js_on_load`, server functions)으로 부분만 재디자인할 수도 있다.
- 비채택:
  - Chainlit, Mesop: 유지보수 중단.
  - Streamlit: Docker 전용이라 유료 정책 영향을 받고, 디자인 자유도도 낮다.
  - 순수 FastAPI + Docker: gr.Server보다 나은 점이 크지 않은데 유료 정책 위험이 있다. HF 밖에서 호스팅한다면 그때 다시 검토한다.

## 출처
- https://huggingface.co/docs/hub/spaces-overview (원문: https://github.com/huggingface/hub-docs/blob/main/docs/hub/spaces-overview.md)
- https://huggingface.co/docs/hub/spaces-gpus (48시간 sleep, CPU Basic 사양)
- https://huggingface.co/docs/hub/spaces-sdks-streamlit , https://huggingface.co/docs/hub/main/spaces-changelog
- https://huggingface.co/docs/hub/spaces-sdks-docker (secrets, UID 1000)
- https://discuss.huggingface.co/t/docker-sdk-now-marked-as-paid-when-creating-a-new-space/177580
- https://discuss.huggingface.co/t/official-community-complaint-revert-free-cpu-basic-spaces-and-remove-anti-developer-sdk-restrictions/177703
- https://github.com/gradio-app/gradio/blob/main/CHANGELOG.md (6.0: tuples 삭제, `launch()` 인자 이동, gr.HTML 커스텀 컴포넌트 / 6.10: `gradio.Server`)
- https://gradio.app/guides/server-mode/ (원문: https://github.com/gradio-app/gradio/blob/main/guides/09_gradio-clients-and-lite/08_server-mode.md)
- https://github.com/gradio-app/gradio/blob/main/gradio/components/chatbot.py , https://github.com/gradio-app/gradio/blob/main/gradio/chat_interface.py
- https://github.com/streamlit/docs/blob/main/content/develop/quick-references/release-notes/2026.md
- https://github.com/Chainlit/docs/blob/main/get-started/project-status.mdx , https://github.com/Chainlit/chainlit/blob/main/CHANGELOG.md
- https://github.com/fastapi/fastapi/blob/master/docs/en/docs/tutorial/server-sent-events.md
- https://github.com/mesop-dev/mesop (유지보수 종료 공지)
- PyPI JSON API (https://pypi.org/pypi/<pkg>/json): gradio 6.30.0 (2026-10-08), streamlit 1.66.1, chainlit 2.12.0, fastapi 0.162.0, mesop 1.3.7, nicegui 3.18.0, reflex 0.10.1

*huggingface.co, gradio.app, docs.streamlit.io, docs.chainlit.io는 이 컨테이너에서 접근이 막혀 있었다. 그래서 각 문서의 GitHub 원문과 WebSearch로 확인했다. 저장소 파일은 수정하지 않았다.*
