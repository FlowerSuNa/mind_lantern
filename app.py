import asyncio
import os
from pathlib import Path
from textwrap import dedent

import gradio as gr
from dotenv import load_dotenv
from langchain_core.documents import Document
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage
from langchain_core.prompts import (
    ChatPromptTemplate,
    HumanMessagePromptTemplate,
    MessagesPlaceholder,
    SystemMessagePromptTemplate,
)
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_openai import OpenAIEmbeddings
from langchain_pinecone import PineconeVectorStore

from prompts import CHAT_HUMAN_TEMPLATE, CHAT_SYSTEM_TEMPLATE

# ------------------------- 환경 설정 -------------------------

load_dotenv()

OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY")
GOOGLE_API_KEY = os.environ.get("GOOGLE_API_KEY")
PINECONE_API_KEY = os.environ.get("PINECONE_API_KEY")
PINECONE_INDEX_NAME = os.environ.get("PINECONE_INDEX_NAME", "mind-lantern")
CHAT_MODEL = os.environ.get("CHAT_MODEL", "gemini-2.5-flash")

EXAMPLE_QUESTIONS = [
    "계속 불만이 생겨요. 어떻게 해야 할까요?",
    "마음에 안드는 사람이 있어요. 어떻게 하면 좋을까요?",
    "욕심이 계속 많아져요. 욕심이 왜 많아질까요? 욕심을 멈출 수 있을까요?",
    "자존감이 떨어지고, 마음이 늘 불안합니다. 어떻게 해야 할까요?",
]

# ------------------------- 모델 -------------------------


def get_embeddings(api_key: str) -> OpenAIEmbeddings:
    """임베딩 모델 반환 (인덱스와 같은 모델·차원 필수)"""
    return OpenAIEmbeddings(
        model="text-embedding-3-small",
        openai_api_key=api_key,
        dimensions=1536,
    )


def get_llm(api_key: str) -> ChatGoogleGenerativeAI:
    """LLM 모델 반환"""
    return ChatGoogleGenerativeAI(
        model=CHAT_MODEL,
        google_api_key=api_key,
        temperature=0.1,
    )


# ------------------------- 검색·프롬프트 -------------------------


def get_retriever(embeddings: OpenAIEmbeddings):
    """Pinecone MMR 검색기 반환"""
    vectorstore = PineconeVectorStore(
        index_name=PINECONE_INDEX_NAME,
        embedding=embeddings,
        pinecone_api_key=PINECONE_API_KEY,
    )
    return vectorstore.as_retriever(
        search_type="mmr",
        search_kwargs={
            "k": 3,  # 검색할 문서의 수
            "fetch_k": 8,  # mmr 알고리즘에 전달할 문서의 수 (fetch_k > k)
            "lambda_mult": 0.3,  # 다양성 정도 (1은 최소 다양성, 0은 최대 다양성, 기본값 0.5)
        },
    )


def get_prompt() -> ChatPromptTemplate:
    """대화 프롬프트 정의"""
    return ChatPromptTemplate.from_messages(
        [
            SystemMessagePromptTemplate.from_template(CHAT_SYSTEM_TEMPLATE),
            MessagesPlaceholder(variable_name="history"),
            HumanMessagePromptTemplate.from_template(CHAT_HUMAN_TEMPLATE),
        ]
    )


def parsing_documents(documents: list[Document]) -> dict[str, str]:
    """검색 결과를 프롬프트용 문맥과 출처 링크 목록으로 변환"""
    context, urls = [], []
    for doc in documents:
        title = doc.metadata.get("title")
        context.append(f"주제 : {title}\n{doc.page_content}")
        urls.append(f"[{title}]({doc.metadata.get('video_url')})")
    return {"context": "\n\n".join(context), "urls": "\n".join(urls)}


def to_chatbot(history: list[BaseMessage]) -> list[dict]:
    """대화 기록을 Chatbot 메시지 형식으로 변환"""
    roles = {"human": "user", "ai": "assistant"}
    return [{"role": roles[m.type], "content": m.content} for m in history]


# ------------------------- 이벤트 처리 -------------------------


def confirm_api_key(openai_api_key: str, gemini_api_key: str):
    """API 키 검증 및 모델 로드 (실패 시 기본 키 사용)"""
    try:
        retriever = get_retriever(get_embeddings(openai_api_key))
        llm = get_llm(gemini_api_key)
        llm.get_num_tokens("연결 테스트")
        state = "✓ 키가 저장됐어요. 이제 질문할 수 있어요."
    except Exception as e:
        print(f"✗ API 키 검증 에러 : {e}")
        state = "✗ API 키를 다시 확인해 주세요."
        retriever = get_retriever(get_embeddings(OPENAI_API_KEY))
        llm = get_llm(GOOGLE_API_KEY)
    return state, retriever, llm


async def get_response(message: str, history: list[BaseMessage], retriever, llm):
    """참고 자료 검색 후 답변 스트리밍"""
    chat = to_chatbot(history) + [{"role": "user", "content": message}]
    input_box = gr.update(value=message, interactive=False)

    def render(content: str) -> list[dict]:
        return chat + [{"role": "assistant", "content": content}]

    def status(text: str) -> list[dict]:
        return render(f'<span class="bounce-text">{text}</span>')

    answer, urls = "", ""
    try:
        yield status("↻ 준비하고 있어요…"), history, retriever, llm, input_box
        if retriever is None or llm is None:
            _, retriever, llm = await asyncio.to_thread(
                confirm_api_key, OPENAI_API_KEY, GOOGLE_API_KEY
            )

        yield status("⌕ 참고 자료를 찾고 있어요…"), history, retriever, llm, input_box
        context = parsing_documents(await retriever.ainvoke(message))
        urls = context["urls"]

        yield status("… 답변을 정리하고 있어요"), history, retriever, llm, input_box
        chain = get_prompt() | llm
        async for chunk in chain.astream(
            {"question": message, "history": history, "context": context["context"]}
        ):
            answer += chunk.text
            yield render(answer), history, retriever, llm, input_box

    except Exception as e:  # 키·검색·생성 중 어디서 실패해도 입력창은 복구
        print(f"✗ 답변 생성 에러 : {e}")
        answer = "⚠︎ 지금은 답변을 만들 수 없어요. 잠시 후 다시 시도해 주세요."

    if urls:
        answer += "\n\n[유튜브 출처/링크]\n" + urls
    history += [HumanMessage(content=message), AIMessage(content=answer)]
    yield to_chatbot(history), history, retriever, llm, gr.update(value="", interactive=True)


# ------------------------- 화면 -------------------------


def main():
    """챗봇 화면 구성 및 실행"""
    with gr.Blocks() as demo:
        history = gr.State([])
        retriever, llm = gr.State(None), gr.State(None)

        gr.Markdown("## ☸︎ 법륜스님 [즉문즉설] 스타일 지혜 받기")
        gr.Textbox(
            dedent(
                """
                ▸ 질문을 입력하면 [즉문즉설] 유튜브 내용을 바탕으로 답변해요.
                ※ 이 봇은 법륜스님의 [즉문즉설]에서 영감을 받은 비공식 LLM 프로젝트로, 특정 인물과 무관하며 상업 목적 없이 연구·실험용으로 만들었어요.
                ⚠︎ 기본 키로 동작하지만, 사용량이 많거나 민감한 질문이라면 개인 키를 권장해요.
                """
            ).strip(),
            show_label=False,
        )

        # API 키 인증
        with gr.Accordion("⚿ 개인 키 등록하기", open=False):
            gr.Markdown(
                "질문 임베딩(벡터화)에는 **OpenAI** 텍스트 임베딩 모델을, "
                "답변 생성에는 **Gemini** 모델을 사용해요."
            )
            openai_api_key_box = gr.Textbox(
                placeholder="OpenAI text-embedding-3-small 모델을 사용해요.",
                type="password",
                lines=1,
                label="OpenAI API Key",
            )
            gemini_api_key_box = gr.Textbox(
                placeholder=f"Gemini {CHAT_MODEL} 모델을 사용해요.",
                type="password",
                lines=1,
                label="Gemini API Key",
            )
            key_submit_button = gr.Button("인증하기")
            key_status = gr.Textbox(interactive=False, show_label=False)
            key_submit_button.click(
                fn=confirm_api_key,
                inputs=[openai_api_key_box, gemini_api_key_box],
                outputs=[key_status, retriever, llm],
            )

        # 채팅 창
        with gr.Row():
            with gr.Column(scale=1):
                input_box = gr.Textbox(
                    placeholder="질문을 입력하세요...",
                    lines=20,
                    show_label=False,
                    autofocus=True,
                    autoscroll=True,
                    elem_id="input-box",
                )

                # 예제 질문 버튼 (2개씩 2줄)
                for row in (EXAMPLE_QUESTIONS[:2], EXAMPLE_QUESTIONS[2:]):
                    with gr.Row():
                        for question in row:
                            gr.Button(question, scale=1, elem_classes="same-height").click(
                                fn=lambda q=question: q,  # 기본값으로 클로저 문제 해결
                                outputs=[input_box],
                            )

                send_button = gr.Button("질문하기", variant="primary", scale=1)

            with gr.Column(scale=2):
                chatbot = gr.Chatbot(show_label=False, buttons=["copy"], elem_id="chatbot-box")

        gr.Markdown("⚙︎ LangChain · OpenAI · Gemini · Pinecone · Gradio", elem_id="tool-badge")

        send_button.click(
            fn=get_response,
            inputs=[input_box, history, retriever, llm],
            outputs=[chatbot, history, retriever, llm, input_box],
        )

    demo.launch(css_paths=[Path(__file__).with_name("style.css")])


if __name__ == "__main__":
    main()
