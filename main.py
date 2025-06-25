import gradio as gr
import os
import asyncio

from dotenv import load_dotenv
from textwrap import dedent

from langchain_openai.embeddings import OpenAIEmbeddings
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.vectorstores import Chroma

from langchain_core.prompts import (
    ChatPromptTemplate, 
    SystemMessagePromptTemplate, 
    HumanMessagePromptTemplate,
    MessagesPlaceholder
)
from langchain_core.output_parsers import StrOutputParser
from langchain_core.messages import HumanMessage, AIMessage
from langchain_core.runnables import (
    RunnableMap, 
    RunnablePassthrough, 
    RunnableLambda
)
from langchain_core.documents import Document
from typing import List, Dict

# ------------------------- Environment Setting -------------------------

load_dotenv()

COLLECTION_NAME = "content-250623"
OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY")
GOOGLE_API_KEY = os.environ.get("GOOGLE_API_KEY")

# ------------------------- Confirm and Load Model -------------------------

def get_embeddings(api_key: str):
    """ 임베딩 모델 반환 """
    return OpenAIEmbeddings(
        model="text-embedding-3-small",
        openai_api_key=api_key,
        dimensions=1536
    )

def get_llm(api_key: str):
    """ LLM 모델 반환 """
    return ChatGoogleGenerativeAI(
        model="gemini-2.5-flash",
        google_api_key=api_key,
        temperature=0.1
    )


# ------------------------- Chain Definition -------------------------

def get_retriever(embeddings: OpenAIEmbeddings):
    """ 검색기 반환 """
    # 벡터 스토어 로드
    vectorstore = Chroma(
        embedding_function=embeddings,
        collection_name=COLLECTION_NAME,
        persist_directory="./chroma_db"
    )

    # 검색기 정의
    retriever = vectorstore.as_retriever(
        search_type='mmr',
        search_kwargs={
            'k': 3,                 # 검색할 문서의 수
            'fetch_k': 8,           # mmr 알고리즘에 전달할 문서의 수 (fetch_k > k)
            'lambda_mult': 0.3,     # 다양성을 고려하는 정도 (1은 최소 다양성, 0은 최대 다양성, 기본값은 0.5)
        },
    )
    return retriever


def parsing_documents(documents: List[Document]) -> Dict[str, str]:
    """" 검색 결과를 파싱하여 출력 문자열과 URL 목록 반환 """
    context, urls = [], []
    for doc in documents:
        content = f"주제 : {doc.metadata.get('title')}\n{doc.page_content}"
        context.append(content)

        info = f"[{doc.metadata.get('title')}]({doc.metadata.get('video_url')})"
        urls.append(info)
    return {
        'context': '\n\n'.join(context),
        'urls': '\n'.join(urls)
    }


def get_prompt():
    """ 프롬프트 정의 """
    system_template = dedent(
        """
        당신은 법륜스님처럼 사람들의 고민을 경청하고, 따뜻하면서도 현실적인 조언을 주는 상담자입니다.
        어떤 질문이 와도 판단하거나 비난하지 않고, 상대의 입장에서 공감하며 지혜로운 답변을 합니다.
        답변은 근본적인 깨달음을 전하려고 노력하세요.
        """
    ).strip()
    system_message = SystemMessagePromptTemplate.from_template(template=system_template)

    human_template = dedent(
        """
        다음은 법륜스님의 즉문즉설 강연에서 발췌한 참고 내용입니다:

        --- 참고 발언 시작 ---
        {context}
        --- 참고 발언 끝 ---

        위 내용을 참고하여, 아래 질문에 대해 스님 화법으로 답변하세요.

        질문지 : {question}

        스님 : 
        """
    ).strip()
    human_message = HumanMessagePromptTemplate.from_template(template=human_template)

    return ChatPromptTemplate.from_messages([
        system_message, 
        MessagesPlaceholder(variable_name="history"),
        human_message
    ])


def get_rag_chain(prompt, retriever, llm):
    """ 체인 반환 """
    context_chain = retriever | RunnableLambda(parsing_documents)

    chain = RunnableMap({
        "inputs": RunnablePassthrough(),
        "context_data": lambda x: context_chain.invoke(x["question"])
    }) | RunnableLambda(lambda x: {
        **x["inputs"],
        **x["context_data"]
    }) | {
        "inputs": RunnablePassthrough(),
        "answer": prompt | llm | StrOutputParser()
    } | RunnableLambda(
        lambda x: x["answer"] + "\n\n[유튜브 출처/링크]\n" + x["inputs"]["urls"]   
    )
    return chain

# ------------------------- Action Function -------------------------

def confirm_api_key(openai_api_key: str, gemini_api_key: str):
    """ API Key 검증 및 모델 로드 """

    embeddings = get_embeddings(openai_api_key)
    llm = get_llm(gemini_api_key)

    try: 
        retriever = get_retriever(embeddings)
        llm.get_num_tokens('연결 테스트')
        state = "✅ 키가 저장되었습니다. 이제 질문을 입력할 수 있어요!"

    except Exception as e:
        print(e)
        state = "❌ 올바른 API Key를 입력해주세요."
        embeddings = get_embeddings(OPENAI_API_KEY)
        retriever = get_retriever(embeddings)
        llm = get_llm(GOOGLE_API_KEY)

    prompt = get_prompt()
    chain = get_rag_chain(prompt, retriever, llm)
    return state, chain


def parsing_history(history: List):
    """  """
    pairs = []
    user_message = None
    assistant_message = None

    for message in history:
        role = message.type

        if role == "human":
            # 이전 user_message가 남아있고, ai 없이 다음 human이 오는 경우
            if user_message is not None:
                pairs.append((user_message, None))

            user_message = message.content
            assistant_message = None

        elif role == "ai":
            assistant_message = message.content
            pairs.append((user_message, assistant_message))
            user_message = None
            assistant_message = None

    if user_message is not None:
        pairs.append((user_message, None))

    return pairs


def get_history(message:str , history: List):
    chatbot_display = parsing_history(history)
    return chatbot_display.append((message, None))


async def get_response(message: str, history: List, chain):
    if chain is None:
        _, chain = confirm_api_key(OPENAI_API_KEY, GOOGLE_API_KEY)

    chatbot_display = parsing_history(history)
    yield chatbot_display + [(message, "📝 답변 생성 중...")], history, chain, message

    try:
        response = chain.astream(
            {"question": message, "history": history}
        )
        full_response = ""

        async for chunk in response:
            full_response += chunk
            await asyncio.sleep(0.01)  # 너무 빠를 경우 살짝 지연
            yield chatbot_display + [(message, full_response)], history, chain, message  # 부분 응답을 계속 출력

    except Exception as e:
        print(e)
        full_response = "⚠️ API 사용량이 초과되었습니다."

    history.append(HumanMessage(content=message))
    history.append(AIMessage(content=full_response))
    chatbot_display.append((message, full_response))
    yield chatbot_display, history, chain, "" # 최종 응답


def main():
    """ 챗봇 인터페이스 생성 """
    with gr.Blocks(
        css=dedent(
            """
            .same-height { height: 70px !important; }
                    
            #input-box {
                height: 50vh !important;
                display: flex !important;
                flex-direction: column !important;
            }
                    
            #input-box textarea {
                flex-grow: 1 !important;
                height: 45vh !important;
                box-sizing: border-box !important;
                padding: 12px !important;
                font-size: 16px;
                line-height: 1.5;
                resize: none !important;
                overflow-y: auto !important;
                white-space: pre-wrap !important;
            }

            #chatbot-box {
                height: 70vh !important;     /* 화면 높이의 70% */
                max-height: 70vh !important;
                overflow-y: auto !important;
            }
            """
        )
    ) as demo:
        history = gr.State([])
        chain = gr.State(None)

        gr.Markdown('## 🤖 법륜스님 [즉문즉설] 스타일 지혜 받기')
        gr.Textbox(
            dedent(
                """
                ❤️ 질문을 입력하면 [즉문즉설] 유튜브 내용을 기반으로 답변을 생성합니다.
                📌 이 봇은 법륜스님의 [즉문즉설]에서 영감을 받은 비공식 LLM 프로젝트로, 특정 인물과 무관하며 상업 목적 없이 연구·실험용으로 제작되었습니다.
                ⚠️ 기본 키로 동작하지만, 많은 사용량 또는 민감한 요청은 개인 키 사용을 권장합니다.
                """
            ).strip,
            show_label=False,
        )

        # API Key 인증
        with gr.Accordion("🔐 개인 키 등록하기", open=False):
            gr.Markdown("질문에 대한 임베딩(벡터화)은 **OpenAI**의 텍스트 임베딩 모델을 활용하고, 최종 응답 생성을 위한 LLM은 **Gemini** 모델을 사용합니다.")
            openai_api_key_box = gr.Textbox(
                placeholder="OpenAI text-embedding-3-small 모델을 사용합니다.", 
                type="password", 
                show_label=True,
                lines=1,
                label="OpenAI API Key"
            )
            gemini_api_key_box = gr.Textbox(
                placeholder="Gemini gemini-2.5-flash 모델을 사용합니다.", 
                type="password", 
                show_label=True,
                lines=1,
                label="Gemini API Key"
            )
            key_submit_button = gr.Button("인증하기")
            key_status = gr.Textbox(
                visible=True, 
                interactive=False, 
                label="", 
                show_label=False
            )
            key_submit_button.click(
                fn=confirm_api_key,
                inputs=[openai_api_key_box, gemini_api_key_box],
                outputs=[key_status, chain]
            )

        # 채팅 창
        with gr.Row(): 
            with gr.Column(scale=1):
                # 입력창 정의
                input_box = gr.Textbox(
                    placeholder="질문을 입력하세요...",
                    lines=20,
                    # max_lines=100,
                    # scale=1,
                    show_label=False,
                    autofocus=True,
                    autoscroll=True,
                    elem_id="input-box"
                )

                # 예제 버튼 영역
                example_questions=[
                    "계속 불만이 생겨요. 어떻게 해야 할까요?",
                    "마음에 안드는 사람이 있어요. 어떻게 하면 좋을까요?",
                    "욕심이 계속 많아져요. 욕심이 왜 많아질까요? 욕심을 멈출 수 있을까요?",
                    "자존감이 떨어지고, 마음이 늘 불안합니다. 어떻게 해야 할까요?"
                ]
                with gr.Row():
                    for question in example_questions[:2]:
                        btn = gr.Button(
                            value=question, 
                            scale=1,
                            elem_classes="same-height"
                        )
                        btn.click(
                            fn=lambda q=question: q,  # 기본값으로 클로저 문제 해결
                            outputs=[input_box],
                        )
                with gr.Row():
                    for question in example_questions[2:]:
                        btn = gr.Button(
                            value=question, 
                            scale=1,
                            elem_classes="same-height"
                        )
                        btn.click(
                            fn=lambda q=question: q,  # 기본값으로 클로저 문제 해결
                            outputs=[input_box],
                        )
                
                # 응답 요청 버튼
                with gr.Row():
                    send_button = gr.Button("질문하기", variant="primary", scale=1)

            with gr.Column(scale=2):
                chatbot = gr.Chatbot(
                    show_label=False, 
                    value=[], 
                    show_copy_button=True,
                    elem_id="chatbot-box"
                )

        gr.Markdown("🛠️ LangChain · OpenAI · Gemini · Chroma · Gradio", elem_id="tool-badge")
        
        # 응답 생성
        send_button.click(
            fn=get_response,
            inputs=[input_box, history, chain],
            outputs=[chatbot, history, chain, input_box],
        )
        
    # 데모 실행
    demo.launch()


if __name__ == "__main__":
    main()
