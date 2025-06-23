import gradio as gr
import uuid
import asyncio

from dotenv import load_dotenv
from textwrap import dedent
from pydantic import BaseModel, Field

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
from langchain_core.chat_history import BaseChatMessageHistory
from langchain_core.messages import BaseMessage
from langchain_core.runnables import (
    RunnableMap, 
    RunnablePassthrough, 
    RunnableLambda
)
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_core.documents import Document
from typing import List, Dict


SESSION_STORE = {} # 세션 저장소
COLLECTION_NAME = "content-250623"

class InMemoryHistory(BaseChatMessageHistory, BaseModel):
    """ 메모리 기반 히스토리 구현 """
    messages: list[BaseMessage] = Field(default_factory=list)

    def add_messages(self, messages: list[BaseMessage]) -> None:
        """Add a list of messages to the store"""
        self.messages.extend(messages)

    def clear(self) -> None:
        self.messages = []


def get_session_id():
    return str(uuid.uuid4())[:8]




def get_session_history(session_id: str) -> BaseChatMessageHistory:
    """ 세션 기록 가져오기 """
    if session_id not in SESSION_STORE:
        SESSION_STORE[session_id] = InMemoryHistory()
    return SESSION_STORE[session_id]


def clear_session_history(session_id: str):
    """ 세션 기록 지우기 """
    if session_id in SESSION_STORE:
        SESSION_STORE[session_id].clear()
    return gr.update(value=None)


def get_retriever():
    """ 검색기 반환 """
    # 임베딩 정의
    embeddings = OpenAIEmbeddings(
        model="text-embedding-3-small",
        dimensions=1536
    )

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

        info = f"주제 : {doc.metadata.get('title')}\n{doc.metadata.get('video_url')}"
        urls.append(info)
    return {
        'context': '\n\n'.join(context),
        'urls': '\n'.join(urls)
    }


def get_llm(api_key: str):
    """ LLM 모델 반환 """
    state = "✅ 키가 저장되었습니다. 이제 질문을 입력할 수 있어요!"
    try:
        llm = ChatGoogleGenerativeAI(
            model="gemini-2.5-flash",
            google_api_key=api_key,
            temperature=0.1
        )
        return gr.update(visible=False), gr.update(visible=True), state, llm

    except Exception as e:
        print(e)
        state = "❌ 올바른 Gemini API Key를 입력해주세요"
        return gr.update(visible=True), gr.update(visible=False), state, None


def get_rag_chain(llm, retriever):
    """ 법륜스님처럼 답변하는 체인 반환 """
    # 프롬프트 정의
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
        {content}
        --- 참고 발언 끝 ---

        위 내용을 참고하여, 아래 질문에 대해 스님 화법으로 답변하세요.

        질문지 : {question}

        스님 : 
        """
    ).strip()
    human_message = HumanMessagePromptTemplate.from_template(template=human_template)

    chat_prompt = ChatPromptTemplate.from_messages([
        system_message, 
        MessagesPlaceholder(variable_name="history"),
        human_message
    ])

    # 체인 구성
    chain = RunnableMap({
        "inputs": RunnablePassthrough(),
        "context_data": lambda query: parsing_documents(retriever.invoke(query))
    }) | RunnableLambda(lambda x: {
        "question": x["inputs"]["question"],
        "history": x["inputs"]["history"],
        "context": x["context_data"]["context"],
        "urls": x["context_data"]["urls"]
    }) | {
        "inputs": RunnablePassthrough(),
        "answer": chat_prompt | llm | StrOutputParser()
    } | RunnableLambda(lambda x: x["answer"] + "\n\n[출처]\n" + x["inputs"]["urls"])
    chain_with_history = RunnableWithMessageHistory(
        chain,
        get_session_history,  # 세션별 인메모리 히스토리
        input_messages_key="question",
        history_messages_key="history",
    )
    return chain_with_history
    

def make_chain(llm, retriever):
    chain = get_rag_chain(llm, retriever)

    async def get_response(message:str, session_id:str=None):
        session_id = session_id or get_session_id()
        
        response = chain.astream(
            {"question": message},
            config={"configurable": {"session_id": session_id}}
        )
        full_response = ""

        async for chunk in response:
            full_response += chunk
            await asyncio.sleep(0.01)  # 너무 빠를 경우 살짝 지연
            yield full_response, session_id  # 부분 응답을 계속 출력

        yield full_response, session_id # 최종 응답
    return get_response


def main():
    session_id = gr.State(None)
    llm = gr.State(None) 

    # 검색기 로드
    retriever = get_retriever()

    # 챗봇 인터페이스 생성
    with gr.Blocks() as demo:
        with gr.Column(visible=True) as key_input_area:
            gr.Markdown("🔐 **Gemini API Key를 먼저 입력해주세요.**")
            api_key_box = gr.Textbox(
                placeholder="API Key...", 
                type="password", 
                show_label=True,
                lines=1
            )
            key_submit_button = gr.Button("인증 하기")
            key_status = gr.Textbox(
                visible=True, 
                interactive=False, 
                label="", 
                show_label=False
            )

        with gr.Column(visible=False) as chat_area:
            chatbot = gr.Chatbot(label="법륜스님 [즉문즉설] 스타일 답변", scale=1)
            gr.Markdown("질문을 입력하면 [즉문즉설] 유튜브 내용 기반으로 답변을 합니다.")
            gr.Markdown(
                dedent(
                    """
                    📌 이 봇은 법륜스님의 [즉문즉설]에서 영감을 받은 비공식 LLM 프로젝트로, 특정 인물과 무관하며 상업 목적 없이 연구·실험용으로 제작되었습니다.
                    """
                ).strip()
            )
        
            # 예제 질문 목록
            example_questions=[
                ["계속 불만이 생겨요. 어떻게 해야 할까요?"],
                ["마음에 안드는 사람이 있어요. 어떻게 하면 좋을까요?"],
                ["욕심이 계속 많아져요. 욕심이 왜 많아질까요? 욕심을 멈출 수 있을까요?"]
            ]

            # 입력창 정의
            input_box = gr.Textbox(
                placeholder="질문을 입력하세요...",
                lines=6,
                max_lines=20,
                scale=1,
                show_label=False,
                autofocus=True
            )

            # 예제 버튼 영역
            with gr.Row():
                for idx, question in enumerate(example_questions):
                    gr.Button(value=question).click(
                        fn=lambda q=question: q,  # 기본값으로 클로저 문제 해결
                        outputs=input_box,
                        show_progress=False
                    )
            
            # 버튼 정의
            with gr.Row():
                send_button = gr.Button("답변 받기", variant="primary", scale=1)
                clear_button = gr.Button("이력 삭제", variant="secondary", scale=1)

        gr.Markdown("🛠️ LangChain · Gemini Flash · Chroma · Gradio", elem_id="tool-badge")
        
        # 버튼 클릭 액션
        key_submit_button.click(
            fn=get_llm,
            inputs=api_key_box,
            outputs=[key_input_area, chat_area, key_status, llm]
        )
        response_fn = make_chain(llm, retriever)
        send_button.click(
            fn=response_fn,
            inputs=[input_box, session_id],
            outputs=[chatbot, session_id]
        )
        clear_button.click(fn=clear_session_history, inputs=session_id, outputs=None)

    # 데모 실행
    demo.launch()


if __name__ == "__main__":
    load_dotenv()
    main()
