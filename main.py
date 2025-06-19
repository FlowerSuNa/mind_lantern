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
from langchain_core.runnables import RunnableSequence
from langchain_core.runnables.history import RunnableWithMessageHistory


SESSION_STORE = {} # 세션 저장소

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


def clear_session_history(session_id: str) -> None:
    """ 세션 기록 지우기 """
    if session_id in SESSION_STORE:
        SESSION_STORE[session_id].clear()


def get_retriever():
    """ 검색기 반환 """
    # 임베딩 정의
    embeddings = OpenAIEmbeddings(
        model="text-embedding-3-small",
        dimensions=1024
    )

    # 벡터 스토어 로드
    vectorstore = Chroma(
        embedding_function=embeddings,
        collection_name="content-250618",
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


def parsing_output(docs):
    """ 검색 결과를 파싱하여 반환 """
    output = []
    for doc in docs:
        content = f'주제 : {doc.metadata['source']}\n{doc.page_content}'
        output.append(content)
    return '\n\n'.join(output)

def get_rag_chain():
    """ 법륜스님처럼 답변하는 체인 반환 """
    # LLM 모델 정의
    llm = ChatGoogleGenerativeAI(
        model="gemini-2.5-flash",
        temperature=0.1
    )

    # 검색기 로드
    retriever = get_retriever()

    # 프롬프트 정의
    system_template = dedent("""
        당신은 법륜스님처럼 사람들의 고민을 경청하고, 따뜻하면서도 현실적인 조언을 주는 상담자입니다.
        어떤 질문이 와도 판단하거나 비난하지 않고, 상대의 입장에서 공감하며 지혜로운 답변을 합니다.
        답변은 근본적인 깨달음을 전하려고 노력하세요.
    """).strip()
    system_message = SystemMessagePromptTemplate.from_template(template=system_template)

    human_template = dedent("""
        다음은 법륜스님의 즉문즉설 강연에서 발췌한 참고 내용입니다:

        --- 참고 발언 시작 ---
        {content}
        --- 참고 발언 끝 ---

        위 내용을 참고하여, 아래 질문에 대해 스님처럼 응답해 주세요.

        질문지 : {question}

        스님 : 
    """).strip()
    human_message = HumanMessagePromptTemplate.from_template(template=human_template)

    chat_prompt = ChatPromptTemplate.from_messages([
        system_message, 
        MessagesPlaceholder(variable_name="history"),
        human_message
    ])

    # 체인 구성
    chain = {
        'history': lambda x: x['history'],
        'question': lambda x: x['question'],
        'content': lambda x: parsing_output(retriever.invoke(x['question']))
    } | chat_prompt | llm | StrOutputParser()
    chain_with_history = RunnableWithMessageHistory(
        chain,
        get_session_history,  # 세션별 인메모리 히스토리
        input_messages_key="question",
        history_messages_key="history",
    )
    return chain_with_history


def make_chain(chain): 
    async def get_response(message:str, history:tuple, session_id:str=None):
        if session_id is None:
            session_id =  get_session_id()

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
    chain = get_rag_chain()
    response_fn = make_chain(chain)

    # 챗봇 인터페이스 생성
    with gr.Blocks() as demo:
        session_id = gr.State(None)
        
        gr.ChatInterface(
            # fn=lambda message, h: get_response(message, h, chain, session_id),
            fn=response_fn,
            additional_inputs=[session_id],
            additional_outputs=[session_id],
            title="법륜스님 즉문즉설 내용 기반 답변 봇",
            description="질문을 입력하면 [즉문즉설] 유튜브 내용 기반으로 답변을 합니다.",
            theme=gr.themes.Soft(
                primary_hue="blue",
                secondary_hue="gray",
            ),
            examples=[
                ["막연히 불안해요. 왜 그럴까요?"],
                ["계속 불만이 생겨요. 어떻게 해야 할까요?"],
                ["마음에 안드는 사람이 있어요. 어떻게 하면 좋을까요?"],
            ],
            type="messages",
            textbox=gr.Textbox(placeholder="질문을 입력하세요...", container=True),
        )
        gr.Markdown("🛠️ LangChain · Gemini Flash · Chroma", elem_id="tool-badge")
        clear_button = gr.Button(value="이력 삭제")
        clear_button.click(fn=lambda _: clear_session_history(session_id))

    # 데모 실행
    demo.launch()


if __name__ == "__main__":
    load_dotenv()
    main()
