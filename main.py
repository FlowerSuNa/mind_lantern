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
from langchain_core.messages import HumanMessage, AIMessage
from langchain_core.runnables import (
    RunnableMap, 
    RunnablePassthrough, 
    RunnableLambda
)
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_core.documents import Document
from typing import List, Dict

COLLECTION_NAME = "content-250623"

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


def comfirm_api_key(opnenai_api_key: str, gemini_api_key: str):
    """ API Key 검증 및 모델 로드 """
    state = "✅ 키가 저장되었습니다. 이제 질문을 입력할 수 있어요!"
    try:
        embeddings = get_embeddings(opnenai_api_key)
        llm = get_llm(gemini_api_key)
        return gr.update(visible=False), gr.update(visible=True), state, embeddings, llm

    except Exception as e:
        print(e)
        state = "❌ 올바른 API Key를 입력해주세요"
        return gr.update(visible=True), gr.update(visible=False), state, None, None
    

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

        info = f"주제 : {doc.metadata.get('title')}\n{doc.metadata.get('video_url')}"
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
        lambda x: x["answer"] + "\n\n[출처]\n" + x["inputs"]["urls"]   
    )
    return chain


# ------------------------- Response -------------------------

def make_chain(embeddings, llm):
    prompt = get_prompt()
    retriever = get_retriever(embeddings)
    chain = get_rag_chain(prompt, retriever, llm)

    async def get_response(message: str, history: List):       
        response = chain.astream(
            {"question": message, "history": history},
        )
        full_response = ""

        async for chunk in response:
            full_response += chunk
            await asyncio.sleep(0.01)  # 너무 빠를 경우 살짝 지연
            yield full_response, history  # 부분 응답을 계속 출력

        history.append(HumanMessage(content=message))
        history.append(AIMessage(content=full_response))
        yield full_response, history # 최종 응답
    return get_response


def get_history(history):
    """  """
    chat_history = []
    for message in history:
        role = message.type

        if role == "human":
            _role = "user"
        elif role == "ai":
            _role = "assistant"
        else:
            _role = role

        chat_history.append(gr.ChatMessage(role=_role, content=message.text()))
    return chat_history


def main():
    """ 챗봇 인터페이스 생성 """
    with gr.Blocks() as demo:
        history = gr.State([])
        embeddings = gr.State(None)
        llm = gr.State(None)

        gr.Markdown('## 🤖 법륜스님 [즉문즉설] 스타일 답변받기')
        gr.Markdown('질문을 입력하면 [즉문즉설] 유튜브 내용 기반으로 답변을 생성합니다.')
        gr.Markdown("📌 이 봇은 법륜스님의 [즉문즉설]에서 영감을 받은 비공식 LLM 프로젝트로, 특정 인물과 무관하며 상업 목적 없이 연구·실험용으로 제작되었습니다.")

        # API Key 인증
        with gr.Column(visible=True) as key_input_area:
            gr.Markdown("🔐 **API Key를 먼저 입력해주세요.**")
            gr.Markdown("질문에 대한 임베딩(벡터화)은 **OpenAI**의 텍스트 임베딩 모델을 활용하고, 최종 응답 생성을 위한 LLM은 **Gemini** 모델을 사용합니다.")
            openai_api_key_box = gr.Textbox(
                placeholder="OpenAI API Key...", 
                type="password", 
                show_label=True,
                lines=1
            )
            gemini_api_key_box = gr.Textbox(
                placeholder="Gemini API Key...", 
                type="password", 
                show_label=True,
                lines=1
            )
            key_submit_button = gr.Button("인증하기")
            key_status = gr.Textbox(
                visible=True, 
                interactive=False, 
                label="", 
                show_label=False
            )

        with gr.Column(visible=False) as chat_area:
            with gr.Row(): 
                with gr.Column(scale=1):
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
                    example_questions=[
                        ["계속 불만이 생겨요. 어떻게 해야 할까요?"],
                        ["마음에 안드는 사람이 있어요. 어떻게 하면 좋을까요?"],
                        ["욕심이 계속 많아져요. 욕심이 왜 많아질까요? 욕심을 멈출 수 있을까요?"]
                    ]
                    with gr.Row():
                        for idx, question in enumerate(example_questions):
                            gr.Button(value=question).click(
                                fn=lambda q=question: q,  # 기본값으로 클로저 문제 해결
                                outputs=input_box,
                                show_progress=False
                            )
                    
                    # 버튼 정의
                    with gr.Row():
                        send_button = gr.Button("질문하기", variant="primary", scale=1)

                with gr.Column(scale=2):
                    recent_answer = gr.Textbox(label="최근 답변", interactive=False, lines=5)

            with gr.Accordion("📜 이전 대화 보기", open=False):
                gr.Chatbot(lambda h: get_history(h), inputs=history, label="대화 히스토리",  interactive=False)

        gr.Markdown("🛠️ LangChain · OpenAI · Gemini · Chroma · Gradio", elem_id="tool-badge")
        
        # 버튼 클릭 액션
        key_submit_button.click(
            fn=comfirm_api_key,
            inputs=[openai_api_key_box, gemini_api_key_box],
            outputs=[key_input_area, chat_area, key_status, embeddings, llm]
        )

        if embeddings and llm:
            response_fn = make_chain(embeddings, llm)
            send_button.click(
                fn=response_fn,
                inputs=[input_box, history],
                outputs=[recent_answer, history]
            )
        else:
            send_button.click(
                fn=None,
                inputs=[],
                outputs=[],
                js="alert('⚠️ API Key 인증을 해주세요.')"
            )


    # 데모 실행
    demo.launch()


if __name__ == "__main__":
    load_dotenv()
    main()
