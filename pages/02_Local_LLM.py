from pyexpat import model
import streamlit as st
from langchain_core.messages.chat import ChatMessage
from langchain_openai import ChatOpenAI
from langchain_core.output_parsers import StrOutputParser
from langchain_teddynote.prompts import load_prompt
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from langchain_openai import ChatOpenAI
from langchain_ollama import ChatOllama
from langchain_teddynote import logging
from dotenv import load_dotenv
import os
from retriever import create_retriever
from llama_cpp import Llama

# API KEY 정보로드
load_dotenv()

# 프로젝트 이름을 입력합니다.
logging.langsmith("[Project] PDF RAG")

st.title("Local 모델 기반 RAG 💬")


# 이전 대화를 출력
def print_messages():
    for chat_message in st.session_state["messages"]:
        st.chat_message(chat_message.role).write(chat_message.content)


# 새로운 메시지를 추가
def add_message(role, message):
    st.session_state["messages"].append(ChatMessage(role=role, content=message))


def format_doc(document_list):
    return "\n\n".join([doc.page_content for doc in document_list])


# 체인 생성
def create_chain(retriever, model_name="OpenAI"):
    # 프롬프트 생성
    prompt = load_prompt(
        "D:\\LLM\prompts\\pdf-rag.yaml",
        encoding="utf-8",
    )

    if model_name == "EEVE":
        llm = Llama(
            model_path="D:\\LLM_MODEL\\EEVE-Korean-Instruct-10.8B-v1.0-Q4_1.gguf",
            n_ctx=4096,  # 컨텍스트 길이 증가
            # max_tokens=2048,  # 최대 토큰 수 증가
            temperature=0,  # 생성의 다양성 조절
        )

    elif model_name == "Qwen":
        llm = Llama(
            model_path="D:\\LLM_MODEL\\Qwen2.5-7B-Instruct-kowiki-qa-Q4_0.gguf",
            n_ctx=4096,  # 컨텍스트 길이 증가
            max_tokens=2048,  # 최대 토큰 수 증가
            temperature=0,  # 생성의 다양성 조절
        )

    def process_llm_response(response):
        """LLM 응답을 처리하는 함수"""
        if hasattr(response, "content"):
            return response.content
        elif isinstance(response, dict):
            # Llama 모델의 응답 형식 처리
            if "choices" in response:
                try:
                    return response["choices"][0]["text"]
                except (KeyError, IndexError):
                    return ""
            # 다른 형식의 딕셔너리 응답 처리
            return (
                response.get("content", "") or response.get("text", "") or str(response)
            )
        return str(response)

    def get_context(question: str):
        """검색 결과를 가져오는 함수"""
        docs = retriever.get_relevant_documents(question)
        return "\n\n".join(doc.page_content for doc in docs)

    chain = (
        {"question": RunnablePassthrough(), "context": lambda x: get_context(x)}
        | prompt
        | (lambda x: x.to_string())  # Convert PromptValue to string
        | llm
        | process_llm_response
        | StrOutputParser()
    )
    return chain


# 사이드바 생성
with st.sidebar:
    # 초기화 버튼 생성
    clear_btn = st.button("대화 초기화")

    # 모델 선택 메뉴
    selected_model = st.selectbox("LLM 선택", ["EEVE", "Qwen"], index=0)


# 처음 1번만 실행하기 위한 코드
if "messages" not in st.session_state:
    # 대화기록을 저장
    st.session_state["messages"] = []

if "chain" not in st.session_state:
    # DB에서 retriever 생성
    retriever = create_retriever()
    # 사이드바를 통해 생성한 모델로 chain 생성
    chain = create_chain(retriever, model_name=selected_model)
    st.session_state["chain"] = chain


# 초기화 버튼이 눌리면...
if clear_btn:
    st.session_state["messages"] = []

# 이전 대화 기록 출력
print_messages()

# 사용자의 입력
user_input = st.chat_input("궁금한 내용을 물어보세요!")

# 경고 메시지를 띄우기 위한 빈 영역
warning_msg = st.empty()
# st.session_state["chain"] = chain

# 만약에 사용자 입력이 들어오면...
if user_input:
    # chain 을 생성
    chain = st.session_state["chain"]

    # 사용자의 입력
    st.chat_message("user").write(user_input)
    # 스트리밍 호출
    response = chain.stream(user_input)

    with st.chat_message("assistant"):
        # 빈 공간(컨테이너)을 만들어서, 여기에 토큰을 스트리밍 출력
        container = st.empty()

        ai_answer = ""
        for token in response:
            ai_answer += token
            container.markdown(ai_answer)

    # 대화기록을 저장한다.
    add_message("user", user_input)
    add_message("assistant", ai_answer)
