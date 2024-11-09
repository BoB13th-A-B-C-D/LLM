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
logging.langsmith("[Project] RAG")

st.title("GPT 기반 LLM")


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
def create_chain(retriever, model_name):
    prompt = load_prompt(
        "prompts/pdf-rag.yaml",
        encoding="utf-8",
    )

    llm = ChatOpenAI(temperature=0, model="gpt-4o-mini", streaming=True)

    def process_llm_response(response):
        """LLM 응답을 처리하는 함수"""
        if hasattr(response, "content"):
            return response.content
        elif isinstance(response, dict):
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


# 처음 1번만 실행하기 위한 코드
if "messages" not in st.session_state:
    st.session_state["messages"] = []

if "chain" not in st.session_state:
    retriever = create_retriever()
    # chain 생성
    chain = create_chain(retriever, model_name="gpt-4o-mini")
    st.session_state["chain"] = chain

with st.sidebar:
    clear_btn = st.button("대화 초기화")

if clear_btn:
    st.session_state["messages"] = []

print_messages()

user_input = st.chat_input("궁금한 내용을 물어보세요!")

# 경고 메시지를 띄우기 위한 빈 영역
warning_msg = st.empty()

if user_input:
    chain = st.session_state["chain"]

    st.chat_message("user").write(user_input)
    response = chain.stream(user_input)

    with st.chat_message("assistant"):
        container = st.empty()

        ai_answer = ""
        for token in response:
            ai_answer += token
            container.markdown(ai_answer)

    # 대화기록을 저장한다.
    add_message("user", user_input)
    add_message("assistant", ai_answer)
