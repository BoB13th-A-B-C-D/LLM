from pathlib import Path

from dotenv import load_dotenv
from kiwipiepy import Kiwi
from langchain.retrievers import (
    BM25Retriever,
    ContextualCompressionRetriever,
    EnsembleRetriever,
)
from langchain.retrievers.document_compressors import CrossEncoderReranker
from langchain.retrievers.multi_vector import MultiVectorRetriever, SearchType
from langchain.storage import LocalFileStore
from langchain_chroma import Chroma
from langchain_community.cross_encoders import HuggingFaceCrossEncoder
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_teddynote import logging

import pickle


def create_retriever(file_path):
    # MultiVector Retriever 로드
    model_name = "BAAI/bge-m3"
    model_kwargs = {"device": "cpu"}
    encode_kwargs = {"normalize_embeddings": True}

    hf_embeddings = HuggingFaceEmbeddings(
        model_name=model_name, model_kwargs=model_kwargs, encode_kwargs=encode_kwargs
    )

    DB_PATH = "./MultiVector_plain_final_db"
    STORE_PATH = "./local_docstore"
    id_key = "doc_id"

    # 저장된 Chroma DB 로드
    loaded_vectorstore = Chroma(
        collection_name="multivector_plain",
        embedding_function=hf_embeddings,
        persist_directory=DB_PATH,
    )

    # 저장된 docstore 로드
    store = LocalFileStore(STORE_PATH)

    # 검색기 초기화
    multivector_retriever = MultiVectorRetriever(
        vectorstore=loaded_vectorstore,
        byte_store=store,
        id_key="doc_id",
    )

    def load_split_docs(file_path="./split_documents.pkl"):
        """저장된 분할 문서들을 로드"""
        if not Path(file_path).exists():
            raise FileNotFoundError(f"{file_path}를 찾을 수 없습니다.")
        with open(file_path, "rb") as f:
            loaded_docs = pickle.load(f)
        return loaded_docs

    docs = load_split_docs()

    with open("doc_ids.pkl", "rb") as f:
        doc_ids = pickle.load(f)

    multivector_retriever.docstore.mset(list(zip(doc_ids, docs)))

    # 검색 유형을 MMR(Maximal Marginal Relevance)로 설정
    multivector_retriever.search_type = SearchType.mmr

    # KIWI + BM25 Retriever 로드
    # 토큰화 함수를 생성
    kiwi = Kiwi()

    def kiwi_tokenize(text):
        return [token.form for token in kiwi.tokenize(text)]

    # bm25 retriever
    bm25_kiwi_retriever = BM25Retriever.from_documents(
        docs, preprocess_func=kiwi_tokenize
    )
    bm25_kiwi_retriever.k = 3  # BM25Retriever의 검색 결과 개수

    # Ensemble Retriever 로드
    ensemble_retriever = EnsembleRetriever(
        retrievers=[bm25_kiwi_retriever, multivector_retriever],
        weights=[0.2, 0.8],  # weight는 추가 조정 필요
    )

    # Reranker 설정
    rerank_compressor = CrossEncoderReranker(
        model=HuggingFaceCrossEncoder(model_name="BAAI/bge-reranker-v2-m3"), top_n=3
    )

    # 최종 리트리버
    retriever = ContextualCompressionRetriever(
        base_compressor=rerank_compressor, base_retriever=ensemble_retriever
    )

    return retriever
