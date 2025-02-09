import os
import json
import asyncio
import nest_asyncio
from dotenv import load_dotenv
from typing_extensions import List, TypedDict
from IPython.display import Image, display
import sys

# Windows에서 UTF-8 강제 설정
if sys.platform.startswith('win'):
    import locale
    sys.stdout.reconfigure(encoding='utf-8')
    asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())

# LangChain 관련 임포트
from langchain import hub
from langchain_core.documents import Document
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_chroma import Chroma
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import TextLoader
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser

# LangGraph 관련 임포트
from langgraph.graph import StateGraph, END

# 사전 정의: 특정 표현을 다른 표현으로 변환하기 위한 사전
dictionary = ['사람과 관련된 표현 -> 거주자']

# 환경변수 로드
load_dotenv()

# asyncio 이벤트 루프 설정
nest_asyncio.apply()

# 텍스트 스플리터 설정
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=300,
    chunk_overlap=20,
    separators=["\n\n", "\n"]
)

# 문서 로드 및 분할
def load_and_split_documents(text_path):
    loader = TextLoader(text_path, encoding='utf-8')
    return loader.load_and_split(text_splitter)

# 벡터 스토어 설정
def setup_vector_store(document_list):
    embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
    return Chroma.from_documents(
        documents=document_list,
        embedding=embeddings,
        collection_name="jop_posting_collection",
        persist_directory="./jop_posting_collection"
    )

# AgentState 정의
class AgentState(TypedDict):
    query: str
    context: List[Document]
    answer: str
    should_rewrite: bool
    rewrite_count: int
    answers: List[str]  # 각 시도마다의 답변을 저장할 리스트

# Retrieve 노드
def retrieve(state: AgentState):
    query = state['query']
    docs = retrieval.invoke(query)
    return {'context': docs}

# Verify 노드 (검증)
def verify(state: AgentState) -> dict:
    context = state['context']
    query = state['query']
    
    # 재작성 시도 횟수가 없으면 0으로 초기화
    rewrite_count = state.get('rewrite_count', 0)
    
    # 이미 3번 이상 재작성을 시도했다면 가장 좋은 답변 선택
    if rewrite_count >= 3:
        return {
            "should_rewrite": False,
            "rewrite_count": rewrite_count
        }
    
    verify_template = """
    다음 문서들이 사용자의 질문에 답변하기에 충분한 정보를 포함하고 있는지 판단해주세요.
    
    질문: {query}
    
    문서들:
    {context}
    
    답변 형식:
    - 문서가 충분한 정보를 포함하고 있다면 "YES"
    - 문서가 충분한 정보를 포함하고 있지 않다면 "NO"
    
    답변:
    """
    
    verify_prompt = PromptTemplate(
        template=verify_template,
        input_variables=["query", "context"]
    )
    
    verify_chain = verify_prompt | llm | StrOutputParser()
    response = verify_chain.invoke({
        "query": query,
        "context": "\n\n".join([doc.page_content for doc in context])
    })
    
    return {
        "should_rewrite": "NO" in response.upper(),
        "rewrite_count": rewrite_count + 1,
        "answers": state.get('answers', [])
    }

# Rewrite 노드 (질문 재작성)
def rewrite(state: AgentState) -> dict:
    """
    사용자의 질문을 사전을 참고하여 변경합니다.

    Args:
        state (AgentState): 사용자의 질문을 포함한 에이전트의 현재 state.

    Returns:
        AgentState: 변경된 질문을 포함하는 state를 반환합니다.
    """
    query = state['query']
    
    # 프롬프트 템플릿 생성
    rewrite_prompt = PromptTemplate.from_template(f"""
사용자의 질문을 보고, 우리의 사전을 참고해서 사용자의 질문을 변경해주세요.
이때 반드시 사전에 있는 규칙을 적용해야 합니다.

사전: {dictionary}

질문: {{query}}

변경된 질문을 출력해주세요:
""")
    
    # 리라이트 체인을 구성
    rewrite_chain = rewrite_prompt | llm | StrOutputParser()

    # 질문을 변경
    response = rewrite_chain.invoke({'query': query})
    
    return {'query': response}

# Generate 노드 (답변 생성)
def generate(state: AgentState) -> dict:
    query = state['query']
    context = state['context']
    rewrite_count = state.get('rewrite_count', 0)
    answers = state.get('answers', [])
    
    # 답변 생성을 위한 프롬프트
    generate_prompt = PromptTemplate.from_template("""
질문에 대한 답변을 100토큰 이내로 간단명료하게 작성해주세요.

질문: {question}

참고할 문서:
{context}

답변:
""")
    
    rag_chain = generate_prompt | llm | StrOutputParser()
    response = rag_chain.invoke({
        "question": query,
        "context": context
    })
    
    # 답변 목록에 현재 답변 추가
    answers.append(response)
    
    # 3회 시도 완료 후에는 최종 답변 선택
    if rewrite_count >= 3:
        # 답변 선택을 위한 프롬프트
        select_prompt = PromptTemplate.from_template("""
다음은 같은 질문에 대한 여러 답변들입니다. 가장 정확하고 명확한 답변을 선택해주세요.

질문: {question}

답변들:
{answers}

가장 좋은 답변을 요약해서 출력해주세요.
""")
        
        select_chain = select_prompt | llm | StrOutputParser()
        final_answer = select_chain.invoke({
            "question": query,
            "answers": "\n".join([f"{i+1}. {ans}" for i, ans in enumerate(answers)])
        })
        return {'answer': final_answer, 'answers': answers}
    
    return {'answer': response, 'answers': answers}

def router(state: AgentState) -> str:
    """라우터 함수: verify 결과에 따라 다음 노드를 결정"""
    # 재작성 시도 횟수가 3회 이상이면 무조건 generate로
    if state.get("rewrite_count", 0) >= 3:
        return "generate"
    return "rewrite" if state.get("should_rewrite", False) else "generate"

async def main():
    # 문서 로드 및 분할
    text_path = './documents/income_tax.txt'
    document_list = load_and_split_documents(text_path)
    
    # 벡터 스토어 설정
    global vector_store
    vector_store = setup_vector_store(document_list)
    
    # Retrieval 설정
    global retrieval
    retrieval = vector_store.as_retriever(search_kwargs={"k": 3})
    
    # LLM 및 프롬프트 설정
    global prompt, llm
    prompt = hub.pull("rlm/rag-prompt")
    llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)
    
    # 그래프 설정
    workflow = StateGraph(AgentState)
    
    # 노드 추가
    workflow.add_node("retrieve", retrieve)
    workflow.add_node("verify", verify)
    workflow.add_node("rewrite", rewrite)
    workflow.add_node("generate", generate)
    
    # 엣지 추가
    workflow.add_edge("retrieve", "verify")
    workflow.add_conditional_edges(
        "verify",
        router,
        {
            "rewrite": "rewrite",
            "generate": "generate"
        }
    )
    workflow.add_edge("rewrite", "retrieve")
    workflow.add_edge("generate", END)
    
    # 시작점 설정
    workflow.set_entry_point("retrieve")
    
    # 그래프 컴파일
    graph = workflow.compile()
    
    # 테스트 쿼리 실행
    query = input("테스트 문장을 넣으세요 : ")
    initial_state = {'query': query, 'answers': []}  # answers 초기화 추가
    result = graph.invoke(initial_state)
    print("\n결과:", result)

if __name__ == "__main__":
    asyncio.run(main())