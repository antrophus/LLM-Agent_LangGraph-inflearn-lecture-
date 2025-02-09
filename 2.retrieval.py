import os
import json
import asyncio
import nest_asyncio
import markdown
from bs4 import BeautifulSoup
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
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_openai import OpenAIEmbeddings
from langchain_chroma import Chroma
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader, UnstructuredMarkdownLoader, TextLoader

# LangGraph 관련 임포트
from langgraph.graph import StateGraph, START, END

# Zerox 관련 임포트
from pyzerox import zerox

# 환경변수 로드
load_dotenv()

# asyncio 이벤트 루프 설정
nest_asyncio.apply()

# 파일 경로 설정
DOCUMENTS_DIR = "./documents"
os.makedirs(DOCUMENTS_DIR, exist_ok=True)

# PDF 파일 로드
pdf_file_path = './income_tax.pdf'
loader = PyPDFLoader(pdf_file_path)
pages = []

async def load_pdf():
    async for page in loader.alazy_load():
        pages.append(page)

# Zerox를 사용한 PDF 처리
async def process_pdf_with_zerox():
    model = "gemini/gemini-2.0-flash-exp"
    kwargs = {}
    custom_system_prompt = None
    file_path = "./pixel-nvidia.pdf"
    output_dir = "./documents"
    select_pages = None
    
    result = await zerox(
        file_path=file_path,
        model=model,
        output_dir=output_dir,
        custom_system_prompt=custom_system_prompt,
        select_pages=select_pages,
        **kwargs
    )
    return result

# 마크다운을 텍스트로 변환
def convert_markdown_to_text():
    markdown_path = "./documents/income_tax.md"
    text_path = './documents/income_tax.txt'
    
    with open(markdown_path, 'r', encoding='utf-8') as md_file:
        md_content = md_file.read()
    
    html_content = markdown.markdown(md_content)
    soup = BeautifulSoup(html_content, 'html.parser')
    text_content = soup.get_text()
    
    with open(text_path, 'w', encoding='utf-8') as txt_file:
        txt_file.write(text_content)
    
    return text_path

# 텍스트 스플리터 설정
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=1500,
    chunk_overlap=100,
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
        collection_name="income_tax_collection",
        persist_directory="./income_tax_collection"
    )

# AgentState 정의
class AgentState(TypedDict):
    query: str
    context: List[Document]
    answer: str

# Retrieve 노드
def retrieve(state: AgentState):
    query = state['query']
    docs = retrieval.invoke(query)
    return {'context': docs}

# Generate 노드
def generate(state: AgentState):
    query = state['query']
    context = state['context']
    rag_chain = prompt | llm
    response = rag_chain.invoke({"question": query, "context": context})
    return {'answer': response}

async def main():
    try:
        # PDF 로드
        await load_pdf()
        
        # Zerox 처리
        try:
            zerox_result = await process_pdf_with_zerox()
            print("Zerox 처리 완료")
        except Exception as e:
            print(f"Zerox 처리 중 오류 발생: {str(e)}")
        
        # 마크다운을 텍스트로 변환
        text_path = convert_markdown_to_text()
        
        # 문서 로드 및 분할
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
        llm = ChatGoogleGenerativeAI(model="gemini-2.0-flash-exp")
        
        # 그래프 설정
        graph = StateGraph(AgentState)
        graph.add_node('retrieve', retrieve)
        graph.add_node('generate', generate)
        
        # 엣지 추가
        graph.add_edge(START, 'retrieve')
        graph.add_edge('retrieve', 'generate')
        graph.add_edge('generate', END)
        
        # 그래프 컴파일
        workflow = graph.compile()
        
        # 테스트 쿼리 실행
        query = input("테스트 문장을 넣으세요 : ")
        initial_state = {'query': query}
        result = workflow.invoke(initial_state)
        print("\n결과:", result)
        
    except Exception as e:
        print(f"실행 중 오류 발생: {str(e)}")
        raise  # 디버깅을 위해 전체 오류 스택 출력

if __name__ == "__main__":
    asyncio.run(main())