# 검증을 한번 하고 문서가 사용자의 질문과 관련이 있으면 genenrate를 하고 그렇지 않으면 rewrite를 통해서 사용자의 질문을 문맥에 맡게 수정한 다음에 문서를 다시 가져오는 절차를 짚어보도록 하겠다.

# 4개의 노드가 필요하다.
#   1. ritrieve
#   2. generate
#   3. rewrite
#   4. 문서 검증 노드
# ![image.png](attachment:image.png)

# 먼저 생성 해 두었던 chromaDB에 접근하도록 한다.
# 생성할 때는 from_documents 메소드를 사용 했었다. 이거는 chromaDB 가 존재 하지 않는 경우에 사용 하는 방법이다.
# 이미 생성한 chromaDB에 접근하기 위해서는 chroma 클래스를 바로 써야 한다.
# 벡터 생성할 때 쓴 코드를 그대로 가져온다.


from langchain_chroma import Chroma
from langchain_openai import OpenAIEmbeddings

embeddings_fuction = OpenAIEmbeddings(model='text-embedding-3-large')

vector_store = Chroma(
    embedding_function = embeddings_fuction,
    collection_name = 'income_tax_collection',
    persist_directory = './income_tax_collection' # 벡터 스토어를 저장할 디렉토리를 설정해줘야 로컬에 남아 있게 된다.
)

# 이제 chromaDB에 접근 할 수 있게 되었다.
# retrieve도 똑같이 선언해 주도록 하겠다.


retriever = vector_store.as_retriever(search_kwargs={'k':3})


# 이제 state를 선언해서 우리가 원하는 agent를 만들어 보도록 하겠다.
# state는 기존의 코드와 똑같이 선언해 주면 된다.

# state를 만들고 시작해보자.

from typing_extensions import List, TypedDict
from langchain_core.documents import Document
from langgraph.graph import StateGraph # graph_builder까지 선언해 주도록 하겠다.

class AgentState(TypedDict):
    query: str # 질문
    context: List[Document] # 컨텍스트(답변할 때 참고할 문서들: langchain의 Document 타입) - 경로는 3.1의 from langchain_core.documents import Document
    answer: str # 답변


# graph_builder 선언.
graph_builder = StateGraph(AgentState)

# 이제 노드를 추가해 보도록 하겠다.



# retrieve 노드 추가

def retrieve(state: AgentState) -> AgentState:
    query = state['query'] # 사용자의 질문을 받아온 다음
    docs = retriever.invoke(query) # 기반으로 리트리버에 대해서 검색을 하고
    return {'context': docs} # state의 컨텍스트에 넣어준다.


# generate 노드 추가를 위해 llm과 프롬프트를 선언해 주도록 하겠다.

# from langchain import hub
from langchain_google_genai import ChatGoogleGenerativeAI
# prompt = hub.pull('rlm/rag-prompt')
llm = ChatGoogleGenerativeAI(model='gemini-2.0-flash-exp')

# 그런데 이번에는 generate에 쓸 prompt도 있어야 하고, question을 다시 쓸 때 사용자의  - 질문을 수정할 때 쓸 prompt도 있어야 한다. 그리고 문서의 관련성을 판단할 때 쓸 prompt도 있어야 한다.

#  - 따라서 prompt라고 변수를 하나만 정하면 중복이 될 수 있다.
#  - 공통으로 쓸 llm만 미리 선언하고 
#  - 프롬프트는 각각 선언해 주도록 하겠다.


# context 랑 query를 잘 받아와서, LLM을 invoke 하지 않고, 선언한 프롬프트를 활용해야한다.
# langchain의 LCEL 문법: 파이프 활용 해서 연결하는 것. 기반으로 체인을 만들어 준다.

from langchain import hub

generate_prompt = hub.pull('rlm/rag-prompt')

def generate(state: AgentState) -> AgentState:
    context = state['context']
    query = state['query']
    rag_chain = generate_prompt | llm
    response = rag_chain.invoke({'question': query, 'context': context})
    
    return {'answer': response}



# 이제 문서의 관련성을 측정할 노드를 만들어 보겠다.
# langsmith에서 rag-document-relevance라는 프롬프트를 제공해 준다.
# 사용자의 질문과 문서를 받아서 관련성을 측정해서 score를 반환해 준다.
# 이 score가 1이면 관련성이 높고, 0이면 관련성이 낮다.

# set the LANGSMITH_API_KEY environment variable (create key in settings)
from langchain import hub
from typing import Literal
doc_relevance_prompt = hub.pull('langchain-ai/rag-document-relevance')

# def check_doc_relevance(state: AgentState) -> Literal['generate', 'rewrite']:
#     query = state['query']
#     context = state['context']
#     print(f'context == {context}')
#     doc_relevance_chain = doc_relevance_prompt | llm
#     response = doc_relevance_chain.invoke({'question': query, 'documents': context}) # 수정: context를 그대로 전달
#     print(f'doc relevance response: {response}')
    
#     # state를 설정하는 것이 아니라, 다음에 어디로 갈지 결정. 
#     # 따라서 state를 return하면 안되고, score가  1이면 generate로 가고, 아니면 rewrite로 가도록 하겠다.
#     if response['score'] == 1:
#         return 'generate'
    
#     return 'rewrite'

def check_doc_relevance(state: AgentState) -> Literal['generate', 'rewrite']:
    query = state['query']
    context = state['context']
    print(f'context == {context}')
    doc_relevance_chain = doc_relevance_prompt | llm
    response = doc_relevance_chain.invoke({'question': query, 'documents': context})
    print(f'doc relevance response: {response}')
    
    # 수정된 부분: 리스트의 첫 번째 요소에서 'args' 딕셔너리의 'Score' 값을 가져옴
    score = response[0]['args']['Score']
    
    if score == 1:
        return 'generate'
    return 'rewrite'



# rewrite 노드 추가
# rewrite 는 검색을 용이하게 해 주기 위한것, 검색이 왜 안되었는지를 따져봐야 한다.
# 검색이 안되었던 이유는 사용자의 질문이 문서와 맞지 않았기 때문이다.
# 따라서 사용자의 질문을 문서와 맞게 수정해 주는 것이 필요하다.
# 이를 위해서 사용자의 질문을 수정해 주는 프롬프트를 제공해 준다.

# 검색이 잘 되도록 entity를 맞춤 설정해 주는 것이 필요하다.

# langchain_core.prompts 에서 PromptTemplate를 이용해 직접 작성한다다.
from langchain_core.prompts import PromptTemplate

# output type을 str로만 나오게 한다 : 왜냐하면 state를 보면 query가 str이기 때문이다.
from langchain_core.output_parsers import StrOutputParser 


dictionary = ['사람과 관련된 표현 -> 거주자']

rewrite_prompt = PromptTemplate.from_template(f"""
사용자의 질문을 보고, 우리의 사전을 참고해서 사용자의 질문을 변경해주세요
사전: {dictionary}
질문: {{query}}
""")

def rewrite(state: AgentState) -> AgentState:
    query = state['query']
    rewrite_chain = rewrite_prompt | llm | StrOutputParser()

    response = rewrite_chain.invoke({'query': query}) #response 가 BaseMessage 이므로 putput type을 str로 바꿔줘야 한다.
    return {'query': response}


# 필요한 노드들을 다 만들었다.
# 이제 노드들을 등록하고 엣지를 추가해 주도록 하겠다.

graph_builder.add_node('retrieve', retrieve)
# graph_builder.add_node('check_doc_relevance', check_doc_relevance)
graph_builder.add_node('generate', generate)
graph_builder.add_node('rewrite', rewrite)

# 엣지 추가

from langgraph.graph import START, END

graph_builder.add_edge(START, "retrieve")
graph_builder.add_conditional_edges("retrieve", check_doc_relevance)
graph_builder.add_edge("rewrite", "retrieve")
graph_builder.add_edge("generate", END)

graph = graph_builder.compile() 

from IPython.display import Image, display

display(Image(graph.get_graph().draw_mermaid_png()))

initial_state = {'query': '연봉 5천만원 세금'}
print(f'initial_state == {initial_state}')
graph.invoke(initial_state)



