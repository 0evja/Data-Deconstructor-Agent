from langchain_community.document_loaders import WebBaseLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma
from langchain_ollama import OllamaEmbeddings
from langchain_ollama import ChatOllama
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain.callbacks.streamlit import StreamlitCallbackHandler

#  =================== 加载文档 ======================
loader = WebBaseLoader("https://lilianweng.github.io/posts/2023-06-23-agent/")
data = loader.load()

# ================== 分割、嵌入存储 =====================
text_splitter = RecursiveCharacterTextSplitter(chunk_size = 1000, 
                                               chunk_overlap=200,
                                               add_start_index = True
                                               )

all_splits = text_splitter.split_documents(data)

local_embeddings = OllamaEmbeddings(model="nomic-embed-text")
vectorstore = Chroma.from_documents(documents=all_splits, embedding=local_embeddings)

# =================== 定义模型、函数，输出解释器===========================
model = ChatOllama(model="llama3.1:8b")

def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)

parser = StrOutputParser()

# ==================== 问答系统 ==========================
RAG_TEMPLATE = """
You are an assistant for question-answering tasks. Use the following pieces of retrieved context to answer the question.
If you don't know the answer, just say that you don't know. Use three sentences maximum and keep the answer concise.

<context>
{context}
</context>

Answer the following question:

{question}

"""
rag_prompt = ChatPromptTemplate.from_template(RAG_TEMPLATE)
retriever = vectorstore.as_retriever()

# ===================== 构造rag链 ==========================
qa_chain = (
    {"context": retriever | format_docs, "question": RunnablePassthrough()}
    | rag_prompt
    | model
    | parser
)

# ================== 构建agent智能体 ========================
from langchain.prompts import MessagesPlaceholder
from langchain.agents import Tool
from API_read import get_tavily_api
from langchain_community.tools import TavilySearchResults
from langgraph.checkpoint.memory import MemorySaver
from langgraph.prebuilt import create_react_agent
import os

tavily_api = get_tavily_api()
os.environ["TAVILY_API_KEY"] = tavily_api

# 将现有RAG系统封装为工具
rag_tool = Tool(
    name = "Knowledge_Base",
    func = qa_chain.invoke,
    description = """使用工具回答关于智能体系统和任务分解的问题。
    输入应为完整的自然语言问题。
    """
)

# # 定义tavily搜索引擎工具
# tavily_tool = TavilySearchResults(
#     max_results = 3
# )

# ========= 配置 Agent 系统 ===========
# 定义对话记忆
memory = MemorySaver()

# 定义Agent支持的工具列表
tools = [rag_tool]

# 配置Agent提示词
# agent_prompt = ChatPromptTemplate.from_messages([
#     ("system", """你是智能助手，可以访问一下工具:
     
#      {tools}

#      请遵守以下规则：
#      1. 当问题设计任务分解，智能体系统是，必须使用Knowledge_Base工具
#      2. 保持回答专业简洁
#      3. 如果工具返回不知道，请尝试自行回答"""),

#      ("user", "{input}"),
#      ("agent", "{agent_scratchpad}"),  # Escape the placeholder as regular text
#      ("tools", "{tools}") 
# ])

agent = create_react_agent(
    model = model,
    tools = tools,
    checkpointer = memory,
)

config1 = {"configurable": {"thread_id": "0evja"}}

# 使用Agent进行对话
questions = [
    "What is Task Decomposition?",
    "What according to the blog post are common ways of doing it? redo the search",
    "帮我写一个PyTorch实现代码"
]

for question in questions:
    print(f"Question: {question}")
    res = agent.invoke({"input": question}, config=config1)
    print(f"Answer: {res['messages'][-1].content}\n")  # 修正结果解析方式