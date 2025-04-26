# agent_module.py

from langgraph.graph import StateGraph, START, END
from langchain_core.tools import tool
from langchain_ollama import ChatOllama, OllamaEmbeddings
from langchain_community.document_loaders import WebBaseLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma
from typing import TypedDict, List, Optional
import json

# ==================== 模型定义 ====================
model = ChatOllama(model="llama3.1:8b")

# ==================== 加载和分割知识库 ====================
loader = WebBaseLoader("https://lilianweng.github.io/posts/2023-06-23-agent/")
docs = loader.load()

text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
all_splits = text_splitter.split_documents(documents=docs)

# ==================== 创建向量数据库和检索器 ====================
local_embeddings = OllamaEmbeddings(model="nomic-embed-text")
vectorstore = Chroma.from_documents(documents=docs, embedding=local_embeddings)
retriever = vectorstore.as_retriever()

# ==================== 定义工具 ====================
@tool
def Knowledge_retriever(query: str) -> str:
    """查询本地知识库的工具。输入应为明确的问题或关键词。"""
    docs = retriever.get_relevant_documents(query)
    return "\n\n".join([f"{doc.page_content}" for doc in docs])

tools = [Knowledge_retriever]

# ==================== Agent 状态类型定义 ====================
class AgentState(TypedDict):
    input: str
    thoughts: List[str]
    tool_results: List[str]
    current_action: Optional[str]
    current_action_input: Optional[str]
    output: Optional[str]

# ==================== 状态节点定义 ====================
def think_node(state: AgentState) -> dict:
    prompt = f"""
    当前问题：{state['input']}
    已有信息：{state['tool_results'] or '无'}

    请生成JSON格式的响应:
    {{
        "thought": "思考过程",
        "action": "工具名称或FINISH",
        "action_input": "工具参数"
    }}
    """
    response = model.invoke(prompt).content
    try:
        decision = json.loads(response)
    except:
        decision = {"action": "FINISH", "thought": "无法解析输出"}
    
    return {
        "thoughts": state["thoughts"] + [decision.get("thought", "")],
        "current_action": decision.get("action", "FINISH"),
        "current_action_input": decision.get("action_input", "")
    }

def tool_node(state: AgentState) -> dict:
    result = "无效工具"
    action = state.get("current_action", "")
    action_input = state.get("current_action_input", "")
    
    if action == "knowledge_lookup":
        result = Knowledge_retriever(action_input)
    
    return {
        "tool_results": state["tool_results"] + [result],
        "current_action": None,
        "current_action_input": None
    }

def finalize_node(state: AgentState) -> dict:
    prompt = f"""
    根据以下信息生成最终答案：
    问题：{state['input']}
    检索结果：{state['tool_results']}
    思考过程：{" -> ".join(state['thoughts'])}
    """
    return {
        "output": model.invoke(prompt).content,
        "current_action": "FINISH"
    }

# ==================== 是否继续执行判断 ====================
def should_continue(state: AgentState):
    return state.get("current_action") not in ["FINISH", None]

# ==================== 构建 LangGraph 流程图 ====================
workflow = StateGraph(AgentState)
workflow.add_node("think", think_node)
workflow.add_node("act", tool_node)
workflow.add_node("finalize", finalize_node)

workflow.add_edge(START, "think")
workflow.add_edge("think", "act")
workflow.add_conditional_edges("act", should_continue, {
    True: "think",
    False: "finalize"
})
workflow.add_edge("finalize", END)

app = workflow.compile()

# ==================== 封装流式执行接口 ====================
def run_agent_stream(input_query: str):
    initial_state = {
        "input": input_query,
        "thoughts": [],
        "tool_results": [],
        "current_action": None,
        "current_action_input": None,
        "output": None
    }
    for event in app.stream(initial_state):
        yield event

# run_agent_stream(input_query="what is task decomposition?")