from langchain_core.tools import tool
from langgraph.prebuilt import create_react_agent
from langchain_core.messages import SystemMessage
from langgraph.prebuilt.chat_agent_executor import AgentState
from langchain_core.prompts import ChatPromptTemplate
from langgraph.checkpoint.memory import MemorySaver
from langchain_ollama import ChatOllama
from langchain_ollama import OllamaEmbeddings
from langchain_community.document_loaders import TextLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma
from typing import Dict, TypedDict, List, Optional
import json
from langgraph.graph import START, StateGraph, END
from langchain_community.document_loaders import WebBaseLoader


# ==================== 定义模型 ================
model = ChatOllama(model="llama3.1:8b")

# ================ 加载本地知识库文档 ===========
loader = WebBaseLoader("https://lilianweng.github.io/posts/2023-06-23-agent/")
docs = loader.load()

# ================ 分割文档 ====================
text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
all_splits = text_splitter.split_documents(documents=docs)

# ================= 创建向量存储 ================
local_embeddings = OllamaEmbeddings(model="nomic-embed-text")
vectorstore = Chroma.from_documents(documents=docs, embedding=local_embeddings)

# ================== 创建检索器 ==================
retriever = vectorstore.as_retriever()

# ================== 创建工具 ====================
# 定义检索器工具 
@tool 
def Knowledge_retriever(query: str) -> str:
    """当需要查询本地知识库是使用此工具，输入应为明确的搜索关键词或问题。"""
    docs = retriever.get_relevant_documents(query)
    return "\n\n".join([f"{doc.page_content}" for doc in docs])

# 定义工具列表
tools = [Knowledge_retriever]

# ============ 创建langgraph工具流 ===========================

# 修正后的状态定义
class AgentState(TypedDict):
    input: str
    thoughts: List[str]  # 明确列表类型
    tool_results: List[str]
    current_action: Optional[str]  # 新增动作状态字段
    current_action_input: Optional[str]
    output: Optional[str]

# 修正后的决策函数
def should_continue(state: AgentState):
    return state.get("current_action") not in ["FINISH", None]

# 修正后的思考节点
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
        decision = {"action": "FINISH"}
    
    # 只更新需要修改的字段，保留其他状态
    return {
        "thoughts": state["thoughts"] + [decision.get("thought", "")],
        "current_action": decision.get("action", "FINISH"),
        "current_action_input": decision.get("action_input", "")
    }

# 修正后的工具节点
def tool_node(state: AgentState) -> dict:
    result = "无效工具"
    action = state.get("current_action", "")
    action_input = state.get("current_action_input", "")
    
    if action == "knowledge_lookup":
        result = Knowledge_retriever(action_input)
    
    # 清空当前动作状态
    return {
        "tool_results": state["tool_results"] + [result],
        "current_action": None,
        "current_action_input": None
    }

# 修正后的最终节点
def finalize_node(state: AgentState) -> dict:
    prompt = f"""
    根据以下信息生成最终答案：
    问题：{state['input']}
    检索结果：{state['tool_results']}
    思考过程：{" -> ".join(state['thoughts'])}
    """
    return {
        "output": model.invoke(prompt).content,
        "current_action": "FINISH"  # 确保状态终结
    }

# ================ 构建执行图 ====================
workflow = StateGraph(AgentState)
# 构建节点
workflow.add_node("think", think_node)
workflow.add_node("act", tool_node)
workflow.add_node("finalize", finalize_node)

# 构建连接节点的边
workflow.add_edge(START, "think")
workflow.add_edge("think", "act")
workflow.add_conditional_edges(
    "act",
    should_continue,
    {True: "think", False: "finalize"}
)
workflow.add_edge("finalize", END)

app = workflow.compile()

initial_state = {
    "input": "what is task decomposition",
    "thoughts": [],
    "tool_results": [],
    "current_action": None,
    "current_action_input": None,
    "output": None
}

result = app.invoke(initial_state)

print("最终输出:", result["output"])

























# # ==================== 创建提示词 ====================
# system_message = "你是一个企知识助手，请严格根据工具提供的信息回答问题:.\n\n{input}"
# agent_prompt = ChatPromptTemplate.from_messages(system_message)

# # ==================== 创建代理 ======================
# agent = create_react_agent(
#     model=model,
#     tools=tools,
#     prompt=agent_prompt
# )


# # 查询
# query = "what is the value of magic_function(3)?"
# new_query = "Pardon?"



# """
# 加入任意函数
# prompt = ChatPromptTemplate.from_messages(
#     [
#         ("system", "You are a helpful assistant. Respond only in Spanish."),
#         ("placeholder", "{messages}"),
#     ]
# )
# def _modify_state_messages(state: AgentState):
#     return prompt.invoke({"messages": state["messages"]}).to_messages() + [("user", "Also say Pandamonium! after the answer.")]

# app = create_react_agent(model, tools, state_modifier=_modify_state_messages)
# # res = app.invoke({"messages": [("human", query)]})
# # print(res)
# # print(res["messages"][-1].content)

# """

# memory = MemorySaver()
# app = create_react_agent(
#     model = model,
#     tools = tools,
#     state_modifier = system_message,
#     checkpointer = memory,
# )

# config = {"configurable": {"thread_id": "0evja"}}

# res = app.invoke({"messages": [("user", "Hi I am polly! What's the output of magic_function of 3?")]},
#                  config=config)
# print(res["messages"][-1].content)
# print("-----")
# res1 = app.invoke({"messages": [("user", "Remeber my name?")]},
#                  config=config)
# print(res1["messages"][-1].content)
# print("-----")
# res2 = app.invoke({"messages": [("user", "What was that output again?")]},
#                  config=config)
# print(res2["messages"][-1].content)
# print("-----")