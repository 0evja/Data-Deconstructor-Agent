import streamlit as st
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

# 初始化模型和知识库（只需一次）
@st.cache_resource
def init_agent():
    # ==================== 定义模型 ================
    model = ChatOllama(model="llama3.1:8b")

    # ================ 加载知识库文档 ============
    loader = WebBaseLoader("https://lilianweng.github.io/posts/2023-06-23-agent/")
    docs = loader.load()

    # ================ 分割文档 ==================
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    all_splits = text_splitter.split_documents(documents=docs)

    # ================= 创建向量存储 ==============
    local_embeddings = OllamaEmbeddings(model="nomic-embed-text")
    vectorstore = Chroma.from_documents(documents=docs, embedding=local_embeddings)
    retriever = vectorstore.as_retriever()

    # ================== 创建工具 ================
    @tool 
    def Knowledge_retriever(query: str) -> str:
        docs = retriever.get_relevant_documents(query)
        return "\n\n".join([f"{doc.page_content}" for doc in docs])

    tools = [Knowledge_retriever]

    # ============ 创建执行图 ====================
    class AgentState(TypedDict):
        input: str
        thoughts: List[str]
        tool_results: List[str]
        current_action: Optional[str]
        current_action_input: Optional[str]
        output: Optional[str]

    def should_continue(state: AgentState):
        return state.get("current_action") not in ["FINISH", None]

    def think_node(state: AgentState) -> dict:
        prompt = f"""..."""  # 保持原有prompt
        response = model.invoke(prompt).content
        try:
            decision = json.loads(response)
        except:
            decision = {"action": "FINISH"}
        
        # 记录思考过程
        st.session_state.thoughts.append(decision.get("thought", ""))
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
            # 记录工具调用
            st.session_state.tool_calls.append({
                "action": action,
                "input": action_input,
                "result": result[:200] + "..."  # 显示部分结果
            })
        
        return {
            "tool_results": state["tool_results"] + [result],
            "current_action": None,
            "current_action_input": None
        }

    def finalize_node(state: AgentState) -> dict:
        prompt = f"""..."""  # 保持原有prompt
        st.session_state.final_output = model.invoke(prompt).content
        return {"output": st.session_state.final_output}

    # 构建执行图
    workflow = StateGraph(AgentState)
    workflow.add_node("think", think_node)
    workflow.add_node("act", tool_node)
    workflow.add_node("finalize", finalize_node)
    
    workflow.add_edge(START, "think")
    workflow.add_edge("think", "act")
    workflow.add_conditional_edges(
        "act",
        should_continue,
        {True: "think", False: "finalize"}
    )
    workflow.add_edge("finalize", END)
    return workflow.compile()

# Streamlit界面
def main():
    st.title("智能知识库问答系统")
    st.write("基于LangGraph和Ollama的Agent系统")

    # 初始化session状态
    if 'thoughts' not in st.session_state:
        st.session_state.thoughts = []
    if 'tool_calls' not in st.session_state:
        st.session_state.tool_calls = []
    if 'final_output' not in st.session_state:
        st.session_state.final_output = None

    # 问题输入
    question = st.text_input("请输入您的问题：", key="input")
    submit_button = st.button("提交查询")

    # 初始化代理
    app = init_agent()

    if submit_button and question:
        # 重置session状态
        st.session_state.thoughts = []
        st.session_state.tool_calls = []
        st.session_state.final_output = None

        # 执行代理
        initial_state = {
            "input": question,
            "thoughts": [],
            "tool_results": [],
            "current_action": None,
            "current_action_input": None,
            "output": None
        }
        app.invoke(initial_state)

    # 显示处理过程
    with st.expander("思考过程", expanded=True):
        if st.session_state.thoughts:
            for i, thought in enumerate(st.session_state.thoughts, 1):
                st.markdown(f"**Step {i}:** {thought}")
        else:
            st.write("等待问题输入...")

    with st.expander("工具调用记录"):
        if st.session_state.tool_calls:
            for call in st.session_state.tool_calls:
                st.markdown(f"""
                **工具名称**: `{call['action']}`  
                **输入参数**: `{call['input']}`  
                **执行结果**:  
                {call['result']}
                """)
        else:
            st.write("暂无工具调用记录")

    # 显示最终结果
    if st.session_state.final_output:
        st.success("### 最终答案")
        st.markdown(st.session_state.final_output)

if __name__ == "__main__":
    main()