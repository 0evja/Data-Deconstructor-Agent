# app.py

import streamlit as st
from agent_module import run_agent_stream

st.set_page_config(page_title="LangGraph Agent Demo", layout="wide")
st.title("🧠 LangGraph Agent：数据解构师")

if "history" not in st.session_state:
    st.session_state.history = []

query = st.text_input("请输入一个问题（建议英文，例如：What are the approaches to task decomposition?）", key="query")

if st.button("运行 Agent"):
    if query:
        with st.spinner("Agent 正在分析..."):
            placeholder = st.empty()
            thoughts = []
            tool_results = []
            output = None

            for state in run_agent_stream(query):
                if "thoughts" in state:
                    thoughts = state["thoughts"]
                if "tool_results" in state:
                    tool_results = state["tool_results"]
                if "output" in state and state["output"]:
                    output = state["output"]
                
                placeholder.markdown(f"""
                ### 🧠 思考过程:
                {' → '.join(thoughts)}

                ### 🔎 工具调用结果:
                {"<br><br>".join(tool_results)}
                """, unsafe_allow_html=True)

            st.success("✅ Agent 回答完毕")
            st.markdown("### 📝 最终回答：")
            st.markdown(output)
            st.session_state.history.append({"query": query, "output": output})
    else:
        st.warning("请输入问题后再点击运行。")

# 展示历史记录
if st.session_state.history:
    st.sidebar.markdown("### 📜 历史记录")
    for item in reversed(st.session_state.history):
        st.sidebar.markdown(f"**{item['query']}**\n\n{item['output']}\n\n---")
