from langchain_community.tools import TavilySearchResults
from langchain_openai import ChatOpenAI
from API_read import get_base_url, get_openai_key, get_tavily_api, get_langsmith_api
from langchain_core.messages import HumanMessage, SystemMessage, AIMessage
from langchain_core.output_parsers import StrOutputParser
from langgraph.prebuilt import create_react_agent
from langgraph.checkpoint.memory import MemorySaver
import os

tavily_api = get_tavily_api()
langsmith_api = get_langsmith_api()
os.environ["TAVILY_API_KEY"] = tavily_api
os.environ["LANGCHAIN_TRACING_V2"] = "true"
os.environ["LANGCHAIN_API_KEY"] = langsmith_api

# 定义tavily工具，创建实例
tool = TavilySearchResults(
    max_results = 1,
    search_depth = "advanced",
    include_answer = True,
    include_raw_content = True,
    include_images = True,
)
# 定义工具列表
tools = [tool]

# 定义输出解释器
parser = StrOutputParser()

# 直接调用工具返回结果
# res = tool.invoke({"query": "What happened at the last wimbledon"})
# print(res)


# 也可以通过tool_call调用
# model_generated_tool_call = {
#     "args": {"query": "euro 2024 host nation"},
#     "id": "1",
#     "name": "tavily",
#     "type": "tool_call"
# }

# tool_msg = tool.invoke(model_generated_tool_call)
# print(tool_msg.content[:100])


# ============= 通过调用工具来使用语言模型 =================
model = ChatOpenAI(base_url=get_base_url(), api_key=get_openai_key())
# chain1 = model | parser
# res = chain1.invoke([HumanMessage(content="Hi!")])
# print(res)

# 让模型来学习工具
model_with_tools = model.bind_tools(tools)

# 模型调用工具输出，并显示
# res2 = model_with_tools.invoke([HumanMessage(content="Hi!")])
# print(f"Contentstring: {res2.content}")
# print(f"Toolcalls: {res2.tool_calls}")

# res3 = model_with_tools.invoke([HumanMessage(content="What is the weather in SF?")])
# print(f"Contentstring: {res3.content}")
# print(f"Toolcalls: {res3.tool_calls}")

# ===================================== 创建代理 ======================================

# 使用langgraph来构建代理，注意传入的model, 因为create_react_agent在后台会为我们调用.bind_tools
# agent_executor = create_react_agent(model, tools)

# 运行代理
# 未调用工具
# res = agent_executor.invoke({"messages": [HumanMessage(content="hi!")]})
# print(res)
# print(res["messages"][1])

# # 调用工具
# res1 = agent_executor.invoke(
#     {"messages": [HumanMessage(content="What is the weather is sf?")]}
# ) 
# print(res1["messages"])

# # 流式消息
# for chunk in agent_executor.stream(
#     {"messages": [HumanMessage(content="What is the weather in sf?")]}
# ):
#     print(chunk)
#     print("----")


# =============================== 添加内存 =====================================
# 如前所述，该代理是无记忆的，为了给他添加内存，我们传入一个检查点
# 同时我们还必须在调用代理是传入thread_id 以便它知道从哪个线程/对话恢复
memory = MemorySaver()
agent_executor = create_react_agent(model, tools, checkpointer=memory)

config1 = {"configurable": {"thread_id": "0evja"}}

for chunk in agent_executor.stream(
    {"messages": [HumanMessage(content="hi, i am jim")]},
    config = config1,
):
    print(chunk)
    print("------")

for chunk in agent_executor.stream(
    {"messages": [HumanMessage(content="What is my name?")]},
    config = config1,
):
    print(chunk)
    print("-----")

config2 = {"configurable": {"thread_id": "123"}}
for chunk in agent_executor.stream(
    {"messages": [HumanMessage(content="What is my name?")]},
    config = config2
):
    print(chunk)
    print("---------")
