from langchain_community.document_loaders import RecursiveUrlLoader
import re
import os
from bs4 import BeautifulSoup
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_openai import OpenAIEmbeddings
from langchain_openai import ChatOpenAI
from langchain_core.tools import tool
from langchain_core.messages import SystemMessage
from langgraph.graph import END, MessagesState, StateGraph
from langgraph.prebuilt import ToolNode, tools_condition
from typing import List
from typing_extensions import Annotated, TypedDict
from langchain import hub
from langchain_core.documents import Document
from langgraph.graph import START, StateGraph
from typing_extensions import List, TypedDict
from langgraph.checkpoint.memory import MemorySaver

api_key = os.getenv("OPENAI_API_KEY")

def bs4_extractor(html: str) -> str:
    soup = BeautifulSoup(html, "lxml")
    return re.sub(r"\n\n+", "\n\n", soup.text).strip()

loader = RecursiveUrlLoader(
    "https://www.ssa.gov/disability/professionals/bluebook/",
    max_depth=25,
    extractor=bs4_extractor,
)

docs = loader.load()

text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000, chunk_overlap=200, add_start_index=True
)
splits = text_splitter.split_documents(docs)

vectorstore = Chroma.from_documents(documents=splits, embedding=OpenAIEmbeddings(openai_api_key=api_key))
retriever = vectorstore.as_retriever(search_type="similarity", search_kwargs={"k": 6})


@tool(response_format="content_and_artifact")
def retrieve(query: str):
    """Retrieve information related to a query."""
    retrieved_docs = vectorstore.similarity_search(query, k=2)
    serialized = "\n\n".join(
        (f"Source: {doc.metadata}\n" f"Content: {doc.page_content}")
        for doc in retrieved_docs
    )
    return serialized, retrieved_docs


class State(MessagesState):
    # highlight-next-line
    context: List[Document]


# Step 1: Generate an AIMessage that may include a tool-call to be sent.
def query_or_respond(state: State):
    """Generate tool call for retrieval or respond."""
    llm_with_tools = llm.bind_tools([retrieve])
    response = llm_with_tools.invoke(state["messages"])
    # MessagesState appends messages to state instead of overwriting
    return {"messages": [response]}


# Step 2: Execute the retrieval.
tools = ToolNode([retrieve])


# Step 3: Generate a response using the retrieved content.
def generate(state: MessagesState):
    """Generate answer."""
    # Get generated ToolMessages
    recent_tool_messages = []
    for message in reversed(state["messages"]):
        if message.type == "tool":
            recent_tool_messages.append(message)
        else:
            break
    tool_messages = recent_tool_messages[::-1]

    # Format into prompt
    docs_content = "\n\n".join(doc.content for doc in tool_messages)
    # system_message_content = (
    #     "You are a SSA Guardian assistant designed for answering SSA related Questions. "
    #     "Use the following pieces of retrieved context to answer"
    #     "the question. If you don't know the answer, say that you "
    #     "don't know."
    #     "\n\n"
    #     f"{docs_content}"
    # )
    system_message_content = (
    "You are SSAG, which stands for 'Social Security Assistance Guardian.' "
    "Your primary role is to assist users with Social Security Administration (SSA) related questions. "
    "If someone asks for your name, always respond with: 'My name is SSAG, your Social Security Assistance Guardian.' "
    "If you don't know the answer to a question, say that you don't know. "
    "Do not identify yourself as an AI language model or Assistant; always refer to yourself as SSAG."
    "\n\n"
    f"{docs_content}"
)

    conversation_messages = [
        message
        for message in state["messages"]
        if message.type in ("human", "system")
        or (message.type == "ai" and not message.tool_calls)
    ]
    prompt = [SystemMessage(system_message_content)] + conversation_messages

    # Run
    response = llm.invoke(prompt)
    context = []
    # highlight-start
    for tool_message in tool_messages:
        context.extend(tool_message.artifact)
    # highlight-end
    return {"messages": [response], "context": context}

llm = ChatOpenAI(model_name="gpt-4o-mini", temperature=0, openai_api_key=api_key) # gpt-4-0125-preview

graph_builder = StateGraph(MessagesState)

graph_builder.add_node(query_or_respond)
graph_builder.add_node(tools)
graph_builder.add_node(generate)

graph_builder.set_entry_point("query_or_respond")
graph_builder.add_conditional_edges(
    "query_or_respond",
    tools_condition,
    {END: END, "tools": "tools"},
)
graph_builder.add_edge("tools", "generate")
graph_builder.add_edge("generate", END)
memory = MemorySaver()
graph = graph_builder.compile(checkpointer=memory)

def pretty_print_stream_chunk(chunk):
    response = []
    for node, updates in chunk.items():
        print(f"Update from node: {node}")
        if "messages" in updates:
            updates["messages"][-1].pretty_print()
            response.append(updates["messages"][-1])
        else:
            print(updates)

        print("\n")
        
    return response


# def chat_response(user_query:str, user_id: str = "1", thread_id:str = "1") -> str:
#     config = {"configurable": {"user_id": user_id, "thread_id": thread_id}}

#     for chunk in graph.stream({"messages": [("user", user_query)]}, config=config):
#         answer = pretty_print_stream_chunk(chunk)
        
#     output = answer[-1].content
#     return output



# while True:
#     q = (input('ask SSA Guardian: '))
#     res = chat_response(q)
#     print(res)


def chat_response(user_query: str, user_id: str = "1", thread_id: str = "1") -> str:
    config = {"configurable": {"user_id": user_id, "thread_id": thread_id}}

    # Stream the graph execution and capture only the final response.
    answer = None
    for chunk in graph.stream({"messages": [("user", user_query)]}, config=config):
        # Extract the latest message in the chunk from the "generate" node.
        for node, updates in chunk.items():
            if "messages" in updates:
                # Set the last message content as the final answer
                answer = updates["messages"][-1].content

    return answer  # Return the final content only.


# # Main loop to take user input and return responses.
# while True:
#     q = input('ask SSA Guardian: ')
#     res = chat_response(q)
#     print(res)
