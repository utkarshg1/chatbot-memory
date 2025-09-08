from dotenv import load_dotenv
from langchain_groq import ChatGroq
from langchain_core.messages import HumanMessage
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import START, MessagesState, StateGraph

load_dotenv()
model = ChatGroq(model="llama-3.3-70b-versatile")

workflow = StateGraph(state_schema=MessagesState)


def call_model(state: MessagesState):
    response = model.invoke(state["messages"])
    return {"messages": response}


workflow.add_node("model", call_model)
workflow.add_edge(START, "model")

memory = MemorySaver()
app = workflow.compile(checkpointer=memory)

config = {"configurable": {"thread_id": "abc123"}}

# Turn 1
output = app.invoke(
    {"messages": [HumanMessage("Hi, I'm Alice and I'm a data scientist.")]}, config # type: ignore
)
print(output["messages"][-1].content)

# Turn 2 (model should recall Alice + data scientist)
output = app.invoke(
    {"messages": [HumanMessage("What's my name and occupation?")]}, config # type: ignore
)
print(output["messages"][-1].content)