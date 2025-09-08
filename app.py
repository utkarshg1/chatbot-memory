import streamlit as st
from dotenv import load_dotenv
from langchain_groq import ChatGroq
from langchain_core.messages import HumanMessage
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import START, MessagesState, StateGraph

# Load .env (expects GROQ_API_KEY)
load_dotenv()


# --- 1. Cache the compiled graph + model ---
@st.cache_resource
def get_app():
    # Initialize model (no streaming since we want final responses)
    model = ChatGroq(model="llama-3.3-70b-versatile") # type: ignore

    # Build workflow
    workflow = StateGraph(state_schema=MessagesState)

    def call_model(state: MessagesState):
        response = model.invoke(state["messages"])
        return {"messages": response}

    workflow.add_node("model", call_model)
    workflow.add_edge(START, "model")

    # Attach memory (in-process)
    memory = MemorySaver()

    return workflow.compile(checkpointer=memory)


# Get cached app (graph + model)
app = get_app()

# --- 2. Streamlit UI ---
st.title("💬 ChatGroq + LangGraph + Memory (cached)")

# Thread id (could expose in sidebar for multi-thread chat)
config = {"configurable": {"thread_id": "chat-session"}}

# Fetch conversation state from memory
state = app.get_state(config)  # type: ignore
messages = state.values.get("messages", [])

# Replay previous messages
for msg in messages:
    role = "user" if msg.type == "human" else "assistant"
    with st.chat_message(role):
        st.markdown(msg.content)

# Input box
if user_input := st.chat_input("Ask something..."):
    # Show user input
    with st.chat_message("user"):
        st.markdown(user_input)

    # Invoke app with new human message
    output = app.invoke({"messages": [HumanMessage(user_input)]}, config)  # type: ignore
    ai_message = output["messages"][-1]

    # Show assistant reply
    with st.chat_message("assistant"):
        st.markdown(ai_message.content)
