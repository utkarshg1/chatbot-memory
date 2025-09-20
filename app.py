import streamlit as st
from langchain_groq import ChatGroq
from langchain_core.messages import HumanMessage
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import START, MessagesState, StateGraph
import uuid
import os

# Load environment variables from .env (expects GROQ_API_KEY to be set)
os.environ["GROQ_API_KEY"] = st.secrets["GROQ_API_KEY"]


# --- 1. Cache the compiled graph + model ---
@st.cache_resource  # Ensures model + graph are created once and reused across reruns
def get_app():
    # Initialize Groq LLM (non-streaming mode for full responses only)
    model = ChatGroq(model="llama-3.3-70b-versatile")  # type: ignore

    # Create a LangGraph workflow with a MessagesState schema
    workflow = StateGraph(state_schema=MessagesState)

    # Define how the model is called inside the workflow
    def call_model(state: MessagesState):
        # Take conversation history, send it to the model, return its reply
        response = model.invoke(state["messages"])
        return {"messages": response}

    # Add model node to workflow
    workflow.add_edge(START, "model")  # Start → model
    workflow.add_node("model", call_model)

    # Attach in-memory conversation memory (per session)
    memory = MemorySaver()

    # Compile workflow into an executable app
    return workflow.compile(checkpointer=memory)


# Get cached app (LangGraph workflow + Groq model)
app = get_app()


# --- 2. Streamlit UI ---
st.title("💬 ChatGroq + LangGraph + Memory (cached)")


# Initialize a conversation thread ID (persists across Streamlit reruns)
if "thread_id" not in st.session_state:
    st.session_state.thread_id = str(uuid.uuid4())


# Sidebar: allow user to reset conversation
with st.sidebar:
    st.header("Chat Controls")
    if st.button("New Chat", type="primary"):
        # Reset with a new thread ID → starts fresh memory state
        st.session_state.thread_id = str(uuid.uuid4())
        st.rerun()


# Thread configuration (tells LangGraph which conversation thread to use)
config = {"configurable": {"thread_id": st.session_state.thread_id}}

# Retrieve stored state (messages) from memory for this thread
state = app.get_state(config)  # type: ignore
messages = state.values.get("messages", [])


# Replay past conversation so UI shows history
for msg in messages:
    role = "user" if msg.type == "human" else "assistant"
    with st.chat_message(role):
        st.markdown(msg.content)


# Chat input box
if user_input := st.chat_input("Ask something..."):
    # Show user input immediately in chat
    with st.chat_message("user"):
        st.markdown(user_input)

    # Send new human message into LangGraph workflow
    output = app.invoke({"messages": [HumanMessage(user_input)]}, config)  # type: ignore
    ai_message = output["messages"][-1]  # Get the assistant's latest reply

    # Display assistant's response
    with st.chat_message("assistant"):
        st.markdown(ai_message.content)
