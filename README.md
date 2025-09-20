# Overall architecture

```text
 ┌────────────────────────────────────────────────────┐
 │                    Streamlit UI                    │
 │  - st.chat_input                                   │
 │  - st.chat_message                                 │
 │  - session_state (thread_id)                       │
 └─────────────────────────┬──────────────────────────┘
                           │
                           ▼
 ┌────────────────────────────────────────────────────┐
 │                    LangGraph                       │
 │  - StateGraph (workflow: START → model)            │
 │  - MessagesState (stores all chat turns)           │
 │  - MemorySaver (checkpoint per thread_id)          │
 └─────────────────────────┬──────────────────────────┘
                           │
                           ▼
 ┌────────────────────────────────────────────────────┐
 │                     ChatGroq                       │
 │   - Model: llama-3.3-70b-versatile                 │
 │   - Generates assistant replies from messages      │
 └────────────────────────────────────────────────────┘
```

# Conversational cycle per message 
```text
 User types something
        │
        ▼
┌────────────────────────────┐
│ Streamlit: capture input   │
│   st.chat_input            │
└──────────────┬─────────────┘
               │
               ▼
┌────────────────────────────┐
│ LangGraph: update state    │
│   - Adds HumanMessage      │
│   - Passes to workflow     │
└──────────────┬─────────────┘
               │
               ▼
┌────────────────────────────┐
│ Node "model": call_model() │
│   - Sends messages → LLM   │
│   - Receives response      │
└──────────────┬─────────────┘
               │
               ▼
┌────────────────────────────┐
│ MemorySaver: checkpoint    │
│   - Stores conversation    │
│   - Linked by thread_id    │
└──────────────┬─────────────┘
               │
               ▼
┌────────────────────────────┐
│ Streamlit: render reply    │
│   st.chat_message("assistant") │
└────────────────────────────┘
```