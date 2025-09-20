import streamlit as st
from langchain_groq import ChatGroq
from langchain_core.messages import HumanMessage
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import START, MessagesState, StateGraph
import uuid
import os

# Load the GROQ_API_KEY in enviroment variable
os.environ["GROQ_API_KEY"] = st.secrets["GROQ_API_KEY"]


# core logic of chatbot
@st.cache_resource  # Ensures model + graph are created once and reused across reruns
def get_app():
    # Initalzie model
    model = ChatGroq(model="llama-3.3-70b-versatile")

    # Create a new langgraph workflow to store messages
    workflow = StateGraph(state_schema=MessagesState)

    # Define how model is called inside workflow
    def call_model(state: MessagesState):
        response = model.invoke(state["messages"])
        return {"messages": response}

    # Add the model to workflow
    workflow.add_edge(START, "model")
    workflow.add_node("model", call_model)

    # Add a memory saver
    memory = MemorySaver()

    # Compile the workflow
    return workflow.compile(checkpointer=memory)


# Get the app
app = get_app()

# --- Streamlit UI ----
st.title("Llama 3.3 Chatbot")

# intialize a conversational thread_id (this is responsible to show message history)
if "thread_id" not in st.session_state:
    st.session_state.thread_id = str(uuid.uuid4())

# Create a sidebar to start new conversation
with st.sidebar:
    st.header("Chat Controls")
    if st.button("New Chat", type="primary"):
        st.session_state.thread_id = str(uuid.uuid4())
        st.rerun()

# Thread configuration (tells LangGraph which conversation thread to use)
config = {"configurable": {"thread_id": st.session_state.thread_id}}

# retireve the stored messages
state = app.get_state(config)
messages = state.values.get("messages", [])

# Show the previous conversation
for msg in messages:
    role = "user" if msg.type == "human" else "assitant"
    with st.chat_message(role):
        st.markdown(msg.content)

# Show a chat message box
if user_input := st.chat_input("Ask your query here "):
    with st.chat_message("user"):
        st.markdown(user_input)

    # send human message throught the langgraph app
    output = app.invoke({"messages": [HumanMessage(user_input)]}, config)
    ai_message = output["messages"][-1]

    # display assistant response
    with st.chat_message("assistant"):
        st.markdown(ai_message.content)

