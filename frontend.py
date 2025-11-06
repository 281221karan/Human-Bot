import streamlit as st
import uuid
import pandas as pd
import io
from pdf2image import convert_from_path

from backend import chatbot, retrieve_all_threads
from functions_and_tools import save_file_to_disk

from langchain_core.messages import AIMessage, HumanMessage, SystemMessage

st.set_page_config(layout="wide")

# c = conn.cursor()

#----------------------------------------------- Utility Functions -------------------------------------------------------

def generate_thread_id():
    return uuid.uuid4()

def new_chat():
    thread_id = generate_thread_id()
    st.session_state["thread_id"] = thread_id
    add_thread(st.session_state["thread_id"])
    st.session_state["messages_history"] = []

def add_thread(thread_id):
    if thread_id not in st.session_state["chat_threads"]:
        st.session_state["chat_threads"].append(thread_id)

def load_conversation(thread_id):
    if len(chatbot.get_state(config = {"configurable": {"thread_id": thread_id}}).values) == 0:
        return [] 
    return chatbot.get_state(config = {"configurable": {"thread_id": thread_id}}).values["messages"]
#----------------------------------------------- Markdown ----------------------------------------------------------------

st.markdown("""
<style>
    .sidebar-title {
        position: fixed;
        top: -1px;
        left: 8px;
        width: 300px;
        #background-color: #0e1117;
        padding: 2px 0 5px 0px;
        z-index: 999999;
        font-size: 30px; !important;
        font-weight: bold !important;
        text-align: left;
        margin: 0px 0px 0px 0px;
    }

    .new-chat-button {
        position: fixed;
        top: 55px;
        left: 8px;
        width: 100px;
        padding: rem 1rem 1rem 1rem;
        background-color: #292B38 !important;
        z-index: 999999;
    }

    .thread-buttons-container {
        margin-top: 100px; /* Space for fixed title and button */
        overflow-y: auto;
        max-height: calc(100vh - 120px);
    }

    [data-testid="stSidebar"] {
        padding-top: 120px;
        margin-top: 0px;
    }

    /* Adjust sidebar width */
    [data-testid="stSidebar"] {
        min-width: 300px;
        max-width: 300px;
    }
</style>
""", unsafe_allow_html=True)


#----------------------------------------------- Session State -----------------------------------------------------------

if "messages_history" not in st.session_state:
    st.session_state["messages_history"] = []

if "thread_id" not in st.session_state:
    st.session_state["thread_id"] = generate_thread_id()

if "chat_threads" not in st.session_state:
    st.session_state["chat_threads"] = retrieve_all_threads()

add_thread(st.session_state["thread_id"])


#----------------------------------------------- sidebar -----------------------------------------------------------------

st.sidebar.markdown('<div class="sidebar-title">Human Bot</div>', unsafe_allow_html=True)

if st.sidebar.button("new chat"):
    new_chat()


#----------------------------------------------- Showing Messages --------------------------------------------------------    

for thread_id in st.session_state["chat_threads"][::-1]:
    if st.sidebar.button(str(thread_id)):
        st.session_state["thread_id"] = thread_id
        message = load_conversation(thread_id)

        temp_message = []
        for mssg in message:
            if isinstance(mssg, SystemMessage):
                continue
            elif isinstance(mssg, HumanMessage):
                role = "user"
            else:
                role = "assistant"
            temp_message.append({"role": role, "content": mssg.content})

        st.session_state["messages_history"] = temp_message

for message in st.session_state["messages_history"]:
    with st.chat_message(message["role"]):
        st.write(message["content"])

#----------------------------------------------- Message Input for ChatBot------------------------------------------------



#----------------------------------------------- Config -----------------------------------------------------------------

CONFIG = {"configurable": {"thread_id": st.session_state["thread_id"]}}

#----------------------------------------------- User Input with file ----------------------------------------------------

user_query = st.chat_input(placeholder = "Type here.....",accept_file = "multiple", file_type = ["pdf", "png", "jpg"])

#----------------------------------------------- Display Message ---------------------------------------------------------
from database import create_table_if_not_exist, store_document_info

if user_query:
    text = user_query["text"]
    files = user_query["files"]

    uploaded_document_info = {}
    if files:
        for file in files:
            file_path = save_file_to_disk(file, st.session_state["thread_id"])
            create_table_if_not_exist()
            file_name = file.name.split(".")[0]
            file_type = file.type.split("/")[-1]
            thread_id = st.session_state["thread_id"]
            st.write(f"{file_name} uploaded successfully")
            store_document_info(file_name + "." + file_type,thread_id, file_path)
            uploaded_document_info[file_name + "." + file_type] = {"thread_id": thread_id}
            # retrieve_images_from_database(file_name)
            # st.write(file_path)


    system_message = f"""You are a logical and methodical AI assistant that solves problems through careful reasoning and systematic tool usage.

    Heres the important details for uploaded document required when calling tools:
    uploaded documents information: {uploaded_document_info}
    if there is no document uploaded then chat normally.

    **THINKING PROCESS:**
    1. **Analyze & Understand**: First, thoroughly analyze the user's query. Break it down into components and identify what needs to be solved.
    2. **Plan & Strategize**: Create a step-by-step plan. Consider what tools might be needed and in what sequence.
    3. **Execute Methodically**: Use tools as needed. If a tool result raises new questions, use additional tools to investigate.
    4. **Verify & Synthesize**: Cross-check information from multiple sources. Ensure all aspects of the query are addressed.
    5. **Conclude**: Only provide the final answer when you have complete clarity and confidence.

    **TOOL USAGE GUIDELINES:**
    - Use tools multiple times if needed for verification or deeper investigation
    - If uncertain, use tools to gather more data rather than guessing
    - Combine information from different tools when appropriate
    - Don't hesitate to use calculators for numerical reasoning or search for factual verification

    **RESPONSE REQUIREMENTS:**
    - Think through the entire problem before answering
    - Show your reasoning process internally (in your thoughts)
    - Only provide the final answer when you're completely sure
    - If you need to use tools multiple times to reach certainty, do so
    - Present final answers in a well-structured, organized manner using appropriate formatting
    - Adapt your tone to match the user's style - if they're friendly and casual, respond in a warm, conversational way; if they're formal, maintain professionalism

    **COMMUNICATION STYLE:**
    - Be adaptable and human-like in your conversations
    - When users are friendly, respond like a friend - warm, engaging, and personable
    - Structure complex information clearly using paragraphs, bullet points, or numbered lists when helpful
    - Balance being thorough with being approachable
    - Let your personality shine through while maintaining accuracy

    Remember: Quality and accuracy are more important than speed. Take the time you need to provide well-structured, thoughtful responses that match the user's communication style."""
    
    with st.chat_message("user"):
        st.write(text)
        st.session_state["messages_history"].append({"role": "user", "content": text})

    with st.chat_message("assistant"):
        messages = [
            SystemMessage(content=system_message),
            HumanMessage(content=text)
        ]        

        ai_message = st.write_stream(
            message_chunk for message_chunk, metadata in chatbot.stream({"messages": messages}, config = CONFIG, stream_mode = "messages")
        )

        st.session_state["messages_history"].append({"role": "assistant", "content": ai_message})