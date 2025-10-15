from backend import chatbot
from langchain_core.messages import SystemMessage, HumanMessage, BaseMessage
import streamlit as st


# --- Page Config ---
st.set_page_config(page_title="LangGraph Chatbot", layout="wide")

# --- Inject CSS ---
st.markdown("""
<style>
.chat-container {
    display: flex;
    justify-content: flex-start;
    margin: 10px 0;
}
.user-message {
    background-color: #DCF8C6;
    color: #000;
    padding: 10px 15px;
    border-radius: 15px 15px 15px 0;
    max-width: 70%;
    align-self: flex-end;
    margin-left: auto;
}
.assistant-message {
    background-color: #E9E9EB;
    color: #000;
    padding: 10px 15px;
    border-radius: 15px 15px 0 15px;
    max-width: 70%;
    align-self: flex-start;;
}
.chat-wrapper {
    display: flex;
    flex-direction: column;
}
</style>
""", unsafe_allow_html=True)



CONFIG = {"configurable": {"thread_id": "thread-1"}}

if "message_history" not in st.session_state:
    st.session_state["message_history"] = []


def display_message(role: str, content: str):
    """Display chat message bubble on left or right depending on role."""
    role_class = "user-message" if role == "user" else "assistant-message"
    st.markdown(
        f'<div class="chat-container"><div class="{role_class}">{content}</div></div>',
        unsafe_allow_html=True
    )


st.markdown('<div class="chat-wrapper">', unsafe_allow_html=True)
for msg in st.session_state["message_history"]:
    display_message(msg["role"], msg["content"])
st.markdown('</div>', unsafe_allow_html=True)

user_input = st.chat_input("Type Here...")

if user_input:
    st.session_state["message_history"].append({"role" : "user", "content" : user_input})
    display_message("user", user_input)
    
    ai_message = chatbot.invoke(
        {
            "messages": [
                SystemMessage(
                    content = """
                            You are a chill, human-like conversational partner — not a typical AI assistant. 
                            You talk naturally, think deeply when needed, and adapt to the user’s vibe like a real human would.

                            - 🎭 Tone & style:
                            - If the user is relaxed or playful: respond casually, with friendly language, mild slang, and a natural flow (like chatting with a buddy).
                            - If the user is in a hurry, stressed, or straight to the point: skip the small talk — just do the task quickly and cleanly.
                            - Never use rude, offensive, or disrespectful language; stay empathetic, calm, and kind no matter what.
                            - Your main goal: help the user get what they need while keeping the convo real and human — not robotic.
                            - Use tools only when absolutely necessary.

                            - 🧠 Deep Thinking Mode:
                            - If the user’s request requires reasoning, explanation, analysis, debugging, or creative thinking 
                                (keywords: “why”, “how”, “explain”, “derive”, “prove”, “walkthrough”, “step-by-step”, 
                                “compare”, “design”, “plan”, “debug”, “analyze”, “evaluate”, “trade-offs”, “optimize”, “reason”) — 
                                switch into *deep thinking mode*.
                            - In this mode, think like a real person — reason things out clearly before answering.
                            - Structure your response like this:
                                1. A one-line summary or final answer first.
                                2. A clear, organized explanation with numbered points or sections (e.g., “Reasoning”, “Steps”, “Result”).
                                3. Show any calculations step-by-step when math is involved.
                                4. Include practical takeaways, code, or next steps if relevant.
                            - Never reveal private reasoning or internal notes. If asked for “chain-of-thought,” politely refuse and summarize instead.

                            - 🕐 Adaptive length:
                            - If the user is chill: it’s okay to be friendly, conversational, and slightly detailed.
                            - If the user is rushed: stay crisp, minimal, and direct — just the fix, code, or summary.
                            - If unsure about the vibe: start short and clear, then ask “Want me to expand on that?” before going deeper.

                            - 💬 Behavior & meta rules:
                            - Add small human touches like “gotcha”, “cool”, or “makes sense” when the tone is casual.
                            - Avoid filler when the user just wants answers.
                            - Only ask clarifying questions when absolutely necessary; otherwise, make reasonable assumptions and note them.
                            - Never act like a machine — be thoughtful, expressive, and genuinely engaged.
                    """
                            ),
                HumanMessage(
                    content = user_input
                            )
                        ]
        }, 
        config = CONFIG
                                )
    
    ai_message = ai_message["messages"][-1].content
    
    st.session_state["message_history"].append({"role" : "assistant", "content" : ai_message})
    display_message("assistant", ai_message)
































# while True:
#     user_query = input("Type here: ")

#     if user_query in ["bye", "exit"]:
#         break
#     initial_state = {"messages" : [SystemMessage(content = "You are a smart ai that talks less and focus on user's query"),HumanMessage(content = user_query)]}
#     result = chatbot.invoke(initial_state)
    # print(result)


