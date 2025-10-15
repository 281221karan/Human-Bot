from backend import chatbot, retrieve_all_threads
from langchain_core.messages import SystemMessage, HumanMessage, AIMessage, ToolMessage
import streamlit as st
import uuid
import json
import pickle
import os
from PIL import Image
import io
import tempfile
import pdf2image
from PIL import Image, ImageDraw, ImageFont
from pdf2image import convert_from_path

# --- Page Config ---
st.set_page_config(page_title="LangGraph Chatbot", layout="wide")

# --- Inject CSS ---
st.markdown("""
<style>
/* Remove default Streamlit top & bottom space */
header, footer {
    visibility: hidden;
    height: 0;
    margin: 0;
    padding: 0;
}

/* Remove padding around main content */
main > div {
    padding-top: 0 !important;
    padding-bottom: 0 !important;
    margin: 0 !important;
}

/* --- Space for chat content --- */
.chat-wrapper {
    display: flex;
    flex-direction: column;
    margin-top: 20px;
}

/* Chat container and message styles */
.chat-container {
    display: flex;
    justify-content: flex-start;
    margin: 10px 0;
}

/* User messages keep bubble style */
.user-message {
    background-color: #DCF8C6;
    color: #000;
    padding: 10px 15px;
    border-radius: 15px 15px 15px 0;
    max-width: 70%;
    align-self: flex-end;
    margin-left: auto;
}

/* Assistant messages now plain text (no bubble) */
.assistant-message {
    background-color: transparent;
    color: #FFFFFF;
    padding: 0;
    border-radius: 0;
    max-width: 100%;
    align-self: flex-start;
}

/* Wider sidebar with margin */
[data-testid="stSidebar"] {
    width: 400px !important;
    min-width: 400px !important;
    padding: 60px 15px 0px 5px;
}

/* Remove all sidebar padding */
section[data-testid="stSidebar"] > div {
    padding: 0rem !important;
}

.sidebar-title {
    color: #00FFE1;
    font-size: 1.8rem;
    font-weight: 600;
    text-align: left;
    left: 120px;
    top: 120px;
    margin: -120px 0px 30px 0px;
    padding: 0px 0px 0px 0px;
    border-bottom: 5px;
    position: fixed;
    width: 200px;
    background-color: inherit;
    z-index: 1000;
}



/* Tool dropdown styling */
.tool-dropdown-section {
    margin: 10px 0;
    border-left: 2px solid #00FFE1;
    padding-left: 10px;
}

.tool-dropdown-header {
    color: #00FFE1;
    font-size: 0.9rem;
    margin-bottom: 5px;
}
</style>
""", unsafe_allow_html=True)


# Utility functions
def cleanup_pkl_files():
    """
    Delete all .pkl files in the current directory to free up storage
    """
    try:
        pkl_files_deleted = 0
        current_directory = os.getcwd()
        
        # List all files in current directory
        for filename in os.listdir(current_directory):
            if filename.endswith('.pkl'):
                file_path = os.path.join(current_directory, filename)
                try:
                    os.remove(file_path)
                    pkl_files_deleted += 1
                    print(f"Deleted: {filename}")
                except Exception as e:
                    print(f"Error deleting {filename}: {e}")
        
        return pkl_files_deleted
    except Exception as e:
        print(f"Error during PKL cleanup: {e}")
        return 0

def generate_thread_id():
    return uuid.uuid4()

def new_chat():
    # Clean up previous PKL files first
    deleted_count = cleanup_pkl_files()
    
    # Generate new thread ID
    thread_id = generate_thread_id()
    st.session_state["thread_id"] = thread_id
    add_thread(st.session_state["thread_id"])
    st.session_state["message_history"] = []
    st.session_state["message_tools"] = {}
    
    # Clear any uploaded images from session state
    if "images" in st.session_state:
        del st.session_state["images"]
    
    # Increment uploader key to reset the file uploader
    st.session_state["file_uploader_key"] = st.session_state.get("file_uploader_key", 0) + 1
    
    # Show success message with cleanup info
    if deleted_count > 0:
        st.sidebar.success(f"🧹 Cleared {deleted_count} previous documents and started new chat!")
    else:
        st.sidebar.success("🆕 Started new chat!")
    
    # Force rerun to refresh the UI
    st.rerun()

def add_thread(thread_id):
    if thread_id not in st.session_state["chat_threads"]:
        st.session_state["chat_threads"].append(thread_id)

def load_conversation(thread_id):
    state = chatbot.get_state(config={"configurable": {"thread_id": thread_id}})
    return state.values.get("messages", [])

def extract_message_with_tools(messages):
    """Extract messages with their associated tools"""
    chat_messages = []
    current_tools = []
    message_tools = {}
    
    for i, msg in enumerate(messages):
        if isinstance(msg, SystemMessage):
            continue
            
        if isinstance(msg, HumanMessage):
            if current_tools:
                message_tools[len(chat_messages) - 1] = current_tools.copy()
                current_tools = []
            chat_messages.append({"role": "user", "content": msg.content})
            
        elif isinstance(msg, AIMessage):
            if hasattr(msg, 'tool_calls') and msg.tool_calls:
                for tool_call in msg.tool_calls:
                    current_tools.append({
                        'name': tool_call.get('name', 'unknown_tool'),
                        'input': tool_call.get('args', {}),
                        'output': None,
                        'completed': False
                    })
            
            if msg.content:
                chat_messages.append({"role": "assistant", "content": msg.content})
                if current_tools:
                    message_tools[len(chat_messages) - 1] = current_tools.copy()
                    current_tools = []
                    
        elif isinstance(msg, ToolMessage):
            tool_name = getattr(msg, 'name', 'tool')
            tool_result = getattr(msg, 'content', '')
            
            for tools in message_tools.values():
                for tool in tools:
                    if tool['name'] == tool_name and not tool['completed']:
                        tool['output'] = tool_result
                        tool['completed'] = True
                        break
    
    return chat_messages, message_tools

def safe_json_display(data):
    """Safely display data as JSON or text"""
    try:
        if isinstance(data, (dict, list)):
            st.json(data)
        else:
            parsed = json.loads(data)
            st.json(parsed)
    except (json.JSONDecodeError, TypeError):
        st.text(str(data))

def display_message(role: str, content: str):
    """Display chat message bubble on left or right depending on role."""
    role_class = "user-message" if role == "user" else "assistant-message"
    st.markdown(
        f'<div class="chat-container"><div class="{role_class}">{content}</div></div>',
        unsafe_allow_html=True
    )
    
def display_message_with_tools(role: str, content: str, message_index: int = None, tools: list = None):
    """Display chat message with optional tools"""
    display_message(role, content)
    
    if role == "assistant" and tools:
        display_tool_history(tools, message_index)

def display_tool_history(tool_executions, message_index=None):
    """Display tool history in dropdown format"""
    if tool_executions:
        st.markdown("#### 🛠️ Tools Used")
        
        for i, tool_exec in enumerate(tool_executions):
            with st.expander(f"✅ {tool_exec['name']} - Completed", expanded=False):
                st.write("**Input:**")
                safe_json_display(tool_exec['input'])
                
                st.write("**Output:**")
                if tool_exec['output'] is not None:
                    safe_json_display(tool_exec['output'])
                else:
                    st.text("No output available")


def load_document(uploaded_file):
    """
    Load supported document formats and convert them to images
    Supports: PDF, PNG, JPG, JPEG, TXT
    """
    images = []
    
    # Get file extension
    file_extension = uploaded_file.name.split('.')[-1].lower()
    
    try:
        if file_extension == 'pdf':
            # Handle PDF files
            images = process_pdf(uploaded_file)
            
        elif file_extension in ['png', 'jpg', 'jpeg']:
            # Handle image files directly
            images = process_image(uploaded_file)
            
        elif file_extension == 'txt':
            # Handle text files
            images = process_text(uploaded_file)
            
        else:
            st.error(f"Unsupported file format: {file_extension}. Please upload PDF, PNG, JPG, or TXT files.")
            return []
            
    except Exception as e:
        st.error(f"Error processing file: {str(e)}")
        return []
    
    return images

def process_pdf(uploaded_file):
    """Process PDF files"""
    with tempfile.NamedTemporaryFile(delete=False, suffix='.pdf') as tmp_file:
        tmp_file.write(uploaded_file.getvalue())
        tmp_file_path = tmp_file.name
    
    try:
        # Convert PDF to images
        images = convert_from_path(tmp_file_path, dpi=200, fmt='JPEG')
        return images
    except Exception as e:
        st.error(f"Error converting PDF to images: {str(e)}")
        return []
    finally:
        # Clean up temporary file
        if os.path.exists(tmp_file_path):
            os.unlink(tmp_file_path)

def process_image(uploaded_file):
    """Process image files"""
    try:
        image = Image.open(io.BytesIO(uploaded_file.getvalue()))
        
        # Convert to RGB if necessary
        if image.mode in ('RGBA', 'LA', 'P'):
            image = image.convert('RGB')
        
        return [image]
    except Exception as e:
        st.error(f"Error processing image: {str(e)}")
        return []

def process_text(uploaded_file):
    """Process text files"""
    try:
        text_content = uploaded_file.getvalue().decode('utf-8', errors='ignore')
        return [text_to_image(text_content)]
    except Exception as e:
        st.error(f"Error processing text file: {str(e)}")
        return []

def text_to_image(text_content, max_width=1000):
    """
    Convert text content to an image using PIL
    """
    # Create a temporary image to calculate text size
    temp_img = Image.new('RGB', (1, 1), color='white')
    draw = ImageDraw.Draw(temp_img)
    
    # Try to use available fonts, fallback to default
    try:
        # Try common fonts
        try:
            font = ImageFont.truetype("DejaVuSans.ttf", 14)
        except:
            try:
                font = ImageFont.truetype("arial.ttf", 14)
            except:
                font = ImageFont.load_default()
    except:
        font = ImageFont.load_default()
    
    # Split text into lines that fit within max_width
    lines = []
    for paragraph in text_content.split('\n'):
        if not paragraph.strip():
            lines.append('')
            continue
            
        words = paragraph.split()
        current_line = []
        
        for word in words:
            test_line = ' '.join(current_line + [word])
            bbox = draw.textbbox((0, 0), test_line, font=font)
            text_width = bbox[2] - bbox[0]
            
            if text_width <= max_width:
                current_line.append(word)
            else:
                if current_line:
                    lines.append(' '.join(current_line))
                current_line = [word]
        
        if current_line:
            lines.append(' '.join(current_line))
    
    # Calculate image size
    line_height = 20
    padding = 30
    max_line_width = 0
    
    for line in lines:
        bbox = draw.textbbox((0, 0), line, font=font)
        line_width = bbox[2] - bbox[0]
        max_line_width = max(max_line_width, line_width)
    
    img_width = min(max_line_width + 2 * padding, 1200)
    img_height = len(lines) * line_height + 2 * padding
    
    # Create image
    img = Image.new('RGB', (img_width, img_height), color='white')
    draw = ImageDraw.Draw(img)
    
    # Add a subtle border
    draw.rectangle([5, 5, img_width-5, img_height-5], outline='#e0e0e0', width=2)
    
    # Draw text
    y = padding
    for line in lines:
        if line:  # Only draw non-empty lines
            draw.text((padding, y), line, fill='#333333', font=font)
        y += line_height
    
    return img

def process_uploaded_file(uploaded_file):
    """
    Process uploaded file and return images
    """
    if uploaded_file is not None:
        images = load_document(uploaded_file)
        
        if images:
            file_path = f"{st.session_state['thread_id']}.pkl"
            
            with open(file_path, "wb") as f:
                pickle.dump(images, f)
            
            st.session_state["images"] = images
            st.sidebar.success(f"✅ Loaded {len(images)} pages/images from {uploaded_file.name}")
            
            # Show file info
            file_size = len(uploaded_file.getvalue()) / (1024 * 1024)  # Size in MB
            st.sidebar.info(f"📁 File: {uploaded_file.name}\n"
                          f"📊 Size: {file_size:.2f} MB\n"
                          f"🖼️ Pages/Images: {len(images)}")
        else:
            st.sidebar.error("❌ Failed to process the uploaded file")
# Initialize session state
if "message_history" not in st.session_state:
    st.session_state["message_history"] = []

if "message_tools" not in st.session_state:
    st.session_state["message_tools"] = {}

if "thread_id" not in st.session_state:
    st.session_state["thread_id"] = generate_thread_id()

if "chat_threads" not in st.session_state:
    st.session_state["chat_threads"] = retrieve_all_threads()

add_thread(st.session_state["thread_id"])

# Sidebar
st.sidebar.markdown('<div class="sidebar-title">Human Bot</div>', unsafe_allow_html=True)

if "file_uploader_key" not in st.session_state:
    st.session_state["file_uploader_key"] = 0

uploaded_file = st.sidebar.file_uploader(
    "Upload documents or images", 
    type=["pdf", "png", "jpg", "jpeg", "txt"],
    help="Supported formats: PDF, PNG, JPG, JPEG, TXT",
    key=f"file_uploader_{st.session_state['file_uploader_key']}"  # Dynamic key that changes on new chat
)

if uploaded_file is not None:
    process_uploaded_file(uploaded_file)

if st.sidebar.button("New Chat"):
    new_chat()

st.sidebar.header("Previous Conversations")

# st.sidebar.markdown('<div class="sidebar-title">Human Bot</div>', unsafe_allow_html=True)
# st.sidebar.markdown('<div class="sidebar-content">', unsafe_allow_html=True)

for thread_id in st.session_state["chat_threads"][::-1]:
    if st.sidebar.button(str(thread_id), key=str(thread_id)):
        st.session_state["thread_id"] = thread_id
        messages = load_conversation(thread_id)
        
        chat_messages, message_tools = extract_message_with_tools(messages)
        st.session_state["message_history"] = chat_messages
        st.session_state["message_tools"] = message_tools

# Display all messages with their tools
for i, msg in enumerate(st.session_state["message_history"]):
    tools = st.session_state["message_tools"].get(i, [])
    display_message_with_tools(msg["role"], msg["content"], i, tools)

st.markdown('</div>', unsafe_allow_html=True)

# User input
user_input = st.chat_input("Type Here...")

CONFIG = {"configurable": {"thread_id": st.session_state["thread_id"]}}

if user_input:
    st.session_state["message_history"].append({"role": "user", "content": user_input})
    display_message("user", user_input)
    
    # Get document length from uploaded file or default
    if "images" in st.session_state:
        document_length = len(st.session_state["images"])
    else:
        document_length = 0
    
    
    messages = [
        SystemMessage(
            content=f"""
        🎯 **CORE DIRECTIVE:**  
        Execute user instructions precisely without hesitation. Use tools when available, respond directly when not.

        🔧 **TOOL USAGE (CRITICAL):**  
        - **ALWAYS** use BOTH tools for every query:  
        1. `retrieval_tool` - Get document info first  
        2. `augmentation_and_generation_tool` - Generate response  
        - **Thread ID:** {st.session_state["thread_id"]} (REQUIRED for both tools)  
        - **Retrieval limit:** `number_of_pages_to_retrieve` ≤ {document_length}

        📝 **QUERY PROCESSING:**  
        - **ONE question per retrieval call** - Split multi-part questions  
        - Break complex queries into steps  
        - Answer sequentially in clear, structured format

        🗣️ **COMMUNICATION STYLE:**  
        - Match user's tone: casual, serious, or supportive  
        - Natural human conversation - no robotic phrasing  
        - Straightforward and clear - no confusion or uncertainty

        🚫 **AVOID:**  
        - Overthinking or questioning instructions  
        - Tool selection confusion (USE BOTH)  
        - Complex explanations when simple will do

        💡 **REMEMBER:** You're a capable assistant who understands exactly what's needed and delivers efficiently.
        """
        ),
        HumanMessage(content=user_input)
    ]
    
    # Create a placeholder for tools that will be updated in real-time
    tool_placeholder = st.empty()
    
    def ai_stream_with_realtime_tools():
        current_tool_index = 0
        tool_executions = []
        collected_content = []
        
        def render_tools():
            with tool_placeholder.container():
                if tool_executions:
                    st.markdown("#### 🛠️ Tools Used")
                    for tool_exec in tool_executions:
                        if not tool_exec['completed']:
                            with st.expander(f"🔄 {tool_exec['name']} - Running...", expanded=True):
                                st.markdown('<div class="tool-running">🔄 Tool is currently running...</div>', unsafe_allow_html=True)
                                st.write("**Input:**")
                                safe_json_display(tool_exec['input'])
                        else:
                            with st.expander(f"✅ {tool_exec['name']} - Completed", expanded=False):
                                st.write("**Input:**")
                                safe_json_display(tool_exec['input'])
                                st.write("**Output:**")
                                if tool_exec['output'] is not None:
                                    safe_json_display(tool_exec['output'])
                                else:
                                    st.text("No output available")

        for message_chunk, metadata in chatbot.stream(
            {"messages": messages},
            config=CONFIG,
            stream_mode="messages",
        ):
            if (isinstance(message_chunk, AIMessage) and 
                hasattr(message_chunk, 'tool_calls') and 
                message_chunk.tool_calls):
                
                for tool_call in message_chunk.tool_calls:
                    tool_name = tool_call.get('name', 'unknown_tool')
                    tool_args = tool_call.get('args', {})
                    
                    tool_executions.append({
                        'name': tool_name,
                        'input': tool_args,
                        'output': None,
                        'index': current_tool_index,
                        'completed': False
                    })
                    current_tool_index += 1
                    
                    render_tools()

            elif isinstance(message_chunk, ToolMessage):
                tool_name = getattr(message_chunk, 'name', 'tool')
                tool_result = getattr(message_chunk, 'content', '')
                
                for tool_exec in tool_executions:
                    if tool_exec['name'] == tool_name and not tool_exec['completed']:
                        tool_exec['output'] = tool_result
                        tool_exec['completed'] = True
                        break
                
                render_tools()

            if isinstance(message_chunk, AIMessage) and message_chunk.content:
                collected_content.append(message_chunk.content)
                yield message_chunk.content

        render_tools()
        
        completed_tools = [te for te in tool_executions if te['completed']]
        if completed_tools:
            new_message_index = len(st.session_state["message_history"])
            st.session_state["message_tools"][new_message_index] = completed_tools

    # Display AI response
    ai_message = st.write_stream(ai_stream_with_realtime_tools())
    
    # Add final AI message to history
    st.session_state["message_history"].append({"role": "assistant", "content": ai_message})