# 🧠 Human Bot – Vision-Language AI Assistant

A **multimodal AI assistant** built using **Streamlit**, **LangGraph**, and **Google Gemini**, capable of:
- Conversational reasoning  
- Document understanding (PDF, image-based documents)
- Vision-Language retrieval and generation  
- Search and calculator tool usage  

This system integrates **retrieval-augmented generation (RAG)** with **multimodal reasoning** to provide context-aware, intelligent responses.

---

## 🚀 Features

- 💬 **Conversational Interface:**  
  Natural and contextual chat via Streamlit.

- 🧾 **Multimodal Understanding:**  
  Upload PDFs or images; the system retrieves and interprets relevant pages using NVIDIA’s `llama-nemoretriever` and Qwen’s `Qwen2.5-VL` models.

- 🔍 **RAG-style Retrieval:**  
  Automatically retrieves top-matching pages from uploaded documents before answering.

- 🧮 **Integrated Tools:**  
  - `Calculator`: Basic arithmetic  
  - `DuckDuckGo Search`: Web search queries  
  - `Retrieval` and `Generation`: Vision-language inference

- 🧠 **Stateful Conversations:**  
  Each chat session (thread) is stored in a SQLite database using **LangGraph checkpoints**.

---

## 🏗️ Project Structure

```
├── frontend.py              # Streamlit UI for chat and file upload
├── backend.py               # LangGraph agent logic and workflow
├── functions_and_tools.py   # Tool definitions (retrieval, generation, search, etc.)
├── database.py              # SQLite-based document management
├── requirements.txt         # Python dependencies
```

---

## ⚙️ Setup Instructions

### 1. Clone the Repository
```bash
git clone https://github.com/281221karan/Human-Bot.git
cd human-bot
```

### 2. Create a Virtual Environment
```bash
python -m venv venv
source venv/bin/activate    # Linux/Mac
venv\Scripts\activate       # Windows
```

### 3. Install Dependencies
Create a `requirements.txt` (example below 👇) and install:
```bash
pip install -r requirements.txt
```

> **Note:**  
> - `poppler` is required for `pdf2image`.  
>   Install via:
>   ```bash
>   sudo apt install poppler-utils
>   ```
> - Use CUDA for GPU acceleration (recommended).

---

## 🧩 How It Works

### 🖥️ `frontend.py`
- Provides a **Streamlit**-based interface.
- Allows users to upload PDFs or images.
- Manages multiple chat threads.
- Displays conversation history and handles user input.

### 🧠 `backend.py`
- Builds the **LangGraph pipeline**:
  - Connects the **Gemini LLM** (`ChatGoogleGenerativeAI`).
  - Integrates tools: calculator, search, retrieval, and multimodal generation.
  - Uses SQLite checkpoints to persist chat state.

### 🧰 `functions_and_tools.py`
Defines modular tools for the chatbot:
- `load_document_and_info`: Loads and converts uploaded files.
- `retrieval`: Finds top relevant document pages using `nvidia/llama-nemoretriever-colembed-3b-v1`.
- `augmentation_and_generation`: Generates visual-textual answers with `Qwen2.5-VL-7B-Instruct`.
- `calculator` and `search`: Utility tools.

### 💾 `database.py`
- Manages document metadata storage (file name, path, thread ID).
- Uses SQLite for lightweight persistence.

---

## 🧠 System Workflow

1. User uploads a document or image.
2. The document is saved and indexed in SQLite.
3. The chatbot:
   - Retrieves top-matching pages for the user’s query.
   - Feeds those pages to a vision-language model.
4. Generates a **context-aware response** that references visual content and text.

---

## 🧪 Run the Application

```bash
streamlit run frontend.py
```

Then open the provided local URL (e.g. `http://localhost:8501`) in your browser.

---

## 🧰 Example Usage

1. **Upload** a contract PDF.
2. **Ask:**  
   _“Summarize the termination clause.”_
3. The chatbot retrieves the relevant pages, analyzes them visually + textually, and generates a concise answer.

---

## 📊 Future Enhancements

- Support for long-context retrieval (semantic chunking)
- Integration with local LLMs (Llama, Mistral, etc.)
- Better memory management for large documents
- Support for additional file formats (DOCX, XLSX)
