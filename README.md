# 🤖 Human Bot

A powerful, multimodal chatbot that combines **Vision-Language Models (VLMs)** and **Retrieval-Augmented Generation (RAG)** to answer questions based on uploaded **PDFs** or **images**.

Built using **Streamlit**, this chatbot also supports fallback to **Google Gemini API** when no document is uploaded.

---

## 🚀 Features

- 😎 **Smart** `Google Gemini`
- 🧠 **Vision-Language Understanding** via `Qwen/Qwen2.5-VL-7B-Instruct`
- 🔍 **Image/PDF Retrieval** using `nvidia/llama-nemoretriever-colembed-3b-v1`
- 🗃️ Upload **PDF**, **IMAGES** files
- 🔄 **Document-aware question answering**

---
# SETUP

## CLOUD

Every thing is already setup for you, you just need to click **RUN**

**https://lightning.ai/karan281221/human-bot/studios/human-bot/code?source=copylink**

Just visit the link and follow the **video** given below:

**  **

this setup does not required any type of *GPU*, its running on *cloud*, the only thing that required is **Login** here **https://lightning.ai/**

## LOCALLY
**Requirements**

at least **48GB** GPU memory

1. `git clone https://github.com/281221karan/VL-RAG`

2. `pip install -r requirements.txt`

3. `streamlit run main.py`

it will automatically download the model

---
I think thats all
