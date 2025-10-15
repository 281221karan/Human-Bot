# 🤖 Vision-Language RAG ChatBot

A powerful, multimodal chatbot that combines **Vision-Language Models (VLMs)** and **Retrieval-Augmented Generation (RAG)** to answer questions based on uploaded **PDFs** or **images**.

Built using **Streamlit**, this chatbot also supports fallback to **Google Gemini API** when no document is uploaded.

---

## 🚀 Features

- 🧠 **Vision-Language Understanding** via `Qwen2.5-VL`
- 🔍 **Image/PDF Retrieval** using `ColQwen2.5`
- 🗃️ Upload **PDF**, **IMAGES** files
- 🔄 **Document-aware question answering**
- 📎 **Gemini API fallback** when no document is uploaded
- 💬 Clean UI with

---
# SETUP

## CLOUD

Every thing is already setup for you, you just need to click **RUN**

**https://lightning.ai/bindaas281221/vision-model/studios/vl-rag/code?turnOn=true**

Just visit the link and follow the **video** given below:

**https://www.dropbox.com/scl/fi/nfqubppfxtqn3ww179hbi/VL-RAG-tutorial.mp4?rlkey=00e3jj0ncdevb1vdj2f516o30&st=743wrcca&dl=0**

this setup does not required any type of *GPU*, its running on cloud, the only thing that required is **Login** here **https://lightning.ai/**

## LOCALLY
**Requirements**

at least **48GB** GPU memory

1. `git clone https://github.com/281221karan/VL-RAG`

2. `pip install -r requirements.txt`

3. `streamlit run main.py`

it will automatically download the model

---
I think thats all
