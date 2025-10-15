import torch
import gc
import uuid
import sqlite3
import streamlit as st
import pickle
import os

from pdf2image import convert_from_bytes
from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor, AutoModel
from qwen_vl_utils import process_vision_info

from langgraph.graph import StateGraph, START, END
from typing import TypedDict, Annotated
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.messages import BaseMessage
from langgraph.graph.message import add_messages
from langgraph.checkpoint.sqlite import SqliteSaver

from langgraph.prebuilt import ToolNode, tools_condition
from langchain_core.tools import tool
from langchain_community.tools import DuckDuckGoSearchRun


#utility functions

@st.cache_resource
def retrieval_loader():
    retrieval_model = AutoModel.from_pretrained(
        'nvidia/llama-nemoretriever-colembed-3b-v1',
        device_map='cuda',
        trust_remote_code=True,
        torch_dtype=torch.bfloat16,
        attn_implementation="flash_attention_2",
        revision='50c36f4d5271c6851aa08bd26d69f6e7ca8b870c',
        cache_dir = "./models"
    ).eval()
    return retrieval_model

@st.cache_resource
def vlm_loader():
    
    vlm_model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        "Qwen/Qwen2.5-VL-7B-Instruct",
        torch_dtype=torch.bfloat16,
        attn_implementation="flash_attention_2",
        device_map="auto",
        cache_dir = "./models"
    )
    # default processer
    vlm_processor = AutoProcessor.from_pretrained(
        "Qwen/Qwen2.5-VL-7B-Instruct",
        use_fast = True,
        cache_dir = "./models"
        )
    return vlm_model, vlm_processor

@st.cache_resource
def load_document(uploaded_file):
    """
    Convert a Streamlit UploadedFile PDF into a list of images
    """
    pdf_bytes = uploaded_file.read()
    images = convert_from_bytes(pdf_bytes)
    uploaded_file.seek(0)
    return images

def load_images_from_pickle(thread_id: str):
    file_path = f"{thread_id}.pkl"
    if os.path.exists(file_path):
        # raise FileNotFoundError(f"No pickle file found for thread_id {thread_id}")
        with open(file_path, "rb") as f:
            images = pickle.load(f)
    return images

#tools

@tool
def retrieval(query: str, number_of_pages_to_retrieve: int, thread_id: str) -> list[int]:
    """
    Retrieve indices of images most relevant to the user's query using embedding similarity.
    
    This tool processes the query and document images through a multimodal embedding model,
    computes similarity scores between query and image embeddings, and returns the indices
    of the most relevant pages.
    
    Args:
        query (str): The search query or question to find relevant content in the document.
        number_of_pages_to_retrieve (int): Number of top relevant pages to return. 
            Must be a positive integer <= total number of pages in the document.
        thread_id (str): The id to retrieve the images provided in the SystemMessages
    
    Returns:
        list[int]: List of page indices (0-based) corresponding to the most relevant images, 
        sorted by relevance score in descending order.
    
    Raises:
        RuntimeError: If embedding generation, similarity computation, or retrieval fails.
        ValueError: If number_of_pages_to_retrieve exceeds available pages or is invalid.
    
    Example:
        retrieval("find information about machine learning", 3)
        [2, 5, 1]  # Returns indices of the 3 most relevant pages
    
    Process:
        1. Generate query embeddings using forward_queries()
        2. Generate passage embeddings from images using forward_passages() 
        3. Compute similarity scores between query and passage embeddings
        4. Select top-k pages based on highest similarity scores
        5. Return indices of most relevant pages
    
    Note:
        - The tool uses batch processing for efficiency (batch_size=8)
        - Returned indices are 0-based (first page = index 0)
        - Ensure number_of_pages_to_retrieve does not exceed total document pages
        - Higher scores indicate better relevance to the query
    """
    gc.collect()
    torch.cuda.empty_cache()
    images = load_images_from_pickle(thread_id)
    query_embeddings = retrieval_model.forward_queries([query], batch_size=1)
    passage_embeddings = retrieval_model.forward_passages(images, batch_size=1)
    scores = retrieval_model.get_scores(query_embeddings, passage_embeddings)
    top_result = torch.topk(scores, k=number_of_pages_to_retrieve)
    topk_scores = top_result.values
    images_index = top_result.indices
    gc.collect()
    torch.cuda.empty_cache()
    
    return images_index[0]

@tool
def augmentation_and_generation(images_index: list[int], query: str, thread_id: str = None) -> str:
    """
    Generate a comprehensive answer by analyzing retrieved document pages and answering the user's query.
    
    This tool uses a vision-language model to process selected document pages along with the user's question,
    generating a contextual answer based on both visual content (images) and textual query.
    
    Args:
        images_index (list[int]): List of 0-based indices specifying which document pages to analyze.
            These indices correspond to the most relevant pages retrieved by the retrieval tool.
        query (str): The user's original question or information request that needs to be answered.
        thread_id (str): The id to retrieve the images provided in the SystemMessages
    
    Returns:
        str: A comprehensive, contextual answer generated by analyzing the specified document pages 
        in relation to the user's query. The response combines information extracted from visual 
        content and addresses the specific question.
    
    Raises:
        RuntimeError: If image processing, model inference, or text generation fails.
        ValueError: If images_index contains invalid indices or is empty.
        IndexError: If any image index exceeds available document pages.
    
    Example:
        augmentation_and_generation([2, 5, 1], "What are the main topics discussed?")
        "The document discusses three main topics: machine learning fundamentals on page 2, 
         neural network architectures on page 5, and training methodologies on page 1."
    
    Process:
        1. Memory cleanup and GPU cache optimization
        2. Prepare multimodal content list with specified images and query text
        3. Apply chat template to format the input for the vision-language model
        4. Process vision information and prepare model inputs
        5. Generate response using the vision-language model with controlled output length
        6. Decode and clean up the generated text
        7. Final memory cleanup and resource optimization
    
    Technical Details:
        - Uses Qwen2.5-VL model for multimodal understanding
        - Processes up to 128 new tokens for response generation
        - Automatically handles GPU memory management
        - Supports batch processing of multiple images
        - Applies proper chat formatting for conversational context
    
    Note:
        - Ensure images_index contains valid indices within document bounds
        - The tool automatically handles image preprocessing and feature extraction
        - Response quality depends on relevance of retrieved pages to the query
        - Memory cleanup is performed before and after generation to optimize performance
    """

    gc.collect()
    torch.cuda.empty_cache()
    
    images = load_images_from_pickle(thread_id)

    content_list = []
    
    for i in images_index:
      content_list.append(
          {
              "type": "image", "image": images[i]
          }
                          )
        
    content_list.append(
        {
            "type":"text", "text":query
        }
                        )
    messages = [
        {
            "role": "user",
            "content": content_list,
        }
                ]
    
    text = vlm_processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    
    image_inputs, video_inputs = process_vision_info(messages)
    inputs = vlm_processor(text=[text],images=image_inputs,videos=video_inputs,padding=True,return_tensors="pt",)
    inputs = inputs.to("cuda")

    generated_ids = vlm_model.generate(**inputs, max_new_tokens=10000)
    generated_ids_trimmed = [out_ids[len(in_ids) :] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)]
    
    output_text = vlm_processor.batch_decode(generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False)
    
    gc.collect()
    torch.cuda.empty_cache()
    return output_text

@tool
def calculator(first_num: float, second_num: float, operation: str) -> dict:
    """
    Perform a basic arithmetic operation on two numbers.
    Supported operations: add, sub, mul, div
    """
    try:
        if operation == "add":
            result = first_num + second_num
        elif operation == "sub":
            result = first_num - second_num
        elif operation == "mul":
            result = first_num * second_num
        elif operation == "div":
            if second_num == 0:
                return {"error": "Division by zero is not allowed"}
            result = first_num / second_num
        else:
            return {"error": f"Unsupported operation '{operation}'"}
        
        return {"first_num": first_num, "second_num": second_num, "operation": operation, "result": result}
    except Exception as e:
        return {"error": str(e)}


search = DuckDuckGoSearchRun()


# retrieval_model = retrieval_loader()
# vlm_model, vlm_processor = vlm_loader()


# from pdf2image import convert_from_path
# images = convert_from_path("keep317.pdf")
# output = retrieval("question 1", 3)
# print(output)


model = ChatGoogleGenerativeAI(model="gemini-2.5-flash", api_key="AIzaSyBPI_9TmT9zcVmj45QtclQrK5HD1qT6ssg")

tools = [retrieval, augmentation_and_generation, calculator, search]
model_with_tools = model.bind_tools(tools)

class ChatBotState(TypedDict):
    messages: Annotated[list[BaseMessage], add_messages]

def chat_node(state: ChatBotState):
    messages = state["messages"]
    response = model_with_tools.invoke(messages)
    
    return {"messages": [response]}

tool_node = ToolNode(tools)

conn = sqlite3.connect(database = "chatbot.db", check_same_thread = False)
checkpointer = SqliteSaver(conn = conn)

graph = StateGraph(ChatBotState)

graph.add_node("chat_node", chat_node)
graph.add_node("tools", tool_node)

graph.add_edge(START, "chat_node")
graph.add_conditional_edges("chat_node", tools_condition)
graph.add_edge("tools", "chat_node")

chatbot = graph.compile(checkpointer = checkpointer)

def retrieve_all_threads():
    all_threads = []
    for i in checkpointer.list(None):
        if i.config["configurable"]["thread_id"] not in all_threads:
            all_threads.append(i.config["configurable"]["thread_id"])
    return all_threads