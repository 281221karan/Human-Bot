import torch
import gc
import os
from PIL import Image

from database import retrieve_images_from_database

from pdf2image import convert_from_path
from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor, AutoModel
from qwen_vl_utils import process_vision_info

from langchain_core.tools import tool
from langchain_community.tools import DuckDuckGoSearchRun



#------------------------------------------------------- Functions -----------------------------------------------

def save_file_to_disk(file, thread_id, base_folder="uploaded_files"):

    folder_path = os.path.join(base_folder, str(thread_id))
    os.makedirs(folder_path, exist_ok=True)

    file_path = os.path.join(folder_path, file.name)

    with open(file_path, "wb") as f:
        f.write(file.getvalue())

    return file_path

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

def vlm_loader():
    vlm_model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        "Qwen/Qwen2.5-VL-7B-Instruct",
        torch_dtype=torch.bfloat16,
        attn_implementation="flash_attention_2",
        device_map="auto",
        cache_dir = "./models"
    )
    vlm_processor = AutoProcessor.from_pretrained(
        "Qwen/Qwen2.5-VL-7B-Instruct",
        use_fast = True,
        cache_dir = "./models"
        )
    return vlm_model, vlm_processor

#------------------------------------------------------- Tools ----------------------------------------------------
    
def _load_document_and_info(file_name: str, thread_id):
    thread_id = str(thread_id)
    document_path = retrieve_images_from_database(file_name, thread_id)
    print(document_path)
    if not document_path or len(document_path) == 0:
        raise ValueError(f"No document found for file '{file_name}' and thread '{thread_id}'")
    else:
        for i in document_path:
            path = i[0]
            file_extension = os.path.splitext(path)[1].lower()
            
            if file_extension == '.pdf':
                images = convert_from_path(path)
            elif file_extension in ['.jpg', '.jpeg', '.png', '.gif', '.bmp', '.tiff']:
                    images = [Image.open(path)]
            else:
                print(f"Unsupported file type: {file_extension} for file {path}. Skipping.")
    
    return images, len(images)

@tool
def load_document_and_info(file_name:str, thread_id):
    """
    The tool provide the (length / total number of pages in the document) in the uploaded file.
    
    Args:
    file_name: name of the file you need the information of
    thread_id: session thread_id

    Returns:
    images: images variable not of your use.
    length of images: total number of pages in uploaded file. When calling the retrieving tool this will help you to decide the number_of_pages_to_retrieve.
    """
    return _load_document_and_info(file_name, thread_id)

@tool
def retrieval(file_name:str, query: str, number_of_pages_to_retrieve: int, thread_id: str) -> list[int]:
    """
    Retrieve the most relevant document pages for a given query using a vision-language retrieval model.
    required load_document_and_info tool before calling this tool
    Args:
        file_name (str): The name of the stored document to perform retrieval on.
        query (str): The user's search query used to find relevant content.
        number_of_pages_to_retrieve (int): The number of top-matching pages to return. 
            Must be less than or equal to the total number of pages in the document.
        thread_id (str): Unique identifier associated with the user's session or document set.

    Returns:
        list[int]: A list of indices representing the top-ranked document pages most relevant to the query.

    Notes:
        - The function clears GPU memory before and after inference to optimize performance.
        - It uses the retrieval model to embed both the query and document pages, 
          computes similarity scores, and returns indices of the highest-scoring pages.
        - The `load_images` function must return all pages of the document as image tensors or PIL images.

    Example:
        >>> retrieval("contract.pdf", "termination clause", 3, "thread_123")
        [2, 5, 8]
    """

    gc.collect()
    torch.cuda.empty_cache()
    retrieval_model = retrieval_loader()
    images, length = _load_document_and_info(file_name, thread_id)
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
def augmentation_and_generation(file_name: str, images_index: list[int], query: str, thread_id: str = None) -> str:
    
    """
    Generate a multimodal response by combining retrieved document pages (images) 
    with the user's query using a Vision-Language Model (VLM).
    required load_document_and_info tool before calling this tool

    Args:
        file_name (str): The name of the stored document to retrieve images from.
        images_index (list[int]): A list of page indices representing the most relevant document pages.
        query (str): The user's text query or prompt for content generation.
        thread_id (str, optional): Unique session or document identifier. Defaults to None.

    Returns:
        str: The generated textual response from the Vision-Language Model based on the input query and document context.

    Notes:
        - Clears GPU memory before and after execution to optimize performance.
        - Loads the specified Vision-Language Model (VLM) and its processor.
        - Converts retrieved document pages into image inputs and pairs them with the user query.
        - Constructs a multimodal message for generation using `apply_chat_template`.
        - Produces detailed, contextually grounded text output related to the retrieved images and user query.

    Example:
        >>> augmentation_and_generation(
        ...     file_name="contract.pdf",
        ...     images_index=[2, 5, 8],
        ...     query="Summarize the termination clause",
        ...     thread_id="thread_123"
        ... )
        "The termination clause states that either party may end the contract with a 30-day notice..."
    """
    
    gc.collect()
    torch.cuda.empty_cache()
    vlm_model, vlm_processor = vlm_loader()
    
    images, length = _load_document_and_info(file_name, thread_id)

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