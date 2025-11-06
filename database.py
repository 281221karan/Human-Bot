import sqlite3
conn = sqlite3.connect(database = "chatbot.db", check_same_thread = False)

c = conn.cursor()

def create_table_if_not_exist():
    c.execute("""
        CREATE TABLE IF NOT EXISTS document_info (
                file_name TEXT,
                thread_id TEXT,
                file_path TEXT
            )
            """)
    conn.commit()

def store_document_info(file_name : str, thread_id, file_path: str):
    thread_id = str(thread_id)
    if not check_if_image_data_exist_in_database(file_path, thread_id):
        c.execute("INSERT INTO document_info (file_name, thread_id, file_path) VALUES (?, ?, ?)", (file_name, thread_id, file_path))
        conn.commit()

def check_if_image_data_exist_in_database(file_path:str, thread_id):
    c.execute("SELECT EXISTS(SELECT 1 FROM document_info WHERE file_path = ? AND thread_id = ?)", (file_path,thread_id))
    exists = c.fetchone()[0]
    return bool(exists)
    
def retrieve_images_from_database(file_name:str, thread_id):
    thread_id = str(thread_id)
    c.execute("SELECT file_path FROM document_info WHERE thread_id = ? AND file_name = ?", (thread_id,file_name))
    result = c.fetchall()
    return result