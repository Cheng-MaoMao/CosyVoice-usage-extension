# filepath: /C:/Users/TianXuan/Documents/GitHub/AliCosyVoice-usage-extension/webui.py
import gradio as gr
from main import ai_chat, VectorDB, semantic_search_and_respond

# 创建向量数据库实例
vector_db = VectorDB()

def chat_with_ai(user_input: str, session_id: str) -> str:
    """
    与AI进行对话的函数。

    Args:
        user_input (str): 用户输入的文本。
        session_id (str): 会话标识符。

    Returns:
        str: AI的回复。
    """
    # 调用AI聊天函数
    response = ai_chat(user_input, "", session_id)
    return response

def semantic_search(query: str) -> str:
    """
    执行语义搜索并生成回答的函数。

    Args:
        query (str): 用户的查询文本。

    Returns:
        str: LLM生成的回答。
    """
    response = semantic_search_and_respond(query, vector_db, debug=True)
    return response

# Gradio界面
with gr.Blocks() as demo:
    gr.Markdown("# AI 聊天界面")
    
    with gr.Row():
        with gr.Column():
            user_input = gr.Textbox(label="输入您的问题", placeholder="在这里输入...")
            session_id = gr.Textbox(label="会话ID", placeholder="输入会话ID（可选）")
            submit_button = gr.Button("发送")
            chat_output = gr.Textbox(label="AI 回复", interactive=False)

            submit_button.click(chat_with_ai, inputs=[user_input, session_id], outputs=chat_output)

    with gr.Row():
        search_input = gr.Textbox(label="语义搜索", placeholder="输入搜索内容...")
        search_button = gr.Button("搜索")
        search_output = gr.Textbox(label="搜索结果", interactive=False)

        search_button.click(semantic_search, inputs=search_input, outputs=search_output)

# 启动Gradio应用
if __name__ == "__main__":
    demo.launch(server_name="0.0.0.0", server_port=7860)