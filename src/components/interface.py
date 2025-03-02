# filepath: /C:/Users/TianXuan/Documents/GitHub/AliCosyVoice-usage-extension/webui.py
import gradio as gr
from main import ai_chat, VectorDB  # 确保从 main.py 导入所需的函数和类

# 创建向量数据库实例
vector_db = VectorDB()

def chat_with_ai(user_input: str, session_id: str):
    """
    与 AI 进行聊天的函数。
    
    Args:
        user_input (str): 用户输入的文本。
        session_id (str): 会话标识符。

    Returns:
        str: AI 的响应文本。
    """
    embedding_prompt = "请根据用户输入生成响应。"  # 这里可以根据需要调整
    response = ai_chat(user_input, embedding_prompt, session_id)
    return response

def launch_gradio_interface():
    """
    启动 Gradio 界面。
    """
    with gr.Blocks() as demo:
        gr.Markdown("# AI 聊天界面")
        
        with gr.Row():
            user_input = gr.Textbox(label="输入您的问题", placeholder="在这里输入...")
            session_id = gr.Textbox(label="会话 ID", value="session_1", visible=False)  # 隐藏的会话 ID

        submit_button = gr.Button("发送")
        output = gr.Textbox(label="AI 响应", interactive=False)

        submit_button.click(chat_with_ai, inputs=[user_input, session_id], outputs=output)

    demo.launch()

if __name__ == "__main__":
    launch_gradio_interface()