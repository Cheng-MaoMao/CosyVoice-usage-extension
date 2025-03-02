# filepath: /C:/Users/TianXuan/Documents/GitHub/AliCosyVoice-usage-extension/webui.py
import gradio as gr
from main import ai_chat  # 假设 ai_chat 函数在 main.py 中

def chat_with_ai(user_input, embedding_prompt, session_id):
    """
    与 AI 进行聊天的接口函数。
    
    Args:
        user_input (str): 用户输入的文本。
        embedding_prompt (str): 嵌入提示文本。
        session_id (str): 会话标识符。

    Returns:
        str: AI 的响应。
    """
    response = ai_chat(user_input, embedding_prompt, session_id)
    return response

def launch_gradio_interface():
    """
    启动 Gradio Web UI。
    """
    with gr.Blocks() as demo:
        gr.Markdown("# AI 聊天界面")
        
        with gr.Row():
            user_input = gr.Textbox(label="输入您的问题", placeholder="在这里输入...")
            embedding_prompt = gr.Textbox(label="嵌入提示", placeholder="可选的嵌入提示...")
            session_id = gr.Textbox(label="会话 ID", placeholder="会话标识符...")

        submit_button = gr.Button("发送")
        output = gr.Textbox(label="AI 的回答", interactive=False)

        submit_button.click(
            chat_with_ai,
            inputs=[user_input, embedding_prompt, session_id],
            outputs=output
        )

    demo.launch()

if __name__ == "__main__":
    launch_gradio_interface()