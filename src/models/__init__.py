# filepath: /C:/Users/TianXuan/Documents/GitHub/AliCosyVoice-usage-extension/webui.py
import gradio as gr
from main import ai_chat, VectorDB

# 创建向量数据库实例
vector_db = VectorDB()

def chat_with_ai(user_input: str, session_id: str):
    # 调用 ai_chat 函数与 AI 进行对话
    embedding_prompt = "这里是一些上下文信息，可以根据需要进行修改。"  # 你可以根据需要修改这个提示
    response = ai_chat(user_input, embedding_prompt, session_id)
    return response

def launch_gradio_interface():
    with gr.Blocks() as demo:
        gr.Markdown("# AI 聊天界面")
        
        with gr.Row():
            with gr.Column():
                user_input = gr.Textbox(label="输入您的问题", placeholder="在这里输入...")
                session_id = gr.Textbox(label="会话 ID", placeholder="输入会话 ID (可选)", value="session_1")
                submit_button = gr.Button("发送")
                
            with gr.Column():
                output = gr.Textbox(label="AI 的回复", interactive=False)

        submit_button.click(chat_with_ai, inputs=[user_input, session_id], outputs=output)

    demo.launch()

if __name__ == "__main__":
    launch_gradio_interface()