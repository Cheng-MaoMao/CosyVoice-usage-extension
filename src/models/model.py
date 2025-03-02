# filepath: /C:/Users/TianXuan/Documents/GitHub/AliCosyVoice-usage-extension/webui.py
import gradio as gr
from main import ai_chat  # 假设 ai_chat 函数在 main.py 中

def chat_with_ai(user_input, session_id):
    # 调用 ai_chat 函数获取 AI 的响应
    response = ai_chat(user_input, "", session_id)
    return response

def launch_gradio_interface():
    # 创建 Gradio 接口
    with gr.Blocks() as demo:
        gr.Markdown("## AI 聊天助手")
        
        with gr.Row():
            with gr.Column():
                user_input = gr.Textbox(label="输入您的问题", placeholder="在这里输入...")
                session_id = gr.Textbox(label="会话 ID", placeholder="输入会话 ID", value="session_1")
                submit_button = gr.Button("发送")
                
            with gr.Column():
                output = gr.Textbox(label="AI 的回答", interactive=False)

        submit_button.click(chat_with_ai, inputs=[user_input, session_id], outputs=output)

    demo.launch()

if __name__ == "__main__":
    launch_gradio_interface()