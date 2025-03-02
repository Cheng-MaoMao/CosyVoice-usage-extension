import gradio as gr
import ai_chat
import uuid
from typing import List
import json
import os

def update_model_config(chat_url: str, chat_key: str, chat_model: str,
                       text_url: str, text_key: str, text_model: str,
                       embed_url: str, embed_key: str, embed_model: str):
    # 更新模型配置
    ai_chat.api_url = chat_url
    ai_chat.headers["Authorization"] = f"Bearer {chat_key}"
    ai_chat.chat_model = chat_model
    
    ai_chat.text_api_url = text_url
    ai_chat.text_headers = {
        "Authorization": f"Bearer {text_key}",
        "Content-Type": "application/json"
    }
    ai_chat.text_model = text_model
    
    ai_chat.url_embedding = embed_url
    ai_chat.embed_headers = {
        "Authorization": f"Bearer {embed_key}",
        "Content-Type": "application/json"
    }
    ai_chat.embed_model = embed_model
    
    return "模型配置已更新"

def build_knowledge_base(embedding_list: str, vector_db: ai_chat.VectorDB) -> str:
    """构建知识库"""
    if not embedding_list:
        return "知识库为空，请添加文本或网页URL"
    
    try:
        # 清空现有知识库
        if hasattr(vector_db, 'texts'):
            vector_db.texts = []
        if hasattr(vector_db, 'embeddings'):
            vector_db.embeddings = []
            
        # 分割嵌入提示列表
        items = embedding_list.strip().split('\n')
        
        texts = []
        urls = []
        
        # 分类处理文本和URL
        for item in items:
            if item.startswith("网页URL: "):
                urls.append(item.replace("网页URL: ", "").strip())
            else:
                texts.append(item.strip())
        
        # 添加文本到向量数据库
        if texts:
            vector_db.add_texts(texts)
            
        # 批量处理网页
        if urls:
            ai_chat.batch_analyze_webpages(urls, vector_db)
        
        # 保存向量数据库到本地
        vector_db.save_db()
            
        return f"知识库构建完成，包含 {len(vector_db.texts) if hasattr(vector_db, 'texts') else 0} 条文本"
    except Exception as e:
        return f"构建知识库失败: {str(e)}"

def chat_with_ai(user_input: str, embedding_prompt: str, session_id: str, use_kb: bool, vector_db: ai_chat.VectorDB = None):
    """处理聊天请求"""
    if not session_id:  
        session_id = str(uuid.uuid1())
        
    # 根据开关决定是否使用知识库
    knowledge_prompt = ""
    if use_kb and vector_db and hasattr(vector_db, 'texts') and vector_db.texts:
        try:
            knowledge_prompt = ai_chat.semantic_search_and_respond(user_input, vector_db, False)
        except Exception as e:
            print(f"知识库搜索失败: {str(e)}")
    
    # 获取 AI 的回复
    response = ai_chat.ai_chat(user_input, knowledge_prompt, session_id)
    
    # 生成语音
    ai_chat.send_audio_info_to_ai(response)
    audio_path = ai_chat.gradio_api_use()
    if audio_path:
        try:
            ai_chat.audio_play(audio_path)
        except Exception as e:
            print(f"播放音频失败: {str(e)}")
            
    return response

def add_embedding(text_input: str, current_embeddings: str) -> str:
    # 添加新的嵌入提示到当前列表
    if not current_embeddings:
        return text_input
    return current_embeddings + "\n" + text_input

def add_webpage(url_input: str, current_embeddings: str) -> str:
    # 添加网页URL到当前列表
    if not current_embeddings:
        return "网页URL: " + url_input
    return current_embeddings + "\n" + "网页URL: " + url_input

def generate_session_id() -> str:
    # 生成新的会话ID
    return str(uuid.uuid1())

def save_config(embedding_list: str, chat_url: str, chat_key: str, chat_model: str,
               text_url: str, text_key: str, text_model: str,
               embed_url: str, embed_key: str, embed_model: str) -> str:
    """保存配置到文件"""
    config = {
        "embedding_list": embedding_list,
        "chat": {
            "url": chat_url,
            "key": chat_key,
            "model": chat_model
        },
        "text": {
            "url": text_url,
            "key": text_key,
            "model": text_model
        },
        "embed": {
            "url": embed_url,
            "key": embed_key,
            "model": embed_model
        }
    }
    
    config_dir = os.path.join(os.path.dirname(__file__), "config")
    os.makedirs(config_dir, exist_ok=True)
    config_path = os.path.join(config_dir, "config.json")
    
    try:
        with open(config_path, 'w', encoding='utf-8') as f:
            json.dump(config, f, ensure_ascii=False, indent=2)
        return "配置已保存"
    except Exception as e:
        return f"保存配置失败: {str(e)}"

def load_config() -> dict:
    """加载配置文件"""
    config_path = os.path.join(os.path.dirname(__file__), "config", "config.json")
    default_config = {
        "embedding_list": "",
        "chat": {
            "url": "https://api.siliconflow.cn/v1/chat/completions",
            "key": "",
            "model": "deepseek-ai/DeepSeek-R1"
        },
        "text": {
            "url": "https://api.siliconflow.cn/v1/chat/completions",
            "key": "",
            "model": "Qwen/Qwen2.5-72B-Instruct"
        },
        "embed": {
            "url": "https://api.siliconflow.cn/v1/embeddings",
            "key": "",
            "model": "BAAI/bge-m3"
        }
    }
    
    try:
        if os.path.exists(config_path):
            with open(config_path, 'r', encoding='utf-8') as f:
                config = json.load(f)
            return config
        return default_config
    except Exception as e:
        print(f"加载配置失败: {str(e)}")
        return default_config

def launch_gradio_interface():
    # 加载配置
    config = load_config()
    
    # 创建向量数据库实例并尝试加载已保存的数据
    vector_db = ai_chat.VectorDB()
    vector_db.load_db()  # 添加这行来加载已保存的数据库
    
    # 缓存上一次的嵌入提示和知识库状态
    state = {
        "last_embedding": "",
        "kb_built": False
    }
    
    def handle_kb_switch(switch_value, embedding_content):
        """处理知识库开关变化"""
        if switch_value:
            if not state["kb_built"] or embedding_content != state["last_embedding"]:
                state["last_embedding"] = embedding_content
                result = build_knowledge_base(embedding_content, vector_db)
                state["kb_built"] = True
                return result
            return "使用已加载的知识库"
        state["kb_built"] = False
        return "知识库已禁用"
    
    # 启动时自动更新模型配置
    update_model_config(
        config["chat"]["url"], config["chat"]["key"], config["chat"]["model"],
        config["text"]["url"], config["text"]["key"], config["text"]["model"],
        config["embed"]["url"], config["embed"]["key"], config["embed"]["model"]
    )
    
    with gr.Blocks() as demo:
        gr.Markdown("## AI 聊天助手")
        
        with gr.Tab("聊天界面"):
            with gr.Row():
                # 左侧面板：聊天输入和会话控制
                with gr.Column(scale=2):
                    session_id = gr.Textbox(label="会话 ID", interactive=True)
                    generate_id_btn = gr.Button("生成新会话ID")
                    user_input = gr.Textbox(label="输入您的问题", placeholder="在这里输入...")
                    submit_button = gr.Button("发送")
                    output = gr.Textbox(label="AI 的回复（语音已自动播放）", interactive=False)
                
                # 右侧面板：嵌入提示管理
                with gr.Column(scale=1):
                    use_kb_switch = gr.Checkbox(
                        label="启用知识库",
                        value=False,
                        interactive=True
                    )
                    embedding_list = gr.TextArea(
                        label="当前嵌入提示列表",
                        interactive=False,
                        value=config["embedding_list"]
                    )
                    text_input = gr.Textbox(label="添加文本嵌入", placeholder="输入要嵌入的文本...")
                    add_text_btn = gr.Button("添加文本")
                    url_input = gr.Textbox(label="添加网页URL", placeholder="输入网页URL...")
                    add_url_btn = gr.Button("添加网页")
                    build_kb_btn = gr.Button("构建知识库")
                    kb_status = gr.Textbox(label="知识库状态", interactive=False)
        
        with gr.Tab("模型配置"):
            with gr.Row():
                with gr.Column():
                    gr.Markdown("### 聊天模型配置")
                    chat_url = gr.Textbox(
                        label="API URL",
                        value=config["chat"]["url"],
                        interactive=True
                    )
                    chat_key = gr.Textbox(
                        label="API Key",
                        value=config["chat"]["key"],
                        interactive=True
                    )
                    chat_model = gr.Textbox(
                        label="模型名称",
                        value=config["chat"]["model"],
                        interactive=True
                    )
                
                with gr.Column():
                    gr.Markdown("### 文本优化模型配置")
                    text_url = gr.Textbox(
                        label="API URL",
                        value=config["text"]["url"],
                        interactive=True
                    )
                    text_key = gr.Textbox(
                        label="API Key",
                        value=config["text"]["key"],
                        interactive=True
                    )
                    text_model = gr.Textbox(
                        label="模型名称",
                        value=config["text"]["model"],
                        interactive=True
                    )
                
                with gr.Column():
                    gr.Markdown("### 嵌入式模型配置")
                    embed_url = gr.Textbox(
                        label="API URL",
                        value=config["embed"]["url"],
                        interactive=True
                    )
                    embed_key = gr.Textbox(
                        label="API Key",
                        value=config["embed"]["key"],
                        interactive=True
                    )
                    embed_model = gr.Textbox(
                        label="模型名称",
                        value=config["embed"]["model"],
                        interactive=True
                    )
            
            with gr.Row():
                update_btn = gr.Button("更新模型配置")
                save_btn = gr.Button("保存当前配置")
                config_status = gr.Textbox(label="配置状态", interactive=False)
            
            # 更新和保存配置的事件处理
            update_btn.click(
                fn=update_model_config,
                inputs=[
                    chat_url, chat_key, chat_model,
                    text_url, text_key, text_model,
                    embed_url, embed_key, embed_model
                ],
                outputs=config_status
            )
            
            save_btn.click(
                fn=save_config,
                inputs=[
                    embedding_list,
                    chat_url, chat_key, chat_model,
                    text_url, text_key, text_model,
                    embed_url, embed_key, embed_model
                ],
                outputs=config_status
            )

        # 设置按钮点击事件
        generate_id_btn.click(
            fn=generate_session_id,
            outputs=session_id
        )
        
        submit_button.click(
            fn=lambda x, y, z, w: chat_with_ai(x, y, z, w, vector_db),
            inputs=[user_input, embedding_list, session_id, use_kb_switch],
            outputs=output
        )

        add_text_btn.click(
            fn=add_embedding,
            inputs=[text_input, embedding_list],
            outputs=embedding_list
        )

        add_url_btn.click(
            fn=add_webpage,
            inputs=[url_input, embedding_list],
            outputs=embedding_list
        )

        def update_and_build_kb(embedding_content, use_kb):
            """更新并构建知识库"""
            if not use_kb:
                state["kb_built"] = False
                return "知识库未启用"
            
            if embedding_content != state["last_embedding"]:
                state["last_embedding"] = embedding_content
                state["kb_built"] = True
                return build_knowledge_base(embedding_content, vector_db)
            elif not state["kb_built"]:
                state["kb_built"] = True
                return build_knowledge_base(embedding_content, vector_db)
            return "知识库未发生变化，使用已有知识库"

        # 构建知识库按钮事件
        build_kb_btn.click(
            fn=update_and_build_kb,
            inputs=[embedding_list, use_kb_switch],
            outputs=[kb_status]
        )

        # 添加知识库开关状态变化事件
        use_kb_switch.change(
            fn=handle_kb_switch,
            inputs=[use_kb_switch, embedding_list],
            outputs=kb_status
        )

        # 页面加载时自动生成会话ID
        demo.load(
            fn=generate_session_id,
            outputs=session_id
        )

    # 启动 Gradio 应用
    demo.launch()

if __name__ == "__main__":
    launch_gradio_interface()