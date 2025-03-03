# 🌸 AI Voice Chat Assistant

[![Python Version](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/)

> ✨ A smart voice interaction extension project based on [CosyVoice](https://github.com/FunAudioLLM/CosyVoice)

[简体中文](./README.md)|[English]

## 🌟 Project Overview

This system integrates multi-modal AI capabilities to achieve:

- 🗣️ Intelligent dialogue generation
- 📚 Semantic retrieval from a knowledge base
- 🔊 Emotional voice synthesis
- 🌐 User-friendly WebUI

<details>
  <summary>View Program Flowchart</summary>
  <img src="./images/流程图.PNG" alt="Flowchart">
</details>

## 🚀 Quick Start

### 📥 Prerequisites

1. Deploy the [CosyVoice](https://github.com/FunAudioLLM/CosyVoice) voice synthesis service
2. Python 3.10+ environment
3. Valid large model API key

## 🔧 Features

| Function Module | Supported Technology           | Feature Description         |
| --------------- | ------------------------------ | -------------------------- |
| Intelligent Dialogue Generation | DeepSeek-R1/Qwen2.5/Others | 🧠 Multi-turn scenario dialogue |
| Semantic Retrieval | BGE-M3 Embedding Model + FAISS | 🔍 Fast knowledge base matching |
| Voice Synthesis | CosyVoice API                  | 🎵 Emotional voice generation |
| Web Analysis | Selenium + BeautifulSoup         | 🌐 Dynamic web content scraping |
| Graphical Interface | Gradio                      | 🖥️ User-friendly WebUI |

## 📖 User Guide

*Recommended to use the WebUI*

### Command Line Method

#### 1. Add Large Model URL and Key


# Large Model API Configuration
api_url = "https://api.siliconflow.cn/v1/chat/completions"
headers = {
    "Authorization": "Bearer sk-jgxgrpjdrxmmtghsjmplqkdclxcjegasofsrfbfcwkyiaekc",
    "Content-Type": "application/json"
}
# Modify the following global variables in ai_chat.py
chat_model = "deepseek-ai/DeepSeek-R1"
text_model = "Qwen/Qwen2.5-72B-Instruct"
embed_model = "BAAI/bge-m3"
text_api_url = "https://api.siliconflow.cn/v1/chat/completions"
url_embedding = "https://api.siliconflow.cn/v1/embeddings"


#### 2. Prepare the Knowledge Base


vector_db = VectorDB()  # Create a vector database instance
vector_db.add_texts(texts)  # Add text to the knowledge base
batch_analyze_webpages(webpage_urls, vector_db)  # Add web content to the knowledge base


*After the initial build, comment out the addition code to speed up subsequent runs*

#### 3. Modify CosyVoice Code


# Modify webui.py (Disable streaming - Required)
audio_output = gr.Audio(label="Synthesized Audio", autoplay=True, streaming=False)
# Modify cosyvoice\cli\frontend.py (Disable slicing - Optional)
def text_normalize(self, text, split=True, text_frontend=True):
    if isinstance(text, Generator):
        logging.info('get tts_text generator, will skip text_normalize!')
        return [text]
    if text_frontend is False:
        return [text] if split is True else text
    text = text.strip()
    if self.use_ttsfrd:
        texts = [i["text"] for i in json.loads(self.frd.do_voicegen_frd(text))["sentences"]]
        text = ''.join(texts)
    else:
        if contains_chinese(text):
            text = self.zh_tn_model.normalize(text)
            text = text.replace("\n", "")
            text = replace_blank(text)
            text = replace_corner_mark(text)
            text = text.replace(".", "。")
            text = text.replace(" - ", "，")
            text = remove_bracket(text)
            text = re.sub(r'[，,、]+$', '。', text)
        else:
            text = self.en_tn_model.normalize(text)
            text = spell_out_number(text, self.inflect_parser)
      
    # Removed split_paragraph slicing, return the entire text directly
    return [text] if split is True else text


#### 4. Start the Service


# Start CosyVoice first
python webui.py --port 50000 --model_dir pretrained_models/models_name
# Then run ai_chat.py
python ai_chat.py


### WebUI Method

#### 1. Modify CosyVoice Code


# Modify webui.py (Disable streaming - Required)
audio_output = gr.Audio(label="Synthesized Audio", autoplay=True, streaming=False)
# Modify cosyvoice\cli\frontend.py (Disable slicing - Optional)
def text_normalize(self, text, split=True, text_frontend=True):
    if isinstance(text, Generator):
        logging.info('get tts_text generator, will skip text_normalize!')
        return [text]
    if text_frontend is False:
        return [text] if split is True else text
    text = text.strip()
    if self.use_ttsfrd:
        texts = [i["text"] for i in json.loads(self.frd.do_voicegen_frd(text))["sentences"]]
        text = ''.join(texts)
    else:
        if contains_chinese(text):
            text = self.zh_tn_model.normalize(text)
            text = text.replace("\n", "")
            text = replace_blank(text)
            text = replace_corner_mark(text)
            text = text.replace(".", "。")
            text = text.replace(" - ", "，")
            text = remove_bracket(text)
            text = re.sub(r'[，,、]+$', '。', text)
        else:
            text = self.en_tn_model.normalize(text)
            text = spell_out_number(text, self.inflect_parser)
      
    # Removed split_paragraph slicing, return the entire text directly
    return [text] if split is True else text


#### 2. Start the Service


# Start CosyVoice first
python webui.py --port 50000 --model_dir pretrained_models/models_name
# Then run app.py
python app.py


## 🙏 Acknowledgments

This project is developed based on the following excellent open-source projects:

- [CosyVoice](https://github.com/FunAudioLLM/CosyVoice) for voice synthesis code
- [Gradio](https://gradio.app/) for the user-friendly WebUI framework

Thanks to Bilibili UP主 TinyLight微光小明 for sharing the reference audio of Alisa.

- [TinyLight微光小明](https://space.bilibili.com/13264090)

## ⚠️ Notes

1. Ensure that the CosyVoice service is correctly deployed and running on port 50000 (the port number can be modified)
2. Reference audio files should be named in the following format: `【情感】语音内容.wav`

## 🤝 Contributing

We welcome improvement suggestions through Issues or PRs!

## 📄 License

This project is licensed under the [Apache License 2.0](LICENSE)