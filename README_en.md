# 🌸 AI Voice Chat Assistant

[![Python Version](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/)

> ✨ An intelligent voice interaction extension project based on [CosyVoice](https://github.com/FunAudioLLM/CosyVoice)

[简体中文](./README.md)|[English]

## 🌟 Project Overview

This system integrates multimodal AI capabilities to achieve:

- 🗣️ Intelligent dialogue generation
- 📚 Knowledge base semantic retrieval
- 🔊 Emotional speech synthesis
- 🌐 Automated web content analysis

<details>
  <summary>View Flowchart</summary>
  <img src="./images/流程图.PNG" alt="Flowchart">
</details>

## 🚀 Quick Start

### 📥 Prerequisites

1. Deploy [CosyVoice](https://github.com/FunAudioLLM/CosyVoice) speech synthesis service
2. Python 3.10+ environment
3. Valid LLM API key

### ⚙️ Installation Steps

## 🔧 Features

| Module          | Supported Technologies       | Description                 |
| --------------- | ----------------------------- | --------------------------- |
| Dialogue Engine | DeepSeek-R1/Qwen2.5/Others    | 🧠 Contextual conversations |
| Semantic Search | BGE-M3 Embedding + FAISS      | 🔍 Fast knowledge matching  |
| Speech Synthesis| CosyVoice API                 | 🎵 Emotional voice generation |
| Web Analysis    | Selenium + BeautifulSoup      | 🌐 Dynamic web scraping     |

## 📖 Usage Guide

### 1. Configure API Settings
```python
# LLM API Configuration
api_url = "https://api.siliconflow.cn/v1/chat/completions"
headers = {
    "Authorization": "Bearer sk-jgxgrpjdrxmmtghsjmplqkdclxcjegasofsrfbfcwkyiaekc",
    "Content-Type": "application/json"
}
# Modify LLM usage in these functions:
ai_chat()
embedding_model()
get_llm_response()
send_audio_info_to_ai()
```

### 2. Prepare Knowledge Base

```python
vector_db = VectorDB()  # Create vector database instance
vector_db.add_texts(texts) # Add texts to knowledge base
batch_analyze_webpages(webpage_urls, vector_db) # Add web content to knowledge base
```

*Comment out these lines after initial setup to improve performance*

### 3. Modify CosyVoice Code
```python
# Modify webui.py (disable streaming - mandatory)
audio_output = gr.Audio(label="Synthesized Audio", autoplay=True, streaming=False)

# Modify cosyvoice\cli\frontend.py (disable slicing - optional)
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
            # ... (keep original normalization logic)
            
    # Removed split_paragraph slicing, return full text directly
    return [text] if split is True else text
```

### 4. Launch Service

```python
# First launch CosyVoice
python webui.py --port 50000 --model_dir pretrained_models/CosyVoice-300M

# Then run main.py
python main.py
```

## 📂 Project Structure

```
ai-voice-assistant/
├── reference_audio/      # Reference audio library
├── generated_audio/      # Synthesized audio storage
├── core/                 # Core logic modules
│   ├── main.py           # Main program
```

## ⚠️ Important Notes

1. Ensure CosyVoice service is properly deployed on port 50000 (configurable)
2. Reference audio files must follow naming format: `【Emotion】SpeechContent.wav`

## 🤝 Contribution

Welcome to submit improvements via Issues or PRs!

## 🙏 Acknowledgments

This project is developed based on these excellent open-source projects:

- [CosyVoice](https://github.com/FunAudioLLM/CosyVoice)

Thanks to the audio shared by Bilibili UP主 TinyLight微光小明 for reference on Alishia

- [TinyLight微光小明](https://space.bilibili.com/13264090)

## 📄 License

This project is licensed under [Apache License 2.0](LICENSE)