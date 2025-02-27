# 🌸 AI 语音聊天助手

[![Python Version](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/)

> ✨ 基于 [CosyVoice](https://github.com/FunAudioLLM/CosyVoice) 的智能语音交互扩展项目

[简体中文]|[English](./README_en.md)

## 🌟 项目简介

本系统通过集成多模态AI能力，实现：

- 🗣️ 智能对话生成
- 📚 知识库语义检索
- 🔊 情感化语音合成
- 🌐 网页内容自动化分析

<details>
  <summary>查看程序流程图</summary>
  <img src="./images/流程图.PNG" alt="流程图">
</details>

## 🚀 快速开始

### 📥 前置要求

1. 部署 [CosyVoice](https://github.com/FunAudioLLM/CosyVoice) 语音合成服务
2. Python 3.10+ 环境
3. 有效的 大模型API 密钥

### ⚙️ 安装步骤

## 🔧 功能特性

| 功能模块     | 支持技术                   | 特性描述            |
| ------------ | -------------------------- | ------------------- |
| 智能对话生成 | DeepSeek-R1/Qwen2.5/Others | 🧠 多轮情景对话     |
| 语义检索     | BGE-M3 嵌入模型 + FAISS    | 🔍 知识库快速匹配   |
| 语音合成     | CosyVoice API              | 🎵 情感化语音生成   |
| 网页分析     | Selenium + BeautifulSoup   | 🌐 动态网页内容抓取 |

## 📖 使用指南

### 1. 添加大模型url和密钥

```python
# 大模型 API配置
api_url = "https://api.siliconflow.cn/v1/chat/completions"
headers = {
    "Authorization": "Bearer sk-jgxgrpjdrxmmtghsjmplqkdclxcjegasofsrfbfcwkyiaekc",
    "Content-Type": "application/json"
}
# 在以下函数中修改需要使用的大模型
ai_chat()
embedding_model()
get_llm_response()
send_audio_info_to_ai()
```

### 2. 准备知识库

```python
vector_db = VectorDB()  # 创建向量数据库实例
vector_db.add_texts(texts) # 将文字添加进知识库
batch_analyze_webpages(webpage_urls, vector_db) # 将网页内容添加到知识库
```

*运行一次构建成功后请注释添加代码以加快运行速度*

### 3. 修改CosyVoice代码

```python
 #修改webui.py(关闭流式传输--必须)
 audio_output = gr.Audio(label="合成音频", autoplay=True, streaming=False)
 #修改cosyvoice\cli\frontend.py(关闭切片--非必需)
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
          
     # 移除了split_paragraph切片处理,直接返回整个文本
     return [text] if split is True else text
```

### 4. 启动服务

```python
# 先启动CosyVoice
python webui.py --port 50000 --model_dir pretrained_models/CosyVoice-300M
# 再运行main.py
python main.py
```

## 📂 项目结构

```
ai-voice-assistant/
├── reference_audio/      # 参考音频库
├── generated_audio/      # 合成音频存储
├── core/                 # 核心逻辑模块
│   ├── main.py           # 主要代码
```

## ⚠️ 注意事项

1. 确保 CosyVoice 服务已正确部署并运行在 50000 端口(可自行修改端口号)
2. 参考音频文件需按格式命名：`【情感】语音内容.wav`

## 🤝 参与贡献

欢迎通过 Issue 或 PR 提交改进建议！

## 🙏 致谢

本项目基于以下优秀的开源项目开发：

- [CosyVoice](https://github.com/FunAudioLLM/CosyVoice)

感谢B站UP主TinyLight微光小明分享的爱莉希雅参考音频
- [TinyLight微光小明](https://space.bilibili.com/13264090)

## 📄 开源协议

本项目采用 [Apache License 2.0](LICENSE)
