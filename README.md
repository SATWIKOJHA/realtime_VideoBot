# 🎥 Real-Time Streaming Video Bot

![Streaming Video Bot](https://img.shields.io/badge/Python-3.8%2B-blue) ![FastAPI](https://img.shields.io/badge/FastAPI-0.68%2B-green) ![CUDA](https://img.shields.io/badge/CUDA-12.4-orange) ![TensorRT](https://img.shields.io/badge/TensorRT-8.6%2B-red)

A **real-time streaming video bot** that generates talking avatar videos with **ultra-low latency** using:
- **Piper TTS** for lightning-fast, natural-sounding speech
- **Wav2Lip** for lip-sync animation
- **Streaming architecture** for immediate video playback
- **Traditional batch mode** for complete video downloads

Perfect for virtual assistants, customer service bots, and interactive AI applications!

## 🚀 Features

### 🎙️ **Natural Voice**
- **Amy Medium** female voice with adjustable speed (`length_scale=1.7` for slow, girlish speech)
- **Ultra-fast TTS**: 0.2s generation time vs 13s with traditional VITS
- **Configurable speech speed**: From fast (0.8x) to very slow (2.0x)

### ⚡ **Dual Generation Modes**
1. **Real-Time Streaming**: Watch video as it generates (0.5s to first frame)
2. **Traditional Batch**: Complete video file for download/sharing

### 🖥️ **Technical Highlights**
- **TensorRT FP16 acceleration** for Wav2Lip (40% faster)
- **GPU-optimized** for NVIDIA V100/A100/H100
- **Memory efficient** streaming with chunked processing
- **Web-ready** with FastAPI and MJPEG streaming

### 📊 **Performance Comparison**
| Metric | Traditional VITS | This Bot (Piper) |
|--------|------------------|------------------|
| **TTS Latency** | 13s (50s audio) | **0.2s** (50s audio) |
| **Total Pipeline** | 25s | **3-5s** |
| **Time to First Frame** | N/A | **0.5s** |
| **Voice Quality** | High | Medium-High |
| **Speech Speed** | Fixed | **Configurable** |

## 🛠️ Installation

### Prerequisites
- **Python 3.8+**
- **NVIDIA GPU** with CUDA 12.x (V100/A100/H100 recommended)
- **16GB+ RAM**
- **Docker** (optional but recommended)

### 1. Clone Repository
```bash
git clone https://github.com/yourusername/streaming-video-bot.git
cd streaming-video-bot
```

### 2. Create Virtual Environment
```bash
python -m venv venv
source venv/bin/activate  # Linux/Mac
# venv\Scripts\activate  # Windows
```

### 3. Install Dependencies
```bash
# Core dependencies
pip install fastapi uvicorn python-multipart jinja2 requests numpy scipy opencv-python

# GPU dependencies (CUDA 12.x)
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
pip install onnxruntime-gpu==1.16.3 pycuda tensorrt

# TTS and Wav2Lip
pip install piper-tts face-alignment
```

### 4. Download Models
```bash
# Create models directory
mkdir -p piper_models checkpoints static

# Download Piper TTS (Amy Medium - girlish voice)
cd piper_models
wget https://huggingface.co/rhasspy/piper-voices/resolve/main/en/en_US/amy/medium/en_US-amy-medium.onnx
wget https://huggingface.co/rhasspy/piper-voices/resolve/main/en/en_US/amy/medium/en_US-amy-medium.onnx.json

# Download Wav2Lip checkpoint
cd ../checkpoints
wget https://iiitaphyd-my.sharepoint.com/personal/radrabha_m_research_iiit_ac_in/_layouts/15/download.aspx?share=EdjI7bZlg3lNmE64a7RmR7BBIl6wP9N6l6pARvqM6mb6UA -O wav2lip_gan.pth

# Create default avatar
# Place your avatar.png in static/ directory
cp /path/to/your/avatar.png static/avatar.png
```

### 5. Install Wav2Lip Dependencies
```bash
# Clone Wav2Lip repository
git clone https://github.com/Rudrabha/Wav2Lip.git
cp -r Wav2Lip/models.py Wav2Lip/audio.py .
# Install face detection
pip install face-alignment
```

## 🚀 Usage

### 1. Start the Server
```bash
uvicorn main:app --host 0.0.0.0 --port 8000 --reload
```

### 2. Access Web Interface
Open your browser: **http://localhost:8000**

### 3. API Endpoints

#### 📡 **Streaming Mode (Real-Time)**
```bash
# GET request with prompt parameter
curl "http://localhost:8000/stream?prompt=Hello%20world"

# With custom avatar
curl "http://localhost:8000/stream?prompt=Hello%20world&avatar_url=/static/custom_avatar.png"
```

**HTML Integration:**
```html
<img src="/stream?prompt=Hello%20world" />
```

#### 📦 **Traditional Mode (Batch)**
```bash
# POST request for complete video
curl -X POST http://localhost:8000/generate \
  -H "Content-Type: application/json" \
  -d '{"text": "Hello world", "avatar": "/static/avatar.png"}'
```

### 4. Configuration
Edit the top section of `video_bot_wav2lip.py` and `streaming_video_bot.py`:

```python
# Voice settings
TTS_LENGTH_SCALE = 1.7  # 1.0=normal, 1.7=very slow/girlish

# Model paths
PIPER_MODEL_PATH = "piper_models/en_US-amy-medium.onnx"
WAV2LIP_CHECKPOINT_PATH = "checkpoints/wav2lip_gan.pth"

# LLM settings
LLM_ENDPOINT = "http://your-llm-endpoint:8001/v1/chat/completions"
LLM_MODEL = "/path/to/your/model"
```

## 🎨 Customization

### 🗣️ **Voice Options**
Download different Piper voices from [Hugging Face](https://huggingface.co/rhasspy/piper-voices):

| Voice | Command |
|-------|---------|
| **Kristin** (young female) | `wget https://huggingface.co/rhasspy/piper-voices/.../en_US-kristin-medium.onnx` |
| **Joe** (male) | `wget https://huggingface.co/rhasspy/piper-voices/.../en_US-joe-medium.onnx` |
| **Kathleen** (female) | `wget https://huggingface.co/rhasspy/piper-voices/.../en_US-kathleen-medium.onnx` |

### ⚙️ **Performance Tuning**
- **Faster streaming**: Reduce `max_words=12` in `split_into_chunks()`
- **Higher quality**: Use `en_US-amy-high.onnx` (slower but better quality)
- **Lower latency**: Reduce Wav2Lip batch size from 16 to 8

### 🎭 **Avatar Customization**
- Replace `static/avatar.png` with your custom avatar
- Ensure face is clearly visible and well-lit
- Square aspect ratio recommended (512x512)

## 📊 Performance Optimization

### GPU Memory Management
For long responses, monitor VRAM usage:
```bash
# Check GPU memory
nvidia-smi

# Reduce batch sizes if running out of memory
# In streaming_video_bot.py:
# - Reduce Wav2Lip batch size (line ~280)
# - Reduce chunk size (line ~45)
```

### TensorRT Acceleration
Build a custom TensorRT engine for maximum Wav2Lip performance:
```bash
# Install TensorRT tools
pip install onnx-graphsurgeon polygraphy

# Convert PyTorch to ONNX, then to TensorRT
# (Advanced users only)
```

## 🐛 Troubleshooting

### Common Issues

**1. "No module named 'pycuda'"**
```bash
pip install pycuda
# If compilation fails, install system dependencies:
sudo apt-get install build-essential python3-dev
```

**2. "CUDA out of memory"**
- Reduce `wav2lip_batch_size` in traditional pipeline
- Reduce streaming batch size from 16 to 8
- Use lower quality TTS model (`en_US-amy-low.onnx`)

**3. AudioChunk errors**
- Ensure you're using compatible Piper TTS version
- The code handles both old and new Piper versions automatically

**4. LLM connection timeout**
- Verify LLM endpoint is accessible
- Increase timeout in `requests.post(timeout=30)`

### Debugging
Enable verbose logging:
```python
# In both pipeline files
logging.basicConfig(level=logging.DEBUG)
```

## 📄 Project Structure
```
streaming-video-bot/
├── main.py                 # FastAPI application
├── video_bot_wav2lip.py    # Traditional batch pipeline
├── streaming_video_bot.py  # Real-time streaming pipeline
├── templates/
│   └── index.html          # Web interface
├── static/
│   ├── avatar.png          # Default avatar
│   ├── avatars/            # Uploaded avatars
│   └── videos/             # Generated videos
├── piper_models/           # TTS models
├── checkpoints/            # Wav2Lip models
└── temp/                   # Temporary files
```

## 🤝 Contributing
Contributions are welcome! Please open an issue or submit a pull request for:
- New voice models
- Performance improvements
- Additional features (emotion detection, multiple avatars, etc.)

## 📜 License
This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments
- [Piper TTS](https://github.com/rhasspy/piper) - Ultra-fast neural TTS
- [Wav2Lip](https://github.com/Rudrabha/Wav2Lip) - Lip-sync technology
- [Coqui TTS](https://github.com/coqui-ai/TTS) - Original TTS implementation
- [FastAPI](https://fastapi.tiangolo.com/) - Web framework

---

**Made with ❤️ for real-time AI video generation** 🚀

[![Deploy](https://img.shields.io/badge/Deploy-Docker-blue?logo=docker)](https://docs.docker.com/)
[![Python](https://img.shields.io/badge/Python-3.8%2B-blue?logo=python)](https://www.python.org/)
[![CUDA](https://img.shields.io/badge/CUDA-12.4-orange?logo=nvidia)](https://developer.nvidia.com/cuda-toolkit)
