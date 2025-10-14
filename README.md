pm
🎥 Real-Time Streaming Video Bot
Streaming Video Bot
FastAPI
CUDA
TensorRT

A real-time streaming video bot that generates talking avatar videos with ultra-low latency using:

Piper TTS for lightning-fast, natural-sounding speech
Wav2Lip for lip-sync animation
Streaming architecture for immediate video playback
Traditional batch mode for complete video downloads
Perfect for virtual assistants, customer service bots, and interactive AI applications!

🚀 Features
🎙️ Natural Voice
Amy Medium female voice with adjustable speed (length_scale=1.7 for slow, girlish speech)
Ultra-fast TTS: 0.2s generation time vs 13s with traditional VITS
Configurable speech speed: From fast (0.8x) to very slow (2.0x)
⚡ Dual Generation Modes
Real-Time Streaming: Watch video as it generates (0.5s to first frame)
Traditional Batch: Complete video file for download/sharing
🖥️ Technical Highlights
TensorRT FP16 acceleration for Wav2Lip (40% faster)
GPU-optimized for NVIDIA V100/A100/H100
Memory efficient streaming with chunked processing
Web-ready with FastAPI and MJPEG streaming


Installation
Prerequisites
Python 3.8+
NVIDIA GPU with CUDA 12.x (V100/A100/H100 recommended)
16GB+ RAM

