#!/usr/bin/env python3
"""
ULTRA-FAST Video Bot: LLM → PIPER TTS → Wav2Lip
- Amy medium quality (young female voice)
- Very slow speech speed (length_scale=1.7)
- Supports longer content (up to 80 words)
"""

import os
import sys
import time
import logging
import numpy as np
import torch
import requests
from scipy.io.wavfile import write as write_wav
import subprocess
import shutil
import cv2
import re

# --- PIN TO GPU 0 ---
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

# --- CONFIGURATION ---
LLM_ENDPOINT = "http://49.50.117.66:8001/v1/chat/completions"
LLM_MODEL = "/workspace/models/scout-fp4"

# PIPER TTS (AMY MEDIUM + VERY SLOW SPEED)
PIPER_MODEL_PATH = "piper_models/en_US-amy-medium.onnx"
PIPER_CONFIG_PATH = "piper_models/en_US-amy-medium.onnx.json"
TTS_OUTPUT_PATH = "response.wav"
SAMPLE_RATE = 22050
TTS_LENGTH_SCALE = 1.7  # Very slow speech

WAV2LIP_CHECKPOINT_PATH = "checkpoints/wav2lip_gan.pth"
WAV2LIP_TRT_ENGINE = "wav2lip_fp16.engine"
AVATAR_IMAGE_PATH = "static/avatar.png"
OUTPUT_VIDEO_PATH = "static/videos/output.mp4"

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# Logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(name)s | %(levelname)s | %(message)s",
    handlers=[
        logging.FileHandler("video_bot_piper.log"),
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger("VideoBotPiper")

sys.path.insert(0, os.path.abspath("."))

# --- Wav2Lip Modules ---
try:
    from models import Wav2Lip
    import audio
except ImportError as e:
    logger.error(f"Failed to import Wav2Lip modules: {e}")
    sys.exit(1)

# --- LLM STAGE (80 WORDS MAX) ---
class LLMStage:
    @staticmethod
    def process(prompt: str) -> str:
        start = time.time()
        payload = {
            "model": LLM_MODEL,
            "messages": [
                {
                    "role": "system",
                    "content": "You are a helpful, friendly assistant. Respond in plain text with NO markdown. Keep responses under 80 words for natural slow speech."
                },
                {"role": "user", "content": prompt}
            ],
            "max_tokens": 80,
            "temperature": 0.7
        }
        try:
            resp = requests.post(LLM_ENDPOINT, json=payload, timeout=20)
            resp.raise_for_status()
            content = resp.json()["choices"][0]["message"]["content"].strip()
            content = re.sub(r'[*#=]{2,}', '', content)
            content = re.sub(r'\s+', ' ', content).strip()
            logger.info(f"[LLM] latency={time.time()-start:.2f}s | response='{content[:50]}...'")
            return content
        except Exception as e:
            logger.error(f"[LLM] FAILED: {e}")
            return "I'm sorry, I couldn't process that."

# --- PIPER TTS STAGE (NO TRUNCATION - FULL 80 WORDS) ---
class TTSStage:
    def __init__(self):
        logger.info("[TTS] Loading Piper TTS (Amy Medium - Young Female Voice)...")
        from piper import PiperVoice
        self.model = PiperVoice.load(
            model_path=PIPER_MODEL_PATH,
            config_path=PIPER_CONFIG_PATH,
            use_cuda=(DEVICE == "cuda")
        )
        logger.info(f"[TTS] ✅ Piper TTS loaded on {'GPU' if DEVICE == 'cuda' else 'CPU'}")

    def process(self, text: str, output_path: str = TTS_OUTPUT_PATH) -> str:
        start = time.time()
        try:
            # 🔥 REMOVED TRUNCATION - Allow full LLM response
            # words = text.split()
            # if len(words) > 20:  # ← THIS WAS THE PROBLEM!
            #     text = ' '.join(words[:20]) + "..."
            
            # VERY SLOW SPEECH: length_scale=1.7
            try:
                wav_gen = self.model.synthesize(text, length_scale=TTS_LENGTH_SCALE)
            except TypeError:
                original_scale = getattr(self.model.config, 'length_scale', 1.0)
                self.model.config.length_scale = TTS_LENGTH_SCALE
                wav_gen = self.model.synthesize(text)
                self.model.config.length_scale = original_scale
            
            # AudioChunk handling
            audio_samples = []
            for chunk in wav_gen:
                if hasattr(chunk, 'audio_float_array'):
                    audio_samples.extend(chunk.audio_float_array)
                elif hasattr(chunk, 'audio'):
                    audio_samples.extend(chunk.audio)
                else:
                    audio_samples.append(chunk)
            
            wav = np.array(audio_samples, dtype=np.float32)
            wav = (wav * 32767).astype(np.int16)
            write_wav(output_path, SAMPLE_RATE, wav)
            duration = len(wav) / SAMPLE_RATE
            logger.info(f"[TTS] latency={time.time()-start:.2f}s | duration={duration:.2f}s")
            return output_path
        except Exception as e:
            logger.error(f"[TTS] error: {e}")
            return None

# --- WAV2LIP ANIMATION STAGE (OPTIMIZED FOR LONGER VIDEOS) ---
import face_detection

USE_TENSORRT = False
try:
    import tensorrt as trt
    import pycuda.driver as cuda
    import pycuda.autoinit
    USE_TENSORRT = True
    logger.info("✅ TensorRT and PyCUDA available")
except ImportError as e:
    logger.warning(f"TensorRT not available: {e}. Falling back to PyTorch.")

class Wav2LipAnimationStage:
    def __init__(self, avatar_path=AVATAR_IMAGE_PATH):
        self.device = DEVICE
        self.img_size = 96
        self.wav2lip_batch_size = 32  # ← Reduced for longer videos (was 64)
        self.fps = 20.0
        self.pads = [0, 10, 0, 0]
        self.nosmooth = False

        self.face_detector = face_detection.FaceAlignment(
            face_detection.LandmarksType._2D,
            flip_input=False,
            device=self.device
        )
        self._precompute_avatar_face(avatar_path)

        self.use_trt = False
        if USE_TENSORRT and os.path.isfile(WAV2LIP_TRT_ENGINE):
            try:
                self._load_tensorrt_engine(WAV2LIP_TRT_ENGINE)
                self.use_trt = True
                logger.info("✅ Using TENSORRT FP16 for Wav2Lip")
            except Exception as e:
                logger.error(f"Failed to load TensorRT engine: {e}")
        
        if not self.use_trt:
            logger.info("⚠️ Using PyTorch Wav2Lip (fallback)")
            self._load_pytorch_model()

    def _load_pytorch_model(self):
        model = Wav2Lip()
        checkpoint = torch.load(WAV2LIP_CHECKPOINT_PATH, map_location=self.device)
        s = checkpoint["state_dict"]
        new_s = {k.replace('module.', ''): v for k, v in s.items()}
        model.load_state_dict(new_s, strict=False)
        model = model.to(self.device).eval()
        if self.device == "cuda":
            model = model.half()
        self.model = model

    def _load_tensorrt_engine(self, engine_path):
        logger.info(f"Loading TensorRT FP16 engine: {engine_path}")
        with open(engine_path, "rb") as f:
            runtime = trt.Runtime(trt.Logger(trt.Logger.WARNING))
            self.trt_engine = runtime.deserialize_cuda_engine(f.read())
        self.trt_context = self.trt_engine.create_execution_context()
        logger.info("✅ TensorRT FP16 engine loaded")

    def _get_smoothened_boxes(self, boxes, T=5):
        for i in range(len(boxes)):
            if i + T > len(boxes):
                window = boxes[len(boxes) - T:]
            else:
                window = boxes[i: i + T]
            boxes[i] = np.mean(window, axis=0)
        return boxes

    def _face_detect(self, images):
        batch_size = 16
        while True:
            try:
                predictions = []
                for i in range(0, len(images), batch_size):
                    batch = np.array(images[i:i + batch_size])
                    preds = self.face_detector.get_detections_for_batch(batch)
                    predictions.extend(preds)
                break
            except RuntimeError:
                if batch_size == 1:
                    raise RuntimeError("Image too large for GPU.")
                batch_size //= 2
        results = []
        pady1, pady2, padx1, padx2 = self.pads
        for rect, img in zip(predictions, images):
            if rect is None:
                raise ValueError("No face detected!")
            y1 = max(0, int(rect[1]) - pady1)
            y2 = min(img.shape[0], int(rect[3]) + pady2)
            x1 = max(0, int(rect[0]) - padx1)
            x2 = min(img.shape[1], int(rect[2]) + padx2)
            results.append([x1, y1, x2, y2])
        boxes = np.array(results)
        if not self.nosmooth:
            boxes = self._get_smoothened_boxes(boxes)
        return boxes

    def _precompute_avatar_face(self, avatar_path):
        full_frame = cv2.imread(avatar_path)
        if full_frame is None:
            raise ValueError(f"Avatar image not found: {avatar_path}")
        boxes = self._face_detect([full_frame])
        x1, y1, x2, y2 = boxes[0]
        face_crop = full_frame[y1:y2, x1:x2]
        self.precomputed_face = cv2.resize(face_crop, (self.img_size, self.img_size))
        self.precomputed_coords = (y1, y2, x1, x2)
        self.full_frame = full_frame
        self.frame_h, self.frame_w = full_frame.shape[:2]
        logger.info("✅ Avatar face precomputed")

    def _load_audio(self, audio_path):
        wav = audio.load_wav(audio_path, 16000)
        mel = audio.melspectrogram(wav)
        if np.isnan(mel.reshape(-1)).sum() > 0:
            mel = np.nan_to_num(mel, nan=1e-7)
        return mel

    def _split_mel_chunks(self, mel):
        mel_chunks = []
        mel_idx_multiplier = 80.0 / self.fps
        i = 0
        while True:
            start_idx = int(i * mel_idx_multiplier)
            if start_idx + 16 > mel.shape[1]:
                mel_chunks.append(mel[:, mel.shape[1] - 16:])
                break
            mel_chunks.append(mel[:, start_idx: start_idx + 16])
            i += 1
        return mel_chunks

    def _run_batch_pytorch(self, img_batch, mel_batch, frame_batch, coords_batch, video_writer):
        img_batch = np.asarray(img_batch)
        mel_batch = np.asarray(mel_batch)

        img_tensor = torch.FloatTensor(img_batch).permute(0, 3, 1, 2).to(self.device)
        mel_tensor = torch.FloatTensor(mel_batch).unsqueeze(1).to(self.device)

        if self.device == "cuda":
            img_tensor = img_tensor.half()
            mel_tensor = mel_tensor.half()

        with torch.no_grad():
            pred = self.model(mel_tensor, img_tensor)

        pred = pred.float().cpu().numpy().transpose(0, 2, 3, 1) * 255.0
        self._write_frames(pred, frame_batch, coords_batch, video_writer)

    def _run_batch_tensorrt(self, img_batch, mel_batch, frame_batch, coords_batch, video_writer):
        batch_size = len(img_batch)
        img_batch = np.asarray(img_batch, dtype=np.float32)
        mel_batch = np.asarray(mel_batch, dtype=np.float32)

        img_nchw = img_batch.transpose(0, 3, 1, 2)
        mel_nchw = mel_batch[:, np.newaxis, :, :]

        img_flat = img_nchw.ravel()
        mel_flat = mel_nchw.ravel()
        output_flat = np.empty(batch_size * 3 * 96 * 96, dtype=np.float32)

        img_gpu = cuda.mem_alloc(img_flat.nbytes)
        mel_gpu = cuda.mem_alloc(mel_flat.nbytes)
        out_gpu = cuda.mem_alloc(output_flat.nbytes)

        cuda.memcpy_htod(mel_gpu, mel_flat)
        cuda.memcpy_htod(img_gpu, img_flat)
        self.trt_context.execute_v2(bindings=[int(mel_gpu), int(img_gpu), int(out_gpu)])
        cuda.memcpy_dtoh(output_flat, out_gpu)

        pred = output_flat.reshape(batch_size, 3, 96, 96).transpose(0, 2, 3, 1) * 255.0
        self._write_frames(pred, frame_batch, coords_batch, video_writer)

        img_gpu.free()
        mel_gpu.free()
        out_gpu.free()

    def _write_frames(self, pred, frame_batch, coords_batch, video_writer):
        for p, f, c in zip(pred, frame_batch, coords_batch):
            y1, y2, x1, x2 = c
            p_resized = cv2.resize(p.astype(np.uint8), (x2 - x1, y2 - y1))
            f[y1:y2, x1:x2] = p_resized
            video_writer.write(f)

    def process(self, avatar_image_path: str, audio_path: str, output_path: str = OUTPUT_VIDEO_PATH):
        start_time = time.time()
        temp_dir = "temp_w2l"
        os.makedirs(temp_dir, exist_ok=True)
        temp_video = os.path.join(temp_dir, "result.avi")

        try:
            mel = self._load_audio(audio_path)
            mel_chunks = self._split_mel_chunks(mel)
            logger.info(f"[Wav2Lip] Generating {len(mel_chunks)} frames @ {self.fps} FPS")

            face = self.precomputed_face.copy()
            y1, y2, x1, x2 = self.precomputed_coords
            frame_h, frame_w = self.frame_h, self.frame_w

            out = cv2.VideoWriter(temp_video, cv2.VideoWriter_fourcc(*'DIVX'), self.fps, (frame_w, frame_h))

            img_batch, mel_batch, frame_batch, coords_batch = [], [], [], []
            for i, m in enumerate(mel_chunks):
                img_masked = face.copy()
                img_masked[self.img_size//2:, :] = 0
                img_combined = np.concatenate((img_masked, face), axis=2) / 255.0

                img_batch.append(img_combined)
                mel_batch.append(m)
                frame_batch.append(self.full_frame.copy())
                coords_batch.append((y1, y2, x1, x2))

                if len(img_batch) >= self.wav2lip_batch_size:
                    if self.use_trt:
                        self._run_batch_tensorrt(img_batch, mel_batch, frame_batch, coords_batch, out)
                    else:
                        self._run_batch_pytorch(img_batch, mel_batch, frame_batch, coords_batch, out)
                    img_batch, mel_batch, frame_batch, coords_batch = [], [], [], []

            if img_batch:
                if self.use_trt:
                    self._run_batch_tensorrt(img_batch, mel_batch, frame_batch, coords_batch, out)
                else:
                    self._run_batch_pytorch(img_batch, mel_batch, frame_batch, coords_batch, out)

            out.release()

            ffmpeg_cmd = [
                'ffmpeg', '-y',
                '-i', temp_video,
                '-i', audio_path,
                '-map', '0:v', '-map', '1:a',
                '-c:v', 'libx264', '-preset', 'ultrafast', '-tune', 'zerolatency', '-crf', '23',
                '-c:a', 'aac', '-b:a', '128k', '-ar', str(SAMPLE_RATE), '-ac', '1',
                '-pix_fmt', 'yuv420p', '-movflags', '+faststart', '-shortest',
                output_path
            ]
            subprocess.run(ffmpeg_cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, check=True)

            logger.info(f"[Wav2Lip] latency={time.time()-start_time:.2f}s | saved={output_path}")
            return output_path

        except Exception as e:
            logger.exception(f"[Wav2Lip] error: {e}")
            return None
        finally:
            if os.path.exists(temp_dir):
                shutil.rmtree(temp_dir)

# --- MAIN PIPELINE ---
class VideoBotPipeline:
    def __init__(self):
        logger.info(f"🚀 VideoBotPipeline | initialized on {DEVICE}")
        self.llm = LLMStage()
        self.tts = TTSStage()
        self.animator = Wav2LipAnimationStage()

    def run(self):
        logger.info("✅ Ready. Type 'quit' to exit.")
        try:
            while True:
                user_input = input("\nYou: ").strip()
                if not user_input or user_input.lower() in {"quit", "exit"}:
                    break

                logger.info(f"Processing: '{user_input[:30]}...'")
                t0 = time.time()

                response = self.llm.process(user_input)
                audio_file = self.tts.process(response)
                if audio_file:
                    video_file = self.animator.process(AVATAR_IMAGE_PATH, audio_file, OUTPUT_VIDEO_PATH)
                    if video_file:
                        logger.info(f"✅ Output: {video_file}")

                total = time.time() - t0
                logger.info(f"⏱️ TOTAL LATENCY: {total:.2f}s")

        except KeyboardInterrupt:
            logger.info("Interrupted")
        finally:
            logger.info("Shutdown complete")

def ensure_files_exist():
    required = [
        AVATAR_IMAGE_PATH,
        WAV2LIP_CHECKPOINT_PATH,
        PIPER_MODEL_PATH,
        PIPER_CONFIG_PATH
    ]
    for f in required:
        if not os.path.isfile(f):
            logger.error(f"Missing: {f}")
            sys.exit(1)

if __name__ == "__main__":
    ensure_files_exist()
    VideoBotPipeline().run()